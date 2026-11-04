"""Explicit, bounded acquisition from the fixed integrated public endpoints.

Only URLs in the source registry are fetched, only on an explicit request, only
over HTTPS, within size/time bounds and the refresh/backoff schedule. Redirects
are refused before they are followed unless they stay on the registry host over
HTTPS. A failure is recorded and never replaced with fabricated observations; a
payload with any invalid record is rejected whole rather than partially admitted.
"""
from __future__ import annotations

import hashlib
import json
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

from gig.intel.events import normalize_eonet, normalize_usgs
from gig.intel.sources import FETCHABLE, SOURCES
from gig.intel.store import IntelStore

Opener = Callable[[str, int, int], tuple[int, bytes]]
NORMALIZERS = {"USGS_EARTHQUAKES": normalize_usgs, "NASA_EONET": normalize_eonet}


class RedirectRejected(Exception):
    pass


class _SameHostRedirect(urllib.request.HTTPRedirectHandler):
    def __init__(self, host: str):
        self.host = host

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # type: ignore[no-untyped-def]
        target = urllib.parse.urlsplit(urllib.parse.urljoin(req.full_url, newurl))
        if target.scheme != "https" or target.hostname != self.host:
            raise RedirectRejected("redirect left the approved host")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _reject_constant(value: str) -> Any:
    raise ValueError(f"non-finite JSON constant {value}")


def _default_opener(url: str, timeout: int, max_bytes: int) -> tuple[int, bytes]:
    host = urllib.parse.urlsplit(url).hostname or ""
    opener = urllib.request.build_opener(_SameHostRedirect(host))
    request = urllib.request.Request(url, headers={"User-Agent": "GIG-Quant-Terminal-research/1.0",
                                                   "Accept": "application/json"})
    with opener.open(request, timeout=timeout) as response:  # noqa: S310 - fixed HTTPS registry URL
        final = urllib.parse.urlsplit(response.geturl())
        if final.scheme != "https" or final.hostname != host:
            raise RedirectRejected("redirect left the approved host")
        body = response.read(max_bytes + 1)
        if len(body) > max_bytes:
            raise ValueError("response exceeds the size bound")
        return response.status, body


def refresh(store: IntelStore, source_id: str, *, now: datetime | None = None,
            opener: Opener | None = None, force_schedule: bool = False) -> dict[str, Any]:
    if source_id not in FETCHABLE:
        raise ValueError("source is not an integrated fetchable provider")
    source = SOURCES[source_id]
    now = now or datetime.now(UTC)
    allowed = store.next_allowed(source_id)
    if allowed is not None and now < allowed and not force_schedule:
        return {"source_id": source_id, "state": "DEFERRED", "next_allowed_at": allowed.isoformat(),
                "reason": "Refresh interval or failure backoff in effect."}
    attempted = now.isoformat()

    def failed(code: str, http_status: int | None = None) -> dict[str, Any]:
        store.log_fetch(source_id, attempted_at=attempted, ok=False, http_status=http_status,
                        error_code=code, raw_sha256=None, received=0, new_revisions=0)
        result: dict[str, Any] = {"source_id": source_id, "state": "FAILED", "error_code": code}
        if http_status is not None:
            result["http_status"] = http_status
        return result

    try:
        status, body = (opener or _default_opener)(source["url"], 20, source["max_bytes"])
    except urllib.error.HTTPError as exc:
        return failed("HTTP_STATUS", exc.code)
    except RedirectRejected:
        return failed("REDIRECT_REJECTED")
    except ValueError:
        return failed("INVALID_PAYLOAD")
    except OSError:
        return failed("TIMEOUT_OR_NETWORK")
    if status != 200:
        return failed("HTTP_STATUS", status)
    raw_sha = hashlib.sha256(body).hexdigest()
    try:
        records = NORMALIZERS[source_id](json.loads(body, parse_constant=_reject_constant), acquired_at=attempted)
    except (ValueError, TypeError, KeyError, AttributeError, IndexError, OverflowError, RecursionError):
        return failed("INVALID_PAYLOAD")
    added = store.append(records, classification="PROVIDER_OBSERVATION", ingested_at=datetime.now(UTC).isoformat())
    store.log_fetch(source_id, attempted_at=attempted, ok=True, http_status=status, error_code=None,
                    raw_sha256=raw_sha, received=len(records), new_revisions=added)
    return {"source_id": source_id, "state": "ACQUIRED", "received": len(records), "new_revisions": added,
            "raw_sha256": raw_sha, "acquired_at": attempted}
