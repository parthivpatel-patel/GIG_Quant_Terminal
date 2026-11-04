"""Explicit, bounded acquisition from the fixed integrated public endpoints.

Only URLs in the source registry are fetched, only on an explicit request, only
over HTTPS, within size/time bounds and the refresh/backoff schedule. A failure
is recorded and never replaced with fabricated observations.
"""
from __future__ import annotations

import hashlib
import json
import urllib.error
import urllib.request
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

from gig.intel.events import normalize_eonet, normalize_usgs
from gig.intel.sources import FETCHABLE, SOURCES
from gig.intel.store import IntelStore

Opener = Callable[[str, int, int], tuple[int, bytes]]
NORMALIZERS = {"USGS_EARTHQUAKES": normalize_usgs, "NASA_EONET": normalize_eonet}


def _default_opener(url: str, timeout: int, max_bytes: int) -> tuple[int, bytes]:
    request = urllib.request.Request(url, headers={"User-Agent": "GIG-Quant-Terminal-research/1.0",
                                                   "Accept": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310 - fixed HTTPS registry URL
        final = response.geturl()
        if not final.startswith("https://") or final.split("/")[2] != url.split("/")[2]:
            raise ValueError("redirect left the approved host")
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
    try:
        status, body = (opener or _default_opener)(source["url"], 20, source["max_bytes"])
        if status != 200:
            raise urllib.error.HTTPError(source["url"], status, "unexpected status", None, None)  # type: ignore[arg-type]
        raw_sha = hashlib.sha256(body).hexdigest()
        records = NORMALIZERS[source_id](json.loads(body), acquired_at=attempted)
    except urllib.error.HTTPError as exc:
        store.log_fetch(source_id, attempted_at=attempted, ok=False, http_status=exc.code,
                        error_code="HTTP_STATUS", raw_sha256=None, received=0, new_revisions=0)
        return {"source_id": source_id, "state": "FAILED", "error_code": "HTTP_STATUS", "http_status": exc.code}
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        code = "TIMEOUT_OR_NETWORK" if isinstance(exc, OSError) else "INVALID_PAYLOAD"
        store.log_fetch(source_id, attempted_at=attempted, ok=False, http_status=None,
                        error_code=code, raw_sha256=None, received=0, new_revisions=0)
        return {"source_id": source_id, "state": "FAILED", "error_code": code}
    added = store.append(records, classification="PROVIDER_OBSERVATION", ingested_at=datetime.now(UTC).isoformat())
    store.log_fetch(source_id, attempted_at=attempted, ok=True, http_status=status, error_code=None,
                    raw_sha256=raw_sha, received=len(records), new_revisions=added)
    return {"source_id": source_id, "state": "ACQUIRED", "received": len(records), "new_revisions": added,
            "raw_sha256": raw_sha, "acquired_at": attempted}
