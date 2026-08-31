"""Tiny HTTP helper. Providers inject `get` in tests so CI never hits the network."""

from __future__ import annotations

import gzip
from collections.abc import Callable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

GetFn = Callable[[str, dict[str, str] | None], str]


def _redact(url: str) -> str:
    for key in ("token=", "api_key=", "apikey="):
        if key in url.lower():
            i = url.lower().find(key)
            return url[: i + len(key)] + "***"
    return url


def decode_http_body(raw: bytes) -> str:
    if raw[:2] == b"\x1f\x8b":
        raw = gzip.decompress(raw)
    return raw.decode("utf-8")


def http_get(url: str, headers: dict[str, str] | None = None, timeout: float = 30.0) -> str:
    hdrs = {"User-Agent": "GIG-Trading-Algorithm/0.1"}
    if headers:
        hdrs.update(headers)
    req = Request(url, headers=hdrs)
    try:
        with urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
        return decode_http_body(raw)
    except HTTPError as exc:
        raise RuntimeError(f"HTTP {exc.code} for {_redact(url)}") from exc
    except URLError as exc:
        raise RuntimeError(f"network error for {_redact(url)}: {exc.reason}") from exc
