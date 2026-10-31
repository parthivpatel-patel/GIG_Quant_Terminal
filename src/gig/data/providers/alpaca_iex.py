"""Alpaca IEX snapshots — free live quotes (IEX tape, not full SIP)."""

from __future__ import annotations

import json
from datetime import UTC, datetime

import pandas as pd

from gig.data.http import GetFn, http_get


def parse_snapshots(payload: str, source: str = "alpaca_iex") -> pd.DataFrame:
    raw = json.loads(payload)
    # latest quotes endpoint: {"quotes": {"AAPL": {"bp": ..., "ap": ..., "t": ...}}}
    quotes = raw.get("quotes") or raw.get("snapshots") or raw
    if not isinstance(quotes, dict):
        return pd.DataFrame(columns=["ts", "symbol", "bid", "ask", "last", "source"])
    recs = []
    now = datetime.now(UTC)
    for sym, blob in quotes.items():
        if not isinstance(blob, dict):
            continue
        quote = blob.get("latestQuote") or blob.get("quote") or blob
        trade = blob.get("latestTrade") or blob.get("trade") or {}
        recs.append(
            {
                "ts": now,
                "symbol": str(sym).upper(),
                "bid": _num(quote.get("bp") or quote.get("bid_price")),
                "ask": _num(quote.get("ap") or quote.get("ask_price")),
                "last": _num(trade.get("p") or trade.get("price") or quote.get("p")),
                "source": source,
            }
        )
    return pd.DataFrame(recs)


def _num(x) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def fetch_latest_quotes(
    symbols: list[str],
    api_key: str,
    secret: str,
    data_url: str = "https://data.alpaca.markets",
    *,
    get: GetFn | None = None,
) -> pd.DataFrame:
    if not api_key or not secret:
        raise RuntimeError("Alpaca keys missing — free paper account at https://alpaca.markets")
    getter = get or http_get
    batch = ",".join(symbols[:50])
    url = f"{data_url.rstrip('/')}/v2/stocks/snapshots?symbols={batch}&feed=iex"
    headers = {
        "APCA-API-KEY-ID": api_key,
        "APCA-API-SECRET-KEY": secret,
        "Accept": "application/json",
    }
    return parse_snapshots(getter(url, headers), source="alpaca_iex")
