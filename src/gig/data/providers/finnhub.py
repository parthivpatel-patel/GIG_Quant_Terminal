"""Finnhub free tier — quotes, general news, earnings calendar."""

from __future__ import annotations

import json
from datetime import UTC, date, datetime

import pandas as pd

from gig.data.http import GetFn, http_get
from gig.nlp.news import headlines_to_frame


def parse_quote(payload: str, symbol: str) -> dict:
    q = json.loads(payload)
    return {
        "ts": datetime.fromtimestamp(q["t"], tz=UTC) if q.get("t") else datetime.now(UTC),
        "symbol": symbol.upper(),
        "bid": float("nan"),
        "ask": float("nan"),
        "last": float(q.get("c") or float("nan")),
        "source": "finnhub",
    }


def parse_earnings(payload: str) -> pd.DataFrame:
    data = json.loads(payload)
    rows = data if isinstance(data, list) else data.get("earningsCalendar") or []
    recs = []
    for r in rows:
        recs.append(
            {
                "event_date": date.fromisoformat(str(r.get("date"))) if r.get("date") else None,
                "symbol": str(r.get("symbol") or "").upper(),
                "event_type": "earnings",
                "extra": json.dumps({"epsEstimate": r.get("epsEstimate"), "hour": r.get("hour")}),
            }
        )
    df = pd.DataFrame(recs)
    return df.dropna(subset=["event_date", "symbol"]) if not df.empty else pd.DataFrame(
        columns=["event_date", "symbol", "event_type", "extra"]
    )


def fetch_quotes(symbols: list[str], token: str, *, get: GetFn = http_get) -> pd.DataFrame:
    if not token:
        raise RuntimeError("FINNHUB_API_KEY is empty — free key at https://finnhub.io")
    recs = []
    for sym in symbols:
        url = f"https://finnhub.io/api/v1/quote?symbol={sym}&token={token}"
        try:
            recs.append(parse_quote(get(url, None), sym))
        except Exception:
            continue
    return pd.DataFrame(recs) if recs else pd.DataFrame(
        columns=["ts", "symbol", "bid", "ask", "last", "source"]
    )


def fetch_earnings(token: str, start: date, end: date, *, get: GetFn = http_get) -> pd.DataFrame:
    url = (
        "https://finnhub.io/api/v1/calendar/earnings"
        f"?from={start.isoformat()}&to={end.isoformat()}&token={token}"
    )
    return parse_earnings(get(url, None))


def fetch_news(token: str, *, get: GetFn = http_get) -> pd.DataFrame:
    url = f"https://finnhub.io/api/v1/news?category=general&token={token}"
    raw = json.loads(get(url, None))
    rows = []
    for it in raw if isinstance(raw, list) else []:
        rows.append(
            {
                "symbol": "",
                "title": it.get("headline") or it.get("summary") or "",
                "published": it.get("datetime"),
                "publisher": it.get("source") or "finnhub",
                "link": it.get("url") or "",
            }
        )
    return headlines_to_frame(rows)
