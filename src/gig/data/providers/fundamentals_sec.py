"""
SEC companyfacts → point-in-time book equity / shares.

Uses the XBRL companyfacts API (free). ``asof_date`` is the *filing* date, not
the fiscal period end — that is what keeps the join honest. Empty / missing
tags are skipped; nothing is invented.
"""

from __future__ import annotations

import json
import time
from datetime import date
from typing import Any

import pandas as pd

from gig.data.http import GetFn, http_get

COMPANYFACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"

# Prefer pure stockholders' equity; fall back to broader equity tags.
EQUITY_TAGS = (
    "StockholdersEquity",
    "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
    "CommonStockholdersEquity",
)
SHARES_TAGS = (
    "CommonStockSharesOutstanding",
    "EntityCommonStockSharesOutstanding",
    "WeightedAverageNumberOfSharesOutstandingBasic",
)


def _usd_points(facts: dict[str, Any], tags: tuple[str, ...]) -> list[dict[str, Any]]:
    us_gaap = (facts.get("facts") or {}).get("us-gaap") or {}
    dei = (facts.get("facts") or {}).get("dei") or {}
    pools = [us_gaap, dei]
    out: list[dict[str, Any]] = []
    for tag in tags:
        block = None
        for pool in pools:
            if tag in pool:
                block = pool[tag]
                break
        if not block:
            continue
        units = block.get("units") or {}
        series = units.get("USD") or units.get("shares") or units.get("pure") or []
        for row in series:
            filed = row.get("filed")
            val = row.get("val")
            if not filed or val is None:
                continue
            try:
                asof = date.fromisoformat(str(filed)[:10])
                value = float(val)
            except (TypeError, ValueError):
                continue
            if not (value == value):  # NaN
                continue
            out.append(
                {
                    "asof_date": asof,
                    "value": value,
                    "form": str(row.get("form") or ""),
                    "end": str(row.get("end") or ""),
                    "tag": tag,
                }
            )
        if out:
            break  # first matching tag wins
    return out


def parse_companyfacts(payload: str, symbol: str) -> pd.DataFrame:
    """Extract book_equity + shares_outstanding rows for one issuer."""
    data = json.loads(payload)
    equity = _usd_points(data, EQUITY_TAGS)
    shares = _usd_points(data, SHARES_TAGS)
    rows: list[dict[str, Any]] = []
    for e in equity:
        rows.append(
            {
                "asof_date": e["asof_date"],
                "symbol": symbol.upper(),
                "field": "book_equity",
                "value": e["value"],
                "source": f"sec:{e['tag']}:{e['form']}",
            }
        )
    for s in shares:
        rows.append(
            {
                "asof_date": s["asof_date"],
                "symbol": symbol.upper(),
                "field": "shares_outstanding",
                "value": s["value"],
                "source": f"sec:{s['tag']}:{s['form']}",
            }
        )
    if not rows:
        return pd.DataFrame(columns=["asof_date", "symbol", "field", "value", "source"])
    frame = pd.DataFrame(rows)
    return frame.drop_duplicates(subset=["asof_date", "symbol", "field"], keep="last")


def fetch_book_equity(
    symbols: list[str],
    cik_map: dict[str, str],
    user_agent: str,
    *,
    get: GetFn = http_get,
    pause: float = 0.15,
    max_symbols: int = 80,
) -> pd.DataFrame:
    """
    Pull companyfacts for up to ``max_symbols`` names.

    Caps keep the first ingest under a few minutes; re-runs are incremental
    only in the sense that upsert replaces those symbols' rows.
    """
    frames: list[pd.DataFrame] = []
    headers = {"User-Agent": user_agent}
    seen = 0
    for sym in symbols:
        if seen >= max_symbols:
            break
        cik = cik_map.get(str(sym).upper())
        if not cik:
            continue
        url = COMPANYFACTS_URL.format(cik=cik)
        try:
            frame = parse_companyfacts(get(url, headers), str(sym).upper())
        except Exception:
            continue
        if not frame.empty:
            frames.append(frame)
            seen += 1
        if pause:
            time.sleep(pause)
    if not frames:
        return pd.DataFrame(columns=["asof_date", "symbol", "field", "value", "source"])
    return pd.concat(frames, ignore_index=True)
