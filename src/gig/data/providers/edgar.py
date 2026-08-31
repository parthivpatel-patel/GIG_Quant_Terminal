"""SEC EDGAR — free 8-K / 10-K index. Requires a User-Agent with a real email."""

from __future__ import annotations

import json
import time
from datetime import UTC, date, datetime

import pandas as pd

from gig.data.http import GetFn, http_get

TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"

# Conservative event-study signs for Form 8-K items (unsigned 2.02/7.01/8.01/9.01).
ITEM_SCORES: dict[str, float] = {
    "1.03": -1.00,  # bankruptcy / receivership
    "4.02": -0.80,  # non-reliance on prior financials
    "3.01": -0.75,  # delisting / listing-standard notice
    "2.06": -0.50,  # material impairment
    "2.05": -0.40,  # exit / disposal costs
    "2.04": -0.35,  # obligation acceleration
    "1.02": -0.30,  # agreement termination
    "4.01": -0.30,  # accountant change
    "5.01": -0.25,  # change in control
    "5.02": -0.15,  # officer / director departure
    "3.02": -0.10,  # unregistered equity sale
    "2.03": -0.10,  # new direct financial obligation
    "2.01": 0.10,  # acquisition / disposition completed
    "1.01": 0.10,  # material definitive agreement
}


def score_8k_items(items: str) -> float:
    """Most-extreme signed item; unsigned 8-K (earnings/FD/exhibits-only) = presence."""
    parts = [p.strip() for p in (items or "").replace(";", ",").split(",") if p.strip()]
    if not parts:
        return 0.2
    material = []
    for part in parts:
        key = part.split()[0] if part else ""
        if key.startswith("9."):
            continue
        material.append(ITEM_SCORES.get(key, 0.0))
    if not material:
        return 0.2
    if all(s == 0.0 for s in material):
        return 0.2
    return float(max(material, key=lambda s: abs(s)))


def parse_tickers(payload: str) -> dict[str, str]:
    """Map ticker → zero-padded 10-digit CIK."""
    raw = json.loads(payload)
    out: dict[str, str] = {}
    for row in raw.values():
        ticker = str(row.get("ticker") or "").upper()
        cik = int(row.get("cik_str") or 0)
        if ticker and cik:
            out[ticker] = f"{cik:010d}"
    return out


def parse_submissions(payload: str, symbol: str, form_filter: tuple[str, ...] = ("8-K", "10-K", "10-Q")) -> pd.DataFrame:
    data = json.loads(payload)
    recent = (data.get("filings") or {}).get("recent") or {}
    forms = recent.get("form") or []
    dates = recent.get("filingDate") or []
    acc = recent.get("accessionNumber") or []
    recs = []
    for i, form in enumerate(forms):
        if form not in form_filter:
            continue
        filed = dates[i] if i < len(dates) else None
        if not filed:
            continue
        asof = date.fromisoformat(filed)
        items_list = recent.get("items") or []
        items = items_list[i] if i < len(items_list) else ""
        sent = score_8k_items(str(items)) if str(form).upper().startswith("8-K") else 0.0
        title = f"{form} {items} {symbol}".strip() if items else f"{form} {symbol}"
        recs.append(
            {
                "filed_at": datetime.fromisoformat(filed).replace(tzinfo=UTC)
                if "T" in filed
                else datetime.combine(asof, datetime.min.time(), tzinfo=UTC),
                "asof_date": asof,
                "symbol": symbol,
                "form": form,
                "accession": acc[i] if i < len(acc) else "",
                "title": title,
                "sentiment": sent,
            }
        )
    return pd.DataFrame(recs) if recs else pd.DataFrame(
        columns=["filed_at", "asof_date", "symbol", "form", "accession", "title", "sentiment"]
    )


def fetch_cik_map(*, get: GetFn = http_get, user_agent: str) -> dict[str, str]:
    body = get(TICKERS_URL, {"User-Agent": user_agent})
    return parse_tickers(body)


def fetch_filings(
    symbols: list[str],
    cik_map: dict[str, str],
    user_agent: str,
    *,
    get: GetFn = http_get,
    pause: float = 0.12,
) -> pd.DataFrame:
    frames = []
    headers = {"User-Agent": user_agent}
    for sym in symbols:
        cik = cik_map.get(sym.upper())
        if not cik:
            continue
        url = f"https://data.sec.gov/submissions/CIK{cik}.json"
        try:
            frames.append(parse_submissions(get(url, headers), sym))
        except Exception:
            continue
        if pause:
            time.sleep(pause)
    if not frames:
        return pd.DataFrame(
            columns=["filed_at", "asof_date", "symbol", "form", "accession", "title", "sentiment"]
        )
    return pd.concat(frames, ignore_index=True)
