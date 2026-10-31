"""Universe spec: mega-cap YAML or the full US listed tape from DuckDB."""

from __future__ import annotations

from pathlib import Path

import yaml

# GICS-style sectors used when a vendor lookup is unavailable this session.
DEFAULT_SECTORS: dict[str, str] = {
    "AAPL": "tech",
    "MSFT": "tech",
    "NVDA": "tech",
    "AVGO": "tech",
    "ORCL": "tech",
    "CRM": "tech",
    "ADBE": "tech",
    "AMD": "tech",
    "GOOGL": "communication",
    "META": "communication",
    "NFLX": "communication",
    "DIS": "communication",
    "AMZN": "consumer",
    "TSLA": "consumer",
    "HD": "consumer",
    "MCD": "consumer",
    "NKE": "consumer",
    "SBUX": "consumer",
    "WMT": "staples",
    "COST": "staples",
    "PG": "staples",
    "KO": "staples",
    "PEP": "staples",
    "JPM": "financials",
    "BAC": "financials",
    "GS": "financials",
    "V": "financials",
    "MA": "financials",
    "BRK-B": "financials",
    "UNH": "health",
    "JNJ": "health",
    "LLY": "health",
    "ABBV": "health",
    "MRK": "health",
    "PFE": "health",
    "XOM": "energy",
    "CVX": "energy",
    "COP": "energy",
    "CAT": "industrials",
    "HON": "industrials",
    "GE": "industrials",
    "UNP": "industrials",
    "BA": "industrials",
    "LIN": "materials",
    "APD": "materials",
    "NEE": "utilities",
    "AMT": "real_estate",
    "SPY": "etf_broad",
    "QQQ": "etf_broad",
    "IWM": "etf_broad",
}


def default_symbols() -> list[str]:
    return list(DEFAULT_SECTORS)


def universe_mode(path: Path | None = None) -> str:
    """'us_listed' (all NYSE/Nasdaq/AMEX commons) or 'mega' (YAML sleeve)."""
    if path is None or not path.exists():
        return "mega"
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    src = str(raw.get("source") or "").lower()
    if src in {"us_listed", "us", "all"}:
        return "us_listed"
    return "mega"


def load_universe(path: Path | None = None) -> dict[str, str]:
    if path is None or not path.exists():
        return dict(DEFAULT_SECTORS)
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if str(raw.get("source") or "").lower() in {"us_listed", "us", "all"}:
        return dict(DEFAULT_SECTORS)
    names = raw.get("symbols") or raw
    if isinstance(names, list):
        return {str(s): DEFAULT_SECTORS.get(str(s), "unknown") for s in names}
    if isinstance(names, dict):
        keys = {str(k) for k in names}
        if keys <= {"source", "exclude_etf", "exclude_test"}:
            return dict(DEFAULT_SECTORS)
        return {str(k): str(v) for k, v in names.items() if str(k) not in {"source", "exclude_etf", "exclude_test"}}
    return dict(DEFAULT_SECTORS)


def resolve_universe(path: Path | None, store=None, *, refresh: bool = False) -> dict[str, str]:
    """Symbol → sector. Full US tape comes from `listings` after a refresh."""
    if universe_mode(path) != "us_listed":
        return load_universe(path)
    if store is None:
        return dict(DEFAULT_SECTORS)
    from gig.data.listings import refresh_us_listings

    listings = store.load_listings()
    if refresh or listings is None or listings.empty:
        listings = refresh_us_listings(store)
    if listings is None or listings.empty:
        return dict(DEFAULT_SECTORS)
    col = "yahoo_symbol" if "yahoo_symbol" in listings.columns else "symbol"
    out: dict[str, str] = {}
    for row in listings.itertuples(index=False):
        sym = str(getattr(row, col)).upper()
        out[sym] = DEFAULT_SECTORS.get(sym, "unknown")
    return out
