"""US listed common stocks from Nasdaq Trader (NYSE / Nasdaq / AMEX / Arca).

This is the current tape, not CRSP. Delisted names are missing, so a backtest
on this universe has survivorship bias. Still the right live scan: every
common stock listed today, small/mid/large, excluding ETFs and test issues.
"""

from __future__ import annotations

import re
from datetime import date

import pandas as pd

from gig.data.http import GetFn, http_get

NASDAQ_LISTED_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"
OTHER_LISTED_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt"

EXCHANGE_MAP = {
    "A": "AMEX",
    "N": "NYSE",
    "P": "ARCA",
    "Z": "BATS",
    "V": "IEX",
    "Q": "NASDAQ",
    "G": "NASDAQ",
    "S": "NASDAQ",
}

# Structured products, not common equity. ADRs and REITs stay in.
_SKIP_NAME = re.compile(
    r"(ETF|ETN|ETP|Exchange[- ]Traded|Warrant|Right[s]?\b|Units?\b|Preferred|"
    r"Closed[- ]End|Note[s]?\b|Debenture)",
    re.I,
)

CAP_BREAKS = (
    ("large", 50_000_000.0),
    ("mid", 5_000_000.0),
    ("small", 1_000_000.0),
)


def to_yahoo_symbol(symbol: str) -> str:
    """Nasdaq/NYSE class shares use '.' ; Yahoo uses '-'."""
    return str(symbol).strip().upper().replace("/", "-").replace(".", "-")


def parse_nasdaq_listed(text: str) -> pd.DataFrame:
    recs = []
    for line in text.splitlines():
        if not line or line.startswith("File Creation") or line.startswith("Symbol|"):
            continue
        parts = line.split("|")
        if len(parts) < 7:
            continue
        recs.append(
            {
                "symbol": parts[0].strip().upper(),
                "name": parts[1].strip(),
                "exchange": "NASDAQ",
                "is_etf": _flag(parts[6]),
                "is_test": _flag(parts[3]) if len(parts) > 3 else False,
            }
        )
    return pd.DataFrame(recs)


def parse_other_listed(text: str) -> pd.DataFrame:
    recs = []
    for line in text.splitlines():
        if not line or line.startswith("File Creation") or line.startswith("ACT Symbol"):
            continue
        parts = line.split("|")
        if len(parts) < 7:
            continue
        recs.append(
            {
                "symbol": parts[0].strip().upper(),
                "name": parts[1].strip(),
                "exchange": EXCHANGE_MAP.get(parts[2].strip().upper(), parts[2].strip().upper()),
                "is_etf": _flag(parts[4]),
                "is_test": _flag(parts[6]),
            }
        )
    return pd.DataFrame(recs)


def filter_common_stock(raw: pd.DataFrame) -> pd.DataFrame:
    if raw is None or raw.empty:
        return pd.DataFrame(
            columns=["symbol", "yahoo_symbol", "exchange", "name", "is_etf", "is_test"]
        )
    df = raw.copy()
    df["symbol"] = df["symbol"].astype(str).str.upper().str.strip()
    df = df[df["symbol"].str.len() > 0]
    df = df[~df["symbol"].str.contains(r"[\^$]", regex=True)]
    etf = df["is_etf"].fillna(False).astype(bool)
    test = df["is_test"].fillna(False).astype(bool)
    skip_name = df["name"].fillna("").map(lambda n: bool(_SKIP_NAME.search(str(n))))
    df = df[~etf & ~test & ~skip_name]
    df["yahoo_symbol"] = df["symbol"].map(to_yahoo_symbol)
    df = df.drop_duplicates("yahoo_symbol", keep="first")
    return df.reset_index(drop=True)


def cap_bucket(adv_usd: float) -> str:
    if adv_usd >= CAP_BREAKS[0][1]:
        return "large"
    if adv_usd >= CAP_BREAKS[1][1]:
        return "mid"
    if adv_usd >= CAP_BREAKS[2][1]:
        return "small"
    return "micro"


def cap_buckets_from_adv(adv: pd.Series) -> pd.Series:
    return adv.fillna(0.0).map(cap_bucket)


def fetch_us_listings(*, get: GetFn | None = None) -> pd.DataFrame:
    getter = get or http_get
    nasdaq = parse_nasdaq_listed(getter(NASDAQ_LISTED_URL, None))
    other = parse_other_listed(getter(OTHER_LISTED_URL, None))
    raw = pd.concat([nasdaq, other], ignore_index=True)
    return filter_common_stock(raw)


def refresh_us_listings(store, *, get: GetFn | None = None, asof: date | None = None) -> pd.DataFrame:
    listings = fetch_us_listings(get=get)
    listings["cap_bucket"] = "unknown"
    listings["asof_date"] = asof or date.today()
    store.upsert_listings(listings)
    return listings


def _flag(value: str) -> bool:
    return str(value).strip().upper()[:1] == "Y"
