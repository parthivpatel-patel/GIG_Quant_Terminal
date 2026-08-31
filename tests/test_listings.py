from __future__ import annotations

from datetime import date

import pandas as pd

from gig.data.listings import (
    cap_bucket,
    filter_common_stock,
    parse_nasdaq_listed,
    parse_other_listed,
    to_yahoo_symbol,
)
from gig.data.universe_spec import load_universe, universe_mode
from gig.store import Store

NASDAQ_SAMPLE = """Symbol|Security Name|Market Category|Test Issue|Financial Status|Round Lot Size|ETF|NextShares
AAPL|Apple Inc. - Common Stock|Q|N|N|100|N|N
QQQ|Invesco QQQ Trust|G|N|N|100|Y|N
TEST|Fake Test Issue|Q|Y|N|100|N|N
ZWZZT|NASDAQ TEST STOCK|Q|Y|N|100|N|N
"""

OTHER_SAMPLE = """ACT Symbol|Security Name|Exchange|CQS Symbol|ETF|Round Lot Size|Test Issue|NASDAQ Symbol
IBM|International Business Machines Corporation Common Stock|N|IBM|N|100|N|IBM
BRK.B|Berkshire Hathaway Inc. Class B|N|BRK.B|N|100|N|BRK.B
SPY|SPDR S&P 500 ETF Trust|P|SPY|Y|100|N|SPY
XYZWS|XYZ Inc. Warrant|N|XYZWS|N|100|N|XYZWS
"""


def test_to_yahoo_class_shares():
    assert to_yahoo_symbol("BRK.B") == "BRK-B"


def test_filter_drops_etf_warrant_test():
    raw = pd.concat(
        [parse_nasdaq_listed(NASDAQ_SAMPLE), parse_other_listed(OTHER_SAMPLE)],
        ignore_index=True,
    )
    out = filter_common_stock(raw)
    syms = set(out["yahoo_symbol"])
    assert "AAPL" in syms
    assert "IBM" in syms
    assert "BRK-B" in syms
    assert "QQQ" not in syms
    assert "SPY" not in syms
    assert "TEST" not in syms
    assert "ZWZZT" not in syms
    assert "XYZWS" not in syms


def test_cap_buckets():
    assert cap_bucket(80_000_000) == "large"
    assert cap_bucket(8_000_000) == "mid"
    assert cap_bucket(2_000_000) == "small"
    assert cap_bucket(100_000) == "micro"


def test_listings_roundtrip(tmp_path):
    store = Store(tmp_path / "q.duckdb")
    store.init()
    raw = parse_nasdaq_listed(NASDAQ_SAMPLE)
    other = parse_other_listed(OTHER_SAMPLE)
    df = filter_common_stock(pd.concat([raw, other], ignore_index=True))
    df["cap_bucket"] = "unknown"
    df["asof_date"] = date(2024, 1, 2)
    n = store.upsert_listings(df)
    assert n >= 3
    loaded = store.load_listings()
    assert "AAPL" in set(loaded["yahoo_symbol"])
    assert loaded["is_etf"].sum() == 0


def test_universe_yaml_us_listed(tmp_path):
    p = tmp_path / "u.yaml"
    p.write_text("source: us_listed\nexclude_etf: true\n", encoding="utf-8")
    assert universe_mode(p) == "us_listed"
    mega = load_universe(p)
    assert "AAPL" in mega


def test_update_cap_buckets(tmp_path):
    store = Store(tmp_path / "q2.duckdb")
    store.init()
    df = filter_common_stock(parse_nasdaq_listed(NASDAQ_SAMPLE))
    df["cap_bucket"] = "unknown"
    df["asof_date"] = date(2024, 1, 2)
    store.upsert_listings(df)
    store.update_cap_buckets(pd.Series({"AAPL": "large"}))
    loaded = store.load_listings()
    row = loaded[loaded["yahoo_symbol"] == "AAPL"].iloc[0]
    assert row["cap_bucket"] == "large"


def test_universe_yaml_mega_not_confused_with_source():
    from pathlib import Path

    assert universe_mode(Path("configs/universe.yaml")) in {"us_listed", "mega"}
