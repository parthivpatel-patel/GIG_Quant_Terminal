"""SEC companyfacts parser stays offline and PIT-honest (filed date)."""

from datetime import date

from gig.data.providers.fundamentals_sec import parse_companyfacts


SAMPLE = """
{
  "facts": {
    "us-gaap": {
      "StockholdersEquity": {
        "units": {
          "USD": [
            {"end": "2023-12-31", "val": 1000000000, "filed": "2024-02-01", "form": "10-K"},
            {"end": "2024-06-30", "val": 1100000000, "filed": "2024-08-01", "form": "10-Q"}
          ]
        }
      },
      "CommonStockSharesOutstanding": {
        "units": {
          "shares": [
            {"end": "2023-12-31", "val": 500000000, "filed": "2024-02-01", "form": "10-K"},
            {"end": "2024-06-30", "val": 510000000, "filed": "2024-08-01", "form": "10-Q"}
          ]
        }
      }
    }
  }
}
"""


def test_parse_companyfacts_uses_filed_date():
    df = parse_companyfacts(SAMPLE, "AAPL")
    assert set(df["field"]) == {"book_equity", "shares_outstanding"}
    assert set(df["symbol"]) == {"AAPL"}
    assert date(2024, 2, 1) in set(df["asof_date"])
    assert date(2023, 12, 31) not in set(df["asof_date"])  # period end is not asof
    book = df.loc[df["field"] == "book_equity"].sort_values("asof_date")
    assert float(book.iloc[-1]["value"]) == 1_100_000_000
    # Same filing date twice must collapse to one PK row.
    assert df.duplicated(subset=["asof_date", "symbol", "field"]).sum() == 0


def test_upsert_fundamentals_dedupes(tmp_path, monkeypatch):
    import duckdb
    import pandas as pd
    from gig.config import get_settings
    from gig.store import Store

    db = tmp_path / "f.duckdb"
    duckdb.connect(str(db)).close()
    monkeypatch.setenv("GIG_DB_PATH", str(db))
    get_settings.cache_clear()
    store = Store(db)
    store.init()
    n = store.upsert_fundamentals(
        pd.DataFrame(
            [
                {"asof_date": date(2024, 1, 1), "symbol": "AAA", "field": "book_equity", "value": 1.0},
                {"asof_date": date(2024, 1, 1), "symbol": "AAA", "field": "book_equity", "value": 2.0},
            ]
        )
    )
    assert n == 1
    got = store.load_fundamentals()
    assert len(got) == 1
    assert float(got.iloc[0]["value"]) == 2.0
    get_settings.cache_clear()



def test_book_to_price_needs_both_fields():
    import numpy as np
    import pandas as pd

    from gig.factors.fundamentals import BookToPrice
    from gig.types import MarketPanel

    idx = pd.bdate_range("2024-01-02", periods=5)
    close = pd.DataFrame({"AAA": 10.0}, index=idx)
    panel = MarketPanel(
        close=close,
        volume=close * 0 + 1e6,
        sectors=pd.Series({"AAA": "tech"}),
        fundamentals=pd.DataFrame(
            [
                {"asof_date": date(2024, 1, 2), "symbol": "AAA", "field": "book_equity", "value": 1e9},
                {
                    "asof_date": date(2024, 1, 2),
                    "symbol": "AAA",
                    "field": "shares_outstanding",
                    "value": 1e8,
                },
            ]
        ),
    )
    scores = BookToPrice().compute(panel)
    # book/market = 1e9 / (10 * 1e8) = 1.0
    assert abs(float(scores.iloc[-1]["AAA"]) - 1.0) < 1e-9

    panel.fundamentals = panel.fundamentals.loc[
        panel.fundamentals["field"] == "book_equity"
    ]
    empty = BookToPrice().compute(panel)
    assert np.isnan(empty.to_numpy()).all()
