from __future__ import annotations

from datetime import date

import pandas as pd

from gig.data.providers.synthetic import SyntheticProvider
from gig.nlp.lexicon import lexicon_score
from gig.nlp.news import headlines_to_frame, news_factor
from gig.store import Store


def test_lexicon_signs():
    assert lexicon_score("Company beats estimates, record profit") > 0
    assert lexicon_score("Fraud probe and bankruptcy warning") < 0
    assert lexicon_score("") == 0.0


def test_news_is_as_of_date():
    rows = [
        {
            "symbol": "AAA",
            "title": "beats estimates",
            "published": "2020-06-15T20:00:00Z",
            "publisher": "test",
            "link": "",
        }
    ]
    news = headlines_to_frame(rows)
    dates = pd.bdate_range("2020-06-01", periods=30)
    close = pd.DataFrame({"AAA": range(30, 60)}, index=dates)
    fac = news_factor(news, close)
    before = fac.loc[dates[dates < "2020-06-15"], "AAA"]
    assert before.isna().all()
    after = fac.loc[dates[dates >= "2020-06-15"], "AAA"]
    assert after.notna().any()
    assert (after.dropna() > 0).all()


def test_weekend_news_maps_to_next_session():
    rows = [
        {
            "symbol": "AAA",
            "title": "beats estimates",
            "published": "2020-06-14T12:00:00Z",  # Sunday
            "publisher": "test",
            "link": "",
        }
    ]
    news = headlines_to_frame(rows)
    dates = pd.bdate_range("2020-06-12", periods=3)  # Fri, Mon, Tue
    close = pd.DataFrame({"AAA": [1.0, 2.0, 3.0]}, index=dates)
    fac = news_factor(news, close)
    assert pd.isna(fac.loc[dates[0], "AAA"])
    assert fac.loc[dates[1], "AAA"] > 0


def test_duckdb_roundtrip(tmp_path):
    store = Store(tmp_path / "q.duckdb")
    store.init()
    panel = SyntheticProvider(n_names=8, n_days=40, seed=1).load_panel(date(2019, 1, 2), date(2019, 6, 1))
    n = store.upsert_bars(panel)
    store.upsert_universe(panel.sectors, asof=date(2019, 6, 1))
    assert n > 0
    loaded = store.load_panel(date(2019, 1, 2), date(2019, 6, 1), panel.symbols())
    assert loaded is not None
    assert set(loaded.symbols()) == set(panel.symbols())
    assert store.stats()["bars"] == n


def test_macro_and_quotes_tables(tmp_path):
    store = Store(tmp_path / "q2.duckdb")
    store.init()
    macro = pd.DataFrame({"dt": [date(2020, 1, 2)], "series_id": ["VIXCLS"], "value": [12.0]})
    assert store.upsert_macro(macro) == 1
    assert store.load_macro().iloc[0]["series_id"] == "VIXCLS"
    quotes = pd.DataFrame(
        {
            "ts": [pd.Timestamp("2020-01-02", tz="UTC")],
            "symbol": ["AAPL"],
            "bid": [1.0],
            "ask": [1.1],
            "last": [1.05],
            "source": ["test"],
        }
    )
    assert store.upsert_quotes(quotes) == 1
    assert store.stats()["quotes"] == 1
