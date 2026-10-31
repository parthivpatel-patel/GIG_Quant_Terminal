from __future__ import annotations

from datetime import date

import pandas as pd

from gig.data.providers.alpaca_iex import parse_snapshots
from gig.data.providers.edgar import parse_submissions, parse_tickers
from gig.data.providers.finnhub import parse_earnings, parse_quote
from gig.data.providers.fred import parse_observations
from gig.factors.macro import vix_gross_scale


def test_gzip_http_body():
    import gzip

    from gig.data.http import decode_http_body

    raw = gzip.compress(b'{"ok": true}')
    assert raw[:2] == b"\x1f\x8b"
    assert decode_http_body(raw) == '{"ok": true}'


def test_alpaca_nested_snapshot_last():
    payload = '{"AAPL": {"latestQuote": {"bp": 10.0, "ap": 10.1}, "latestTrade": {"p": 10.05}}}'
    df = parse_snapshots(payload)
    assert df.iloc[0]["last"] == 10.05
    assert df.iloc[0]["bid"] == 10.0
    payload = """{"observations": [
        {"date": "2020-01-02", "value": "12.5"},
        {"date": "2020-01-03", "value": "."}
    ]}"""
    df = parse_observations(payload, "VIXCLS")
    assert len(df) == 1
    assert float(df.iloc[0]["value"]) == 12.5


def test_edgar_maps_cik_and_filters_forms():
    tickers = '{"0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple"}}'
    assert parse_tickers(tickers)["AAPL"] == "0000320193"
    subs = """{"filings": {"recent": {
        "form": ["8-K", "4"],
        "filingDate": ["2024-01-15", "2024-01-16"],
        "accessionNumber": ["000-1", "000-2"],
        "items": ["1.03,9.01", ""],
        "primaryDocument": ["a.htm", "b.htm"]
    }}}"""
    df = parse_submissions(subs, "AAPL")
    assert list(df["form"]) == ["8-K"]
    assert df.iloc[0]["asof_date"] == date(2024, 1, 15)
    assert df.iloc[0]["sentiment"] == -1.0


def test_alpaca_snapshot_parse():
    payload = '{"quotes": {"MSFT": {"bp": 400.1, "ap": 400.2, "p": 400.15}}}'
    df = parse_snapshots(payload)
    assert df.iloc[0]["symbol"] == "MSFT"
    assert df.iloc[0]["bid"] == 400.1
    assert df.iloc[0]["source"] == "alpaca_iex"


def test_finnhub_quote_and_calendar():
    q = parse_quote('{"c": 150.0, "t": 1700000000}', "aapl")
    assert q["symbol"] == "AAPL"
    assert q["last"] == 150.0
    cal = parse_earnings('{"earningsCalendar": [{"date": "2024-06-03", "symbol": "MSFT", "hour": "amc"}]}')
    assert cal.iloc[0]["event_type"] == "earnings"
    assert cal.iloc[0]["symbol"] == "MSFT"


def test_vix_scale_uses_lag():
    idx = pd.bdate_range("2020-01-02", periods=4)
    vix = pd.Series([10.0, 10.0, 30.0, 30.0], index=idx)
    scale = vix_gross_scale(vix)
    assert scale.iloc[0] == 1.0
    # day-3 uses prior VIX=30 → 0.5
    assert scale.iloc[3] == 0.5


def test_injected_http_fetches():
    from gig.data.providers.alpaca_iex import fetch_latest_quotes
    from gig.data.providers.finnhub import fetch_quotes
    from gig.data.providers.fred import fetch_series

    def fred_get(url, headers):
        return '{"observations": [{"date": "2020-01-02", "value": "1.25"}]}'

    df = fetch_series("DGS10", "demo", get=fred_get)
    assert df.iloc[0]["value"] == 1.25

    def alpaca_get(url, headers):
        assert "APCA-API-KEY-ID" in headers
        return '{"quotes": {"AAPL": {"bp": 1.0, "ap": 1.1}}}'

    q = fetch_latest_quotes(["AAPL"], "pk", "sk", get=alpaca_get)
    assert q.iloc[0]["symbol"] == "AAPL"

    def fh_get(url, headers):
        return '{"c": 99.0, "t": 1700000000}'

    fq = fetch_quotes(["MSFT"], "tok", get=fh_get)
    assert fq.iloc[0]["last"] == 99.0


def test_filings_factor_asof():
    from gig.factors.filings import filings_factor

    dates = pd.bdate_range("2020-06-01", periods=10)
    close = pd.DataFrame({"AAA": range(10, 20), "BBB": range(20, 30)}, index=dates)
    filings = pd.DataFrame(
        {
            "filed_at": [pd.Timestamp("2020-06-05", tz="UTC")],
            "asof_date": [date(2020, 6, 5)],
            "symbol": ["AAA"],
            "form": ["8-K"],
            "title": ["8-K 1.03 AAA"],
            "sentiment": [0.5],
        }
    )
    fac = filings_factor(filings, close, window=5)
    assert fac.loc[dates[dates < "2020-06-05"], "AAA"].fillna(0).eq(0).all()
    assert fac.loc[pd.Timestamp("2020-06-05"), "AAA"] == 0.5
    assert fac.loc[pd.Timestamp("2020-06-05"), "BBB"] == 0.0


def test_8k_item_scores():
    from gig.data.providers.edgar import score_8k_items

    assert score_8k_items("1.03,9.01") == -1.0
    assert score_8k_items("2.02,9.01") == 0.2
    assert score_8k_items("") == 0.2
