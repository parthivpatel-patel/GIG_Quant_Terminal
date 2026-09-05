"""Service-layer payloads must stay JSON-safe and never invent data."""

import json
import math

import duckdb
import pandas as pd
import pytest

from gig.service import snapshot

# Vendor keys arrive from .env via unprefixed names, and Settings falls back to
# data/qalpha.duckdb when the configured path is missing. Both are convenient in
# a shell and both would let these tests read the production lake, so the fixture
# creates a real (empty) DuckDB file and blanks every key.
_BLANK_ENV = (
    "ALPACA_API_KEY",
    "ALPACA_SECRET_KEY",
    "FINNHUB_API_KEY",
    "FRED_API_KEY",
    "GIG_ALPACA_API_KEY",
    "GIG_ALPACA_SECRET_KEY",
    "GIG_FINNHUB_API_KEY",
    "GIG_FRED_API_KEY",
)


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    from gig.config import get_settings

    db_path = tmp_path / "svc.duckdb"
    duckdb.connect(str(db_path)).close()

    monkeypatch.setenv("GIG_DB_PATH", str(db_path))
    monkeypatch.setenv("GIG_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("GIG_RESULTS_DIR", str(tmp_path / "results"))
    for name in _BLANK_ENV:
        monkeypatch.setenv(name, "")

    get_settings.cache_clear()
    snapshot.clear_cache()
    settings = get_settings()
    settings.ensure_dirs()
    assert settings.db_path == db_path, "fixture failed to isolate the lake"

    yield settings

    snapshot.clear_cache()
    get_settings.cache_clear()


def _seed(settings, n_names=40, n_days=520, seed=7):
    """Write a synthetic panel into the isolated store and return it."""
    from gig.data.providers.synthetic import SyntheticProvider
    from gig.store import Store

    provider = SyntheticProvider(n_names=n_names, n_days=n_days, seed=seed)
    panel = provider.load_panel(
        provider.start, provider.start.replace(year=provider.start.year + 3)
    )
    store = Store(settings.db_path)
    store.init()
    store.upsert_bars(panel)
    store.upsert_universe(panel.sectors, asof=pd.Timestamp(panel.close.index[-1]).date())
    return panel


def test_finite_filter_drops_nan_and_inf():
    assert snapshot._f(1.5) == 1.5
    assert snapshot._f(float("nan")) is None
    assert snapshot._f(float("inf")) is None
    assert snapshot._f(None) is None
    assert snapshot._f("abc") is None


def test_zscore_is_centred_and_clipped():
    z = snapshot._zscore(pd.Series([1.0, 2.0, 3.0, 4.0]))
    assert abs(float(z.mean())) < 1e-9
    assert z.abs().max() <= 3.0
    assert (snapshot._zscore(pd.Series([2.0, 2.0, 2.0])) == 0).all()


def test_status_reports_install_without_secrets(isolated):
    payload = snapshot.service_status()
    assert payload["version"]
    assert set(payload["keys"]) == {"fred", "finnhub", "alpaca", "edgar"}
    assert payload["keys"]["alpaca"] is False
    assert payload["limits"]["max_gross_leverage"] > 0
    assert payload["db_ready"] is False
    assert "ml" in payload
    assert "ollama" in payload
    assert "alpaca_api_key" not in json.dumps(payload).lower()


def test_research_snapshot_absent_gives_hint(isolated):
    payload = snapshot.research_snapshot()
    assert payload["available"] is False
    assert "gig backtest" in payload["hint"]


def test_research_snapshot_is_nan_tolerant(isolated):
    (isolated.results_dir / "equity_ls_last.json").write_text(
        '{"source": "yahoo", "config_hash": "abc", '
        '"metrics": {"sharpe": 0.12, "ann_vol": NaN}, '
        '"factor_ic": {"momentum_12_1": 0.02, "news_sentiment": NaN}, '
        '"factor_ic_n": {"momentum_12_1": 500}}',
        encoding="utf-8",
    )
    payload = snapshot.research_snapshot()
    assert payload["available"] is True
    assert payload["metrics"]["sharpe"] == pytest.approx(0.12)
    assert payload["metrics"]["ann_vol"] is None
    assert payload["factor_ic"]["news_sentiment"] is None
    assert "NaN" not in json.dumps(payload)


def test_experiment_history_skips_bad_lines(isolated):
    (isolated.results_dir / "experiments.jsonl").write_text(
        '{"ts": "2026-01-01T00:00:00", "name": "equity_ls", "config_hash": "h1", '
        '"config": {"n_long": 10}, "metrics": {"sharpe": 0.5, "dsr": NaN}}\n'
        "not json at all\n"
        '{"ts": "2026-01-02T00:00:00", "name": "equity_ls", "config_hash": "h2", '
        '"config": {"n_long": 20}, "metrics": {"sharpe": -0.2}}\n',
        encoding="utf-8",
    )
    payload = snapshot.experiment_history()
    assert payload["available"] is True
    assert len(payload["runs"]) == 2
    assert payload["runs"][0]["dsr"] is None
    assert payload["runs"][-1]["sharpe"] == pytest.approx(-0.2)


def test_factor_cloud_on_empty_lake_is_honest(isolated):
    payload = snapshot.factor_cloud(limit=30)
    assert payload["available"] is False
    assert payload["points"] == []
    assert "ingest" in payload["hint"]


def test_factor_cloud_axes_book_and_limits(isolated):
    """Populated store: finite axes, dollar-neutral book, limits respected."""
    _seed(isolated)
    payload = snapshot.factor_cloud(limit=40, lookback_days=3000)

    assert payload["available"] is True
    assert payload["points"]

    text = json.dumps(payload)
    assert "NaN" not in text and "Infinity" not in text

    for point in payload["points"]:
        for axis in ("momentum", "reversal", "ivol", "score"):
            value = point[axis]
            assert value is None or math.isfinite(value)
        assert point["side"] in {"long", "short", "flat"}

    # Synthetic names clear $1M ADV comfortably, so the floor stays on.
    assert payload["adv_floor"] == 1_000_000.0

    book = payload["book"]
    assert book["longs"] > 0 and book["shorts"] > 0
    assert book["gross"] > 0
    assert book["gross"] <= isolated.max_gross_leverage + 1e-6
    assert abs(book["net"]) < 0.05  # cash-neutral under the optimizer
    assert book["max_name"] <= isolated.max_name_weight + 1e-6
    assert payload["construction"] in {"risk_constrained", "quantile_equal_weight"}
    if payload["construction"] == "risk_constrained":
        # The point of the whole module: common-factor share must collapse.
        assert payload["risk"]["available"] is True
        assert payload["risk"]["systematic_share"] < 0.15
    assert "momentum_12_1" in payload["factor_ic"]


def test_adv_floor_stands_down_instead_of_emptying_the_book(isolated):
    """An unreachable floor must be dropped and reported, not silently applied."""
    _seed(isolated)
    payload = snapshot.factor_cloud(limit=40, lookback_days=3000, min_adv_usd=1e15)
    assert payload["adv_floor"] == 0.0
    assert payload["book"]["longs"] > 0


def test_book_snapshot_sorts_by_conviction(isolated):
    _seed(isolated, seed=11)
    payload = snapshot.book_snapshot(limit=40)

    assert payload["available"] is True
    longs = payload["longs"]
    assert longs and all(p["weight"] > 0 for p in longs)
    assert all(p["weight"] < 0 for p in payload["shorts"])
    # Listed by alpha score descending — the UI's conviction order, not weight.
    scores = [p["score"] for p in longs if p["score"] is not None]
    assert scores == sorted(scores, reverse=True)


def test_live_tape_labels_the_fallback_as_delayed(isolated):
    """With no vendor key the tape must say 'stored close', never 'live'."""
    _seed(isolated, n_names=12, n_days=80)
    payload = snapshot.live_tape(limit=5)

    assert payload["delayed"] is True
    assert payload["source"] == "duckdb_last_close"
    assert 0 < len(payload["quotes"]) <= 5
    assert "NaN" not in json.dumps(payload)


def test_cache_holds_until_cleared():
    calls = {"n": 0}

    def build():
        calls["n"] += 1
        return {"n": calls["n"]}

    first = snapshot._cached("k", 60.0, build)
    assert snapshot._cached("k", 60.0, build) is first
    assert calls["n"] == 1

    snapshot.clear_cache()
    assert snapshot._cached("k", 60.0, build) is not first
    assert calls["n"] == 2
