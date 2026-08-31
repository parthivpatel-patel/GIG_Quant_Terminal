from __future__ import annotations

import numpy as np
import pandas as pd

from gig.assets.futures import roll_return, tsmom_signal
from gig.portfolio.optimizer import mean_variance_weights
from gig.research.experiment import ExperimentLog, config_hash
from gig.risk.exposures import factor_exposures
from gig.strategies.futures_tsmom import FuturesTSMOM
from gig.strategies.options_vrp import VarianceRiskPremium


def test_tsmom_is_causal():
    idx = pd.bdate_range("2020-01-02", periods=400)
    rng = np.random.default_rng(0)
    r = pd.Series(rng.normal(0.0004, 0.01, 400), index=idx)
    sig = tsmom_signal(r)
    assert sig.iloc[:251].isna().all()
    # First defined value cannot depend on the last return
    r2 = r.copy()
    r2.iloc[-1] = 9.9
    sig2 = tsmom_signal(r2)
    defined = sig.dropna().index[:-1]
    pd.testing.assert_series_equal(sig.loc[defined], sig2.loc[defined])


def test_roll_yield_sign():
    idx = pd.bdate_range("2020-01-02", periods=5)
    nearby = pd.Series([100, 101, 102, 103, 104.0], index=idx)
    deferred = pd.Series([99, 100, 101, 102, 103.0], index=idx)
    assert (roll_return(nearby, deferred) > 0).all()


def test_futures_strategy_runs():
    idx = pd.bdate_range("2018-01-02", periods=500)
    rng = np.random.default_rng(1)
    r = pd.Series(0.0005 + rng.normal(0, 0.008, 500), index=idx)
    result = FuturesTSMOM().run(r)
    assert "sharpe" in result.metrics


def test_vrp_strategy_runs():
    idx = pd.bdate_range("2020-01-02", periods=80)
    rv = pd.Series(0.18, index=idx)
    iv = pd.Series(np.linspace(0.12, 0.25, 80), index=idx)
    pnl = pd.Series(0.001, index=idx)
    result = VarianceRiskPremium().run(rv, iv, pnl)
    assert len(result.returns) > 10


def test_mean_variance_respects_name_cap():
    mu = pd.Series({"A": 0.1, "B": 0.05, "C": -0.04, "D": -0.02})
    cov = pd.DataFrame(np.eye(4) * 0.02, index=mu.index, columns=mu.index)
    w = mean_variance_weights(mu, cov, max_name=0.05, max_gross=1.0, max_net=0.2)
    assert w.abs().max() <= 0.05 + 1e-8
    assert w.abs().sum() <= 1.0 + 1e-6


def test_experiment_log(tmp_path):
    log = ExperimentLog(tmp_path / "e.jsonl")
    digest = log.record("x", {"k": 1}, {"sharpe": 0.5})
    assert digest == config_hash({"k": 1})
    assert (tmp_path / "e.jsonl").read_text(encoding="utf-8").strip()


def test_exposures_keys():
    idx = pd.bdate_range("2020-01-02", periods=80)
    names = ["A", "B"]
    rng = np.random.default_rng(0)
    rets = pd.DataFrame(rng.normal(0, 0.01, (80, 2)), index=idx, columns=names)
    w = pd.Series({"A": 0.5, "B": -0.5})
    out = factor_exposures(w, rets, pd.Series({"A": "t", "B": "t"}))
    assert set(out) >= {"trailing_beta", "net", "gross"}
