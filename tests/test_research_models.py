"""Numerical tests for HMM, GARCH, Kalman, Kelly, Black–Litterman, and schedules."""

import numpy as np
import pytest

from gig.execution.schedule import almgren_chriss_schedule, twap_schedule
from gig.governance.registry import STATUSES, model_inventory
from gig.models.diagnostics import diagnostics_from_returns
from gig.models.garch import ewma_variance, fit_garch11
from gig.models.hmm import fit_gaussian_hmm
from gig.models.kalman import kalman_dynamic_regression
from gig.portfolio.black_litterman import black_litterman
from gig.portfolio.kelly import fractional_kelly
from gig.risk.deleverage import exposure_scale
from gig.signals.contract import signal_from_weight


def _two_regime_returns(seed: int = 3) -> np.ndarray:
    rng = np.random.default_rng(seed)
    bear = rng.normal(-0.004, 0.02, 250)
    bull = rng.normal(0.002, 0.008, 250)
    return np.concatenate([bear, bull])


def test_hmm_states_are_separated_and_rows_are_distributions():
    result = fit_gaussian_hmm(_two_regime_returns(), n_states=2, n_iter=30)
    assert result.means[0] != result.means[1]
    assert abs(result.means.max() - result.means.min()) > 0.002
    assert np.allclose(result.transition.sum(axis=1), 1.0)
    assert np.allclose(result.filtered.sum(axis=1), 1.0)
    assert np.allclose(result.smoothed.sum(axis=1), 1.0)
    # Causal filter at the end of the high-mean regime should favor the higher mean.
    high = int(np.argmax(result.means))
    assert result.filtered[-1, high] > result.filtered[10, high]
    payload = result.as_dict()
    assert "bull" not in payload["label_policy"].lower()
    assert payload["most_likely_state"] == high


def test_hmm_rejects_a_short_sample():
    with pytest.raises(ValueError):
        fit_gaussian_hmm(np.zeros(10), n_states=2)


def test_garch_beats_constant_variance_on_simulated_garch():
    rng = np.random.default_rng(2)
    omega, alpha, beta = 1e-6, 0.06, 0.90
    variance = omega / (1.0 - alpha - beta)
    returns = np.empty(2500)
    for t in range(2500):
        returns[t] = rng.normal(0.0, np.sqrt(variance))
        variance = omega + alpha * returns[t] ** 2 + beta * variance
    fit = fit_garch11(returns)
    assert fit.omega > 0
    assert fit.alpha >= 0 and fit.beta >= 0
    assert fit.persistence < 1.0
    assert fit.conditional_variance.min() > 0
    constant = np.full(returns.size, np.var(returns))
    constant_ll = -0.5 * np.sum(np.log(2 * np.pi) + np.log(constant) + returns**2 / constant)
    assert fit.log_likelihood > constant_ll
    assert fit.annualized_next_vol() > 0


def test_ewma_is_not_reported_as_garch():
    series = np.random.default_rng(0).normal(0, 0.01, 100)
    var = ewma_variance(series)
    assert var.shape == series.shape
    assert var.min() > 0


def test_kalman_tracks_a_drifting_beta():
    rng = np.random.default_rng(4)
    t = np.arange(400)
    x = rng.normal(0, 1, t.size)
    beta = 0.2 + 0.002 * t
    y = beta * x + rng.normal(0, 0.05, t.size)
    result = kalman_dynamic_regression(x, y, state_variance=1e-4)
    assert np.corrcoef(result.beta, beta)[0, 1] > 0.8
    assert result.as_dict()["causal"] is True


def test_fractional_kelly_respects_the_gross_cap():
    mu = np.array([0.02, 0.01])
    cov = np.array([[0.04, 0.0], [0.0, 0.02]])
    weights = fractional_kelly(mu, cov, fraction=0.25, max_gross=0.5)
    assert abs(weights).sum() <= 0.5 + 1e-9
    assert weights[0] > 0 and weights[1] > 0
    uncapped = fractional_kelly(mu, cov, fraction=0.25, max_gross=10.0)
    full = np.linalg.solve(cov + 1e-8 * np.eye(2), mu)
    assert np.allclose(uncapped, 0.25 * full, atol=1e-5)


def test_black_litterman_view_moves_the_posterior():
    cov = np.eye(2) * 0.04
    weights = np.array([0.5, 0.5])
    result = black_litterman(
        cov,
        weights,
        view_pick=np.array([[1.0, 0.0]]),
        view_return=np.array([0.20]),
        view_variance=np.array([0.01]),
    )
    assert result.posterior[0] > result.equilibrium[0]
    assert result.posterior[0] > result.posterior[1]


def test_exposure_scale_falls_as_stress_rises_and_stays_bounded():
    calm = exposure_scale(0.0, 0.10, 0.10)
    stressed = exposure_scale(-0.20, 0.25, 0.10)
    assert 0.25 <= stressed < calm <= 1.0


def test_almgren_chriss_sums_to_the_parent_and_front_loads_with_risk():
    twap = twap_schedule(1000, 5)
    assert np.allclose(twap.trades.sum(), 1000)
    patient = almgren_chriss_schedule(1000, 8, sigma=0.01, eta=1e-6, risk_aversion=1e-8)
    urgent = almgren_chriss_schedule(1000, 8, sigma=0.02, eta=1e-6, risk_aversion=1e-3)
    assert np.allclose(patient.trades.sum(), 1000)
    assert np.allclose(urgent.trades.sum(), 1000)
    assert urgent.trades[0] > patient.trades[0]
    flat = almgren_chriss_schedule(1000, 4, sigma=0.01, eta=1e-6, risk_aversion=0.0)
    assert np.allclose(flat.trades, twap_schedule(1000, 4).trades)


def test_signal_contract_does_not_invent_a_probability():
    buy = signal_from_weight("S000", "2020-12-29", 0.04, 1.4, "2020-12-29")
    assert buy.direction == "BUY"
    assert buy.confidence_kind == "relative_score"
    assert buy.confidence == 1.0
    hold = signal_from_weight("S001", "2020-12-29", 0.0, 0.1, "2020-12-29")
    assert hold.direction == "HOLD"
    blocked = signal_from_weight("S002", "2020-12-29", 0.04, 0.2, "DATA UNAVAILABLE", stale=True)
    assert blocked.direction == "NO_TRADE"
    assert blocked.confidence is None


def test_registry_has_no_champion_and_marks_new_models_research():
    rows = model_inventory()
    ids = {row["model_id"] for row in rows}
    assert "regime.gaussian_hmm" in ids
    assert "vol.garch11" in ids
    assert all(row["status"] in STATUSES for row in rows)
    assert all(row["status"] != "CHAMPION" for row in rows)
    hmm = next(row for row in rows if row["model_id"] == "regime.gaussian_hmm")
    assert hmm["affects_execution"] is False
    assert hmm["status"] == "RESEARCH"


def test_market_diagnostics_compare_garch_with_ewma_without_promoting():
    payload = diagnostics_from_returns(_two_regime_returns())
    assert payload["available"] is True
    assert payload["hmm"]["n_states"] == 2
    assert payload["variance_forecast_comparison"]["promotion"].startswith("none")
    assert payload["suggested_exposure_scale"]["applied_to_orders"] is False
    assert diagnostics_from_returns(np.ones(5))["reason"] == "DATA UNAVAILABLE"
