"""The risk model must reproduce known covariance structure, not just run."""

import numpy as np
import pandas as pd
import pytest

from gig.risk.factor_model import (
    TRADING_DAYS,
    fit_statistical_risk_model,
    risk_report,
)


def _one_factor_panel(n_names=30, n_days=800, beta_lo=0.8, beta_hi=1.2, seed=0):
    """Returns driven by a single common factor plus independent noise."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2021-01-04", periods=n_days)
    symbols = [f"S{i:02d}" for i in range(n_names)]
    betas = np.linspace(beta_lo, beta_hi, n_names)

    factor = rng.normal(0.0, 0.01, size=n_days)
    idio = rng.normal(0.0, 0.005, size=(n_days, n_names))
    values = factor[:, None] * betas[None, :] + idio
    return pd.DataFrame(values, index=dates, columns=symbols), betas


def test_returns_none_on_unusable_input():
    assert fit_statistical_risk_model(pd.DataFrame()) is None
    assert fit_statistical_risk_model(None) is None

    short = pd.DataFrame(np.random.default_rng(0).normal(size=(10, 5)))
    assert fit_statistical_risk_model(short, min_obs=120) is None


def test_first_factor_captures_a_one_factor_market():
    panel, _ = _one_factor_panel()
    model = fit_statistical_risk_model(panel, n_factors=3)
    assert model is not None

    # One driver plus small noise: the leading PC should dominate.
    assert model.explained > 0.6
    first_pc_share = float(
        np.square(model.exposures["pc1"]).sum() / np.square(model.exposures).to_numpy().sum()
    )
    assert first_pc_share > 0.9


def test_model_covariance_tracks_the_sample_diagonal():
    """B B' + D must reproduce each asset's own variance by construction."""
    panel, _ = _one_factor_panel()
    model = fit_statistical_risk_model(panel, n_factors=3)
    modelled = np.diag(model.covariance().to_numpy())
    sample = panel.var(ddof=1).reindex(model.symbols).to_numpy()
    assert np.allclose(modelled, sample, rtol=1e-8, atol=1e-12)


def test_long_short_book_has_less_risk_than_long_only():
    """A dollar-neutral book must net out the common factor a long book carries."""
    panel, _ = _one_factor_panel()
    model = fit_statistical_risk_model(panel, n_factors=3)
    symbols = model.symbols

    long_only = pd.Series(0.0, index=symbols)
    long_only.iloc[:10] = 0.1

    neutral = pd.Series(0.0, index=symbols)
    neutral.iloc[:10] = 0.1
    neutral.iloc[-10:] = -0.1

    assert model.volatility(neutral) < model.volatility(long_only)

    long_split = model.decompose(long_only)
    neutral_split = model.decompose(neutral)
    # Betas span 0.8-1.2, so equal dollars still leave residual factor exposure.
    # The book is more idiosyncratic than long-only, not purely idiosyncratic.
    assert long_split["systematic_share"] > neutral_split["systematic_share"]
    assert neutral_split["specific_share"] > long_split["specific_share"]


def test_equal_beta_neutral_book_is_almost_pure_specific_risk():
    """With identical betas, dollar-neutral cancels the factor almost exactly."""
    panel, _ = _one_factor_panel(beta_lo=1.0, beta_hi=1.0, seed=2)
    # Correctly specified: one driver, one factor. Extra PCs would fit noise the
    # neutral book cannot cancel, leaving a few percent of residual systematic.
    model = fit_statistical_risk_model(panel, n_factors=1)
    symbols = model.symbols

    neutral = pd.Series(0.0, index=symbols)
    neutral.iloc[:10] = 0.1
    neutral.iloc[-10:] = -0.1

    split = model.decompose(neutral)
    assert split["specific_share"] > 0.98
    assert split["systematic_share"] < 0.02


def test_marginal_contributions_sum_to_volatility():
    """MCTR is only meaningful if the parts add up to the whole."""
    panel, _ = _one_factor_panel(seed=4)
    model = fit_statistical_risk_model(panel, n_factors=4)
    weights = pd.Series(
        np.linspace(-0.05, 0.05, len(model.symbols)), index=model.symbols
    )

    total = float(model.marginal_contributions(weights).sum())
    assert total == pytest.approx(model.volatility(weights, annualized=False), rel=1e-10)


def test_volatility_matches_a_direct_quadratic_form():
    panel, _ = _one_factor_panel(seed=9)
    model = fit_statistical_risk_model(panel, n_factors=5)
    weights = pd.Series(
        np.random.default_rng(1).normal(0, 0.02, len(model.symbols)), index=model.symbols
    )

    cov = model.covariance().to_numpy()
    w = weights.reindex(model.symbols).to_numpy()
    direct = float(np.sqrt(w @ cov @ w) * np.sqrt(TRADING_DAYS))
    assert model.volatility(weights) == pytest.approx(direct, rel=1e-10)


def test_concentrated_book_shows_up_as_few_effective_names():
    """Ten names at equal weight is ten effective names; one big name is not."""
    panel, _ = _one_factor_panel(seed=11)
    symbols = list(panel.columns)

    spread = pd.Series(0.0, index=symbols)
    spread.iloc[:10] = 0.1
    spread.iloc[-10:] = -0.1
    spread_report = risk_report(spread, panel)
    assert spread_report["available"] is True
    assert spread_report["effective_names"] == pytest.approx(20.0, rel=1e-6)

    lopsided = pd.Series(0.0, index=symbols)
    lopsided.iloc[0] = 0.9
    lopsided.iloc[1:10] = 0.011
    assert risk_report(lopsided, panel)["effective_names"] < 3.0


def test_risk_report_ranks_contributors_and_compares_to_realized():
    panel, _ = _one_factor_panel(seed=13)
    symbols = list(panel.columns)
    weights = pd.Series(0.0, index=symbols)
    weights.iloc[:5] = 0.2
    weights.iloc[-5:] = -0.2

    report = risk_report(weights, panel, sectors=pd.Series("tech", index=symbols))
    assert report["available"] is True
    assert report["n_held"] == 10
    assert report["ex_ante_vol"] > 0
    assert report["realized_vol"] > 0

    # A statistical model fitted on the same window should land near realized.
    assert report["ex_ante_vol"] == pytest.approx(report["realized_vol"], rel=0.2)

    # Ranked by absolute risk share. Signed shares can offset, so their
    # magnitudes need not sum to one; `marginal_contributions` covers the
    # additivity of the underlying decomposition.
    shares = [abs(c["share"]) for c in report["top_contributors"]]
    assert shares == sorted(shares, reverse=True)
    assert all(c["symbol"] in symbols for c in report["top_contributors"])
    assert all(c["weight"] != 0 for c in report["top_contributors"])


def test_risk_report_unavailable_without_history():
    tiny = pd.DataFrame(np.random.default_rng(0).normal(size=(5, 3)))
    assert risk_report(pd.Series([0.1, -0.1, 0.0]), tiny)["available"] is False
