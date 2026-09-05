"""
The optimizer's claims are checkable, so they are checked.

"Factor-neutral" and "vol-targeted" are exact statements about the output, not
descriptions of intent. Each is asserted here against a panel with known
structure, together with the regression these tests exist for: the equal-weight
quantile book carried the overwhelming majority of its variance in common-factor
exposure, and the replacement must not.
"""

import numpy as np
import pandas as pd
import pytest

from gig.portfolio.construct import dollar_neutral_quantiles
from gig.portfolio.optimize import (
    OptimizerConfig,
    constraint_summary,
    optimize_book,
)
from gig.risk.factor_model import TRADING_DAYS, fit_statistical_risk_model

N_NAMES = 60
SECTORS = ["tech", "energy", "health", "financials"]


def _panel(n_names=N_NAMES, n_days=700, idio=0.03, seed=0):
    """
    Two common factors plus fat idiosyncratic noise.

    The idiosyncratic vol is deliberately large so a 10% ex-ante target is
    reachable inside the gross cap; with tiny specific risk every test would
    just measure the gross constraint.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2021-01-04", periods=n_days)
    symbols = [f"S{i:02d}" for i in range(n_names)]

    market = rng.normal(0.0, 0.012, size=n_days)
    style = rng.normal(0.0, 0.008, size=n_days)
    beta_m = rng.uniform(0.7, 1.4, n_names)
    beta_s = rng.uniform(-1.0, 1.0, n_names)
    noise = rng.normal(0.0, idio, size=(n_days, n_names))

    values = market[:, None] * beta_m + style[:, None] * beta_s + noise
    returns = pd.DataFrame(values, index=dates, columns=symbols)
    sectors = pd.Series([SECTORS[i % len(SECTORS)] for i in range(n_names)], index=symbols)
    return returns, sectors


def _factor_bet_panel(n_names=N_NAMES, n_days=700, seed=3):
    """
    Realistic proportions: factor vol comparable to name vol, wide beta spread.

    `_panel` deliberately uses fat idiosyncratic noise so a vol target is
    reachable inside the gross cap, but that also means specific risk dominates
    for any book. Reproducing the systematic-risk regression needs the opposite
    regime, which is the one real equities are in.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2021-01-04", periods=n_days)
    symbols = [f"S{i:02d}" for i in range(n_names)]

    market = rng.normal(0.0, 0.012, size=n_days)
    beta = rng.uniform(0.5, 1.6, n_names)
    noise = rng.normal(0.0, 0.008, size=(n_days, n_names))

    returns = pd.DataFrame(market[:, None] * beta + noise, index=dates, columns=symbols)
    sectors = pd.Series([SECTORS[i % len(SECTORS)] for i in range(n_names)], index=symbols)
    return returns, sectors


def _alpha(symbols, seed=7):
    rng = np.random.default_rng(seed)
    return pd.Series(rng.normal(size=len(symbols)), index=symbols)


@pytest.fixture(scope="module")
def fitted():
    returns, sectors = _panel()
    model = fitted_model = fit_statistical_risk_model(returns, n_factors=5)
    assert fitted_model is not None
    return returns, sectors, model


def test_factor_exposure_is_zero_to_machine_precision(fitted):
    """
    The closed form makes neutrality exact, not approximate.

    This is the property that distinguishes it from residualizing the signal and
    hoping: a book built by an iterative solver would land near zero, and "near"
    is where a 94% systematic share hides.
    """
    returns, sectors, model = fitted
    book = optimize_book(_alpha(model.symbols), model, sectors=sectors)

    assert book.available
    assert float(book.factor_exposure.abs().max()) < 1e-12
    assert book.systematic_share < 1e-12
    assert book.specific_share == pytest.approx(1.0, abs=1e-9)


def test_book_is_cash_and_sector_neutral(fitted):
    returns, sectors, model = fitted
    book = optimize_book(_alpha(model.symbols), model, sectors=sectors)

    assert abs(book.net) < 1e-12
    by_sector = book.weights.groupby(sectors.reindex(book.weights.index)).sum()
    assert by_sector.abs().max() < 1e-12
    # Neutrality with real weights, not by holding nothing.
    assert book.gross > 0.5
    assert book.n_held > 40


def test_ex_ante_vol_hits_the_target_when_nothing_else_binds(fitted):
    returns, sectors, model = fitted
    config = OptimizerConfig(target_vol=0.10, max_gross=4.0, max_name=0.50)
    book = optimize_book(_alpha(model.symbols), model, sectors=sectors, config=config)

    assert book.ex_ante_vol == pytest.approx(0.10, rel=1e-6)
    assert book.binding == "vol_target"

    # And the reported figure is the model's own quadratic form, not a proxy.
    direct = np.sqrt(
        book.weights.reindex(model.symbols).fillna(0.0)
        @ model.covariance().to_numpy()
        @ book.weights.reindex(model.symbols).fillna(0.0)
    ) * np.sqrt(TRADING_DAYS)
    assert book.ex_ante_vol == pytest.approx(float(direct), rel=1e-9)


def test_target_vol_scales_the_book_linearly(fitted):
    returns, sectors, model = fitted
    config = OptimizerConfig(target_vol=0.05, max_gross=4.0, max_name=0.50)
    half = optimize_book(_alpha(model.symbols), model, sectors=sectors, config=config)
    double = optimize_book(
        _alpha(model.symbols),
        model,
        sectors=sectors,
        config=OptimizerConfig(target_vol=0.20, max_gross=4.0, max_name=0.50),
    )
    assert double.gross == pytest.approx(4.0 * half.gross, rel=1e-6)


def test_gross_cap_wins_when_the_target_would_need_leverage(fitted):
    """A vol target is not permission to lever up to reach it."""
    returns, sectors, model = fitted
    config = OptimizerConfig(target_vol=5.0, max_gross=2.0, max_name=0.10)
    book = optimize_book(_alpha(model.symbols), model, sectors=sectors, config=config)

    assert book.gross == pytest.approx(2.0, rel=1e-6)
    assert book.ex_ante_vol < 5.0
    assert book.binding in {"gross", "name_cap"}


def test_name_cap_is_never_violated(fitted):
    returns, sectors, model = fitted
    config = OptimizerConfig(target_vol=0.60, max_gross=6.0, max_name=0.03)
    book = optimize_book(_alpha(model.symbols), model, sectors=sectors, config=config)

    assert book.max_name <= 0.03 + 1e-12
    # Enforcing the box must not smuggle factor exposure back in: the clip is a
    # trade, and the projection has to hedge it.
    assert float(book.factor_exposure.abs().max()) < 1e-6
    assert abs(book.net) < 1e-6


def test_weights_follow_the_alpha_ordering(fitted):
    """Highest residual alpha long, lowest short. Otherwise it is not the signal."""
    returns, sectors, model = fitted
    alpha = _alpha(model.symbols)
    book = optimize_book(alpha, model, sectors=sectors)

    held = book.weights[book.weights.abs() > 0]
    corr = float(np.corrcoef(alpha.reindex(held.index), held)[0, 1])
    assert corr > 0.5
    assert book.weights[alpha.idxmax()] > 0
    assert book.weights[alpha.idxmin()] < 0


def test_alpha_scale_does_not_change_the_book(fitted):
    """Only the shape of the signal matters; the level is absorbed by the target."""
    returns, sectors, model = fitted
    alpha = _alpha(model.symbols)
    base = optimize_book(alpha, model, sectors=sectors)
    scaled = optimize_book(alpha * 250.0 + 3.0, model, sectors=sectors)
    pd.testing.assert_series_equal(base.weights, scaled.weights, rtol=1e-9)


def test_low_specific_risk_names_get_the_larger_weight(fitted):
    """
    The D^-1 tilt is the mean-variance part of the solution.

    Two names with the same residual alpha should not get the same weight if one
    is twice as volatile.
    """
    returns, sectors, model = fitted
    quiet, loud = model.symbols[0], model.symbols[1]

    doctored = model.specific_var.copy()
    doctored[quiet] = 0.0004
    doctored[loud] = 0.0016
    patched = type(model)(
        exposures=model.exposures,
        specific_var=doctored,
        n_obs=model.n_obs,
        n_factors=model.n_factors,
        explained=model.explained,
    )

    alpha = pd.Series(0.0, index=model.symbols)
    alpha[quiet] = 1.0
    alpha[loud] = 1.0
    alpha.iloc[2:] = -2.0 / (len(alpha) - 2)

    # Floor disabled: it exists to stop a suspiciously quiet name from
    # dominating, which is the very effect being measured here.
    book = optimize_book(
        alpha,
        patched,
        sectors=None,
        config=OptimizerConfig(max_name=1.0, specific_var_floor_ratio=0.0),
    )
    assert abs(book.weights[quiet]) > 1.5 * abs(book.weights[loud])


def test_specific_variance_floor_stops_degenerate_names_taking_the_book():
    """
    Names whose measured specific risk has collapsed must not own the book.

    Halted, stale, and duplicated series all produce this: near-zero measured
    specific variance, which the inverse-variance tilt reads as "free risk
    budget". A single such name is self-limiting -- it dominates the GLS fit,
    so its own residual alpha goes to zero -- but a handful of them are not,
    which is why the floor is here.
    """
    returns, sectors = _panel(seed=21)
    quiet = list(returns.columns[:5])
    returns[quiet] = returns[quiet] * 0.001
    model = fit_statistical_risk_model(returns, n_factors=5)
    alpha = _alpha(model.symbols, seed=31)

    # Name cap lifted so this measures the weighting metric alone. In the
    # configured book the 5% cap would mask most of the effect, which is a
    # second line of defence rather than a reason not to have the first.
    shared = {"max_name": 100.0, "max_gross": 1e6, "target_vol": 0.10}
    floored = optimize_book(alpha, model, sectors=sectors, config=OptimizerConfig(**shared))
    unfloored = optimize_book(
        alpha,
        model,
        sectors=sectors,
        config=OptimizerConfig(**shared, specific_var_floor_ratio=0.0),
    )

    def concentration(book):
        return float(book.weights[quiet].abs().sum() / book.gross)

    assert concentration(unfloored) > 0.95     # five names are the entire book
    assert concentration(floored) < 0.60
    assert unfloored.effective_names < 3
    assert floored.effective_names > 5 * unfloored.effective_names

    # The floor changes the weighting metric, never the reported risk: both
    # books are still measured against the model's own specific variance.
    assert floored.ex_ante_vol == pytest.approx(0.10, rel=1e-6)
    assert unfloored.ex_ante_vol == pytest.approx(0.10, rel=1e-6)


def test_optimizer_fixes_the_systematic_risk_of_the_quantile_book():
    """
    The regression this module was written for.

    A live run of the equal-weight quantile book showed ~94% of its variance in
    common-factor exposure while calling itself dollar-neutral, which is how it
    ran at 40% vol. The mechanism is that the alpha is *correlated with* the
    factor loadings -- low idiosyncratic vol is a low-beta bet, momentum is a
    style bet -- so ranking on it and taking the extremes puts on a factor
    position deliberately, and counting dollars does not detect that.

    Modelled here by ranking on the leading factor loading itself. Same alpha,
    same risk model, same limits: the constrained book must move the systematic
    share to zero and predict materially lower volatility.
    """
    returns, sectors = _factor_bet_panel()
    model = fit_statistical_risk_model(returns, n_factors=5)
    alpha = model.exposures["pc1"].copy()

    naive = dollar_neutral_quantiles(alpha, n_long=15, n_short=15, gross_leverage=2.0)
    naive_summary = constraint_summary(naive, model, sectors=sectors)

    config = OptimizerConfig(target_vol=0.10, max_gross=2.0, max_name=0.05)
    book = optimize_book(alpha, model, sectors=sectors, config=config)

    assert naive_summary["net"] == pytest.approx(0.0, abs=1e-9)  # dollar-neutral...
    assert naive_summary["systematic_share"] > 0.80              # ...and a factor bet
    assert book.systematic_share < 1e-9
    assert book.ex_ante_vol < 0.5 * naive_summary["ex_ante_vol"]
    # Risk-aware weighting also spreads the book across more independent bets.
    naive_effective = 1.0 / np.square(naive.abs() / naive.abs().sum()).sum()
    assert book.effective_names > naive_effective


def test_constraint_summary_describes_a_book_it_did_not_build(fitted):
    returns, sectors, model = fitted
    # 20 names capped at 5% each is 1.0 gross, not the 2.0 requested: the name
    # cap binds first, which is worth pinning since it also bites in the tests.
    naive = dollar_neutral_quantiles(_alpha(model.symbols), n_long=10, n_short=10)
    summary = constraint_summary(naive, model, sectors=sectors)

    assert summary["gross"] == pytest.approx(1.0, rel=1e-9)
    assert summary["net"] == pytest.approx(0.0, abs=1e-9)
    assert summary["max_name"] == pytest.approx(0.05, rel=1e-9)
    assert summary["worst_factor_exposure"] > 0
    assert 0.0 <= summary["systematic_share"] <= 1.0
    assert "worst_sector_net" in summary


def test_thin_or_degenerate_cross_sections_return_nothing(fitted):
    returns, sectors, model = fitted

    assert not optimize_book(pd.Series(dtype=float), model).available
    # Fewer eligible names than the minimum: no book rather than a token one.
    few = _alpha(model.symbols[:5])
    assert not optimize_book(few, model, min_names=20).available
    # A flat signal carries no information, so there is nothing to express.
    flat = pd.Series(1.0, index=model.symbols)
    assert not optimize_book(flat, model).available


def test_names_missing_from_the_risk_model_are_dropped(fitted):
    """An unmeasured name is not a low-risk name."""
    returns, sectors, model = fitted
    alpha = _alpha(model.symbols)
    alpha["UNKNOWN"] = 5.0

    book = optimize_book(alpha, model, sectors=sectors)
    assert book.weights.get("UNKNOWN", 0.0) == 0.0
    assert "UNKNOWN" in book.weights.index  # reported as flat, not silently gone


def test_sector_neutrality_can_be_switched_off(fitted):
    returns, sectors, model = fitted
    config = OptimizerConfig(neutralize_sectors=False)
    book = optimize_book(_alpha(model.symbols), model, sectors=sectors, config=config)

    by_sector = book.weights.groupby(sectors.reindex(book.weights.index)).sum()
    # Factors stay neutral; sector nets are now free to be non-zero.
    assert float(book.factor_exposure.abs().max()) < 1e-12
    assert by_sector.abs().max() > 1e-6
