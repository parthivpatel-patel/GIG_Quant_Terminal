"""Scenario propagation and liquidation-horizon risk on caller-supplied inputs.

A stated shock to some assets is propagated to the rest through the supplied
covariance, and a position's value-at-risk is extended over the days needed to
sell it plus the cost of crossing the spread. Losses are positive numbers.
Results are model estimates under the stated assumptions, not limits.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.stats import chi2, norm

from gig import checks
from gig.risk.var_models import CONVENTION, _covariance, _exposures


def conditional_stress(weights, covariance, shocked_assets, shocks) -> dict:
    """Portfolio return when some assets move by stated amounts and the rest follow their conditional mean.

    Returns are treated as jointly normal with zero mean, so the unshocked
    assets move by ``cov[u, s] @ inv(cov[s, s]) @ shock``.
    """
    cov = _covariance(covariance)
    n = cov.shape[0]
    w = _exposures(weights, n)
    if not isinstance(shocked_assets, list) or not 1 <= len(shocked_assets) < n:
        raise ValueError("shocked_assets must list at least one and fewer than all asset positions")
    shocked = [checks.integer(index, "shocked_assets", 0, n-1) for index in shocked_assets]
    if len(set(shocked)) != len(shocked):
        raise ValueError("shocked_assets must not repeat")
    shock = checks.array(shocks, "shocks", dimensions=1, maximum=n)
    if shock.shape != (len(shocked),) or np.abs(shock).max() > 10.:
        raise ValueError("shocks must hold one return per shocked asset, each within -10 to 10")
    free = [index for index in range(n) if index not in shocked]
    within = cov[np.ix_(shocked, shocked)]
    across = cov[np.ix_(free, shocked)]
    solved = np.linalg.solve(within, shock)
    implied = across @ solved
    residual = cov[np.ix_(free, free)]-across @ np.linalg.solve(within, across.T)
    moves = np.zeros(n)
    moves[shocked], moves[free] = shock, implied
    direct = float(w[shocked] @ shock)
    total = float(w @ moves)
    distance = float(math.sqrt(shock @ solved))
    return {"method": "CONDITIONAL_NORMAL_SHOCK_PROPAGATION", "assets": n, "shocked_assets": shocked,
            "asset_returns": moves.tolist(), "implied_returns": {str(index): float(value) for index, value in zip(free, implied, strict=True)},
            "portfolio_return": total, "direct_return": direct, "propagated_return": total-direct,
            "contributions": (w*moves).tolist(),
            "residual_volatility": float(math.sqrt(max(w[free] @ residual @ w[free], 0.))),
            "shock_mahalanobis_distance": distance,
            "shock_tail_probability": float(chi2.sf(distance*distance, len(shocked))),
            "assumption_limits": "ZERO_MEAN_JOINT_NORMAL_LINEAR_PROPAGATION_STRESS_CORRELATIONS_NOT_MODELLED",
            "classification": "MODEL_ESTIMATE"}


def liquidity_adjusted_var(position_value, volatility, *, average_daily_value, max_participation=.2, spread=0.,
                           spread_volatility=0., spread_multiplier=3., confidence=.99) -> dict:
    """Value-at-risk of one position sold evenly over the days its size requires, plus half-spread cost.

    Daily volatility and relative bid-ask spread are decimals. The position is
    sold in equal daily pieces at ``max_participation`` of average daily traded
    value, so day ``i`` of ``n`` still carries ``(n-i+1)/n`` of it. The spread
    term follows Bangia et al. (1999): half of mean spread plus a multiple of
    its volatility.
    """
    value = checks.number(position_value, "position_value", positive=True, high=1e15)
    sigma = checks.number(volatility, "volatility", positive=True, high=5.)
    traded = checks.number(average_daily_value, "average_daily_value", positive=True, high=1e15)
    participation = checks.number(max_participation, "max_participation", positive=True, high=1.)
    mean_spread = checks.number(spread, "spread", low=0., high=1.)
    spread_sigma = checks.number(spread_volatility, "spread_volatility", low=0., high=1.)
    multiple = checks.number(spread_multiplier, "spread_multiplier", low=0., high=10.)
    level = checks.confidence(confidence)
    days = math.ceil(value/(participation*traded)-1e-12)
    if days > 2520:
        raise ValueError("liquidation would take more than ten trading years at this participation")
    z = float(norm.ppf(level))
    horizon_factor = math.sqrt((days+1)*(2*days+1)/(6*days))
    one_day = z*sigma*value
    price_risk = one_day*horizon_factor
    spread_cost = .5*value*min(mean_spread+multiple*spread_sigma, 1.)
    return {"method": "LINEAR_LIQUIDATION_VAR_PLUS_EXOGENOUS_SPREAD", "loss_convention": CONVENTION, "confidence": level,
            "liquidation_days": days, "horizon_factor": horizon_factor, "one_day_var": one_day,
            "liquidation_var": price_risk, "spread_cost": spread_cost, "liquidity_adjusted_var": price_risk+spread_cost,
            "liquidity_share": 1-one_day/(price_risk+spread_cost),
            "immediate_sale_equivalent_days": horizon_factor**2,
            "assumption_limits": "NORMAL_IID_RETURNS_EQUAL_DAILY_SALES_NO_PRICE_IMPACT_OR_VOLUME_UNCERTAINTY",
            "classification": "MODEL_ESTIMATE"}
