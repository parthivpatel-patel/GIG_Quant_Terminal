"""Validated historical loss, drawdown and full-position stress analytics.

Return inputs are arithmetic returns, not percent points. These are research
estimates; no routine grants order approval or implies economic validation.
"""
from __future__ import annotations

import numpy as np

MAX_OBSERVATIONS = 1_000_000
MAX_ASSETS = 512
MAX_PANEL_ELEMENTS = 4_000_000


def _vector(values, name: str, *, minimum: int = 1) -> np.ndarray:
    raw = np.asarray(values)
    if raw.ndim != 1 or not minimum <= len(raw) <= MAX_OBSERVATIONS:
        raise ValueError(f"{name} must contain {minimum}..1000000 observations")
    array = np.array(raw, dtype=float, copy=True)
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite vector")
    return array


def historical_loss_risk(returns, confidence: float = 0.99, *, minimum: int = 20) -> dict:
    """Empirical VaR/ES in positive-loss convention, without dropping NaNs.

    VaR uses the inverse empirical CDF. ES integrates the worst ``(1-c)*n``
    observations, fractionally weighting its boundary observation. This avoids
    overstating the tail sample at a noninteger confidence cutoff. Negative VaR
    and ES indicate gains even in the corresponding tail; they are not clipped.
    """
    if not np.isfinite(confidence) or not 0 < confidence < 1:
        raise ValueError("confidence must lie strictly between 0 and 1")
    if type(minimum) is not int or minimum < 1:
        raise ValueError("minimum must be a positive integer")
    values = _vector(returns, "returns", minimum=minimum)
    losses = -values
    ordered = np.sort(losses)[::-1]
    mass = (1-confidence)*len(ordered)
    complete = min(len(ordered), int(np.floor(mass)))
    fraction = mass-complete
    # Divide before summation: a valid tail mean can fit binary64 even when its
    # unnormalized sum overflows (e.g. twenty losses of 1e308).
    total = float(np.sum(ordered[:complete]/mass, dtype=np.longdouble))
    if fraction > 0 and complete < len(ordered):
        total += (fraction/mass) * float(ordered[complete])
    if not np.isfinite(total):
        raise ValueError("expected shortfall arithmetic overflow")
    return {"var": float(np.quantile(losses, confidence, method="inverted_cdf")),
            "expected_shortfall": total, "confidence": float(confidence),
            "observations": len(values), "tail_mass": mass,
            "units": "arithmetic_return_loss", "method": "empirical_fractional_tail"}


def drawdown_report(returns) -> dict:
    """Compounded wealth drawdowns against an initial wealth of one."""
    values = _vector(returns, "returns")
    if np.any(values < -1):
        raise ValueError("arithmetic returns cannot be below -100 percent")
    wealth = np.cumprod(1+values)
    if not np.isfinite(wealth).all():
        raise ValueError("compounded wealth overflow")
    high = np.maximum.accumulate(np.r_[1., wealth])[1:]
    drawdowns = wealth/high-1
    durations = np.zeros(len(values), dtype=int)
    for i, value in enumerate(drawdowns):
        if value < -1e-14:
            durations[i] = (durations[i-1] if i else 0)+1
    return {"max_drawdown": float(drawdowns.min()), "current_drawdown": float(drawdowns[-1]),
            "max_duration_observations": int(durations.max()),
            "current_duration_observations": int(durations[-1]),
            "wealth": wealth.tolist(), "drawdowns": drawdowns.tolist(),
            "units": "fraction_of_running_peak"}


def stress_positions(quantities, prices, shocks, *, nav: float) -> dict:
    """Revalue linear equity positions under asset-specific arithmetic shocks.

    ``shocks`` has shape (scenarios, instruments). P&L excludes financing,
    nonlinear derivatives, transaction costs and any assumed rebalancing.
    """
    qty = _vector(quantities, "quantities")
    px = _vector(prices, "prices")
    scenarios = np.array(shocks, dtype=float, copy=True)
    if len(qty) != len(px) or len(qty) > MAX_ASSETS or np.any(px <= 0):
        raise ValueError("position/pricing dimensions must match and prices must be positive")
    if (scenarios.ndim != 2 or scenarios.shape[1] != len(qty) or not len(scenarios)
            or scenarios.size > MAX_PANEL_ELEMENTS):
        raise ValueError("shocks must be a nonempty scenarios by instruments matrix")
    if not np.isfinite(scenarios).all() or np.any(scenarios < -1):
        raise ValueError("equity price shocks must be finite and at least -100 percent")
    if not np.isfinite(nav) or nav <= 0:
        raise ValueError("NAV must be finite and positive")
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        notionals = qty*px
        pnl = scenarios @ notionals
        nav_returns = pnl/nav
    if not all(np.isfinite(v).all() for v in (notionals, pnl, nav_returns)):
        raise ValueError("stress valuation overflow")
    return {"pnl": pnl.tolist(), "nav_return": nav_returns.tolist(),
            "worst_pnl": float(pnl.min()), "base_notional": notionals.tolist(),
            "valuation": "linear_equity_constant_positions", "units": "NAV_currency"}


def portfolio_risk(returns, weights, *, confidence: float = 0.99, annualization: int = 252) -> dict:
    """Static-weight risk on a common, complete return panel.

    Assets must already share point-in-time identity and valuation timestamps.
    No filling, aligning or inferred cash/sector exposures occurs here.
    """
    data = np.array(returns, dtype=float, copy=True)
    allocation = _vector(weights, "weights")
    if (data.ndim != 2 or data.shape[1] != len(allocation) or not 20 <= data.shape[0] <= MAX_OBSERVATIONS
            or len(allocation) > MAX_ASSETS or data.size > MAX_PANEL_ELEMENTS):
        raise ValueError("returns must be a complete >=20 by assets panel")
    if not np.isfinite(data).all() or type(annualization) is not int or annualization <= 0:
        raise ValueError("returns and annualization must be valid")
    series = data @ allocation
    loss = historical_loss_risk(series, confidence)
    drawdown = drawdown_report(series)
    covariance = np.atleast_2d(np.cov(data, rowvar=False, ddof=1))*annualization
    variance = float(allocation @ covariance @ allocation)
    contribution = allocation*(covariance @ allocation)
    if not np.isfinite(covariance).all() or not np.isfinite(variance) or not np.isfinite(contribution).all():
        raise ValueError("portfolio covariance overflow")
    return {"loss_risk": loss, "drawdown": drawdown,
            "annualized_volatility": float(np.sqrt(max(variance, 0))),
            "variance_contributions": contribution.tolist(),
            "risk_fractions": (contribution/variance).tolist() if variance > 0 else None,
            "gross_exposure": float(np.abs(allocation).sum()),
            "net_exposure": float(allocation.sum()), "max_absolute_weight": float(np.abs(allocation).max()),
            "annualization": annualization, "weight_assumption": "constant_rebalanced_weights"}
