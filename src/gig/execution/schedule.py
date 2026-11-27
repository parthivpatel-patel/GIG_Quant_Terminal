"""Execution schedule research under explicit impact assumptions.

Almgren-Chriss (2000) optimal liquidation in discrete time, a participation
capped volume schedule, and a seeded implementation-shortfall simulation of
any schedule under the same price model. Impact and volatility parameters are
caller assumptions; nothing here routes, sizes or approves an order.
"""
from __future__ import annotations

import math

import numpy as np

from gig import checks

MAX_INTERVALS = 1000
MAX_DRAWS = 4_000_000


def _impact(volatility, permanent, temporary, fixed, interval: float) -> tuple[float, float, float, float, float]:
    sigma = checks.number(volatility, "volatility", low=0., high=1e9)
    gamma = checks.number(permanent, "permanent_impact", low=0., high=1e9)
    eta = checks.number(temporary, "temporary_impact", positive=True, high=1e9)
    epsilon = checks.number(fixed, "fixed_cost", low=0., high=1e9)
    adjusted = eta-.5*gamma*interval
    if adjusted <= 0:
        raise ValueError("temporary_impact must exceed half the permanent impact accrued per interval")
    return sigma, gamma, eta, epsilon, adjusted


def _moments(trades: np.ndarray, interval: float, sigma: float, gamma: float, epsilon: float,
             adjusted: float) -> tuple[float, float]:
    total = trades.sum()
    holdings = total-np.cumsum(trades)
    expected = .5*gamma*total**2 + epsilon*np.abs(trades).sum() + adjusted/interval*np.sum(trades**2)
    return float(expected), float(sigma**2*interval*np.sum(holdings**2))


def _trajectory(shares: float, horizon: float, intervals: int, kappa: float) -> np.ndarray:
    times = np.linspace(0., horizon, intervals+1)
    if kappa == 0:
        return shares*(1-times/horizon)
    # sinh(k(T-t))/sinh(kT) written with decaying exponentials only.
    return shares*np.exp(-kappa*times)*(-np.expm1(-2*kappa*(horizon-times)))/(-np.expm1(-2*kappa*horizon))


def almgren_chriss(shares, horizon, intervals, *, volatility, temporary_impact, permanent_impact=0.,
                   fixed_cost=0., risk_aversion=0., frontier=None) -> dict:
    """Mean-variance optimal liquidation of ``shares`` over ``horizon``.

    Price follows an arithmetic random walk with ``volatility`` per square
    root of time; trading ``n`` shares in an interval ``tau`` moves the price
    permanently by ``permanent_impact*n`` and costs ``fixed_cost +
    temporary_impact*n/tau`` per share temporarily.
    """
    x = checks.number(shares, "shares", positive=True, high=1e12)
    t = checks.number(horizon, "horizon", positive=True, high=1e6)
    n = checks.integer(intervals, "intervals", 1, MAX_INTERVALS)
    tau = t/n
    sigma, gamma, _, epsilon, adjusted = _impact(volatility, permanent_impact, temporary_impact, fixed_cost, tau)
    aversions = [checks.number(risk_aversion, "risk_aversion", low=0., high=1e9)]
    if frontier is not None:
        if not isinstance(frontier, list) or not 1 <= len(frontier) <= 32:
            raise ValueError("frontier must list 1 to 32 risk aversions")
        aversions += [checks.number(value, "frontier risk aversion", low=0., high=1e9) for value in frontier]

    def plan(aversion: float) -> dict:
        curvature = aversion*sigma**2/adjusted
        kappa = math.acosh(.5*curvature*tau**2+1)/tau if curvature > 0 else 0.
        holdings = _trajectory(x, t, n, kappa)
        holdings[-1] = 0.
        trades = -np.diff(holdings)
        expected, variance = _moments(trades, tau, sigma, gamma, epsilon, adjusted)
        return {"risk_aversion": aversion, "kappa": kappa, "half_life": math.log(2)/kappa if kappa else None,
                "expected_cost": expected, "cost_variance": variance, "cost_standard_deviation": math.sqrt(variance),
                "objective": expected+aversion*variance, "holdings": holdings.tolist(), "trades": trades.tolist()}

    chosen = plan(aversions[0])
    uniform = _moments(np.full(n, x/n), tau, sigma, gamma, epsilon, adjusted)
    result = {"method": "ALMGREN_CHRISS_2000_DISCRETE", "shares": x, "horizon": t, "intervals": n, "interval": tau,
              **chosen, "cost_per_share": chosen["expected_cost"]/x,
              "uniform_expected_cost": uniform[0], "uniform_cost_variance": uniform[1],
              "assumption_limits": "LINEAR_IMPACT_ARITHMETIC_PRICE_NO_FILL_UNCERTAINTY",
              "classification": "MODEL_ESTIMATE"}
    if frontier is not None:
        result["frontier"] = [{key: row[key] for key in ("risk_aversion", "kappa", "expected_cost", "cost_variance")}
                              for row in map(plan, aversions[1:])]
    return result


def participation_schedule(shares, volumes, *, max_participation=.1) -> dict:
    """Whole-share volume-proportional slices capped at a participation rate.

    Shares the cap cannot place are reported as unfilled, never forced.
    """
    total = checks.integer(shares, "shares", 1, 10**12)
    volume = checks.array(volumes, "volumes", dimensions=1, maximum=MAX_INTERVALS)
    rate = checks.number(max_participation, "max_participation", positive=True, high=1.)
    if np.any(volume < 0) or volume.sum() <= 0:
        raise ValueError("volumes must be nonnegative with a positive total")
    capacity = np.floor(rate*volume+1e-9).astype(np.int64)
    placed = min(total, int(capacity.sum()))
    ideal = placed*volume/volume.sum()
    slices = np.minimum(np.floor(ideal+1e-9).astype(np.int64), capacity)
    while slices.sum() < placed:
        spare = np.where(slices < capacity)[0]
        best = spare[np.argmax((ideal-slices)[spare])]
        slices[best] += 1
    with np.errstate(divide="ignore", invalid="ignore"):
        used = np.where(volume > 0, slices/volume, 0.)
    return {"method": "VOLUME_PROPORTIONAL_PARTICIPATION_CAP", "shares": total, "max_participation": rate,
            "slices": slices.tolist(), "participation": used.tolist(), "scheduled_shares": placed,
            "unfilled_shares": total-placed, "feasible": placed == total,
            "volume_limits": "VOLUMES_ARE_CALLER_FORECASTS_NOT_OBSERVED_LIQUIDITY",
            "classification": "MODEL_ESTIMATE"}


def simulate_shortfall(trades, interval, *, volatility, temporary_impact, permanent_impact=0., fixed_cost=0.,
                       paths=10000, seed=0) -> dict:
    """Seeded implementation shortfall of a schedule under the Almgren-Chriss price model."""
    n = checks.array(trades, "trades", dimensions=1, maximum=MAX_INTERVALS)
    tau = checks.number(interval, "interval", positive=True, high=1e6)
    if not n.any():
        raise ValueError("trades must not be all zero")
    sigma, gamma, eta, epsilon, adjusted = _impact(volatility, permanent_impact, temporary_impact, fixed_cost, tau)
    count, seed = checks.integer(paths, "paths", 100, 200000), checks.integer(seed, "seed", 0, 2**32-1)
    if count*len(n) > MAX_DRAWS:
        raise ValueError("paths times intervals exceeds four million draws")
    holdings = n.sum()-np.cumsum(n)
    # Trade k fills against the price after k-1 shocks, so shock j is borne by holdings[j].
    shocks = np.random.default_rng(seed).standard_normal((count, len(n)))
    certain = gamma*float(np.sum(n*(np.cumsum(n)-n))) + epsilon*float(np.abs(n).sum()) + eta/tau*float(np.sum(n**2))
    shortfall = certain-sigma*math.sqrt(tau)*(shocks @ holdings)
    expected, variance = _moments(n, tau, sigma, gamma, epsilon, adjusted)
    deviation = float(shortfall.std(ddof=1))
    return {"method": "IMPLEMENTATION_SHORTFALL_SIMULATION", "paths": count, "seed": seed, "intervals": len(n),
            "shares": float(n.sum()), "mean": float(shortfall.mean()), "standard_deviation": deviation,
            "standard_error": deviation/math.sqrt(count),
            "quantiles": {str(q): float(np.quantile(shortfall, q)) for q in (.05, .5, .95, .99)},
            "analytic_expected_cost": expected, "analytic_standard_deviation": math.sqrt(variance),
            "assumption_limits": "LINEAR_IMPACT_ARITHMETIC_PRICE_NO_FILL_UNCERTAINTY",
            "simulation_state": "SEEDED_MONTE_CARLO", "classification": "MODEL_ESTIMATE"}
