"""Cost-aware rebalancing, maximum diversification and benchmark tracking.

Each problem is convex and solved by SLSQP. A solution is returned only after
its Karush-Kuhn-Tucker conditions are re-checked with nonnegative multipliers;
otherwise the function raises. Weights are long-only and fully invested.
Outputs are research estimates on caller assumptions, never orders.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.optimize import linprog, lsq_linear, minimize

from gig import checks
from gig.risk.var_models import MAX_ASSETS, _covariance

ACTIVE = 1e-7
STATIONARITY = 2e-6


def _vector(value, n: int, name: str) -> np.ndarray:
    result = checks.array(value, name, dimensions=1, maximum=MAX_ASSETS)
    if result.shape != (n,):
        raise ValueError(f"{name} must hold one value per asset")
    return result


def _weights(value, n: int, name: str) -> np.ndarray:
    result = _vector(value, n, name)
    if result.min() < 0 or abs(result.sum()-1) > 1e-9:
        raise ValueError(f"{name} must be nonnegative and sum to 1")
    return result


def _cap(value, n: int) -> float:
    cap = checks.number(value, "max_weight", high=1.)
    if cap < 1/n:
        raise ValueError("max_weight must permit a long-only fully invested portfolio")
    return cap


def _solve(objective, gradient, start, lower, upper, equal, equal_value, at_most=None, at_most_value=None) -> np.ndarray:
    """Minimize a convex function under linear constraints and verify the optimum."""
    size = len(start)
    at_most = np.zeros((0, size)) if at_most is None else at_most
    at_most_value = np.zeros(0) if at_most_value is None else at_most_value
    constraints = [{"type": "eq", "fun": lambda x: equal @ x-equal_value, "jac": lambda _x: equal}]
    if len(at_most):
        constraints.append({"type": "ineq", "fun": lambda x: at_most_value-at_most @ x, "jac": lambda _x: -at_most})
    best = None
    for attempt in (start, np.clip(start+1e-3, lower, upper)):
        result = minimize(objective, attempt, jac=gradient, method="SLSQP", bounds=list(zip(lower, upper, strict=True)),
                          constraints=constraints, options={"maxiter": 1000, "ftol": 1e-15})
        x = np.clip(result.x, lower, upper)
        if np.abs(equal @ x-equal_value).max() > 1e-8 or (len(at_most) and (at_most @ x-at_most_value).max() > 1e-8):
            continue
        g = gradient(x)
        columns = [equal.T]
        free = equal.shape[0]
        binding = at_most[at_most @ x-at_most_value >= -ACTIVE] if len(at_most) else at_most
        columns.append(binding.T)
        columns.append(np.eye(size)[:, x >= upper-ACTIVE])
        columns.append(-np.eye(size)[:, x <= lower+ACTIVE])
        matrix = np.hstack(columns)
        low = np.r_[np.full(free, -np.inf), np.zeros(matrix.shape[1]-free)]
        multipliers = lsq_linear(matrix, -g, bounds=(low, np.full(matrix.shape[1], np.inf)), tol=1e-14)
        residual = float(np.abs(matrix @ multipliers.x+g).max())
        if residual <= STATIONARITY*max(1., float(np.abs(g).max())):
            return x
        best = residual if best is None else min(best, residual)
    raise ValueError("no verified optimum: the constraints are infeasible or the solver did not converge"
                     + ("" if best is None else f" (stationarity residual {best:.2e})"))


def rebalance_with_costs(current_weights, expected_returns, covariance, *, risk_aversion, cost, max_weight=1.,
                         max_turnover=None) -> dict:
    """Mean-variance rebalance from current weights with proportional trading costs.

    Maximizes ``mu'w - risk_aversion/2 * w'Cov w - cost'|w - current|``. ``cost``
    is a decimal per unit of weight traded, one number or one per asset.
    ``max_turnover`` bounds one-way turnover ``sum(|w - current|)/2``.
    """
    cov = _covariance(covariance)
    n = cov.shape[0]
    held = _weights(current_weights, n, "current_weights")
    mu = _vector(expected_returns, n, "expected_returns")
    aversion = checks.number(risk_aversion, "risk_aversion", positive=True, high=1e6)
    costs = np.full(n, checks.number(cost, "cost", low=0., high=1.)) if not isinstance(cost, list) else _vector(cost, n, "cost")
    if costs.min() < 0 or costs.max() > 1:
        raise ValueError("cost must be between 0 and 1 for every asset")
    cap = _cap(max_weight, n)
    if held.max() > cap+1e-9:
        raise ValueError("max_weight is below a current weight")
    eye = np.eye(n)
    trade = np.hstack([eye, -eye])
    at_most, at_most_value = [np.vstack([trade, -trade])], [np.r_[cap-held, held]]
    if max_turnover is not None:
        turnover_cap = checks.number(max_turnover, "max_turnover", low=0., high=1.)
        at_most.append(np.full((1, 2*n), .5))
        at_most_value.append(np.array([turnover_cap]))

    def utility(w: np.ndarray) -> float:
        return float(mu @ w-.5*aversion*w @ cov @ w)

    def objective(x: np.ndarray) -> float:
        return -utility(held+trade @ x)+float(costs @ (x[:n]+x[n:]))

    def gradient(x: np.ndarray) -> np.ndarray:
        marginal = mu-aversion*cov @ (held+trade @ x)
        return np.r_[-marginal+costs, marginal+costs]

    x = _solve(objective, gradient, np.zeros(2*n), np.zeros(2*n), np.ones(2*n), np.r_[np.ones(n), -np.ones(n)][None, :],
               np.zeros(1), np.vstack(at_most), np.concatenate(at_most_value))
    change = trade @ x
    target = held+change
    paid = float(costs @ np.abs(change))
    return {"method": "MEAN_VARIANCE_PROPORTIONAL_COST_REBALANCE", "assets": n, "weights": target.tolist(),
            "trades": change.tolist(), "traded_assets": int(np.count_nonzero(change)),
            "turnover": float(np.abs(change).sum()/2), "cost": paid,
            "utility_before": utility(held), "utility_after_cost": utility(target)-paid,
            "utility_gain": utility(target)-paid-utility(held),
            "expected_return": float(mu @ target), "volatility": float(math.sqrt(target @ cov @ target)),
            "verification": "KKT_CONDITIONS_CHECKED",
            "input_limits": "EXPECTED_RETURNS_AND_COSTS_ARE_CALLER_ASSUMPTIONS_SINGLE_PERIOD_NO_PRICE_IMPACT",
            "classification": "MODEL_ESTIMATE"}


def maximum_diversification(covariance, *, max_weight=1.) -> dict:
    """Long-only weights maximizing weighted-average volatility over portfolio volatility (Choueifaty-Coignard)."""
    cov = _covariance(covariance)
    n = cov.shape[0]
    cap = _cap(max_weight, n)
    sigma = np.sqrt(np.diag(cov))
    # Minimize y'Cov y with sigma'y = 1; weights are y normalized, and the cap stays linear in y.
    at_most = np.eye(n)-cap if cap < 1 else None
    y = _solve(lambda v: float(v @ cov @ v), lambda v: 2*cov @ v, np.full(n, 1/sigma.sum()), np.zeros(n), np.full(n, np.inf),
               sigma[None, :], np.ones(1), at_most, None if at_most is None else np.zeros(n))
    w = y/y.sum()
    dust = w < 1e-10
    if dust.any() and not dust.all():
        cleaned = w.copy()
        cleaned[dust] = 0.
        cleaned /= cleaned.sum()
        if cleaned.max() <= cap+1e-8:
            w = cleaned
    volatility = float(math.sqrt(w @ cov @ w))
    ratio = float(sigma @ w/volatility)
    return {"method": "MAXIMUM_DIVERSIFICATION_LONG_ONLY", "assets": n, "max_weight": cap, "weights": w.tolist(),
            "diversification_ratio": ratio, "effective_independent_bets": ratio*ratio, "volatility": volatility,
            "weighted_average_volatility": float(sigma @ w),
            "correlation_with_portfolio": (cov @ w/(sigma*volatility)).tolist(),
            "held_assets": int(np.count_nonzero(w)), "verification": "KKT_CONDITIONS_CHECKED",
            "input_limits": "COVARIANCE_IS_A_CALLER_INPUT_NO_EXPECTED_RETURN_VIEW", "classification": "MODEL_ESTIMATE"}


def tracking_error_portfolio(benchmark_weights, covariance, *, excluded=None, max_weight=1., expected_returns=None,
                             minimum_active_return=None) -> dict:
    """Long-only portfolio closest to a benchmark in tracking variance under exclusions, a cap or a required active return."""
    cov = _covariance(covariance)
    n = cov.shape[0]
    benchmark = _weights(benchmark_weights, n, "benchmark_weights")
    cap = checks.number(max_weight, "max_weight", positive=True, high=1.)
    barred = [] if excluded is None else excluded
    if not isinstance(barred, list) or len(barred) >= n:
        raise ValueError("excluded must list fewer than all asset positions")
    barred = [checks.integer(index, "excluded", 0, n-1) for index in barred]
    upper = np.full(n, cap)
    upper[barred] = 0.
    if len(set(barred)) != len(barred) or upper.sum() < 1-1e-12:
        raise ValueError("excluded must not repeat and the remaining assets must be able to hold the whole portfolio")
    if (expected_returns is None) != (minimum_active_return is None):
        raise ValueError("expected_returns and minimum_active_return must be supplied together")
    at_most = at_most_value = None
    mu = None
    if expected_returns is not None:
        mu = _vector(expected_returns, n, "expected_returns")
        required = checks.number(minimum_active_return, "minimum_active_return", low=-10., high=10.)
        at_most, at_most_value = -mu[None, :], np.array([-(required+mu @ benchmark)])
        attainable = linprog(-mu, A_eq=np.ones((1, n)), b_eq=np.ones(1), bounds=list(zip(np.zeros(n), upper, strict=True)))
        if attainable.success and -float(attainable.fun) < required+float(mu @ benchmark)-1e-7:
            raise ValueError("minimum_active_return is not attainable under the weight cap and exclusions")
    start = np.minimum(benchmark, upper)
    room = upper-start
    start = start+room*(1-start.sum())/room.sum() if room.sum() > 0 else start
    w = _solve(lambda v: float((v-benchmark) @ cov @ (v-benchmark)), lambda v: 2*cov @ (v-benchmark), start, np.zeros(n),
               upper, np.ones((1, n)), np.ones(1), at_most, at_most_value)
    active = w-benchmark
    variance = float(active @ cov @ active)
    error = math.sqrt(max(variance, 0.))
    benchmark_variance = float(benchmark @ cov @ benchmark)
    return {"method": "MINIMUM_TRACKING_VARIANCE_LONG_ONLY", "assets": n, "weights": w.tolist(),
            "active_weights": active.tolist(), "tracking_error": error, "active_share": float(np.abs(active).sum()/2),
            "beta_to_benchmark": float(w @ cov @ benchmark/benchmark_variance),
            "tracking_contributions": ((active*(cov @ active))/error).tolist() if error > 1e-12 else [0.]*n,
            "active_return": None if mu is None else float(mu @ active),
            "portfolio_volatility": float(math.sqrt(w @ cov @ w)), "benchmark_volatility": math.sqrt(benchmark_variance),
            "excluded": barred, "verification": "KKT_CONDITIONS_CHECKED",
            "input_limits": "COVARIANCE_AND_EXPECTED_RETURNS_ARE_CALLER_INPUTS_EX_ANTE_ONLY", "classification": "MODEL_ESTIMATE"}
