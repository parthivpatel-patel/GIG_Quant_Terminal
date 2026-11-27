"""Bounded portfolio construction research on caller-supplied inputs.

Covariance shrinkage, hierarchical risk parity, minimum-CVaR allocation and a
long-only efficient frontier. Each function bounds its inputs, re-checks its
own solution and raises instead of returning an unverified solver result.
Outputs are research estimates, never orders or approved allocations.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.optimize import linprog, minimize
from scipy.sparse import csr_matrix, hstack, vstack
from scipy.sparse import eye as sparse_eye
from scipy.spatial.distance import pdist

from gig import checks
from gig.risk.var_models import _covariance, tail_weights

MAX_ASSETS = 64
MAX_OBSERVATIONS = 2000
FEASIBILITY = 1e-7


def _returns(value, name: str = "returns") -> np.ndarray:
    r = checks.array(value, name, dimensions=2, maximum=MAX_OBSERVATIONS*MAX_ASSETS)
    if not 2 <= r.shape[0] <= MAX_OBSERVATIONS or not 2 <= r.shape[1] <= MAX_ASSETS:
        raise ValueError(f"{name} must hold 2 to 2000 rows of 2 to 64 assets")
    return r


def _cap(value, n: int) -> float:
    cap = checks.number(value, "max_weight", high=1.)
    if cap < 1/n:
        raise ValueError("max_weight must permit a long-only fully invested portfolio")
    return cap


def ledoit_wolf(returns) -> dict:
    """Ledoit-Wolf (2004) shrinkage of the sample covariance toward a scaled identity."""
    x = _returns(returns)
    t, n = x.shape
    centred = x-x.mean(axis=0)
    sample = centred.T @ centred/t
    mu = float(np.trace(sample)/n)
    if mu <= 0:
        raise ValueError("returns must vary")
    dispersion = float(np.sum((sample-mu*np.eye(n))**2)/n)
    error = float((np.sum(np.sum(centred**2, axis=1)**2)-t*np.sum(sample**2))/(n*t*t))
    shrinkage = 0. if dispersion == 0 else min(max(error, 0.), dispersion)/dispersion
    shrunk = (1-shrinkage)*sample
    shrunk[np.diag_indices(n)] += shrinkage*mu
    return {"method": "LEDOIT_WOLF_2004_SCALED_IDENTITY", "observations": t, "assets": n,
            "normalization": "DIVISOR_T", "shrinkage": shrinkage, "target_variance": mu,
            "covariance": shrunk.tolist(),
            "sample_condition_number": float(np.linalg.cond(sample)) if t > n else None,
            "condition_number": float(np.linalg.cond(shrunk)), "classification": "MODEL_ESTIMATE"}


def _inverse_variance(cov: np.ndarray, members: list[int]) -> float:
    block = cov[np.ix_(members, members)]
    w = 1/np.diag(block)
    w /= w.sum()
    return float(w @ block @ w)


def hierarchical_risk_parity(covariance) -> dict:
    """Lopez de Prado (2016) hierarchical risk parity.

    Single linkage on the Euclidean distance between correlation-distance
    columns, the resulting leaf order, then recursive bisection with
    inverse-variance cluster variances. No matrix inversion is used.
    """
    cov = _covariance(covariance)
    n = len(cov)
    if n < 2:
        raise ValueError("covariance must hold at least 2 assets")
    sd = np.sqrt(np.diag(cov))
    distance = np.sqrt(.5*(1-np.clip(cov/np.outer(sd, sd), -1, 1)))
    np.fill_diagonal(distance, 0.)
    order = [int(i) for i in leaves_list(linkage(pdist(distance), "single"))]
    weights = np.ones(n)
    clusters = [order]
    while clusters:
        halves = []
        for cluster in clusters:
            if len(cluster) < 2:
                continue
            left, right = cluster[:len(cluster)//2], cluster[len(cluster)//2:]
            vl, vr = _inverse_variance(cov, left), _inverse_variance(cov, right)
            weights[left] *= vr/(vl+vr)
            weights[right] *= vl/(vl+vr)
            halves += [left, right]
        clusters = halves
    naive = 1/np.diag(cov)
    naive /= naive.sum()
    variance = float(weights @ cov @ weights)
    return {"method": "HIERARCHICAL_RISK_PARITY", "linkage": "SINGLE",
            "distance": "EUCLIDEAN_OF_CORRELATION_DISTANCE", "order": order, "weights": weights.tolist(),
            "volatility": math.sqrt(variance),
            "risk_contributions": (weights*(cov @ weights)/variance).tolist(),
            "inverse_variance_weights": naive.tolist(),
            "inverse_variance_volatility": math.sqrt(float(naive @ cov @ naive)),
            "classification": "MODEL_ESTIMATE"}


def minimum_cvar(scenarios, *, confidence: float = .95, max_weight: float = 1.,
                 minimum_return: float | None = None) -> dict:
    """Long-only fully invested minimum-CVaR weights (Rockafellar-Uryasev linear program)."""
    r = _returns(scenarios, "scenarios")
    t, n = r.shape
    level, cap = checks.confidence(confidence), _cap(max_weight, n)
    mean = r.mean(axis=0)
    cost = np.concatenate([np.zeros(n), [1.], np.full(t, 1/((1-level)*t))])
    upper = hstack([-r, -np.ones((t, 1)), -sparse_eye(t)], format="csr")
    bound = np.zeros(t)
    if minimum_return is not None:
        floor = checks.number(minimum_return, "minimum_return")
        upper = vstack([upper, csr_matrix(np.concatenate([-mean, np.zeros(t+1)])[None, :])], format="csr")
        bound = np.append(bound, -floor)
    solution = linprog(cost, A_ub=upper, b_ub=bound, A_eq=np.concatenate([np.ones(n), np.zeros(t+1)])[None, :],
                       b_eq=[1.], bounds=[(0., cap)]*n+[(None, None)]+[(0., None)]*t, method="highs")
    if solution.status == 2:
        raise ValueError("no long-only portfolio meets minimum_return under max_weight")
    if not solution.success:
        raise RuntimeError("minimum-CVaR linear program failed")
    w = solution.x[:n]
    if abs(w.sum()-1) > FEASIBILITY or w.min() < -FEASIBILITY or w.max() > cap+FEASIBILITY:
        raise RuntimeError("minimum-CVaR linear program returned infeasible weights")
    w = np.clip(w, 0., cap)
    losses = -r @ w
    tail, var = tail_weights(losses, level)
    cvar = float(tail @ losses)
    if abs(cvar-solution.fun) > 1e-7*max(1., abs(cvar)):
        raise RuntimeError("minimum-CVaR objective does not match its recomputed tail loss")
    equal = -r.mean(axis=1)
    return {"method": "MINIMUM_CVAR_LINEAR_PROGRAM", "loss_convention": "POSITIVE_LOSS", "confidence": level,
            "scenarios": t, "assets": n, "max_weight": cap, "weights": w.tolist(),
            "cvar": cvar, "var": var, "expected_return": float(mean @ w),
            "equal_weight_cvar": float(tail_weights(equal, level)[0] @ equal),
            "scenario_limits": "IN_SAMPLE_SCENARIOS_ONLY_NO_OUT_OF_SAMPLE_CLAIM",
            "classification": "MODEL_ESTIMATE"}


def efficient_frontier(expected_returns, covariance, *, points: int = 11, max_weight: float = 1.) -> dict:
    """Long-only fully invested minimum-variance portfolios across attainable returns."""
    cov = _covariance(covariance)
    n = len(cov)
    if n < 2:
        raise ValueError("covariance must hold at least 2 assets")
    mu = checks.array(expected_returns, "expected_returns", dimensions=1, maximum=MAX_ASSETS)
    if mu.shape != (n,):
        raise ValueError("expected_returns must hold one value per asset")
    count, cap = checks.integer(points, "points", 2, 50), _cap(max_weight, n)
    scale = float(np.max(np.diag(cov)))
    scaled = cov/scale

    def solve(start: np.ndarray, target: float | None) -> np.ndarray:
        constraints = [{"type": "eq", "fun": lambda w: w.sum()-1, "jac": lambda w: np.ones(n)}]
        if target is not None:
            constraints.append({"type": "eq", "fun": lambda w: mu @ w-target, "jac": lambda w: mu})
        result = minimize(lambda w: w @ scaled @ w, start, jac=lambda w: 2*scaled @ w, method="SLSQP",
                          bounds=[(0., cap)]*n, constraints=constraints, options={"ftol": 1e-15, "maxiter": 1000})
        w = result.x
        residual = max(abs(w.sum()-1), -w.min(), w.max()-cap, 0. if target is None else abs(mu @ w-target))
        if not result.success or residual > FEASIBILITY:
            raise RuntimeError("frontier optimizer failed")
        return np.clip(w, 0., cap)

    def row(w: np.ndarray) -> dict:
        return {"expected_return": float(mu @ w), "volatility": math.sqrt(float(w @ cov @ w)), "weights": w.tolist()}

    lowest = solve(np.full(n, 1/n), None)
    highest, room = np.zeros(n), 1.
    for index in np.argsort(-mu, kind="stable"):
        highest[index] = min(cap, room)
        room -= highest[index]
    low, high = float(mu @ lowest), float(mu @ highest)
    frontier = [row(lowest)]
    if high-low > 1e-12*max(1., abs(high)):
        for step in range(1, count):
            share = step/(count-1)
            target = low+share*(high-low)
            frontier.append(row(highest if step == count-1 else solve((1-share)*lowest+share*highest, target)))
    return {"method": "LONG_ONLY_MEAN_VARIANCE_FRONTIER", "assets": n, "max_weight": cap,
            "minimum_variance": frontier[0], "maximum_return": row(highest), "points": frontier,
            "input_limits": "EXPECTED_RETURNS_ARE_CALLER_ASSUMPTIONS_NOT_FORECAST_EVIDENCE",
            "classification": "MODEL_ESTIMATE"}
