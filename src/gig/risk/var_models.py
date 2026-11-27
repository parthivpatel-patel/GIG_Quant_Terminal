"""Bounded value-at-risk research on caller-supplied inputs.

Losses are positive numbers throughout (``loss_convention``). Parametric and
scenario decompositions are Euler allocations and sum to the portfolio figure.
Nothing here reads positions, marks or limits; results are model estimates.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.special import xlogy
from scipy.stats import binom, chi2, norm

from gig import checks

MAX_ASSETS = 64
MAX_OBSERVATIONS = 5000
MAX_DRAWS = 4_000_000
CONVENTION = "POSITIVE_LOSS"


def _covariance(value, name: str = "covariance") -> np.ndarray:
    cov = checks.array(value, name, dimensions=2, maximum=MAX_ASSETS**2)
    if cov.shape[0] != cov.shape[1] or not np.allclose(cov, cov.T, rtol=0, atol=1e-12):
        raise ValueError(f"{name} must be square and symmetric")
    if np.linalg.eigvalsh(cov).min() <= 0:
        raise ValueError(f"{name} must be positive definite; regularization must be explicit")
    return cov


def _exposures(value, n: int, name: str = "weights") -> np.ndarray:
    result = checks.array(value, name, dimensions=1, maximum=MAX_ASSETS)
    if result.shape != (n,) or not result.any():
        raise ValueError(f"{name} must hold one value per asset and not be all zero")
    return result


def _mean(value, n: int) -> np.ndarray:
    if value is None:
        return np.zeros(n)
    result = checks.array(value, "mean", dimensions=1, maximum=MAX_ASSETS)
    if result.shape != (n,):
        raise ValueError("mean must hold one value per asset")
    return result


def tail_weights(losses: np.ndarray, confidence: float) -> tuple[np.ndarray, float]:
    """Scenario weights of the Rockafellar-Uryasev tail and its value-at-risk.

    The worst ``(1-confidence)*T`` scenarios carry the mass; a fractional
    scenario at the boundary is weighted by its fraction, so the weighted loss
    equals ``min_z z + mean(max(loss-z, 0))/(1-confidence)`` exactly.
    """
    order = np.argsort(-losses, kind="stable")
    tail = (1-confidence)*len(losses)
    whole = min(int(math.floor(tail+1e-9)), len(losses))
    weights = np.zeros(len(losses))
    weights[order[:whole]] = 1.
    fraction = tail-whole
    if whole < len(losses) and fraction > 1e-9:
        weights[order[whole]] = fraction
        boundary = order[whole]
    else:
        boundary = order[whole-1]
    return weights/weights.sum(), float(losses[boundary])


def parametric(weights, covariance, *, mean=None, confidence: float = .99, horizon: float = 1.) -> dict:
    """Normal VaR and expected shortfall with marginal and component figures."""
    cov = _covariance(covariance)
    w = _exposures(weights, len(cov))
    mu = _mean(mean, len(cov))
    level, h = checks.confidence(confidence), checks.number(horizon, "horizon", positive=True, high=1e4)
    z = float(norm.ppf(level))
    shortfall = float(norm.pdf(z)/(1-level))
    sigma = math.sqrt(float(w @ cov @ w))
    gradient = cov @ w/sigma
    rows = {}
    for key, multiple in (("var", z), ("expected_shortfall", shortfall)):
        marginal = -h*mu + multiple*math.sqrt(h)*gradient
        rows[key] = float(w @ marginal)
        rows[f"marginal_{key}"] = marginal.tolist()
        rows[f"component_{key}"] = (w*marginal).tolist()
    return {"method": "NORMAL_PARAMETRIC", "loss_convention": CONVENTION, "confidence": level, "horizon": h,
            "portfolio_volatility": sigma, "normal_quantile": z, **rows,
            "decomposition": "EULER_COMPONENTS_SUM_TO_TOTAL", "classification": "MODEL_ESTIMATE"}


def cornish_fisher(returns, *, confidence: float = .99) -> dict:
    """Skewness and kurtosis adjusted quantile VaR of one return series."""
    r = checks.array(returns, "returns", dimensions=1, maximum=MAX_OBSERVATIONS, minimum=30)
    level = checks.confidence(confidence)
    centred = r-r.mean()
    m2 = float(np.mean(centred**2))
    if m2 <= 0:
        raise ValueError("returns must vary")
    skew, kurt = float(np.mean(centred**3)/m2**1.5), float(np.mean(centred**4)/m2**2-3)
    sd = float(r.std(ddof=1))

    def adjust(z):
        return z + (z**2-1)*skew/6 + (z**3-3*z)*kurt/24 - (2*z**3-5*z)*skew**2/36

    z = float(norm.ppf(1-level))
    grid = adjust(np.linspace(-4, 4, 801))
    return {"method": "CORNISH_FISHER_FOURTH_ORDER", "loss_convention": CONVENTION, "confidence": level,
            "observations": len(r), "mean": float(r.mean()), "standard_deviation": sd,
            "skewness": skew, "excess_kurtosis": kurt,
            "normal_quantile": z, "adjusted_quantile": float(adjust(z)),
            "var": float(-(r.mean()+sd*adjust(z))), "normal_var": float(-(r.mean()+sd*z)),
            "historical_var": float(-np.quantile(r, 1-level)),
            # Outside its domain of validity the expansion is not a quantile function.
            "quantile_function_monotone": bool(np.all(np.diff(grid) > 0)),
            "classification": "MODEL_ESTIMATE"}


def _asset_returns(value) -> np.ndarray:
    r = checks.array(value, "asset_returns", dimensions=2, maximum=MAX_OBSERVATIONS*MAX_ASSETS)
    if not 20 <= r.shape[0] <= MAX_OBSERVATIONS or not 1 <= r.shape[1] <= MAX_ASSETS:
        raise ValueError("asset_returns must hold 20 to 5000 scenarios of 1 to 64 assets")
    return r


def historical_components(asset_returns, weights, *, confidence: float = .99) -> dict:
    """Scenario VaR and expected shortfall with exact per-asset tail contributions."""
    r = _asset_returns(asset_returns)
    w = _exposures(weights, r.shape[1])
    level = checks.confidence(confidence)
    contributions = -r*w
    losses = contributions.sum(axis=1)
    tail, var = tail_weights(losses, level)
    return {"method": "HISTORICAL_SCENARIOS", "loss_convention": CONVENTION, "confidence": level,
            "scenarios": len(r), "tail_scenarios": float((1-level)*len(r)),
            "var": var, "expected_shortfall": float(tail @ losses),
            "component_expected_shortfall": (tail @ contributions).tolist(),
            "worst_loss": float(losses.max()), "decomposition": "TAIL_SCENARIO_AVERAGES_SUM_TO_TOTAL",
            "classification": "MODEL_ESTIMATE"}


def monte_carlo(weights, covariance, *, mean=None, confidence: float = .99, horizon: float = 1.,
                paths: int = 20000, seed: int = 0, degrees_of_freedom: float | None = None) -> dict:
    """Seeded Gaussian or Student-t simulation with the stated covariance."""
    cov = _covariance(covariance)
    w = _exposures(weights, len(cov))
    mu = _mean(mean, len(cov))
    level, h = checks.confidence(confidence), checks.number(horizon, "horizon", positive=True, high=1e4)
    count, seed = checks.integer(paths, "paths", 1000, 200000), checks.integer(seed, "seed", 0, 2**32-1)
    if count*len(cov) > MAX_DRAWS:
        raise ValueError("paths times assets exceeds four million draws")
    rng = np.random.default_rng(seed)
    shocks = rng.standard_normal((count, len(cov))) @ np.linalg.cholesky(cov).T
    if degrees_of_freedom is not None:
        nu = checks.number(degrees_of_freedom, "degrees_of_freedom", low=2.5, high=200)
        shocks *= (np.sqrt((nu-2)/nu)/np.sqrt(rng.chisquare(nu, count)/nu))[:, None]
    contributions = -(h*mu + math.sqrt(h)*shocks)*w
    losses = contributions.sum(axis=1)
    tail, var = tail_weights(losses, level)
    result = {"method": "MONTE_CARLO", "loss_convention": CONVENTION, "confidence": level, "horizon": h,
              "distribution": "GAUSSIAN" if degrees_of_freedom is None else "STUDENT_T_COVARIANCE_MATCHED",
              "paths": count, "seed": seed, "var": var, "expected_shortfall": float(tail @ losses),
              "component_expected_shortfall": (tail @ contributions).tolist(),
              "sampling_error": "NOT_ESTIMATED_RERUN_WITH_OTHER_SEEDS", "classification": "MODEL_ESTIMATE"}
    if degrees_of_freedom is None:
        reference = parametric(w, cov, mean=mu, confidence=level, horizon=h)
        result["analytic_var"], result["analytic_expected_shortfall"] = reference["var"], reference["expected_shortfall"]
    else:
        result["degrees_of_freedom"] = nu
    return result


def backtest(returns, var_forecasts, *, confidence: float = .99) -> dict:
    """Kupiec coverage, Christoffersen independence and a Basel-style zone.

    ``var_forecasts[t]`` is the positive loss forecast made before
    ``returns[t]``; an exception is a return below its negative.
    """
    r = checks.array(returns, "returns", dimensions=1, maximum=MAX_OBSERVATIONS, minimum=20)
    v = checks.array(var_forecasts, "var_forecasts", dimensions=1, maximum=MAX_OBSERVATIONS, minimum=20)
    if r.shape != v.shape or np.any(v <= 0):
        raise ValueError("var_forecasts must be positive losses, one per return")
    level = checks.confidence(confidence)
    p, hits = 1-level, r < -v
    t, x = len(r), int(hits.sum())
    rate = x/t
    coverage = float(-2*(xlogy(t-x, 1-p)+xlogy(x, p)-xlogy(t-x, 1-rate)-xlogy(x, rate)))
    before, after = hits[:-1], hits[1:]
    n00, n01 = int((~before & ~after).sum()), int((~before & after).sum())
    n10, n11 = int((before & ~after).sum()), int((before & after).sum())
    pi = (n01+n11)/(t-1)
    pi01 = n01/(n00+n01) if n00+n01 else 0.
    pi11 = n11/(n10+n11) if n10+n11 else 0.
    independence = float(-2*(xlogy(n00+n10, 1-pi)+xlogy(n01+n11, pi)
                             - xlogy(n00, 1-pi01)-xlogy(n01, pi01)-xlogy(n10, 1-pi11)-xlogy(n11, pi11)))
    coverage, independence = max(coverage, 0.), max(independence, 0.)
    cumulative = float(binom.cdf(x, t, p))
    return {"method": "VAR_EXCEPTION_BACKTEST", "confidence": level, "observations": t, "exceptions": x,
            "expected_exceptions": p*t, "exception_rate": rate,
            "kupiec_statistic": coverage, "kupiec_p_value": float(chi2.sf(coverage, 1)),
            "transitions": {"n00": n00, "n01": n01, "n10": n10, "n11": n11},
            "independence_statistic": independence, "independence_p_value": float(chi2.sf(independence, 1)),
            "conditional_coverage_statistic": coverage+independence,
            "conditional_coverage_p_value": float(chi2.sf(coverage+independence, 2)),
            "cumulative_probability": cumulative,
            # The Basel 250-day 99% table generalized by its own probability thresholds.
            "basel_style_zone": "GREEN" if cumulative < .95 else "YELLOW" if cumulative < .9999 else "RED",
            "test_limits": "ASYMPTOTIC_CHI_SQUARE_LOW_POWER_IN_SHORT_SAMPLES", "classification": "MODEL_ESTIMATE"}
