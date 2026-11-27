"""Fama-MacBeth cross-sectional regression with Newey-West standard errors.

A balanced caller-supplied panel only: no missing-value handling, no
errors-in-variables correction for estimated exposures.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.stats import norm

from gig import checks

MAX_VALUES = 400_000


def newey_west_mean(series: np.ndarray, lags: int) -> np.ndarray:
    """Bartlett-kernel standard error of each column mean, without small-sample correction."""
    t = len(series)
    centred = series-series.mean(axis=0)
    variance = np.sum(centred**2, axis=0)/t
    for lag in range(1, lags+1):
        variance += 2*(1-lag/(lags+1))*np.sum(centred[lag:]*centred[:-lag], axis=0)/t
    return np.sqrt(np.maximum(variance, 0.)/t)


def fama_macbeth(returns, exposures, *, lags: int | None = None) -> dict:
    """Period-by-period OLS of returns on exposures, then time-series inference on the slopes.

    ``returns`` is periods x assets; ``exposures`` is periods x assets x
    factors and must already be known before each period's return.
    """
    r = checks.array(returns, "returns", dimensions=2, maximum=MAX_VALUES)
    x = checks.array(exposures, "exposures", dimensions=3, maximum=MAX_VALUES)
    t, n = r.shape
    if x.shape[:2] != (t, n) or not 1 <= x.shape[2] <= 8 or n < x.shape[2]+2 or not 3 <= t <= 1000:
        raise ValueError("need 3 to 1000 periods, 1 to 8 factors and at least factors+2 assets per period")
    k = x.shape[2]
    width = int(math.floor(4*(t/100)**(2/9))) if lags is None else checks.integer(lags, "lags", 0, t-1)
    premia, fit = np.empty((t, k+1)), np.empty(t)
    for period in range(t):
        design = np.column_stack([np.ones(n), x[period]])
        if np.linalg.matrix_rank(design) < k+1:
            raise ValueError("exposures are collinear in at least one period")
        premia[period], *_ = np.linalg.lstsq(design, r[period], rcond=None)
        residual = r[period]-design @ premia[period]
        total = float(np.sum((r[period]-r[period].mean())**2))
        fit[period] = 1-float(residual @ residual)/total if total > 0 else float("nan")
    mean = premia.mean(axis=0)
    simple, robust = premia.std(axis=0, ddof=1)/math.sqrt(t), newey_west_mean(premia, width)
    with np.errstate(divide="ignore", invalid="ignore"):
        statistic = np.where(robust > 0, mean/robust, np.nan)
    names = ["intercept"]+[f"factor_{index+1}" for index in range(k)]
    rows = [{"term": names[i], "premium": float(mean[i]), "fama_macbeth_standard_error": float(simple[i]),
             "newey_west_standard_error": float(robust[i]),
             "t_statistic": None if np.isnan(statistic[i]) else float(statistic[i]),
             "normal_p_value": None if np.isnan(statistic[i]) else float(2*norm.sf(abs(statistic[i])))}
            for i in range(k+1)]
    return {"method": "FAMA_MACBETH_1973", "periods": t, "assets": n, "factors": k, "newey_west_lags": width,
            "premia": rows, "average_r_squared": None if np.isnan(fit).all() else float(np.nanmean(fit)),
            "period_premia": premia.tolist() if t <= 250 else None,
            "inference_limits": "BALANCED_PANEL_NO_ERRORS_IN_VARIABLES_OR_MULTIPLE_TESTING_CORRECTION",
            "classification": "MODEL_ESTIMATE"}
