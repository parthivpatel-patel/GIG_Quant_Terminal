"""Spearman rank IC and Newey–West t-statistics. Standard CS research metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def spearman_ic(factor: pd.DataFrame, forward_returns: pd.DataFrame) -> pd.Series:
    """Date-by-date Spearman rank correlation between scores and forward returns."""
    a = factor.rank(axis=1, method="average")
    b = forward_returns.reindex_like(factor).rank(axis=1, method="average")
    return a.corrwith(b, axis=1).dropna()


def newey_west_tstat(series: pd.Series, lags: int | None = None) -> tuple[float, float, float]:
    """
    Mean, Newey–West t-stat, and two-sided p-value for a time series (e.g. daily IC).

    Default lag = floor(4 * (T/100)^(2/9)) as in Newey & West (1994).
    """
    x = series.dropna().to_numpy(dtype=float)
    t = x.size
    if t < 8:
        return float("nan"), float("nan"), float("nan")
    mu = float(x.mean())
    if lags is None:
        lags = int(np.floor(4 * (t / 100.0) ** (2.0 / 9.0)))
    u = x - mu
    gamma0 = float(np.dot(u, u) / t)
    var = gamma0
    for lag in range(1, lags + 1):
        gamma = float(np.dot(u[lag:], u[:-lag]) / t)
        weight = 1.0 - lag / (lags + 1)
        var += 2.0 * weight * gamma
    se = np.sqrt(max(var, 0.0) / t)
    if se == 0:
        return mu, float("nan"), float("nan")
    tstat = mu / se
    pvalue = float(2 * (1 - stats.t.cdf(abs(tstat), df=t - 1)))
    return mu, float(tstat), pvalue


def information_ratio(ic: pd.Series) -> float:
    s = ic.dropna()
    if s.std(ddof=1) == 0 or len(s) < 2:
        return float("nan")
    return float(s.mean() / s.std(ddof=1) * np.sqrt(252))


def ic_summary(factor: pd.DataFrame, forward_returns: pd.DataFrame) -> dict[str, float]:
    ic = spearman_ic(factor, forward_returns)
    mu, tstat, pvalue = newey_west_tstat(ic)
    return {
        "ic_mean": mu,
        "ic_std": float(ic.std(ddof=1)) if len(ic) > 1 else float("nan"),
        "ic_ir": information_ratio(ic),
        "ic_tstat_nw": tstat,
        "ic_pvalue_nw": pvalue,
        "ic_hit_rate": float((ic > 0).mean()) if len(ic) else float("nan"),
        "n_obs": float(len(ic)),
    }
