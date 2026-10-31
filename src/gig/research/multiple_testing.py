"""Multiple-testing corrections used when many strategies are screened."""

from __future__ import annotations

import math

import numpy as np
from scipy import stats


def deflated_sharpe(
    sharpe: float,
    n_trials: int,
    n_obs: int,
    skew: float = 0.0,
    kurtosis: float = 3.0,
    sharpe_std: float | None = None,
) -> dict[str, float]:
    """
    Deflated Sharpe Ratio (Bailey & López de Prado, 2014).

    Probability that the observed Sharpe is not the max of `n_trials` noise Sharpes.
    """
    if n_obs < 2 or n_trials < 1:
        return {"dsr": float("nan"), "psr": float("nan")}

    sr_std = sharpe_std or _sharpe_se(sharpe, n_obs, skew, kurtosis)
    if n_trials == 1:
        emax = 0.0
    else:
        # Bailey–López de Prado expected max Sharpe under N independent trials
        emax = sr_std * (
            (1 - np.euler_gamma) * stats.norm.ppf(1 - 1 / n_trials)
            + np.euler_gamma * stats.norm.ppf(1 - 1 / (n_trials * np.e))
        )
    dsr = float(stats.norm.cdf((sharpe - emax) / sr_std)) if sr_std > 0 else float("nan")
    psr = probabilistic_sharpe(sharpe, n_obs, skew, kurtosis, sr_benchmark=0.0)
    return {"dsr": dsr, "psr": psr, "expected_max_sharpe": float(emax)}


def probabilistic_sharpe(
    sharpe: float,
    n_obs: int,
    skew: float = 0.0,
    kurtosis: float = 3.0,
    sr_benchmark: float = 0.0,
) -> float:
    se = _sharpe_se(sharpe, n_obs, skew, kurtosis)
    if se <= 0:
        return float("nan")
    z = (sharpe - sr_benchmark) / se
    return float(stats.norm.cdf(z))


def _sharpe_se(sharpe: float, n_obs: int, skew: float, kurtosis: float) -> float:
    # Lo (2002) / Mertens
    return math.sqrt(
        (1 - skew * sharpe + ((kurtosis - 1) / 4.0) * sharpe**2) / (n_obs - 1)
    )


def bonferroni_pvalue(pvalues: list[float]) -> list[float]:
    m = max(len(pvalues), 1)
    return [min(1.0, p * m) for p in pvalues]
