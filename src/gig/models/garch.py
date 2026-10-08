"""
Gaussian GARCH(1,1) by maximum likelihood, plus a RiskMetrics EWMA baseline.

The recursion is

    sigma^2_t = omega + alpha * r^2_{t-1} + beta * sigma^2_{t-1}

with omega > 0, alpha >= 0, beta >= 0, and alpha + beta < 1.

A better in-sample likelihood than EWMA does not by itself promote the model.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize


@dataclass(frozen=True, slots=True)
class GarchResult:
    omega: float
    alpha: float
    beta: float
    log_likelihood: float
    aic: float
    bic: float
    conditional_variance: np.ndarray
    next_variance: float
    success: bool

    @property
    def persistence(self) -> float:
        return self.alpha + self.beta

    def annualized_next_vol(self, periods: int = 252) -> float:
        return float(np.sqrt(max(self.next_variance, 0.0) * periods))

    def as_dict(self) -> dict:
        return {
            "model": "GARCH(1,1)",
            "innovations": "gaussian",
            "omega": self.omega,
            "alpha": self.alpha,
            "beta": self.beta,
            "persistence": self.persistence,
            "log_likelihood": self.log_likelihood,
            "aic": self.aic,
            "bic": self.bic,
            "next_variance": self.next_variance,
            "next_vol_annualized": self.annualized_next_vol(),
            "optimizer_success": self.success,
            "status": "RESEARCH",
        }


def _path(returns: np.ndarray, omega: float, alpha: float, beta: float) -> np.ndarray:
    n = returns.size
    var = np.empty(n)
    var[0] = float(np.var(returns))
    for t in range(1, n):
        var[t] = omega + alpha * returns[t - 1] ** 2 + beta * var[t - 1]
    return np.maximum(var, 1e-12)


def _gaussian_ll(returns: np.ndarray, variance: np.ndarray) -> float:
    return float(-0.5 * np.sum(np.log(2.0 * np.pi) + np.log(variance) + returns**2 / variance))


def ewma_variance(returns: np.ndarray, lam: float = 0.94) -> np.ndarray:
    """RiskMetrics EWMA. `lam` is the decay on yesterday's variance, not a fitted GARCH beta."""
    series = np.asarray(returns, dtype=float)
    series = series[np.isfinite(series)]
    if series.size < 2:
        raise ValueError("not enough returns for EWMA")
    if not 0.0 < lam < 1.0:
        raise ValueError("lambda must lie in (0, 1)")
    var = np.empty(series.size)
    var[0] = float(np.var(series))
    shock = 1.0 - lam
    for t in range(1, series.size):
        var[t] = lam * var[t - 1] + shock * series[t - 1] ** 2
    return np.maximum(var, 1e-12)


def fit_garch11(returns: np.ndarray) -> GarchResult:
    """Fit GARCH(1,1). Raises if the sample is too short for a likelihood."""
    series = np.asarray(returns, dtype=float)
    series = series[np.isfinite(series)]
    if series.size < 40:
        raise ValueError("not enough finite returns to fit GARCH")
    unconditional = float(max(np.var(series), 1e-8))

    def unpack(theta: np.ndarray) -> tuple[float, float, float]:
        omega = float(np.exp(theta[0]))
        # softmax-style split of a persistence capped below 1
        persistence = 0.999 / (1.0 + np.exp(-theta[1]))
        share = 1.0 / (1.0 + np.exp(-theta[2]))
        alpha = float(persistence * share)
        beta = float(persistence * (1.0 - share))
        return omega, alpha, beta

    def objective(theta: np.ndarray) -> float:
        omega, alpha, beta = unpack(theta)
        variance = _path(series, omega, alpha, beta)
        return -_gaussian_ll(series, variance)

    theta0 = np.array(
        [
            np.log(unconditional * 0.05),
            2.0,  # persistence near 0.9
            -2.0,  # alpha small relative to beta
        ]
    )
    opt = minimize(objective, theta0, method="L-BFGS-B")
    omega, alpha, beta = unpack(opt.x)
    variance = _path(series, omega, alpha, beta)
    ll = _gaussian_ll(series, variance)
    n_params = 3
    aic = -2.0 * ll + 2.0 * n_params
    bic = -2.0 * ll + n_params * np.log(series.size)
    next_var = omega + alpha * series[-1] ** 2 + beta * variance[-1]
    return GarchResult(
        omega=omega,
        alpha=alpha,
        beta=beta,
        log_likelihood=ll,
        aic=float(aic),
        bic=float(bic),
        conditional_variance=variance,
        next_variance=float(max(next_var, 1e-12)),
        success=bool(opt.success),
    )


def qlike(realized_sq: np.ndarray, forecast: np.ndarray) -> float:
    """Patton QLIKE. Lower is a better variance forecast. Forecasts must be positive."""
    f = np.maximum(np.asarray(forecast, dtype=float), 1e-12)
    r2 = np.asarray(realized_sq, dtype=float)
    return float(np.mean(np.log(f) + r2 / f))
