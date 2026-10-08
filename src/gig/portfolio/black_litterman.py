"""
Black–Litterman posterior expected returns.

Equilibrium returns are π = δ Σ w. Views are P μ = q with uncertainty Ω.
The posterior is the precision-weighted blend of equilibrium and views.
It is an input to an optimizer, not a set of portfolio weights.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class BlackLittermanResult:
    equilibrium: np.ndarray
    posterior: np.ndarray
    posterior_covariance: np.ndarray

    def as_dict(self) -> dict:
        return {
            "model": "black_litterman",
            "equilibrium": self.equilibrium.tolist(),
            "posterior": self.posterior.tolist(),
            "status": "RESEARCH",
        }


def black_litterman(
    covariance: np.ndarray,
    market_weights: np.ndarray,
    view_pick: np.ndarray,
    view_return: np.ndarray,
    view_variance: np.ndarray,
    risk_aversion: float = 2.5,
    tau: float = 0.05,
) -> BlackLittermanResult:
    """
    `view_pick` is P (n_views × n_assets). `view_return` is q.
    `view_variance` is the diagonal of Ω.
    """
    sigma = np.asarray(covariance, dtype=float)
    weights = np.asarray(market_weights, dtype=float)
    pick = np.asarray(view_pick, dtype=float)
    target = np.asarray(view_return, dtype=float)
    omega_diag = np.asarray(view_variance, dtype=float)
    n = weights.size
    if sigma.shape != (n, n):
        raise ValueError("covariance shape does not match market weights")
    if pick.ndim != 2 or pick.shape[1] != n or pick.shape[0] != target.size:
        raise ValueError("views must be n_views × n_assets")
    if omega_diag.shape != (target.size,) or np.any(omega_diag <= 0):
        raise ValueError("view_variance must be positive and match the views")
    if risk_aversion <= 0 or tau <= 0:
        raise ValueError("risk_aversion and tau must be positive")

    equilibrium = risk_aversion * sigma @ weights
    tau_sigma = tau * sigma
    omega = np.diag(omega_diag)
    precision = np.linalg.inv(tau_sigma)
    view_precision = pick.T @ np.linalg.inv(omega) @ pick
    posterior_cov = np.linalg.inv(precision + view_precision)
    posterior = posterior_cov @ (
        precision @ equilibrium + pick.T @ np.linalg.inv(omega) @ target
    )
    return BlackLittermanResult(
        equilibrium=equilibrium,
        posterior=posterior,
        posterior_covariance=posterior_cov,
    )
