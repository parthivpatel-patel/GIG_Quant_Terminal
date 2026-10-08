"""
Random-walk Kalman regression for a dynamic hedge ratio.

Observation: y_t = alpha_t + beta_t * x_t + noise
State:       [beta, alpha] follows a random walk

The filter is causal. Smoothed states are not returned, so a weight at t
does not depend on observations after t.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class KalmanRegressionResult:
    beta: np.ndarray
    alpha: np.ndarray
    spread: np.ndarray
    log_likelihood: float
    observation_variance: float
    state_variance: float

    def as_dict(self) -> dict:
        return {
            "model": "kalman_dynamic_regression",
            "beta_last": float(self.beta[-1]),
            "alpha_last": float(self.alpha[-1]),
            "spread_last": float(self.spread[-1]),
            "log_likelihood": self.log_likelihood,
            "observation_variance": self.observation_variance,
            "state_variance": self.state_variance,
            "causal": True,
            "status": "RESEARCH",
        }


def kalman_dynamic_regression(
    x: np.ndarray,
    y: np.ndarray,
    observation_variance: float | None = None,
    state_variance: float = 1e-5,
) -> KalmanRegressionResult:
    """Filter a two-state hedge ratio. `state_variance` is the random-walk variance per step."""
    xs = np.asarray(x, dtype=float)
    ys = np.asarray(y, dtype=float)
    if xs.shape != ys.shape or xs.ndim != 1:
        raise ValueError("x and y must be 1-d and the same length")
    mask = np.isfinite(xs) & np.isfinite(ys)
    xs = xs[mask]
    ys = ys[mask]
    if xs.size < 20:
        raise ValueError("not enough observations for a Kalman regression")
    if state_variance <= 0:
        raise ValueError("state_variance must be positive")

    if observation_variance is None:
        design = np.column_stack([xs, np.ones_like(xs)])
        coef, _, _, _ = np.linalg.lstsq(design, ys, rcond=None)
        resid = ys - design @ coef
        observation_variance = float(max(np.var(resid), 1e-8))
    if observation_variance <= 0:
        raise ValueError("observation_variance must be positive")

    state = np.zeros(2)
    cov = np.eye(2)
    eye = np.eye(2)
    q = eye * state_variance
    beta = np.empty(xs.size)
    alpha = np.empty(xs.size)
    spread = np.empty(xs.size)
    ll = 0.0
    for t in range(xs.size):
        cov = cov + q
        obs = np.array([xs[t], 1.0])
        pred = float(obs @ state)
        innov = float(ys[t] - pred)
        scale = float(obs @ cov @ obs + observation_variance)
        gain = cov @ obs / scale
        state = state + gain * innov
        cov = (eye - np.outer(gain, obs)) @ cov
        cov = 0.5 * (cov + cov.T)
        beta[t] = state[0]
        alpha[t] = state[1]
        spread[t] = float(ys[t] - obs @ state)
        ll += -0.5 * (np.log(2.0 * np.pi) + np.log(scale) + innov * innov / scale)
    return KalmanRegressionResult(
        beta=beta,
        alpha=alpha,
        spread=spread,
        log_likelihood=float(ll),
        observation_variance=float(observation_variance),
        state_variance=float(state_variance),
    )
