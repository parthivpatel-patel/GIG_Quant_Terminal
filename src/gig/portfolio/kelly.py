"""
Fractional Kelly sizing.

The continuous approximation is f* = Σ^{-1} μ. The returned vector is scaled by
`fraction` (default quarter-Kelly) and then capped so gross exposure cannot
exceed `max_gross`. Kelly never raises a limit that the caller already set.
"""

from __future__ import annotations

import numpy as np


def fractional_kelly(
    expected_return: np.ndarray,
    covariance: np.ndarray,
    fraction: float = 0.25,
    max_gross: float = 1.0,
) -> np.ndarray:
    """Return weights. `expected_return` is per period, matching `covariance`."""
    mu = np.asarray(expected_return, dtype=float)
    cov = np.asarray(covariance, dtype=float)
    if mu.ndim != 1:
        raise ValueError("expected_return must be 1-d")
    if cov.shape != (mu.size, mu.size):
        raise ValueError("covariance must be square and match expected_return")
    if not 0.0 < fraction <= 1.0:
        raise ValueError("fraction must lie in (0, 1]")
    if max_gross <= 0:
        raise ValueError("max_gross must be positive")
    if not np.isfinite(mu).all() or not np.isfinite(cov).all():
        raise ValueError("inputs must be finite")
    # Ridge keeps a singular sample covariance from inventing infinite size.
    ridge = 1e-8 * np.eye(mu.size)
    raw = np.linalg.solve(cov + ridge, mu)
    weights = fraction * raw
    gross = float(np.abs(weights).sum())
    if gross > max_gross:
        weights = weights * (max_gross / gross)
    return weights
