"""Mean-variance with gross/net/name constraints. Used as a risk overlay, not alpha."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize


def mean_variance_weights(
    expected_excess: pd.Series,
    covariance: pd.DataFrame,
    max_name: float = 0.05,
    max_gross: float = 2.0,
    max_net: float = 0.10,
    risk_aversion: float = 4.0,
) -> pd.Series:
    """
    Max  μ'w - (λ/2) w'Σw  s.t.  sum|w| ≤ G, |sum w| ≤ N, |w_i| ≤ m.

    Falls back to clipped proportional weights if the solver fails.
    """
    names = list(expected_excess.index)
    mu = expected_excess.reindex(names).fillna(0).to_numpy()
    cov = covariance.reindex(index=names, columns=names).fillna(0).to_numpy()
    cov = 0.5 * (cov + cov.T)
    n = len(names)
    if n == 0:
        return pd.Series(dtype=float)

    def objective(w: np.ndarray) -> float:
        return float(-(mu @ w) + 0.5 * risk_aversion * (w @ cov @ w))

    bounds = [(-max_name, max_name)] * n
    cons = [
        {"type": "ineq", "fun": lambda w: max_gross - np.abs(w).sum()},
        {"type": "ineq", "fun": lambda w: max_net - abs(w.sum())},
    ]
    w0 = np.clip(mu / (np.abs(mu).sum() + 1e-12) * min(1.0, max_gross), -max_name, max_name)
    res = minimize(objective, w0, method="SLSQP", bounds=bounds, constraints=cons, options={"maxiter": 400})
    w = res.x if res.success else w0
    return pd.Series(w, index=names)
