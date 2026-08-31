"""Historical VaR and Expected Shortfall from a PnL series."""

from __future__ import annotations

import numpy as np
import pandas as pd


def historical_var_es(
    returns: pd.Series,
    confidence: float = 0.99,
) -> dict[str, float]:
    r = returns.dropna().to_numpy(dtype=float)
    if r.size < 20:
        return {"var": float("nan"), "es": float("nan"), "n": float(r.size)}
    q = np.quantile(r, 1 - confidence)
    tail = r[r <= q]
    es = float(tail.mean()) if tail.size else float(q)
    return {"var": float(q), "es": es, "n": float(r.size)}
