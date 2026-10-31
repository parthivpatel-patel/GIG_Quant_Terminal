"""Combine factors with expanding-window IC weights. No future IC used at date t."""

from __future__ import annotations

import numpy as np
import pandas as pd


def ic_weighted_combine(
    factors: dict[str, pd.DataFrame],
    forward_returns: pd.DataFrame,
    min_history: int = 63,
    ic_halflife: int | None = 126,
) -> pd.DataFrame:
    """
    Weight each factor by its expanding (or EWM) Spearman IC vs next-period returns.

    `forward_returns` must already be shifted so that row t is the return earned
    *after* scores at t are known (typically close-to-close from t to t+h).
    """
    names = list(factors)
    aligned = [factors[n].reindex_like(forward_returns) for n in names]
    ics = pd.DataFrame(index=forward_returns.index, columns=names, dtype=float)

    fr = forward_returns.rank(axis=1, method="average")
    for name, fac in zip(names, aligned, strict=True):
        rk = fac.rank(axis=1, method="average")
        ics[name] = rk.corrwith(fr, axis=1)

    if ic_halflife:
        w = ics.ewm(halflife=ic_halflife, min_periods=min_history).mean()
    else:
        w = ics.expanding(min_periods=min_history).mean()

    # Shift weights by 1 so today's combination cannot use today's IC
    w = w.shift(1).clip(lower=0)
    w = w.div(w.sum(axis=1).replace(0, np.nan), axis=0)

    combo = pd.DataFrame(np.nan, index=forward_returns.index, columns=forward_returns.columns)
    for name, fac in zip(names, aligned, strict=True):
        contrib = fac.mul(w[name], axis=0)
        combo = combo.add(contrib, fill_value=0.0)
    available = np.zeros(combo.shape, dtype=bool)
    for fac in aligned:
        available |= fac.notna().to_numpy()
    return combo.where(pd.DataFrame(available, index=combo.index, columns=combo.columns))
