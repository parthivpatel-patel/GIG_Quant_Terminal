"""Simple CS exposures: beta to equal-weight market and net sector weights."""

from __future__ import annotations

import pandas as pd


def factor_exposures(
    weights: pd.Series,
    asset_returns: pd.DataFrame,
    sectors: pd.Series,
    window: int = 63,
) -> dict[str, float]:
    """
    `asset_returns` is date × symbol. Uses the last `window` rows to estimate
    trailing beta of the current book vs the equal-weight universe.
    """
    hist = asset_returns.iloc[-window:]
    book = hist.mul(weights.reindex(hist.columns).fillna(0.0), axis=1).sum(axis=1)
    market = hist.mean(axis=1)
    cov = book.cov(market)
    var = market.var()
    beta = float(cov / var) if var and var > 0 else float("nan")
    return {
        "trailing_beta": beta,
        "net": float(weights.sum()),
        "gross": float(weights.abs().sum()),
        "n_long": float((weights > 0).sum()),
        "n_short": float((weights < 0).sum()),
    }
