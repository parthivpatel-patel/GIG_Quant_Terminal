"""Long/short book construction from a single cross-section of alpha scores."""

from __future__ import annotations

import numpy as np
import pandas as pd


def dollar_neutral_quantiles(
    scores: pd.Series,
    n_long: int = 15,
    n_short: int = 15,
    max_name_weight: float = 0.05,
    gross_leverage: float = 2.0,
) -> pd.Series:
    """
    Equal-weight (then clipped) long top `n_long` and short bottom `n_short`.

    Gross leverage 2.0 means +1 long book and -1 short book. Net is ~0.
    Names with NaN scores are ineligible.
    """
    s = scores.replace([np.inf, -np.inf], np.nan).dropna()
    weights = pd.Series(0.0, index=scores.index)
    if len(s) < n_long + n_short:
        return weights

    ranked = s.sort_values(ascending=False)
    longs = ranked.index[:n_long]
    shorts = ranked.index[-n_short:]

    target = gross_leverage / 2.0
    long_w = min(target / n_long, max_name_weight)
    short_w = min(target / n_short, max_name_weight)
    weights.loc[longs] = long_w
    weights.loc[shorts] = -short_w
    return weights.reindex(scores.index).fillna(0.0)


def sector_net_exposure(weights: pd.Series, sectors: pd.Series) -> pd.Series:
    aligned = sectors.reindex(weights.index).fillna("unknown")
    return weights.groupby(aligned).sum()
