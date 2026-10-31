"""Macro overlay: shrink gross when VIX / HY spreads are elevated. Uses t-1 only."""

from __future__ import annotations

import pandas as pd


def vix_gross_scale(vix: pd.Series) -> pd.Series:
    """
    Scale in (0.5, 1.0]. Uses prior session VIX so close_t decision does not
    need same-bar VIX print.
    """
    lagged = vix.astype(float).shift(1)
    scale = pd.Series(1.0, index=vix.index)
    scale = scale.mask(lagged > 20, 0.75)
    scale = scale.mask(lagged > 25, 0.50)
    return scale.fillna(1.0)


def series_to_trading_index(macro: pd.DataFrame, trading_index: pd.DatetimeIndex, series_id: str) -> pd.Series:
    sub = macro.loc[macro["series_id"] == series_id, ["dt", "value"]].copy()
    if sub.empty:
        return pd.Series(dtype=float)
    sub["dt"] = pd.to_datetime(sub["dt"])
    s = sub.drop_duplicates("dt").set_index("dt")["value"].sort_index()
    return s.reindex(trading_index).ffill()
