"""Point-in-time membership. A name is tradable on date t only if listed at t."""

from __future__ import annotations

import pandas as pd


def point_in_time_mask(
    close: pd.DataFrame,
    listing_start: pd.Series | None = None,
    listing_end: pd.Series | None = None,
    min_history: int = 252,
    min_price: float = 1.0,
    min_adv: pd.DataFrame | None = None,
    min_adv_usd: float = 1_000_000.0,
) -> pd.DataFrame:
    """
    Boolean mask aligned to `close`. True iff the name is in the tradable universe.

    Listing dates are inclusive. History is counted on non-NaN closes only so
    backfilled IPO pads cannot sneak into the universe.
    """
    tradable = close.notna() & (close > min_price)
    if listing_start is not None:
        for sym, start in listing_start.items():
            if sym in tradable.columns:
                tradable.loc[tradable.index < pd.Timestamp(start), sym] = False
    if listing_end is not None:
        for sym, end in listing_end.items():
            if sym in tradable.columns:
                tradable.loc[tradable.index > pd.Timestamp(end), sym] = False

    history = close.notna().cumsum()
    tradable &= history >= min_history

    if min_adv is not None:
        tradable &= min_adv.reindex_like(close).fillna(0) >= min_adv_usd

    return tradable
