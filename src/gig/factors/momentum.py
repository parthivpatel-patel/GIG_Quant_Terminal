"""Jegadeesh–Titman 12-1 momentum and 1-month short-term reversal."""

from __future__ import annotations

import pandas as pd

from gig.factors.base import Factor
from gig.types import MarketPanel


class Momentum12m1(Factor):
    """Skip the most recent 21 sessions to avoid short-term reversal contamination."""

    name = "momentum_12_1"

    def __init__(self, lookback: int = 252, skip: int = 21) -> None:
        self.lookback = lookback
        self.skip = skip

    def compute(self, panel: MarketPanel) -> pd.DataFrame:
        px = panel.close
        lagged = px.shift(self.skip)
        past = px.shift(self.lookback)
        return lagged / past - 1.0


class ShortTermReversal(Factor):
    """Negative 21-day return. Classic microstructure / liquidity effect."""

    name = "strev_21d"

    def __init__(self, window: int = 21) -> None:
        self.window = window

    def compute(self, panel: MarketPanel) -> pd.DataFrame:
        return -(panel.close / panel.close.shift(self.window) - 1.0)
