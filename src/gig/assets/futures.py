"""
Futures utilities for time-series momentum (Moskowitz, Ooi, Pedersen 2012).

A `FuturesCurve` is nearby + deferred closes. Excess return uses the nearby
contract and a simple roll: when the nearby tenor expires, splice on the
deferred using the roll date's ratio so the series is continuous.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(slots=True)
class FuturesCurve:
    nearby: pd.Series
    deferred: pd.Series
    roll_dates: pd.DatetimeIndex

    def continuous_price(self) -> pd.Series:
        px = self.nearby.astype(float).copy()
        ratio = (self.deferred / self.nearby).reindex(px.index)
        adj = 1.0
        for rd in self.roll_dates:
            if rd in ratio.index and np.isfinite(ratio.loc[rd]):
                adj *= float(ratio.loc[rd])
                px.loc[px.index > rd] = px.loc[px.index > rd] / float(ratio.loc[rd]) * adj
        return px


def roll_return(nearby: pd.Series, deferred: pd.Series) -> pd.Series:
    """Log roll yield ≈ ln(nearby / deferred). Positive = backwardation."""
    return np.log(nearby / deferred)


def tsmom_signal(excess_return: pd.Series, lookback: int = 252, vol_lookback: int = 60) -> pd.Series:
    """
    Sign of trailing excess return, scaled to target 1 (unit) ex-ante vol.

    Position_t = sign(R_{t-lookback:t}) * (σ_target / σ_t), using only past data.
    """
    mom = excess_return.rolling(lookback, min_periods=lookback).sum()
    sign = np.sign(mom.shift(1))
    vol = excess_return.rolling(vol_lookback, min_periods=vol_lookback).std().shift(1)
    vol = vol.replace(0, np.nan)
    target = 0.15 / np.sqrt(252)
    return sign * (target / vol)
