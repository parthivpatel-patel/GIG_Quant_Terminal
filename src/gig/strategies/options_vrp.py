"""Variance-risk-premium tilt: short vol when IV > RV, long vol when cheap."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from gig.assets.options import rv_iv_signal
from gig.backtest.metrics import performance_metrics
from gig.types import BacktestResult


@dataclass
class VarianceRiskPremium:
    entry_threshold: float = 0.02

    def run(self, realized_vol: pd.Series, implied_vol: pd.Series, vol_pnl: pd.Series) -> BacktestResult:
        """
        `vol_pnl` is the daily PnL of a long-vol unit (e.g. delta-hedged straddle).
        Position is +1 (long vol) if IV-RV < -threshold, -1 if IV-RV > threshold.
        """
        spread = pd.Series(
            [rv_iv_signal(rv, iv) for rv, iv in zip(realized_vol, implied_vol, strict=True)],
            index=realized_vol.index,
        )
        pos = pd.Series(0.0, index=spread.index)
        pos[spread > self.entry_threshold] = -1.0
        pos[spread < -self.entry_threshold] = 1.0
        pnl = pos.shift(1) * vol_pnl
        net = pnl.dropna()
        equity = (1 + net).cumprod()
        return BacktestResult(
            equity=equity,
            returns=net,
            weights=pd.DataFrame({"VOL": pos.reindex(net.index)}),
            turnover=pos.diff().abs().reindex(net.index).fillna(0),
            metrics=performance_metrics(net),
        )
