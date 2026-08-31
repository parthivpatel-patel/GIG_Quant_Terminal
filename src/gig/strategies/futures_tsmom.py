"""Moskowitz–Ooi–Pedersen time-series momentum on a futures excess-return series."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from gig.assets.futures import tsmom_signal
from gig.backtest.costs import TransactionCostModel
from gig.backtest.metrics import performance_metrics
from gig.types import BacktestResult


@dataclass
class FuturesTSMOM:
    lookback: int = 252
    vol_lookback: int = 60
    cost_model: TransactionCostModel | None = None

    def run(self, excess_returns: pd.Series) -> BacktestResult:
        pos = tsmom_signal(excess_returns, self.lookback, self.vol_lookback)
        # Trade next session: position known at t earns r_{t+1}
        pnl = pos.shift(1) * excess_returns
        to = pos.diff().abs().fillna(0.0)
        costs = TransactionCostModel() if self.cost_model is None else self.cost_model
        net = pnl - to.map(lambda t: costs.turnover_cost(float(abs(t) / 2.0)))
        net = net.dropna()
        equity = (1 + net).cumprod()
        weights = pd.DataFrame({"FUT": pos.reindex(net.index)})
        return BacktestResult(
            equity=equity,
            returns=net,
            weights=weights,
            turnover=to.reindex(net.index),
            metrics=performance_metrics(net, turnover=to.reindex(net.index)),
        )
