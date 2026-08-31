"""
Vectorized long/short cross-sectional backtest.

Execution assumption (no lookahead):
  score_t uses prices through close_t
  weights are formed at close_t
  PnL is earned on the *next* session's close-to-close return
  costs are charged on one-way turnover at the rebalance
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from gig.backtest.costs import TransactionCostModel
from gig.backtest.metrics import performance_metrics
from gig.portfolio.construct import dollar_neutral_quantiles
from gig.types import BacktestResult


class CrossSectionalBacktest:
    def __init__(
        self,
        n_long: int = 20,
        n_short: int = 20,
        rebalance_every: int = 5,
        cost_model: TransactionCostModel | None = None,
        max_name_weight: float = 0.05,
        gross_leverage: float = 2.0,
    ) -> None:
        self.n_long = n_long
        self.n_short = n_short
        self.rebalance_every = rebalance_every
        self.cost_model = cost_model or TransactionCostModel()
        self.max_name_weight = max_name_weight
        self.gross_leverage = gross_leverage

    def run(
        self,
        scores: pd.DataFrame,
        close: pd.DataFrame,
        tradable: pd.DataFrame | None = None,
        dollar_volume: pd.DataFrame | None = None,
        config_hash: str = "",
        gross_scale: pd.Series | None = None,
    ) -> BacktestResult:
        close, scores = close.align(scores, join="inner", axis=0)
        close, scores = close.align(scores, join="inner", axis=1)
        fwd = close.pct_change().shift(-1)  # return from t to t+1, known only after t

        if tradable is not None:
            tradable = tradable.reindex_like(scores).fillna(False)
            scores = scores.where(tradable)

        weights = pd.DataFrame(0.0, index=scores.index, columns=scores.columns)
        rebal_mask = np.zeros(len(scores), dtype=bool)
        rebal_mask[:: self.rebalance_every] = True

        last = pd.Series(0.0, index=scores.columns)
        for i, _ts in enumerate(scores.index):
            if rebal_mask[i]:
                last = dollar_neutral_quantiles(
                    scores.iloc[i],
                    n_long=self.n_long,
                    n_short=self.n_short,
                    max_name_weight=self.max_name_weight,
                    gross_leverage=self.gross_leverage,
                )
            weights.iloc[i] = last

        if gross_scale is not None:
            scale = gross_scale.reindex(weights.index).fillna(1.0).astype(float)
            weights = weights.mul(scale, axis=0)

        # Turnover: 0.5 * sum |Δw|  (one-way)
        delta = weights.diff().abs().sum(axis=1).fillna(0.0)
        turnover = 0.5 * delta

        if dollar_volume is not None:
            adv = dollar_volume.reindex_like(weights).rolling(21, min_periods=5).mean()
            name_trade = weights.diff().abs().fillna(0.0)
            # Approximate participation with equal NAV=1
            part = (name_trade / adv.replace(0, np.nan)).mean(axis=1).fillna(0.01).clip(0, 0.2)
            costs = pd.Series(
                [self.cost_model.turnover_cost(t, p) for t, p in zip(turnover, part, strict=True)],
                index=weights.index,
            )
        else:
            costs = turnover.map(lambda t: self.cost_model.turnover_cost(float(t)))

        gross = (weights * fwd).sum(axis=1)
        net = gross - costs
        net = net.iloc[:-1]  # last row has no next return
        equity = (1 + net).cumprod()
        metrics = performance_metrics(net, turnover=turnover.reindex(net.index))
        return BacktestResult(
            equity=equity,
            returns=net,
            weights=weights.loc[net.index],
            turnover=turnover.reindex(net.index),
            metrics=metrics,
            config_hash=config_hash,
        )
