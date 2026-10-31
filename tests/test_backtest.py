from __future__ import annotations

import numpy as np
import pandas as pd

from gig.backtest.costs import TransactionCostModel
from gig.backtest.simulator import CrossSectionalBacktest
from gig.portfolio.construct import dollar_neutral_quantiles


def test_costs_increase_with_participation():
    m = TransactionCostModel(spread_bps=4, commission_bps=1, impact_eta=0.10)
    small = m.one_way_bps(10_000, adv=1e8)
    large = m.one_way_bps(5_000_000, adv=1e8)
    assert large > small
    assert small >= 5.0


def test_quantiles_are_dollar_neutral():
    rng = np.random.default_rng(0)
    scores = pd.Series(rng.normal(size=50), index=[f"S{i}" for i in range(50)])
    w = dollar_neutral_quantiles(
        scores, n_long=10, n_short=10, gross_leverage=2.0, max_name_weight=0.15
    )
    assert abs(w.sum()) < 1e-8
    assert abs(w.abs().sum() - 2.0) < 1e-8


def test_backtest_does_not_earn_same_bar_return():
    dates = pd.bdate_range("2020-01-02", periods=80)
    names = [f"S{i:02d}" for i in range(20)]
    rng = np.random.default_rng(0)
    rets = rng.normal(0.001, 0.02, size=(80, 20))
    close = pd.DataFrame(100 * np.exp(np.cumsum(rets, axis=0)), index=dates, columns=names)
    # Perfect lookahead score = today's return (illegal). Simulator must still
    # apply scores to *next* bar, so this should not produce a huge Sharpe.
    scores = close.pct_change()
    result = CrossSectionalBacktest(n_long=5, n_short=5, rebalance_every=1).run(scores, close)
    # If lookahead leaked, Sharpe would explode. Bound it.
    assert result.metrics["sharpe"] < 8.0
    assert len(result.returns) == len(close) - 1
