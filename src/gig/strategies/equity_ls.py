"""
Flagship strategy: US-style cross-sectional equity long/short.

Pipeline:
  1. Point-in-time universe
  2. 12-1 momentum + short-term reversal + low idiosyncratic vol
  3. Optional walk-forward GBDT (OOS scores only)
  4. Optional news sentiment (headlines dated ≤ t)
  5. Sector neutralization
  6. Expanding IC-weighted combination (weights lagged one day)
  7. Factor-neutral, vol-targeted book (or dollar-neutral quantiles), costs,
     next-bar PnL
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from gig.backtest.costs import TransactionCostModel
from gig.backtest.simulator import CrossSectionalBacktest
from gig.data.universe import point_in_time_mask
from gig.factors.combine import ic_weighted_combine
from gig.factors.momentum import Momentum12m1, ShortTermReversal
from gig.factors.neutralize import neutralize
from gig.factors.volatility import IdiosyncraticVol
from gig.portfolio.optimize import OptimizerConfig
from gig.research.experiment import config_hash
from gig.research.ic import ic_summary
from gig.types import BacktestResult, MarketPanel


@dataclass
class EquityLongShort:
    n_long: int = 20
    n_short: int = 20
    rebalance_every: int = 5
    min_history: int = 252
    min_price: float = 1.0
    min_adv_usd: float = 0.0
    use_ml: bool = True
    use_news: bool = True
    cost_model: TransactionCostModel = field(default_factory=TransactionCostModel)
    # None reverts to equal-weight quantiles, which is kept as the baseline the
    # risk-constrained book has to beat rather than deleted.
    optimizer: OptimizerConfig | None = None
    risk_lookback: int = 252
    risk_refit_every: int = 21
    risk_factors: int = 5

    def config(self) -> dict[str, Any]:
        cfg = {
            "strategy": "equity_ls",
            "n_long": self.n_long,
            "n_short": self.n_short,
            "rebalance_every": self.rebalance_every,
            "min_history": self.min_history,
            "use_ml": self.use_ml,
            "use_news": self.use_news,
            "min_adv_usd": self.min_adv_usd,
            "construction": "risk_constrained" if self.optimizer else "quantile_equal_weight",
        }
        if self.optimizer is not None:
            cfg.update(
                {
                    **self.optimizer.as_dict(),
                    "risk_lookback": self.risk_lookback,
                    "risk_refit_every": self.risk_refit_every,
                    "risk_factors": self.risk_factors,
                }
            )
        return cfg

    def scores(self, panel: MarketPanel) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, dict[str, float]]]:
        tradable = point_in_time_mask(
            panel.close,
            min_history=self.min_history,
            min_price=self.min_price,
            min_adv=panel.dollar_volume,
            min_adv_usd=self.min_adv_usd,
        )
        raw = {
            "momentum_12_1": Momentum12m1().compute(panel),
            "strev_21d": ShortTermReversal().compute(panel),
            "ivol_63d": IdiosyncraticVol().compute(panel),
        }
        fwd = panel.close.pct_change().shift(-1)
        neutralized = {k: neutralize(v, panel.sectors, tradable=tradable) for k, v in raw.items()}

        if self.use_ml:
            try:
                from gig.ml.ranker import walk_forward_scores

                ml = walk_forward_scores(neutralized, fwd)
                neutralized["ml_gbdt"] = neutralize(ml, panel.sectors, tradable=tradable)
            except Exception:
                pass

        if self.use_news and panel.news is not None and not panel.news.empty:
            from gig.nlp.news import news_factor

            ns = news_factor(panel.news, panel.close)
            neutralized["news_sentiment"] = neutralize(ns, panel.sectors, tradable=tradable)

        if panel.filings is not None and not panel.filings.empty:
            from gig.factors.filings import filings_factor

            ff = filings_factor(panel.filings, panel.close)
            neutralized["filings_8k"] = neutralize(ff, panel.sectors, tradable=tradable)

        ic_table = {k: ic_summary(v, fwd) for k, v in neutralized.items()}
        combo = ic_weighted_combine(neutralized, fwd, min_history=63)
        combo = combo.where(tradable)
        return combo, tradable, ic_table

    def backtest(self, panel: MarketPanel) -> BacktestResult:
        combo, tradable, ic_table = self.scores(panel)
        n_names = int(tradable.any(axis=0).sum()) if len(tradable) else self.n_long
        n_side = max(5, min(self.n_long, n_names // 4))
        engine = CrossSectionalBacktest(
            n_long=n_side,
            n_short=n_side,
            rebalance_every=self.rebalance_every,
            cost_model=self.cost_model,
            optimizer=self.optimizer,
            risk_lookback=self.risk_lookback,
            risk_refit_every=self.risk_refit_every,
            risk_factors=self.risk_factors,
        )
        result = engine.run(
            combo,
            panel.close,
            tradable=tradable,
            dollar_volume=panel.dollar_volume,
            config_hash=config_hash(self.config()),
            gross_scale=_macro_scale(panel),
            sectors=panel.sectors,
        )
        result.factor_ic = {k: v["ic_mean"] for k, v in ic_table.items()}
        result.factor_ic_n = {k: v["n_obs"] for k, v in ic_table.items()}
        return result


def _macro_scale(panel: MarketPanel) -> pd.Series | None:
    if panel.macro is None or panel.macro.empty:
        return None
    from gig.factors.macro import series_to_trading_index, vix_gross_scale

    vix = series_to_trading_index(panel.macro, pd.DatetimeIndex(panel.close.index), "VIXCLS")
    if vix.empty or vix.notna().sum() < 5:
        return None
    return vix_gross_scale(vix)
