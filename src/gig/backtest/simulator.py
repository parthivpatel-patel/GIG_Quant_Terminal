"""
Vectorized long/short cross-sectional backtest.

Execution assumption (no lookahead):
  score_t uses prices through close_t
  weights are formed at close_t
  PnL is earned on the *next* session's close-to-close return
  costs are charged on one-way turnover at the rebalance

Two weighting rules are available. The equal-weight quantile rule is the
baseline and takes gross leverage as given. The risk-constrained rule fits a
factor model on a trailing window and solves for a factor-neutral book at a
volatility target; its risk model is refit on a schedule from data ending at
``t``, so the constraint set is point-in-time like everything else here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from gig.backtest.costs import TransactionCostModel
from gig.backtest.metrics import performance_metrics
from gig.portfolio.construct import dollar_neutral_quantiles
from gig.portfolio.optimize import OptimizerConfig, optimize_book
from gig.risk.factor_model import RiskModel, fit_statistical_risk_model
from gig.types import BacktestResult


class RollingRiskModel:
    """
    Factor model refit on a schedule from a trailing window of returns.

    Refitting daily is wasted computation: the leading eigenvectors of a 252-day
    covariance barely move overnight, which is why commercial risk models ship
    on a monthly cycle. Refitting every ``refit_every`` sessions keeps the same
    point-in-time discipline at a fraction of the cost, and a fit failure
    returns ``None`` rather than a stale model from a different regime.
    """

    def __init__(
        self,
        returns: pd.DataFrame,
        lookback: int = 252,
        refit_every: int = 21,
        n_factors: int = 5,
        min_obs: int = 120,
    ) -> None:
        self._returns = returns
        self._lookback = int(lookback)
        self._refit_every = max(1, int(refit_every))
        self._n_factors = int(n_factors)
        self._min_obs = int(min_obs)
        self._fitted_at: int | None = None
        self._model: RiskModel | None = None
        self.fits = 0

    def at(self, i: int) -> RiskModel | None:
        """Model usable at close of row ``i``, fit only on rows ``<= i``."""
        if i + 1 < self._min_obs:
            return None
        if self._fitted_at is not None and i - self._fitted_at < self._refit_every:
            return self._model
        window = self._returns.iloc[max(0, i + 1 - self._lookback) : i + 1]
        self._model = fit_statistical_risk_model(
            window, n_factors=self._n_factors, min_obs=self._min_obs
        )
        self._fitted_at = i
        self.fits += 1
        return self._model


class CrossSectionalBacktest:
    def __init__(
        self,
        n_long: int = 20,
        n_short: int = 20,
        rebalance_every: int = 5,
        cost_model: TransactionCostModel | None = None,
        max_name_weight: float = 0.05,
        gross_leverage: float = 2.0,
        optimizer: OptimizerConfig | None = None,
        risk_lookback: int = 252,
        risk_refit_every: int = 21,
        risk_factors: int = 5,
    ) -> None:
        self.n_long = n_long
        self.n_short = n_short
        self.rebalance_every = rebalance_every
        self.cost_model = cost_model or TransactionCostModel()
        self.max_name_weight = max_name_weight
        self.gross_leverage = gross_leverage
        self.optimizer = optimizer
        self.risk_lookback = risk_lookback
        self.risk_refit_every = risk_refit_every
        self.risk_factors = risk_factors

    def run(
        self,
        scores: pd.DataFrame,
        close: pd.DataFrame,
        tradable: pd.DataFrame | None = None,
        dollar_volume: pd.DataFrame | None = None,
        config_hash: str = "",
        gross_scale: pd.Series | None = None,
        sectors: pd.Series | None = None,
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

        risk: RollingRiskModel | None = None
        if self.optimizer is not None:
            risk = RollingRiskModel(
                close.pct_change(),
                lookback=self.risk_lookback,
                refit_every=self.risk_refit_every,
                n_factors=self.risk_factors,
            )
        diagnostics: list[dict[str, float | str]] = []

        last = pd.Series(0.0, index=scores.columns)
        for i, ts in enumerate(scores.index):
            if rebal_mask[i]:
                last = self._rebalance(scores.iloc[i], i, ts, risk, sectors, diagnostics)
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

        diag = pd.DataFrame(diagnostics).set_index("date") if diagnostics else None
        if diag is not None and "ex_ante_vol" in diag.columns:
            metrics["ex_ante_vol_mean"] = float(diag["ex_ante_vol"].mean())
            metrics["systematic_share_mean"] = float(diag["systematic_share"].mean())
            metrics["risk_model_fits"] = float(risk.fits) if risk is not None else 0.0

        return BacktestResult(
            equity=equity,
            returns=net,
            weights=weights.loc[net.index],
            turnover=turnover.reindex(net.index),
            metrics=metrics,
            config_hash=config_hash,
            diagnostics=diag,
        )

    def _rebalance(
        self,
        row: pd.Series,
        i: int,
        ts: pd.Timestamp,
        risk: RollingRiskModel | None,
        sectors: pd.Series | None,
        diagnostics: list[dict[str, float | str]],
    ) -> pd.Series:
        """
        One cross-section of weights.

        Falls back to the quantile rule whenever the risk model cannot be
        fit -- early in the sample, or on a cross-section too thin to estimate a
        covariance. Substituting a stale or degenerate model would be worse than
        admitting the constraint set is unavailable for those dates.
        """
        if risk is not None:
            model = risk.at(i)
            if model is not None:
                book = optimize_book(row, model, sectors=sectors, config=self.optimizer)
                if book.available:
                    diagnostics.append({"date": ts, **book.diagnostics()})
                    return book.weights
        return dollar_neutral_quantiles(
            row,
            n_long=self.n_long,
            n_short=self.n_short,
            max_name_weight=self.max_name_weight,
            gross_leverage=self.gross_leverage,
        )
