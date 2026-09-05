"""
Today's intended book.

This is deliberately the *only* place a live target is produced. It runs the
same `EquityLongShort.scores` path the backtest runs and the same optimizer,
against the latest cross-section in the lake, so a position that appears in the
account can always be traced back to a factor value and a weight the research
code would have chosen. A trading loop that computes its own signals separately
from the backtest is two strategies with one name.

Nothing here talks to a broker or places an order.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any

import pandas as pd

from gig.config import get_settings
from gig.risk.factor_model import RiskModel


@dataclass(slots=True)
class TargetBook:
    """Weights the strategy wants, plus everything needed to price the trades."""

    asof: date
    weights: pd.Series          # signed target weights, non-zero names only
    sectors: pd.Series
    prices: pd.Series           # last stored close per name
    adv: pd.Series              # 21-session average dollar volume
    construction: str
    universe: int
    diagnostics: dict[str, Any] = field(default_factory=dict)
    risk_model: RiskModel | None = None
    available: bool = True
    hint: str = ""

    @property
    def gross(self) -> float:
        return float(self.weights.abs().sum())

    @property
    def net(self) -> float:
        return float(self.weights.sum())

    def to_dict(self) -> dict[str, Any]:
        return {
            "asof": str(self.asof) if self.asof else None,
            "construction": self.construction,
            "universe": self.universe,
            "n_names": int(len(self.weights)),
            "gross": self.gross,
            "net": self.net,
            "diagnostics": self.diagnostics,
        }


def _empty(hint: str) -> TargetBook:
    return TargetBook(
        asof=None,  # type: ignore[arg-type]
        weights=pd.Series(dtype=float),
        sectors=pd.Series(dtype=object),
        prices=pd.Series(dtype=float),
        adv=pd.Series(dtype=float),
        construction="none",
        universe=0,
        available=False,
        hint=hint,
    )


def build_target_book(
    limit: int = 300,
    lookback_days: int = 640,
    use_optimizer: bool | None = None,
    use_ml: bool = False,
    use_news: bool = False,
    min_adv_usd: float = 1_000_000.0,
) -> TargetBook:
    """
    Build the target book from the latest cross-section in DuckDB.

    ``use_ml`` defaults off for the live path: the walk-forward GBDT is the
    slowest stage and its scores are only meaningful with a stored train-end
    date, so turning it on for trading is an explicit decision rather than a
    default. The ADV floor is dropped when too few names could clear it, and the
    applied value is reported, matching the terminal's behaviour.
    """
    from gig.data.lake import load_liquid_panel
    from gig.pipeline import optimizer_config
    from gig.portfolio.construct import dollar_neutral_quantiles
    from gig.portfolio.optimize import optimize_book
    from gig.risk.factor_model import fit_statistical_risk_model
    from gig.strategies.equity_ls import EquityLongShort

    settings = get_settings()
    panel, _store = load_liquid_panel(limit=limit, lookback_days=lookback_days)
    if panel is None or panel.close.empty:
        return _empty("python -m gig universe refresh && python -m gig ingest")

    n_names = len(panel.symbols())
    n_side = max(5, min(50, n_names // 6))
    adv21 = (
        panel.dollar_volume.tail(21).mean()
        if panel.dollar_volume is not None and not panel.dollar_volume.empty
        else pd.Series(dtype=float)
    )
    liquid = int((adv21 >= min_adv_usd).sum()) if len(adv21) else 0
    adv_floor = min_adv_usd if liquid >= 2 * n_side + 2 else 0.0

    strategy = EquityLongShort(
        n_long=n_side,
        n_short=n_side,
        min_adv_usd=adv_floor,
        use_ml=use_ml,
        use_news=use_news,
    )
    combo, tradable, _ic = strategy.scores(panel)
    if combo.empty:
        return _empty("not enough history in DuckDB for a 12-1 momentum cross-section")

    asof = pd.Timestamp(combo.index[-1])
    scores = combo.loc[asof]
    returns = panel.close.pct_change().tail(settings.risk_lookback)

    config = optimizer_config(settings, use_optimizer)
    model = None
    diagnostics: dict[str, Any] = {"adv_floor": adv_floor, "n_side": n_side}

    def equal_weight(note: str) -> tuple[pd.Series, str]:
        """Baseline book, recording why the constrained one was not used."""
        diagnostics["fallback"] = note
        return (
            dollar_neutral_quantiles(
                scores,
                n_long=n_side,
                n_short=n_side,
                max_name_weight=settings.max_name_weight,
                gross_leverage=settings.max_gross_leverage,
            ),
            "quantile_equal_weight",
        )

    if config is None:
        weights, construction = equal_weight("optimizer disabled")
    else:
        model = fit_statistical_risk_model(returns, n_factors=settings.risk_factors)
        if model is None:
            weights, construction = equal_weight("risk model could not be fit")
        else:
            book = optimize_book(scores, model, sectors=panel.sectors, config=config)
            if not book.available:
                weights, construction = equal_weight("optimizer infeasible on this cross-section")
            else:
                weights = book.weights
                construction = "risk_constrained"
                diagnostics.update(book.diagnostics())
                diagnostics["factor_exposure"] = {
                    str(k): float(v) for k, v in book.factor_exposure.items()
                }

    held = weights[weights.abs() > 1e-12]
    prices = panel.close.loc[asof].reindex(held.index)
    return TargetBook(
        asof=asof.date(),
        weights=held.astype(float),
        sectors=panel.sectors.reindex(held.index).fillna("unknown"),
        prices=prices.astype(float),
        adv=adv21.reindex(held.index) if len(adv21) else pd.Series(dtype=float),
        construction=construction,
        universe=n_names,
        diagnostics=diagnostics,
        risk_model=model,
    )
