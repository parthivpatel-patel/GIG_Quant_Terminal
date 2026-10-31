"""
Position reconciliation and trade-list construction.

The account is the source of truth for what is held, never a local record of
what was intended. Every run reads live positions and computes the difference
against the target, so a partial fill, a manual intervention, or a missed
session self-corrects on the next rebalance instead of compounding into a
position the strategy never asked for.

Three filters keep the list sane, and each one exists because of a specific
failure mode:

* a **no-trade band**, because a drifting weight that is within tolerance costs
  more in spread to correct than it costs in tracking error to leave;
* a **minimum notional**, because odd-lot trades pay a fixed cost against
  nearly no risk reduction;
* a **participation cap** against ADV, because the cost model's square-root
  impact term is only credible at low participation, and an order that moves
  the price invalidates the backtest that justified it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from gig.types import Side


@dataclass(frozen=True, slots=True)
class ReconcileConfig:
    no_trade_band: float = 0.0025      # weight drift tolerated before trading
    min_trade_notional: float = 250.0
    max_participation: float = 0.02    # share of one session's ADV
    max_order_pct_nav: float = 0.06    # fat-finger guard on a single order
    allow_fractional: bool = False


@dataclass(frozen=True, slots=True)
class Trade:
    symbol: str
    side: Side
    shares: float
    price: float
    current_weight: float
    target_weight: float
    note: str = ""

    @property
    def notional(self) -> float:
        return float(self.shares * self.price)

    @property
    def signed_notional(self) -> float:
        return self.notional if self.side == "buy" else -self.notional

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "side": self.side,
            "shares": float(self.shares),
            "price": float(self.price),
            "notional": self.notional,
            "current_weight": self.current_weight,
            "target_weight": self.target_weight,
            "note": self.note,
        }


@dataclass(slots=True)
class TradePlan:
    nav: float
    current: pd.Series
    target: pd.Series
    trades: list[Trade] = field(default_factory=list)
    skipped: list[dict[str, Any]] = field(default_factory=list)

    @property
    def turnover(self) -> float:
        """One-way turnover as a share of NAV, the same convention as the backtest."""
        if self.nav <= 0:
            return 0.0
        return float(sum(abs(t.notional) for t in self.trades) / self.nav / 2.0)

    @property
    def buy_notional(self) -> float:
        return float(sum(t.notional for t in self.trades if t.side == "buy"))

    @property
    def sell_notional(self) -> float:
        return float(sum(t.notional for t in self.trades if t.side == "sell"))

    def post_trade_weights(self) -> pd.Series:
        """Weights implied if every trade fills at its reference price."""
        out = self.current.copy()
        for trade in self.trades:
            delta = trade.signed_notional / self.nav if self.nav > 0 else 0.0
            out.loc[trade.symbol] = float(out.get(trade.symbol, 0.0) + delta)
        return out

    def estimated_cost_bps(self, cost_model=None, adv: pd.Series | None = None) -> float:
        """Blended one-way cost on the traded notional, using the backtest's model."""
        from gig.backtest.costs import TransactionCostModel

        model = cost_model or TransactionCostModel()
        traded = sum(abs(t.notional) for t in self.trades)
        if traded <= 0:
            return 0.0
        total = 0.0
        for trade in self.trades:
            name_adv = float(adv.get(trade.symbol, 0.0)) if adv is not None else 0.0
            total += abs(trade.notional) * model.one_way_bps(abs(trade.notional), name_adv)
        return float(total / traded)

    def to_dict(self) -> dict[str, Any]:
        return {
            "nav": self.nav,
            "n_trades": len(self.trades),
            "turnover": self.turnover,
            "buy_notional": self.buy_notional,
            "sell_notional": self.sell_notional,
            "trades": [t.to_dict() for t in self.trades],
            "skipped": self.skipped,
        }


def current_weights(
    positions: dict[str, float],
    prices: pd.Series,
    nav: float,
) -> pd.Series:
    """
    Signed market value over NAV, per name.

    A held name with no price available is reported as NaN rather than 0, so it
    surfaces as a data problem instead of silently looking flat and inviting the
    reconciler to buy a second position on top of the one already there.
    """
    if nav <= 0 or not positions:
        return pd.Series(dtype=float)
    rows = {}
    for symbol, qty in positions.items():
        price = prices.get(symbol, np.nan)
        rows[str(symbol)] = float(qty) * float(price) / nav if pd.notna(price) else np.nan
    return pd.Series(rows, dtype=float)


def build_trade_plan(
    target: pd.Series,
    positions: dict[str, float],
    prices: pd.Series,
    nav: float,
    adv: pd.Series | None = None,
    config: ReconcileConfig | None = None,
) -> TradePlan:
    """
    Difference the target against live positions and size the orders in shares.

    The symbol set is the union of held and targeted names: an exit is as much a
    trade as an entry, and dropping held names that fell out of the target is
    how a book slowly becomes something other than the strategy.
    """
    config = config or ReconcileConfig()
    current = current_weights(positions, prices, nav)
    target = target[target.abs() > 1e-12] if len(target) else pd.Series(dtype=float)

    plan = TradePlan(nav=float(nav), current=current, target=target)
    if nav <= 0:
        plan.skipped.append({"symbol": "*", "reason": "nav is zero or unknown"})
        return plan

    symbols = sorted(set(target.index.astype(str)) | set(current.index.astype(str)))
    for symbol in symbols:
        target_w = float(target.get(symbol, 0.0))
        current_w = current.get(symbol, 0.0)
        price = prices.get(symbol, np.nan)

        if pd.isna(price) or float(price) <= 0:
            plan.skipped.append(
                {"symbol": symbol, "reason": "no price", "target_weight": target_w}
            )
            continue
        price = float(price)

        if pd.isna(current_w):
            plan.skipped.append(
                {"symbol": symbol, "reason": "held but unpriced — resolve before trading"}
            )
            continue
        current_w = float(current_w)

        drift = target_w - current_w
        # An exit always trades: leaving a name the strategy dropped is a
        # position with no thesis, however small the drift looks.
        exiting = target_w == 0.0 and current_w != 0.0
        if not exiting and abs(drift) < config.no_trade_band:
            plan.skipped.append(
                {"symbol": symbol, "reason": "within no-trade band", "drift": drift}
            )
            continue

        notional = drift * nav
        if abs(notional) < config.min_trade_notional and not exiting:
            plan.skipped.append(
                {"symbol": symbol, "reason": "below minimum notional", "notional": notional}
            )
            continue

        cap_nav = config.max_order_pct_nav * nav
        note = ""
        if abs(notional) > cap_nav > 0:
            notional = math.copysign(cap_nav, notional)
            note = "capped at max order size"

        if adv is not None and config.max_participation > 0:
            name_adv = float(adv.get(symbol, np.nan)) if symbol in adv.index else np.nan
            if pd.notna(name_adv) and name_adv > 0:
                cap_adv = config.max_participation * name_adv
                if abs(notional) > cap_adv:
                    notional = math.copysign(cap_adv, notional)
                    note = "capped at ADV participation limit"

        shares = abs(notional) / price
        shares = shares if config.allow_fractional else math.floor(shares)
        if shares <= 0:
            plan.skipped.append(
                {"symbol": symbol, "reason": "rounds to zero shares", "notional": notional}
            )
            continue

        plan.trades.append(
            Trade(
                symbol=symbol,
                side="buy" if notional > 0 else "sell",
                shares=float(shares),
                price=price,
                current_weight=current_w,
                target_weight=target_w,
                note=note or ("exit" if exiting else ""),
            )
        )

    plan.trades.sort(key=lambda t: abs(t.notional), reverse=True)
    return plan


def drift_report(
    target: pd.Series,
    positions: dict[str, float],
    prices: pd.Series,
    nav: float,
) -> pd.DataFrame:
    """Side-by-side of held versus intended weight. Read-only; places nothing."""
    current = current_weights(positions, prices, nav)
    symbols = sorted(set(target.index.astype(str)) | set(current.index.astype(str)))
    rows = []
    for symbol in symbols:
        t = float(target.get(symbol, 0.0))
        c = current.get(symbol, 0.0)
        c = float(c) if pd.notna(c) else np.nan
        rows.append(
            {
                "symbol": symbol,
                "current_weight": c,
                "target_weight": t,
                "drift": t - c,
                "shares_held": float(positions.get(symbol, 0.0)),
                "price": float(prices.get(symbol, np.nan)),
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    return frame.reindex(frame["drift"].abs().sort_values(ascending=False).index)
