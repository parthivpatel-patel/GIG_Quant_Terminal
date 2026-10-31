"""Three-layer institutional cost model: spread + commission + square-root impact."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TransactionCostModel:
    spread_bps: float = 4.0
    commission_bps: float = 1.0
    impact_eta: float = 0.10

    def one_way_bps(self, trade_value: float, adv: float) -> float:
        """Total one-way cost in basis points of notional."""
        spread = self.spread_bps
        commission = self.commission_bps
        if adv <= 0 or trade_value <= 0:
            impact = 0.0
        else:
            # Almgren–Chriss square-root: η * sqrt(Q / ADV) expressed in bps
            participation = trade_value / adv
            impact = self.impact_eta * (participation**0.5) * 10_000.0
        return spread + commission + impact

    def one_way_fraction(self, trade_value: float, adv: float) -> float:
        return self.one_way_bps(trade_value, adv) / 10_000.0

    def turnover_cost(self, turnover: float, avg_adv_participation: float = 0.01) -> float:
        """
        Portfolio-level cost as a fraction of NAV given one-way turnover.

        `turnover` is 0.5 * sum(|Δw|) (standard one-way). Cost is applied once
        per rebalance on that turnover.
        """
        bps = self.spread_bps + self.commission_bps
        impact = self.impact_eta * (avg_adv_participation**0.5) * 10_000.0
        return turnover * (bps + impact) / 10_000.0
