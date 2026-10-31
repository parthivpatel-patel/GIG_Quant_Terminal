"""In-memory paper broker. Marks to an injected price function — no network."""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from itertools import count

from gig.execution.base import Broker
from gig.types import Fill, Side


class PaperBroker(Broker):
    def __init__(self, price_fn: Callable[[str], float], nav0: float = 1_000_000.0) -> None:
        self._price = price_fn
        self._cash = nav0
        self._qty: dict[str, float] = {}
        self._ids = count(1)
        self.fills: list[Fill] = []

    def submit(
        self,
        symbol: str,
        side: Side,
        quantity: float,
        *,
        limit_price: float | None = None,
    ) -> Fill:
        px = float(limit_price if limit_price is not None else self._price(symbol))
        signed = quantity if side == "buy" else -quantity
        notional = signed * px
        self._cash -= notional
        self._qty[symbol] = self._qty.get(symbol, 0.0) + signed
        fill = Fill(
            timestamp=datetime.now(UTC),
            symbol=symbol,
            side=side,
            quantity=quantity,
            price=px,
            cost=0.0,
            order_id=f"PAPER-{next(self._ids)}",
        )
        self.fills.append(fill)
        return fill

    def positions(self) -> dict[str, float]:
        return dict(self._qty)

    def nav(self) -> float:
        mtm = sum(q * self._price(s) for s, q in self._qty.items())
        return self._cash + mtm
