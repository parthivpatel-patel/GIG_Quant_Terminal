"""In-memory paper broker. Marks to an injected price function — no network."""

from __future__ import annotations

import math
from collections.abc import Callable
from datetime import UTC, datetime
from itertools import count

from gig.execution.base import Broker
from gig.types import Fill, Side


class PaperBroker(Broker):
    def __init__(self, price_fn: Callable[[str], float], nav0: float = 1_000_000.0) -> None:
        if isinstance(nav0, bool) or not math.isfinite(nav0) or nav0 <= 0:
            raise ValueError("simulated starting NAV must be finite and positive")
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
        if side not in {"buy", "sell"}:
            raise ValueError("side must be buy or sell")
        if isinstance(quantity, bool) or not math.isfinite(quantity) or quantity <= 0:
            raise ValueError("quantity must be finite and positive")
        if isinstance(limit_price, bool):
            raise ValueError("limit price must be numeric")
        px = float(limit_price if limit_price is not None else self._price(symbol))
        if not math.isfinite(px) or px <= 0:
            raise ValueError("simulated price must be finite and positive")
        signed = quantity if side == "buy" else -quantity
        notional = signed * px
        if not math.isfinite(notional) or not math.isfinite(self._cash - notional):
            raise ValueError("simulated monetary amount overflows")
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
            classification="SIMULATED",
        )
        self.fills.append(fill)
        return fill

    def positions(self) -> dict[str, float]:
        return dict(self._qty)

    def shortable(self, symbols: list[str]) -> dict[str, bool]:
        """All borrow is an explicit simulator assumption, never broker evidence."""
        return {str(symbol): True for symbol in symbols}

    def cancel_open_orders(self) -> int:
        """This immediate-fill simulator never holds working orders."""
        return 0

    def nav(self) -> float:
        mtm = sum(q * self._price(s) for s, q in self._qty.items())
        return self._cash + mtm
