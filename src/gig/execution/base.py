"""
Broker protocol. Live adapters implement this; backtests do not call it.

``submit`` / ``positions`` / ``nav`` are abstract because no adapter is useful
without them. The rest are operational queries a trading loop needs but a
minimal adapter should not be forced to implement, so they have permissive
defaults and each adapter overrides what it can actually answer. The defaults
are chosen so that a broker which cannot answer a question does not silently
*block* trading — the pre-trade gate decides that — with one exception noted on
``is_open``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd

from gig.types import Fill, Side


class Broker(ABC):
    @abstractmethod
    def submit(
        self,
        symbol: str,
        side: Side,
        quantity: float,
        *,
        limit_price: float | None = None,
    ) -> Fill:
        raise NotImplementedError

    @abstractmethod
    def positions(self) -> dict[str, float]:
        raise NotImplementedError

    @abstractmethod
    def nav(self) -> float:
        raise NotImplementedError

    def is_open(self) -> bool | None:
        """
        Whether the regular session is open.

        ``None`` means "this adapter cannot tell", which the pre-trade gate
        treats as unknown rather than closed. Returning ``False`` on ignorance
        would make an offline paper broker untradeable in tests.
        """
        return None

    def shortable(self, symbols: list[str]) -> dict[str, bool]:
        """Per-name borrow availability. Unknown names default to shortable."""
        return {str(s): True for s in symbols}

    def last_prices(self, symbols: list[str]) -> pd.Series:
        """Reference prices from the broker, or empty to fall back to the lake."""
        return pd.Series(dtype=float)

    def cancel_open_orders(self) -> int:
        """Cancel working orders so a new run starts from a clean book."""
        return 0

    def close_all(self, limit_offset_bps: float = 25.0) -> list[Fill]:
        """
        Flatten every position.

        Implemented against ``positions`` and ``submit`` so any adapter gets a
        working kill switch without writing one.
        """
        fills: list[Fill] = []
        for symbol, qty in self.positions().items():
            if not qty:
                continue
            side: Side = "sell" if qty > 0 else "buy"
            fills.append(self.submit(symbol, side, abs(float(qty))))
        return fills
