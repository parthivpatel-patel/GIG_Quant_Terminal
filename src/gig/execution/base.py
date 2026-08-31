"""Broker protocol. Live adapters implement this; backtests do not call it."""

from __future__ import annotations

from abc import ABC, abstractmethod

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
