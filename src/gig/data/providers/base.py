"""Provider contract. Implementations must not look ahead of `asof`."""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import date

from gig.types import MarketPanel


class DataProvider(ABC):
    @abstractmethod
    def load_panel(
        self,
        start: date,
        end: date,
        symbols: list[str] | None = None,
    ) -> MarketPanel:
        """Return OHLCV through `end` inclusive. No future bars."""
