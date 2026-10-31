"""Factor protocol: map a MarketPanel to a date × symbol score (higher = more long)."""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd

from gig.types import MarketPanel


class Factor(ABC):
    name: str

    @abstractmethod
    def compute(self, panel: MarketPanel) -> pd.DataFrame:
        """Return scores aligned to panel.close. NaN where undefined."""
