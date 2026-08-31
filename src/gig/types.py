"""Shared domain types. Keep these small and serializable."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from enum import StrEnum
from typing import Literal

import pandas as pd

AssetClass = Literal["equity", "future", "option", "fx", "credit"]
Side = Literal["buy", "sell"]


class OrderType(StrEnum):
    MARKET = "market"
    LIMIT = "limit"
    TWAP = "twap"


@dataclass(frozen=True, slots=True)
class Instrument:
    symbol: str
    asset_class: AssetClass
    sector: str = "unknown"
    multiplier: float = 1.0
    currency: str = "USD"


@dataclass(slots=True)
class MarketPanel:
    """Wide panels: index = date, columns = symbol."""

    close: pd.DataFrame
    volume: pd.DataFrame
    sectors: pd.Series
    open_: pd.DataFrame | None = None
    high: pd.DataFrame | None = None
    low: pd.DataFrame | None = None
    dollar_volume: pd.DataFrame | None = None
    news: pd.DataFrame | None = None
    filings: pd.DataFrame | None = None
    macro: pd.DataFrame | None = None

    def symbols(self) -> list[str]:
        return list(self.close.columns)

    def dates(self) -> pd.DatetimeIndex:
        return pd.DatetimeIndex(self.close.index)


@dataclass(frozen=True, slots=True)
class Fill:
    timestamp: datetime
    symbol: str
    side: Side
    quantity: float
    price: float
    cost: float
    order_id: str = ""


@dataclass(slots=True)
class Position:
    symbol: str
    quantity: float
    avg_price: float
    market_value: float = 0.0


@dataclass(slots=True)
class BacktestResult:
    equity: pd.Series
    returns: pd.Series
    weights: pd.DataFrame
    turnover: pd.Series
    metrics: dict[str, float]
    factor_ic: dict[str, float] = field(default_factory=dict)
    factor_ic_n: dict[str, float] = field(default_factory=dict)
    config_hash: str = ""
    asof: date | None = None
