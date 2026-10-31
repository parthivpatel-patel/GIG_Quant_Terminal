"""Load from DuckDB if present; otherwise fetch live Yahoo and persist."""

from __future__ import annotations

from datetime import date

from gig.data.providers.base import DataProvider
from gig.data.providers.yahoo import YahooProvider
from gig.store import Store
from gig.types import MarketPanel


class CachedYahooProvider(DataProvider):
    def __init__(self, store: Store, sector_map: dict[str, str] | None = None) -> None:
        self.store = store
        self.yahoo = YahooProvider(sector_map)

    def load_panel(
        self,
        start: date,
        end: date,
        symbols: list[str] | None = None,
    ) -> MarketPanel:
        names = list(dict.fromkeys(symbols or list(self.yahoo.sector_map)))
        cached = self.store.load_panel(start, end, names)
        have = set(cached.symbols()) if cached is not None and len(cached.close) >= 60 else set()
        missing = [s for s in names if s not in have]
        if cached is not None and not missing:
            return cached
        if missing:
            fresh = self.yahoo.load_panel(start, end, missing)
            self.store.upsert_bars(fresh)
            self.store.upsert_universe(fresh.sectors, asof=end)
        loaded = self.store.load_panel(start, end, names)
        if loaded is None:
            raise RuntimeError("no bars after Yahoo fetch — check network")
        return loaded
