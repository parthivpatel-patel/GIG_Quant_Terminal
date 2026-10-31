"""Live Yahoo Finance adapter. Prices are fetched, then cached in DuckDB."""

from __future__ import annotations

import logging
import time
from datetime import date, timedelta

import pandas as pd

from gig.data.providers.base import DataProvider
from gig.data.universe_spec import DEFAULT_SECTORS
from gig.types import MarketPanel

log = logging.getLogger("gig")

BATCH_SIZE = 80
BATCH_PAUSE = 0.35


class YahooProvider(DataProvider):
    def __init__(self, sector_map: dict[str, str] | None = None, batch_size: int = BATCH_SIZE) -> None:
        self.sector_map = sector_map or dict(DEFAULT_SECTORS)
        self.batch_size = max(1, batch_size)

    def load_panel(
        self,
        start: date,
        end: date,
        symbols: list[str] | None = None,
    ) -> MarketPanel:
        names = list(dict.fromkeys(symbols or list(self.sector_map)))
        try:
            import yfinance as yf  # noqa: F401
        except ImportError as exc:
            raise ImportError("Install yfinance: pip install yfinance") from exc

        frames: list[MarketPanel] = []
        n_batch = (len(names) + self.batch_size - 1) // self.batch_size
        for i in range(0, len(names), self.batch_size):
            batch = names[i : i + self.batch_size]
            idx = i // self.batch_size + 1
            log.info("yahoo bars batch %s/%s (%s names)", idx, n_batch, len(batch))
            try:
                frames.append(_download_batch(batch, start, end, self.sector_map))
            except Exception as exc:
                log.warning("yahoo batch %s skipped: %s", idx, exc)
            if i + self.batch_size < len(names) and BATCH_PAUSE:
                time.sleep(BATCH_PAUSE)
        if not frames:
            raise RuntimeError("yfinance returned no rows — check network or ticker list")
        return _concat_panels(frames, self.sector_map)


def _download_batch(
    names: list[str],
    start: date,
    end: date,
    sector_map: dict[str, str],
) -> MarketPanel:
    import yfinance as yf

    kwargs = dict(
        start=start.isoformat(),
        end=(end + timedelta(days=1)).isoformat(),
        auto_adjust=True,
        progress=False,
        threads=True,
        group_by="ticker",
    )
    try:
        raw = yf.download(names, **kwargs)
    except TypeError:
        kwargs.pop("threads", None)
        kwargs.pop("group_by", None)
        raw = yf.download(names, **kwargs)
    if raw is None or raw.empty:
        raise RuntimeError(f"yfinance empty for {names[:3]}…")
    close = _field(raw, "Close", names)
    volume = _field(raw, "Volume", names)
    close.index = pd.DatetimeIndex(close.index).tz_localize(None)
    volume.index = pd.DatetimeIndex(volume.index).tz_localize(None)
    volume = volume.reindex_like(close)
    sectors = pd.Series({s: sector_map.get(s, "unknown") for s in close.columns}, name="sector")
    return MarketPanel(
        close=close,
        volume=volume,
        sectors=sectors,
        dollar_volume=close * volume,
    )


def _concat_panels(frames: list[MarketPanel], sector_map: dict[str, str]) -> MarketPanel:
    close = pd.concat([f.close for f in frames], axis=1)
    volume = pd.concat([f.volume for f in frames], axis=1)
    close = close.loc[:, ~close.columns.duplicated()].sort_index()
    volume = volume.reindex_like(close)
    dollar = close * volume
    sectors = pd.Series({s: sector_map.get(s, "unknown") for s in close.columns}, name="sector")
    return MarketPanel(close=close, volume=volume, sectors=sectors, dollar_volume=dollar)


def _field(raw: pd.DataFrame, name: str, symbols: list[str]) -> pd.DataFrame:
    if not isinstance(raw.columns, pd.MultiIndex):
        frame = raw[[name]].copy() if name in raw.columns else raw.copy()
        frame.columns = symbols[:1]
        return frame
    levels = [list(raw.columns.unique(i)) for i in range(raw.columns.nlevels)]
    if name in levels[0]:
        return raw[name].copy()
    if name in levels[-1]:
        return raw.xs(name, axis=1, level=-1).copy()
    # group_by=ticker → columns are (symbol, field)
    if name in levels[1]:
        return raw.xs(name, axis=1, level=1).copy()
    raise KeyError(f"{name} not in yfinance columns: {raw.columns[:8]}")
