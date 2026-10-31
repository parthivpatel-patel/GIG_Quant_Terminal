"""
Read helpers over the DuckDB lake.

The terminal, the target-book builder, and the trader all need "the most liquid
N names with enough history to compute 12-1 momentum". Keeping that in one place
means a live book cannot be built from a different universe than the one the UI
is displaying, which is the sort of divergence nobody notices until a position
shows up that the research path never selected.

On Windows, DuckDB takes an exclusive file lock per process. So this module
keeps a **process-local shared connection** and a re-entrant lock: the research
terminal and the paper loop must run in the same process (see
``gig serve --trade-every``), not as two CLIs fighting over ``qalpha.duckdb``.
"""

from __future__ import annotations

import threading
import time
from contextlib import contextmanager
from datetime import date, timedelta
from typing import Iterator

import pandas as pd

from gig.config import get_settings
from gig.types import MarketPanel

_DB_LOCK = threading.RLock()
_SHARED: dict[str, object] = {}


@contextmanager
def db_lock(timeout: float | None = None) -> Iterator[None]:
    """
    Serialize DuckDB use across UI threads and the paper loop.

    ``timeout`` (seconds) fails fast with ``TimeoutError`` so status pills can
    keep serving last-known lake stats instead of hanging behind a rebalance.
    """
    if timeout is None:
        with _DB_LOCK:
            yield
        return
    if not _DB_LOCK.acquire(timeout=float(timeout)):
        raise TimeoutError("database busy")
    try:
        yield
    finally:
        _DB_LOCK.release()


def open_store(*, read_only: bool = False, retries: int = 8, shared: bool = True):
    """
    Connected Store with the schema applied.

    ``read_only`` is accepted for API compatibility but the shared process
    connection is always read-write — DuckDB rejects mixed configs on one file.
    """
    from gig.store import Store

    settings = get_settings()
    settings.ensure_dirs()
    key = str(settings.db_path.resolve())
    last_exc: Exception | None = None

    for attempt in range(max(1, retries)):
        try:
            with _DB_LOCK:
                if shared and key in _SHARED:
                    return _SHARED[key]
                # Shared path ignores read_only so serve + trade share one config.
                store = Store(settings.db_path, read_only=False if shared else read_only)
                store.init()
                if shared:
                    _SHARED[key] = store
                return store
        except Exception as exc:
            last_exc = exc
            if attempt + 1 >= retries:
                break
            time.sleep(0.25 * (attempt + 1))
    assert last_exc is not None
    raise last_exc


def reset_shared_stores() -> None:
    """Close and drop process-local connections (tests / clean shutdown)."""
    with _DB_LOCK:
        for store in list(_SHARED.values()):
            try:
                store.close()  # type: ignore[attr-defined]
            except Exception:
                pass
        _SHARED.clear()


def top_adv_symbols(store, limit: int, window_days: int = 40) -> list[str]:
    """Most liquid names by recent average dollar volume, ranked in SQL."""
    rows = store.con.execute(
        f"""
        SELECT symbol, AVG(dollar_volume) AS adv
        FROM bars
        WHERE dt >= (SELECT MAX(dt) FROM bars) - INTERVAL {int(window_days)} DAY
        GROUP BY symbol
        HAVING adv IS NOT NULL AND adv > 0
        ORDER BY adv DESC
        LIMIT {int(limit)}
        """
    ).df()
    if rows.empty:
        return []
    return [str(s) for s in rows["symbol"].tolist()]


def last_bar_date(store) -> date | None:
    row = store.con.execute("SELECT MAX(dt) FROM bars").fetchone()
    if not row or row[0] is None:
        return None
    return pd.Timestamp(row[0]).date()


def load_liquid_panel(
    limit: int = 220,
    lookback_days: int = 640,
    store=None,
) -> tuple[MarketPanel | None, object]:
    """
    Panel of the `limit` most liquid names, long enough for 12-1 momentum.

    Dates end at the last stored bar rather than today, so a stale lake produces
    a panel that is honestly dated instead of one padded with missing sessions.
    """
    store = store or open_store()
    symbols = top_adv_symbols(store, limit)
    if not symbols:
        return None, store
    end = last_bar_date(store) or date.today()
    panel = store.load_panel(end - timedelta(days=lookback_days), end, symbols)
    return panel, store


def last_closes(store, symbols: list[str] | None = None) -> pd.Series:
    """Most recent close per symbol, as a price fallback when no quote feed is on."""
    rows = store.con.execute(
        """
        SELECT symbol, close FROM bars
        WHERE dt = (SELECT MAX(dt) FROM bars)
        """
    ).df()
    if symbols:
        want = set(map(str, symbols))
        rows = rows[rows["symbol"].astype(str).isin(want)]
    if rows.empty:
        return pd.Series(dtype=float)
    return rows.set_index("symbol")["close"].astype(float)
