"""Refresh Yahoo bars for the most liquid names already in the lake (trade path)."""

from __future__ import annotations

from datetime import date

from gig.config import get_settings
from gig.data.lake import load_liquid_panel
from gig.logging import configure_logging
from gig.pipeline import _fetch_yahoo
from gig.store import Store


def main(limit: int = 350) -> None:
    log = configure_logging(get_settings().log_level)
    settings = get_settings()
    store = Store(settings.db_path)
    store.init()
    panel, _ = load_liquid_panel(limit=limit, lookback_days=40)
    if panel is None or panel.close.empty:
        raise SystemExit("no liquid panel — run universe refresh + ingest first")
    sectors = {s: panel.sectors.get(s, "unknown") for s in panel.symbols()}
    log.info("refreshing %s liquid names via Yahoo", len(sectors))
    fresh = _fetch_yahoo(settings, sectors)
    n = store.upsert_bars(fresh)
    store.upsert_universe(fresh.sectors, asof=date.today())
    mx = store.con.execute("select max(dt) from bars").fetchone()[0]
    print({"names": len(sectors), "bars_upserted": n, "max_dt": str(mx)})


if __name__ == "__main__":
    main()
