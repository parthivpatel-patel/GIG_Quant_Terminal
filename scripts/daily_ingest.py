"""
Daily lake maintenance: refresh liquid bars + vendor feeds.

Schedule this once per weekday morning (before the open) via Task Scheduler.
"""

from __future__ import annotations

import json
import os
import sys


def main() -> int:
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if root not in sys.path:
        sys.path.insert(0, root)

    from gig.config import get_settings
    from gig.data.lake import open_store
    from gig.data.universe_spec import resolve_universe
    from gig.logging import configure_logging
    from gig.pipeline import _ingest_vendor_feeds

    log = configure_logging(get_settings().log_level)
    settings = get_settings()

    # Top up liquid Yahoo bars without a full-tape re-download.
    try:
        import runpy

        runpy.run_path(os.path.join(os.path.dirname(__file__), "refresh_liquid.py"), run_name="__main__")
    except SystemExit as exc:
        if exc.code not in (0, None):
            log.warning("liquid refresh exited %s", exc.code)
    except Exception as exc:
        log.warning("liquid refresh failed: %s: %s", type(exc).__name__, exc)

    store = open_store()
    store.init()
    sectors = resolve_universe(settings.universe_file, store, refresh=False)
    symbols = list(sectors)[:400]
    stats = _ingest_vendor_feeds(
        store,
        settings,
        symbols,
        use_news=True,
        log=log,
        filing_symbols=symbols[:80],
        quote_symbols=symbols[:120],
    )
    print(json.dumps({"ok": True, **stats, "path": str(settings.db_path.resolve())}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
