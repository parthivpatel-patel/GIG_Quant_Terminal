"""gig CLI: research, ingest, db, doctor."""

from __future__ import annotations

import argparse
import json
import sys

from gig import __version__
from gig.config import get_settings
from gig.logging import configure_logging


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="gig",
        description="GIG Trading Algorithm: live data, factors, walk-forward ML, C++ kernels, risk.",
    )
    parser.add_argument("--version", action="version", version=f"gig {__version__}")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("doctor", help="Validate install, DuckDB, and data source")

    p_db = sub.add_parser("db", help="Database commands")
    db_sub = p_db.add_subparsers(dest="db_cmd", required=True)
    db_sub.add_parser("init", help="Create DuckDB file and tables (no SQL server)")
    db_sub.add_parser("stats", help="Row counts")

    p_ing = sub.add_parser("ingest", help="Download Yahoo bars + FRED/EDGAR/Alpaca/Finnhub into DuckDB")
    p_ing.add_argument("--no-news", action="store_true")
    p_ing.add_argument("--all", dest="all_us", action="store_true", help="All NYSE/Nasdaq/AMEX common stocks")
    p_ing.add_argument("--mega", action="store_true", help="36-name mega-cap sleeve only")

    p_uni = sub.add_parser("universe", help="Listed-symbol tape")
    uni_sub = p_uni.add_subparsers(dest="uni_cmd", required=True)
    uni_sub.add_parser("refresh", help="Pull Nasdaq Trader NYSE/Nasdaq/AMEX commons into DuckDB")

    sub.add_parser("quotes", help="Live snapshot (Alpaca IEX, else Finnhub)")

    p_bt = sub.add_parser("backtest", help="Run equity long/short (live Yahoo by default)")
    p_bt.add_argument("--source", choices=["yahoo", "synthetic"], default=None)
    p_bt.add_argument("--seed", type=int, default=42)
    p_bt.add_argument("--no-persist", action="store_true")
    p_bt.add_argument("--no-ml", action="store_true")
    p_bt.add_argument("--no-news", action="store_true")
    sub.add_parser("research", help="Alias for backtest")

    args = parser.parse_args(argv)
    settings = get_settings()
    configure_logging(settings.log_level)

    if args.cmd == "doctor":
        return _doctor()
    if args.cmd == "db":
        return _db(args.db_cmd)
    if args.cmd == "ingest":
        from gig.pipeline import ingest_live

        all_us = True if getattr(args, "all_us", False) else (False if getattr(args, "mega", False) else None)
        stats = ingest_live(use_news=not args.no_news, all_us=all_us)
        print(json.dumps(stats, indent=2, default=str))
        return 0
    if args.cmd == "universe":
        return _universe(args.uni_cmd)
    if args.cmd == "quotes":
        from gig.pipeline import live_quotes

        df = live_quotes()
        if df.empty:
            print("No quotes. Add Alpaca or Finnhub keys to .env — see docs/APIS.md")
            return 1
        print(df.to_string(index=False))
        return 0
    if args.cmd in {"backtest", "research"}:
        from gig.pipeline import run_equity_research

        result = run_equity_research(
            source=getattr(args, "source", None),
            seed=getattr(args, "seed", 42),
            persist=not getattr(args, "no_persist", False),
            use_ml=not getattr(args, "no_ml", False),
            use_news=not getattr(args, "no_news", False),
        )
        print(
            json.dumps(
                {
                    "metrics": result.metrics,
                    "factor_ic": result.factor_ic,
                    "factor_ic_n": result.factor_ic_n,
                },
                indent=2,
                default=float,
            )
        )
        return 0
    return 1


def _db(action: str) -> int:
    from gig.store import Store

    settings = get_settings()
    settings.ensure_dirs()
    store = Store(settings.db_path)
    store.init()
    if action == "init":
        print(f"DuckDB ready: {settings.db_path.resolve()}")
        print("No SQL server. Open this file in DBeaver / DuckDB CLI if you want a GUI.")
        return 0
    print(json.dumps({"path": str(settings.db_path.resolve()), **store.stats()}, indent=2))
    return 0


def _universe(action: str) -> int:
    from gig.data.listings import refresh_us_listings
    from gig.store import Store

    settings = get_settings()
    settings.ensure_dirs()
    store = Store(settings.db_path)
    store.init()
    if action != "refresh":
        return 1
    listings = refresh_us_listings(store)
    counts = listings["exchange"].value_counts().to_dict() if "exchange" in listings.columns else {}
    print(
        json.dumps(
            {"listings": int(len(listings)), "exchanges": counts, "path": str(settings.db_path.resolve())},
            indent=2,
            default=str,
        )
    )
    return 0


def _doctor() -> int:
    settings = get_settings()
    settings.ensure_dirs()
    print(f"GIG Trading Algorithm {__version__}")
    print(f"data_source  {settings.data_source}")
    print(f"data_dir     {settings.data_dir.resolve()}")
    print(f"db_path      {settings.db_path.resolve()}  exists={settings.db_path.exists()}")
    print(f"universe     {settings.universe_file}")
    try:
        from gig.data.universe_spec import universe_mode

        print(f"tape         {universe_mode(settings.universe_file)}")
    except Exception:
        pass
    try:
        import yfinance  # noqa: F401

        print("yfinance     ok")
    except ImportError:
        print("yfinance     MISSING  (pip install yfinance)")
    try:
        import duckdb  # noqa: F401

        print("duckdb       ok  (no SQL server needed)")
    except ImportError:
        print("duckdb       MISSING  (pip install duckdb)")
    try:
        import sklearn  # noqa: F401

        print("sklearn      ok  (walk-forward GBDT)")
    except ImportError:
        print("sklearn      MISSING  (pip install scikit-learn)")
    try:
        from gig._speed import neutralize_cs  # noqa: F401

        print("cpp          ok  (gig._speed)")
    except Exception:
        print("cpp          Python fallback (install a C++ compiler, then pip install -e .)")
    keys = settings.keys_status()
    print(f"fred         {'key set' if keys['fred'] else 'MISSING   free key: https://fred.stlouisfed.org/docs/api/api_key.html'}")
    print(f"finnhub      {'key set' if keys['finnhub'] else 'MISSING   free key: https://finnhub.io/register'}")
    print(f"alpaca       {'key set' if keys['alpaca'] else 'MISSING   free paper: https://alpaca.markets'}")
    print(f"edgar        {'user-agent set' if keys['edgar'] else 'MISSING   set EDGAR_USER_AGENT=GIG research you@email.com'}")
    if settings.db_path.exists():
        try:
            from gig.store import Store

            print("tables       " + json.dumps(Store(settings.db_path).stats()))
        except Exception as exc:
            print(f"tables       error {exc}")
    print("install: ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
