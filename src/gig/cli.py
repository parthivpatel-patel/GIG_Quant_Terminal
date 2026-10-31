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

    p_ops = sub.add_parser("ops", help="Kill switch, freshness, heartbeat")
    ops_sub = p_ops.add_subparsers(dest="ops_cmd", required=True)
    ops_sub.add_parser("status", help="Ops snapshot (JSON)")
    p_halt = ops_sub.add_parser("halt", help="Engage kill switch — suppress submissions")
    p_halt.add_argument("--reason", default="manual halt")
    ops_sub.add_parser("resume", help="Clear kill switch")

    p_bt = sub.add_parser("backtest", help="Run equity long/short (live Yahoo by default)")
    p_bt.add_argument("--source", choices=["yahoo", "synthetic"], default=None)
    p_bt.add_argument("--seed", type=int, default=42)
    p_bt.add_argument("--no-persist", action="store_true")
    p_bt.add_argument("--no-ml", action="store_true")
    p_bt.add_argument("--no-news", action="store_true")
    p_bt.add_argument(
        "--no-optimizer",
        action="store_true",
        help="Equal-weight quantile book instead of the factor-neutral vol-targeted one",
    )
    sub.add_parser("research", help="Alias for backtest")

    p_srv = sub.add_parser("serve", help="Research terminal (3D factor space, book, risk)")
    p_srv.add_argument("--host", default="127.0.0.1")
    p_srv.add_argument("--port", type=int, default=8000)
    p_srv.add_argument("--limit", type=int, default=220, help="Names in the factor cloud")
    p_srv.add_argument(
        "--trade-every",
        type=int,
        default=0,
        help="Embed paper loop in this process (seconds). Needed on Windows with DuckDB.",
    )
    p_srv.add_argument("--trade-limit", type=int, default=120)
    p_srv.add_argument("--trade-force", action="store_true", help="Trade outside regular hours")
    p_srv.add_argument("--trade-yes", action="store_true", help="Actually submit paper orders")

    p_tr = sub.add_parser("trade", help="Alpaca paper trading: plan, execute, monitor, flatten")
    tr_sub = p_tr.add_subparsers(dest="trade_cmd", required=True)
    for name, helptext in (
        ("plan", "Build today's target and print the trade list. Sends nothing."),
        ("run", "Execute the plan on the paper account (requires --yes)"),
    ):
        sp = tr_sub.add_parser(name, help=helptext)
        sp.add_argument("--limit", type=int, default=300, help="Universe size by ADV rank")
        sp.add_argument("--force", action="store_true", help="Trade outside regular hours")
        sp.add_argument("--use-ml", action="store_true", help="Include walk-forward LightGBM/sklearn scores")
        sp.add_argument("--band", type=float, default=None, help="No-trade band in weight terms")
        sp.add_argument("--json", action="store_true", help="Full machine-readable run record")
        if name == "run":
            sp.add_argument("--yes", action="store_true", help="Required to send real orders")

    p_st = tr_sub.add_parser("status", help="Account, target, and per-name drift")
    p_st.add_argument("--limit", type=int, default=300)
    p_st.add_argument("--json", action="store_true")

    p_fl = tr_sub.add_parser("flatten", help="Kill switch: close every position")
    p_fl.add_argument("--yes", action="store_true", help="Required to actually close")

    p_hist = tr_sub.add_parser("history", help="Recent runs from the audit trail")
    p_hist.add_argument("--limit", type=int, default=15)
    p_hist.add_argument("--orders", action="store_true", help="Show orders instead of runs")

    p_loop = tr_sub.add_parser("loop", help="Rebalance on a schedule while the market is open")
    p_loop.add_argument("--every", type=int, default=900, help="Seconds between passes")
    p_loop.add_argument("--limit", type=int, default=300)
    p_loop.add_argument("--yes", action="store_true", help="Required to send real orders")
    p_loop.add_argument("--force", action="store_true", help="Trade outside regular hours")
    p_loop.add_argument("--max-passes", type=int, default=0, help="0 runs until interrupted")

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
    if args.cmd == "ops":
        return _ops(args)
    if args.cmd == "serve":
        from gig.service.app import serve

        return serve(
            host=args.host,
            port=args.port,
            cloud_limit=args.limit,
            trade_every=getattr(args, "trade_every", 0) or 0,
            trade_limit=getattr(args, "trade_limit", 120),
            trade_force=getattr(args, "trade_force", False),
            trade_yes=getattr(args, "trade_yes", False),
        )
    if args.cmd == "trade":
        return _trade(args)
    if args.cmd in {"backtest", "research"}:
        from gig.pipeline import run_equity_research

        result = run_equity_research(
            source=getattr(args, "source", None),
            seed=getattr(args, "seed", 42),
            persist=not getattr(args, "no_persist", False),
            use_ml=not getattr(args, "no_ml", False),
            use_news=not getattr(args, "no_news", False),
            use_optimizer=not getattr(args, "no_optimizer", False),
        )
        payload = {
            "metrics": result.metrics,
            "factor_ic": result.factor_ic,
            "factor_ic_n": result.factor_ic_n,
        }
        if result.diagnostics is not None and not result.diagnostics.empty:
            payload["risk_path"] = {
                col: float(result.diagnostics[col].mean())
                for col in ("ex_ante_vol", "systematic_share", "gross", "effective_names")
                if col in result.diagnostics.columns
            }
        print(json.dumps(payload, indent=2, default=float))
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


def _ops(args) -> int:
    from gig.ops.killswitch import kill_switch
    from gig.ops.status import ops_snapshot

    cmd = args.ops_cmd
    if cmd == "status":
        print(json.dumps(ops_snapshot(), indent=2, default=str))
        return 0
    if cmd == "halt":
        out = kill_switch().halt(reason=getattr(args, "reason", "manual halt"), by="cli")
        print(json.dumps(out, indent=2, default=str))
        return 0
    if cmd == "resume":
        out = kill_switch().resume(by="cli")
        print(json.dumps(out, indent=2, default=str))
        return 0
    return 1


def _open_broker():
    """Alpaca paper adapter, with actionable errors instead of a traceback."""
    from gig.execution.alpaca import AlpacaBroker

    return AlpacaBroker()


def _print_trades(plan, limit: int = 40) -> None:
    if not plan.trades:
        print("  no trades — the account already matches the target")
        return
    print(f"\n  {'SIDE':<5}{'SYMBOL':<8}{'SHARES':>9}{'NOTIONAL':>13}{'CUR_W':>9}{'TGT_W':>9}  NOTE")
    for trade in plan.trades[:limit]:
        print(
            f"  {trade.side.upper():<5}{trade.symbol:<8}{trade.shares:>9,.0f}"
            f"{trade.notional:>13,.0f}{trade.current_weight:>9.3f}"
            f"{trade.target_weight:>9.3f}  {trade.note}"
        )
    if len(plan.trades) > limit:
        print(f"  ... {len(plan.trades) - limit} more")
    print(
        f"\n  estimated one-way cost {plan.estimated_cost_bps():.1f} bps on "
        f"{plan.buy_notional + plan.sell_notional:,.0f} traded"
    )


def _trade(args) -> int:
    cmd = args.trade_cmd

    if cmd == "history":
        from gig.data.lake import open_store

        store = open_store()
        frame = (
            store.load_orders(limit=args.limit)
            if args.orders
            else store.load_trade_runs(limit=args.limit)
        )
        if frame is None or frame.empty:
            print("No trading runs recorded yet. Start with: python -m gig trade plan")
            return 0
        print(frame[[c for c in frame.columns if c != "report"]].to_string(index=False))
        return 0

    try:
        broker = _open_broker()
    except ImportError:
        print('Alpaca SDK missing. Install it with:  pip install -e ".[broker]"')
        return 1
    except RuntimeError as exc:
        print(f"Broker not configured: {exc}")
        return 1

    if cmd == "status":
        from gig.execution.trader import account_status

        status = account_status(broker=broker, limit=args.limit)
        if args.json:
            print(json.dumps(status, indent=2, default=str))
            return 0
        acct = status.get("account", {})
        print(f"NAV          {status['nav']:,.2f}   paper={acct.get('paper', '?')}")
        print(f"positions    {status['n_positions']}   market_open={status['market_open']}")
        print(f"target       {json.dumps(status['target'], default=str)}")
        print(f"worst drift  {status['worst_drift']:.4f}")
        if status["drift"]:
            print(f"\n  {'SYMBOL':<8}{'CUR_W':>9}{'TGT_W':>9}{'DRIFT':>9}{'SHARES':>10}")
            for row in status["drift"][:25]:
                print(
                    f"  {row['symbol']:<8}{row['current_weight']:>9.3f}"
                    f"{row['target_weight']:>9.3f}{row['drift']:>9.3f}{row['shares_held']:>10,.0f}"
                )
        return 0

    if cmd == "flatten":
        from gig.execution.trader import flatten

        out = flatten(broker=broker, dry_run=not args.yes)
        print(json.dumps(out, indent=2, default=str))
        if out["dry_run"]:
            print("\nDry run. Re-run with --yes to actually close these positions.")
        return 0

    if cmd == "loop":
        return _trade_loop(args, broker)

    from gig.execution.reconcile import ReconcileConfig
    from gig.execution.trader import run_once

    if cmd == "run" and not args.yes:
        print("`trade run` sends orders to the broker. Re-run with --yes to confirm,")
        print("or use `python -m gig trade plan` to see the trade list without trading.")
        return 1

    reconcile = ReconcileConfig(no_trade_band=args.band) if args.band is not None else None
    run = run_once(
        dry_run=(cmd == "plan"),
        limit=args.limit,
        broker=broker,
        force=args.force,
        use_ml=args.use_ml,
        reconcile=reconcile,
    )
    if args.json:
        print(json.dumps(run.to_dict(), indent=2, default=str))
        return 2 if run.blocked else 0
    print(run.report())
    _print_trades(run.plan)
    if run.blocked:
        print("\nBlocked by the pre-trade gate. Nothing was sent.")
        return 2
    if cmd == "plan":
        print("\nDry run. Re-run as `trade run --yes` to send these orders.")
    return 0


def _trade_loop(args, broker) -> int:
    """Rebalance on a fixed interval until interrupted."""
    import time

    from gig.execution.trader import run_once

    if not args.yes:
        print("Loop starts in dry-run mode. Add --yes to send orders.")
    passes = 0
    try:
        while True:
            run = run_once(
                dry_run=not args.yes,
                limit=args.limit,
                broker=broker,
                force=getattr(args, "force", False),
            )
            print(run.report(), flush=True)
            passes += 1
            if args.max_passes and passes >= args.max_passes:
                return 0
            if broker.is_open() is False:
                print(f"market closed — next pass in {args.every}s", flush=True)
            time.sleep(max(30, args.every))
    except KeyboardInterrupt:
        print(f"\nstopped after {passes} passes")
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
    try:
        import lightgbm  # noqa: F401

        print("lightgbm     ok  (lambdarank walk-forward)")
    except ImportError:
        print('lightgbm     MISSING  (pip install -e ".[ml]" for LambdaRank)')
    try:
        from gig.nlp.ollama import ollama_status

        ol = ollama_status()
        if ol["available"]:
            print(f"ollama       ok  ({ol.get('model')})")
        else:
            print(f"ollama       offline  ({ol.get('hint', 'start ollama')})")
    except Exception as exc:
        print(f"ollama       error {exc}")
    try:
        import fastapi  # noqa: F401
        import uvicorn  # noqa: F401

        print("terminal     ok  (python -m gig serve)")
    except ImportError:
        print('terminal     MISSING  (pip install -e ".[web]")')
    try:
        import alpaca  # noqa: F401

        paper = "paper" in settings.alpaca_base_url
        print(f"broker       ok  ({'paper' if paper else 'LIVE — check ALPACA_BASE_URL'})")
    except ImportError:
        print('broker       MISSING  (pip install -e ".[broker]" for python -m gig trade)')
    keys = settings.keys_status()
    print(f"fred         {'key set' if keys['fred'] else 'MISSING   free key: https://fred.stlouisfed.org/docs/api/api_key.html'}")
    print(f"finnhub      {'key set' if keys['finnhub'] else 'MISSING   free key: https://finnhub.io/register'}")
    print(f"alpaca       {'key set' if keys['alpaca'] else 'MISSING   free paper: https://alpaca.markets'}")
    print(f"edgar        {'user-agent set' if keys['edgar'] else 'MISSING   set EDGAR_USER_AGENT=GIG research you@email.com'}")
    try:
        from gig.ops.killswitch import kill_switch
        from gig.ops.status import ops_snapshot

        ops = ops_snapshot()
        kill = kill_switch().status()
        fresh = ops.get("freshness") or {}
        print(
            f"ops          {ops.get('level')}  bars={fresh.get('asof')} "
            f"age={fresh.get('age_days')}  kill={'ON' if kill.get('engaged') else 'off'}"
        )
        for alert in (ops.get("alerts") or [])[:3]:
            print(f"  alert      [{alert.get('level')}] {alert.get('detail')}")
    except Exception as exc:
        print(f"ops          error {exc}")
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
