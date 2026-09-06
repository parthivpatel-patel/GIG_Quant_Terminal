"""
One-shot launcher for the 24/7 paper research terminal (Windows).

Keeps serve + embedded trade loop in a single process so DuckDB is not locked
twice. Prefer Task Scheduler / NSSM pointing at this script rather than a
separate ``trade loop``.
"""

from __future__ import annotations

import argparse
import os
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="GIG 24/7 paper terminal")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--limit", type=int, default=220)
    parser.add_argument("--trade-every", type=int, default=900)
    parser.add_argument("--trade-limit", type=int, default=120)
    parser.add_argument("--trade-force", action="store_true", help="Allow outside RTH (default on)")
    parser.add_argument("--no-trade-force", action="store_true", help="Require regular session")
    parser.add_argument("--trade-yes", action="store_true", help="Submit paper orders")
    parser.add_argument("--dry-run", action="store_true", help="Force plan-only even with --trade-yes")
    args = parser.parse_args(argv)

    # Ensure package imports resolve when launched from Task Scheduler.
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if root not in sys.path:
        sys.path.insert(0, root)

    from gig.service.app import serve

    force = not args.no_trade_force
    yes = bool(args.trade_yes) and not args.dry_run
    return serve(
        host=args.host,
        port=args.port,
        cloud_limit=args.limit,
        trade_every=args.trade_every,
        trade_limit=args.trade_limit,
        trade_force=force,
        trade_yes=yes,
    )


if __name__ == "__main__":
    raise SystemExit(main())
