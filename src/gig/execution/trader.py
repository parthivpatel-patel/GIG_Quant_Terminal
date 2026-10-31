"""
The trading loop.

One run is: build the target book, read the account, difference the two, put the
result through the risk gate, and only then send orders. Every run writes what
it intended, what the gate said, and what it sent to DuckDB, so any position can
be explained afterwards without relying on anyone's memory.

Two properties are load-bearing:

**Dry run is the default.** ``run_once()`` with no arguments computes and
records a plan and sends nothing. Trading requires passing ``dry_run=False``,
and the CLI requires a further explicit confirmation on top of that.

**Force is narrow.** ``force`` downgrades exactly one check — the market-hours
check — because wanting to rehearse outside the session is legitimate. Limit
breaches and the drawdown halt are not overridable by a flag. If a limit is
wrong, the correct action is to change the limit in configuration, where the
change is visible, rather than to bypass the gate from a command line.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import pandas as pd

from gig.config import get_settings
from gig.execution.algo import ExecutionConfig, limit_price, slice_shares
from gig.execution.base import Broker
from gig.execution.pretrade import PreTradeConfig, PreTradeReport, apply_drops
from gig.execution.reconcile import ReconcileConfig, TradePlan, build_trade_plan
from gig.execution.targets import TargetBook, build_target_book
from gig.logging import configure_logging

FORCEABLE = frozenset({"market_closed"})


@dataclass(slots=True)
class TradeRun:
    """Everything one pass through the loop decided and did."""

    run_id: str
    ts: datetime
    dry_run: bool
    nav: float
    book: TargetBook
    plan: TradePlan
    pretrade: PreTradeReport
    orders: list[dict[str, Any]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    market_open: bool | None = None

    @property
    def submitted(self) -> int:
        return sum(1 for o in self.orders if o.get("status") == "submitted")

    @property
    def blocked(self) -> bool:
        return self.pretrade.blocked

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "ts": self.ts.isoformat(),
            "dry_run": self.dry_run,
            "market_open": self.market_open,
            "nav": self.nav,
            "book": self.book.to_dict(),
            "plan": self.plan.to_dict(),
            "pretrade": self.pretrade.to_dict(),
            "orders": self.orders,
            "submitted": self.submitted,
            "errors": self.errors,
        }

    def report(self) -> str:
        """Human-readable summary for the CLI."""
        lines: list[str] = []
        mode = "DRY RUN — nothing sent" if self.dry_run else "LIVE — orders sent to broker"
        lines.append(f"run {self.run_id}   {mode}")
        if not self.book.available:
            lines.append(f"  no target book: {self.book.hint}")
            return "\n".join(lines)

        diag = self.book.diagnostics
        lines.append(
            f"  target      asof {self.book.asof}  {len(self.book.weights)} names  "
            f"gross {self.book.gross:.2f}  net {self.book.net:+.3f}  ({self.book.construction})"
        )
        if diag.get("ex_ante_vol") is not None:
            lines.append(
                f"  ex-ante     vol {100 * float(diag['ex_ante_vol']):.1f}%  "
                f"systematic {100 * float(diag.get('systematic_share') or 0):.1f}% of variance  "
                f"effective names {float(diag.get('effective_names') or 0):.0f}"
            )
        lines.append(
            f"  account     NAV {self.nav:,.0f}  market_open={self.market_open}  "
            f"held {len(self.plan.current)} names"
        )
        lines.append(
            f"  plan        {len(self.plan.trades)} orders  turnover {100 * self.plan.turnover:.2f}% of NAV  "
            f"buy {self.plan.buy_notional:,.0f}  sell {self.plan.sell_notional:,.0f}"
        )
        for finding in self.pretrade.blocking:
            lines.append(f"  BLOCKED     {finding.code}: {finding.detail}")
        for finding in self.pretrade.warnings:
            lines.append(f"  note        {finding.code}: {finding.detail}")
        for symbol, reason in self.pretrade.dropped.items():
            lines.append(f"  dropped     {symbol}: {reason}")
        if not self.dry_run:
            lines.append(f"  sent        {self.submitted} child orders")
        for err in self.errors:
            lines.append(f"  ERROR       {err}")
        return "\n".join(lines)


def _default_broker() -> Broker:
    from gig.execution.alpaca import AlpacaBroker

    return AlpacaBroker()


def _reference_prices(symbols: list[str], broker: Broker, book: TargetBook) -> pd.Series:
    """
    Price each name, preferring the broker's latest trade over a stored close.

    Order of preference matters: sizing a trade off a close that is a day old
    puts the share count wrong by exactly the overnight move, which on a 5%
    position is a real error. The stored close is a fallback so the loop still
    functions with no market-data entitlement, and the source is recorded.
    """
    from gig.data.lake import last_closes, open_store

    prices = pd.Series(dtype=float)
    try:
        prices = broker.last_prices(symbols)
    except Exception:
        prices = pd.Series(dtype=float)
    prices = prices[prices > 0] if len(prices) else prices

    missing = [s for s in symbols if s not in prices.index]
    if missing:
        fallback = book.prices.reindex(missing).dropna()
        if len(fallback) < len(missing):
            try:
                store = open_store(read_only=True)
                lake = last_closes(store, missing).dropna()
                fallback = fallback.combine_first(lake)
            except Exception:
                pass
        prices = pd.concat([prices, fallback[fallback > 0]])
    return prices[~prices.index.duplicated(keep="first")]


def run_once(
    dry_run: bool = True,
    limit: int = 300,
    broker: Broker | None = None,
    force: bool = False,
    use_ml: bool = False,
    reconcile: ReconcileConfig | None = None,
    execution: ExecutionConfig | None = None,
    gate: PreTradeConfig | None = None,
    record: bool = True,
) -> TradeRun:
    """Compute a target, reconcile against the account, and optionally trade."""
    from gig.execution import pretrade as pretrade_mod
    from gig.ops.killswitch import kill_switch

    log = configure_logging(get_settings().log_level)
    settings = get_settings()
    kill = kill_switch().status()
    if kill.get("engaged") and not dry_run:
        log.warning("kill switch engaged — forcing dry_run (%s)", kill.get("reason"))
        dry_run = True

    reconcile = reconcile or ReconcileConfig()
    execution = execution or ExecutionConfig()
    run_id = f"{datetime.now(UTC):%Y%m%dT%H%M%S}-{uuid.uuid4().hex[:6]}"
    ts = datetime.now(UTC)
    errors: list[str] = []

    book = build_target_book(limit=limit, use_ml=use_ml)
    broker = broker or _default_broker()

    try:
        nav = float(broker.nav())
    except Exception as exc:
        nav = 0.0
        errors.append(f"nav unavailable: {type(exc).__name__}: {exc}")
    try:
        positions = broker.positions()
    except Exception as exc:
        positions = {}
        errors.append(f"positions unavailable: {type(exc).__name__}: {exc}")

    symbols = sorted(set(map(str, book.weights.index)) | set(map(str, positions)))
    prices = _reference_prices(symbols, broker, book) if symbols else pd.Series(dtype=float)

    plan = build_trade_plan(
        book.weights,
        positions,
        prices,
        nav,
        adv=book.adv if len(book.adv) else None,
        config=reconcile,
    )

    try:
        market_open = broker.is_open()
    except Exception:
        market_open = None

    # Only ask about borrow for names a sell would actually take short; the
    # asset endpoint is one call per symbol.
    short_candidates = [
        t.symbol
        for t in plan.trades
        if t.side == "sell" and t.current_weight - (t.notional / nav if nav else 0) < 0
    ]
    shortable = None
    if short_candidates:
        try:
            shortable = broker.shortable(short_candidates)
        except Exception as exc:
            errors.append(f"borrow check skipped: {type(exc).__name__}: {exc}")

    # The high-water mark lives in the audit trail, so it is read on the same
    # condition that writes to it. A run that is not part of the audited
    # sequence has no drawdown history to be measured against, and opening the
    # lake to find that out is a needless dependency.
    high_water = None
    if record:
        try:
            from gig.data.lake import open_store

            high_water = open_store().nav_high_water()
        except Exception:
            pass

    report = pretrade_mod.check(
        plan,
        book,
        settings,
        config=gate,
        market_open=market_open,
        shortable=shortable,
        nav_high_water=high_water,
    )
    if force and report.findings:
        report.findings = [
            f if f.code not in FORCEABLE else type(f)(f.code, f.detail + " [forced]", False)
            for f in report.findings
        ]
    apply_drops(plan, report)

    if kill.get("engaged"):
        from gig.execution.pretrade import Rejection

        report.findings = list(report.findings) + [
            Rejection(
                "kill_switch",
                f"engaged: {kill.get('reason') or 'halted'} — orders suppressed",
                True,
            )
        ]

    run = TradeRun(
        run_id=run_id,
        ts=ts,
        dry_run=dry_run,
        nav=nav,
        book=book,
        plan=plan,
        pretrade=report,
        errors=errors,
        market_open=market_open,
    )

    if not dry_run and not report.blocked and plan.trades:
        _submit(run, broker, execution, log)

    if record:
        _record(run)
    return run


def _submit(run: TradeRun, broker: Broker, execution: ExecutionConfig, log) -> None:
    """Send each parent order as notional-bounded children, recording every one."""
    try:
        cancelled = broker.cancel_open_orders()
        if cancelled:
            log.info("cancelled %s working orders before rebalancing", cancelled)
    except Exception as exc:
        run.errors.append(f"cancel_open_orders failed: {type(exc).__name__}: {exc}")

    for trade in run.plan.trades:
        children = slice_shares(
            trade.shares, trade.price, execution.max_slice_notional, allow_fractional=False
        )
        for qty in children:
            px = (
                limit_price(trade.side, trade.price, execution.limit_offset_bps, execution.tick)
                if execution.use_limit
                else None
            )
            row: dict[str, Any] = {
                "symbol": trade.symbol,
                "side": trade.side,
                "shares": qty,
                "limit_price": px,
                "reference_price": trade.price,
                "notional": qty * trade.price,
                "current_weight": trade.current_weight,
                "target_weight": trade.target_weight,
                "note": trade.note,
            }
            try:
                fill = broker.submit(trade.symbol, trade.side, qty, limit_price=px)
                row["status"] = "submitted"
                row["broker_order_id"] = fill.order_id
            except Exception as exc:
                row["status"] = "rejected"
                row["broker_order_id"] = ""
                row["note"] = f"{row['note']} {type(exc).__name__}: {exc}".strip()
                run.errors.append(f"{trade.symbol}: {type(exc).__name__}: {exc}")
            run.orders.append(row)
            if execution.slice_pause_s > 0 and len(children) > 1:
                time.sleep(execution.slice_pause_s)


def _record(run: TradeRun) -> None:
    """Write the audit trail. A failure here is logged, never fatal."""
    try:
        from gig.data.lake import open_store

        store = open_store()
        store.record_trade_run(
            {
                "ts": run.ts,
                "run_id": run.run_id,
                "asof": run.book.asof,
                "dry_run": run.dry_run,
                "blocked": run.blocked,
                "nav": run.nav,
                "gross": run.book.gross,
                "net": run.book.net,
                "turnover": run.plan.turnover,
                "n_orders": len(run.plan.trades),
                "construction": run.book.construction,
                "report": run.to_dict(),
            }
        )
        store.record_target_book(
            run.run_id,
            run.book.weights,
            sectors=run.book.sectors,
            prices=run.book.prices,
            asof=run.book.asof,
            ts=run.ts,
        )
        rows = run.orders or [
            {**t.to_dict(), "status": "planned", "reference_price": t.price}
            for t in run.plan.trades
        ]
        store.record_orders(run.run_id, rows, ts=run.ts)
    except Exception as exc:
        configure_logging(get_settings().log_level).warning(
            "trade run not persisted: %s: %s", type(exc).__name__, exc
        )


def account_status(broker: Broker | None = None, limit: int = 300) -> dict[str, Any]:
    """Held versus intended, with no side effects. Safe to run any time."""
    from gig.execution.reconcile import drift_report

    broker = broker or _default_broker()
    book = build_target_book(limit=limit)
    if not book.available:
        return {
            "nav": float(broker.nav()) if broker else None,
            "n_positions": len(broker.positions()) if broker else 0,
            "target": book.to_dict(),
            "market_open": broker.is_open() if broker else None,
            "worst_drift": 0.0,
            "drift": [],
            "hint": book.hint or "target book unavailable",
        }
    nav = float(broker.nav())
    positions = broker.positions() or {}
    symbols = sorted(set(map(str, book.weights.index)) | set(map(str, positions)))
    prices = _reference_prices(symbols, broker, book) if symbols else pd.Series(dtype=float)
    if prices is None:
        prices = pd.Series(dtype=float)
    drift = drift_report(book.weights, positions, prices, nav)
    if drift is None:
        drift = pd.DataFrame()

    summary: dict[str, Any] = {
        "nav": nav,
        "n_positions": len(positions),
        "target": book.to_dict(),
        "market_open": broker.is_open(),
        "worst_drift": float(drift["drift"].abs().max()) if not drift.empty else 0.0,
        "drift": drift.head(25).to_dict(orient="records") if not drift.empty else [],
    }
    if hasattr(broker, "account_summary"):
        try:
            summary["account"] = broker.account_summary()
        except Exception:
            pass
    return summary


def flatten(broker: Broker | None = None, dry_run: bool = True) -> dict[str, Any]:
    """
    Kill switch: close every position.

    Deliberately does not consult the target book or the risk gate. When you
    want to be flat, the reason is usually that something upstream is wrong, and
    routing the exit through the same machinery you are trying to escape is how
    a bad situation stays bad.
    """
    broker = broker or _default_broker()
    positions = {s: q for s, q in broker.positions().items() if q}
    if dry_run:
        return {"dry_run": True, "would_close": positions, "n": len(positions)}
    fills = broker.close_all()
    return {
        "dry_run": False,
        "closed": [{"symbol": f.symbol, "side": f.side, "quantity": f.quantity} for f in fills],
        "n": len(fills),
    }
