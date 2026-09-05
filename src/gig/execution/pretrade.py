"""
Pre-trade risk gate.

Every order passes through here before it reaches a broker. The design rule is
that a check either **blocks the whole run** or **drops a single order**, and
nothing in between: a partially-applied risk policy is worse than either
outcome because the resulting book matches no decision anyone made.

Blocking checks are the ones where continuing would mean trading on a premise
that has already failed — a stale lake, a book that breaches a hard limit, a
drawdown past the halt level, a turnover figure that implies the target was
computed wrong. Per-order drops are for name-specific facts, like a short that
the broker cannot borrow.

The checks intentionally re-derive limits from the target rather than trusting
the optimizer's own report. The optimizer is the thing being checked.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any

import pandas as pd


@dataclass(frozen=True, slots=True)
class PreTradeConfig:
    max_stale_days: int = 5          # calendar days between last bar and now
    max_ex_ante_vol: float = 0.25    # annualized, under the fitted risk model
    max_turnover: float = 0.35       # one-way, share of NAV, per run
    max_orders: int = 500
    min_nav: float = 1_000.0
    require_market_open: bool = True


@dataclass(frozen=True, slots=True)
class Rejection:
    code: str
    detail: str
    blocking: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {"code": self.code, "detail": self.detail, "blocking": self.blocking}


@dataclass(slots=True)
class PreTradeReport:
    findings: list[Rejection] = field(default_factory=list)
    dropped: dict[str, str] = field(default_factory=dict)

    @property
    def blocked(self) -> bool:
        return any(f.blocking for f in self.findings)

    @property
    def blocking(self) -> list[Rejection]:
        return [f for f in self.findings if f.blocking]

    @property
    def warnings(self) -> list[Rejection]:
        return [f for f in self.findings if not f.blocking]

    def to_dict(self) -> dict[str, Any]:
        return {
            "blocked": self.blocked,
            "findings": [f.to_dict() for f in self.findings],
            "dropped": self.dropped,
        }


def check(
    plan,
    book,
    settings,
    config: PreTradeConfig | None = None,
    market_open: bool | None = None,
    shortable: dict[str, bool] | None = None,
    nav_high_water: float | None = None,
    now: date | None = None,
) -> PreTradeReport:
    """
    Validate a trade plan against hard limits.

    ``plan`` is a :class:`~gig.execution.reconcile.TradePlan` and ``book`` a
    :class:`~gig.execution.targets.TargetBook`. Returns the report; it does not
    mutate the plan. The caller applies ``report.dropped`` and honours
    ``report.blocked``.
    """
    from gig.risk.limits import check_limits

    config = config or PreTradeConfig()
    report = PreTradeReport()
    today = now or date.today()

    if not book.available:
        report.findings.append(Rejection("no_target", book.hint or "target book unavailable"))
        return report

    if book.asof is not None:
        stale = (today - book.asof).days
        if stale > config.max_stale_days:
            report.findings.append(
                Rejection(
                    "stale_data",
                    f"last bar {book.asof} is {stale} days old (limit {config.max_stale_days}); "
                    "run `python -m gig ingest`",
                )
            )

    if market_open is False and config.require_market_open:
        report.findings.append(
            Rejection("market_closed", "regular session is closed; pass --force to override")
        )

    if plan.nav < config.min_nav:
        report.findings.append(
            Rejection("nav_floor", f"NAV {plan.nav:,.0f} below floor {config.min_nav:,.0f}")
        )

    if nav_high_water and nav_high_water > 0:
        drawdown = plan.nav / nav_high_water - 1.0
        if drawdown < -abs(settings.max_drawdown):
            report.findings.append(
                Rejection(
                    "drawdown_halt",
                    f"NAV is {100 * drawdown:.1f}% below the high-water mark "
                    f"{nav_high_water:,.0f}, past the {100 * settings.max_drawdown:.0f}% halt; "
                    "flatten and review before trading again",
                )
            )

    # Limits are re-derived from the target, not read off the optimizer report.
    breaches = check_limits(
        book.weights,
        book.sectors,
        max_gross=settings.max_gross_leverage,
        max_net=settings.max_net_exposure,
        max_name=settings.max_name_weight,
        max_sector_net=settings.max_sector_net,
    )
    for breach in breaches:
        report.findings.append(
            Rejection(
                "limit_breach",
                f"{breach.name} = {breach.value:.4f} exceeds {breach.limit:.4f}",
            )
        )

    ex_ante = book.diagnostics.get("ex_ante_vol")
    if ex_ante is not None and pd.notna(ex_ante) and ex_ante > config.max_ex_ante_vol:
        report.findings.append(
            Rejection(
                "ex_ante_vol",
                f"target book predicts {100 * ex_ante:.1f}% vol, above the "
                f"{100 * config.max_ex_ante_vol:.0f}% ceiling",
            )
        )

    if plan.turnover > config.max_turnover:
        # An empty account building its first book will always turn over ~gross/2.
        # The cap is there to catch a runaway mid-life rebalance, not to block
        # the initial deployment.
        current_gross = float(plan.current.abs().sum()) if len(plan.current) else 0.0
        if current_gross < 1e-6:
            report.findings.append(
                Rejection(
                    "turnover_bootstrap",
                    f"initial book build turns over {100 * plan.turnover:.1f}% of NAV "
                    "(empty account — per-run turnover cap waived once)",
                    blocking=False,
                )
            )
        else:
            report.findings.append(
                Rejection(
                    "turnover_cap",
                    f"plan turns over {100 * plan.turnover:.1f}% of NAV, above the "
                    f"{100 * config.max_turnover:.0f}% per-run cap — check the target before forcing",
                )
            )

    if len(plan.trades) > config.max_orders:
        report.findings.append(
            Rejection(
                "order_count",
                f"{len(plan.trades)} orders exceeds the {config.max_orders} per-run cap",
            )
        )

    if shortable is not None:
        for trade in plan.trades:
            if trade.side != "sell":
                continue
            delta = -trade.notional / plan.nav if plan.nav > 0 else 0.0
            post = trade.current_weight + delta
            if post < -1e-9 and not shortable.get(trade.symbol, False):
                report.dropped[trade.symbol] = "not shortable at broker"

    if not plan.trades:
        report.findings.append(
            Rejection("no_trades", "book already matches the target", blocking=False)
        )

    return report


def apply_drops(plan, report: PreTradeReport) -> int:
    """Remove dropped symbols from the plan in place. Returns the count removed."""
    if not report.dropped:
        return 0
    keep = [t for t in plan.trades if t.symbol not in report.dropped]
    removed = len(plan.trades) - len(keep)
    for symbol, reason in report.dropped.items():
        plan.skipped.append({"symbol": symbol, "reason": reason})
    plan.trades = keep
    return removed
