"""Integrated, reproducible signal-to-position rehearsal with zero network I/O."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import datetime
from decimal import ROUND_DOWN, Decimal, localcontext
from hashlib import sha256

import pandas as pd

from gig.execution.oms import OfflineOMS, OrderError, _text, _time
from gig.execution.pretrade import PreTradeConfig, check
from gig.execution.reconcile import ReconcileConfig, build_trade_plan
from gig.execution.targets import TargetBook
from gig.portfolio.construct import dollar_neutral_quantiles


@dataclass(frozen=True, slots=True)
class SimulationLimits:
    max_gross_leverage: float = 2.0
    max_net_exposure: float = 0.10
    max_name_weight: float = 0.05
    max_sector_net: float = 0.10
    max_drawdown: float = 0.12


def simulate_signals(
    scores: Mapping[str, float], prices: Mapping[str, float], sectors: Mapping[str, str], *,
    oms: OfflineOMS, run_id: str, asof: datetime, now: datetime,
    nav: float = 1_000_000.0, n_long: int = 1, n_short: int = 1,
    settings: SimulationLimits | None = None, source_id: str = "local-synthetic-scenario",
    data_classification: str = "SYNTHETIC",
) -> dict:
    """Quantile signals → portfolio → risk → OMS → simulated fills → positions.

    The supplied prices are a deterministic fill assumption; the scenario has
    no spread, impact, fees, or economic-validation claim. A working order found
    after interruption blocks rerun until its local state is reconciled. This
    API neither creates nor accepts a broker object and never reads credentials.
    """
    _text(run_id, "run_id")
    _text(source_id, "source_id")
    at, observed = _time(now), _time(asof)
    settings = settings or SimulationLimits()
    if data_classification not in {"SYNTHETIC", "HISTORICAL"}:
        raise OrderError("offline scenarios require SYNTHETIC or HISTORICAL classification")
    if isinstance(nav, bool) or not math.isfinite(nav) or nav <= 0:
        raise OrderError("scenario NAV must be finite and positive")
    if any(isinstance(n, bool) or not isinstance(n, int) or n <= 0 for n in (n_long, n_short)):
        raise OrderError("portfolio side counts must be positive integers")
    if not scores or len(scores) > 1000:
        raise OrderError("scenario requires between 1 and 1000 signals")
    for symbol, score in scores.items():
        _text(symbol, "symbol")
        if isinstance(score, bool) or not math.isfinite(float(score)):
            raise OrderError("signal scores must be finite")
    if len(scores) < n_long + n_short:
        raise OrderError("insufficient signals for disjoint long and short portfolios")
    for symbol, price in prices.items():
        _text(symbol, "symbol")
        if isinstance(price, bool) or not math.isfinite(float(price)) or price <= 0:
            raise OrderError("scenario prices must be finite and positive")
    if set(scores) - prices.keys():
        raise OrderError("every signal requires a reference price")
    if any(not isinstance(symbol, str) or not isinstance(sector, str)
           or not sector or len(sector) > 128 for symbol, sector in sectors.items()):
        raise OrderError("sector identities require bounded nonempty text")
    try:
        intent = {"source_id": source_id, "asof": observed.isoformat(),
                  "data_classification": data_classification, "nav": nav,
                  "n_long": n_long, "n_short": n_short, "settings": asdict(settings),
                  "scores": dict(scores), "prices": dict(prices), "sectors": dict(sectors)}
        intent_sha256 = sha256(json.dumps(intent, sort_keys=True, allow_nan=False,
                                          separators=(",", ":")).encode()).hexdigest()
    except (ValueError, TypeError):
        raise OrderError("scenario inputs require finite canonical values") from None
    base = {"run_id": run_id, "source_id": source_id, "data_classification": data_classification,
            "execution_classification": "SIMULATED", "broker_connected": False,
            "asof": observed.isoformat(), "timestamp": at.isoformat(),
            "market_open_assumption": True, "economically_validated": False,
            "fill_assumptions": "supplied reference prices; zero fees, spread, and impact"}
    previous = oms.connection.execute("SELECT * FROM simulation_runs WHERE run_id=?", (run_id,)).fetchone()
    if previous:
        if previous["intent_sha256"] != intent_sha256:
            raise OrderError("run_id conflicts with admitted scenario intent")
        if previous["state"] == "COMPLETED":
            return {**json.loads(previous["result_json"]), "orders": [], "replayed": True}
        return {**base, "state": "ERROR", "code": "RECONCILIATION_REQUIRED", "orders": []}
    if oms.kill_switch:
        return {**base, "state": "ERROR", "code": "KILL_SWITCH", "orders": []}
    if oms.recover()["requires_reconciliation"]:
        return {**base, "state": "ERROR", "code": "RECONCILIATION_REQUIRED", "orders": []}
    values = pd.Series(dict(scores), dtype=float).sort_index()
    weights = dollar_neutral_quantiles(values, n_long, n_short,
        max_name_weight=settings.max_name_weight, gross_leverage=settings.max_gross_leverage)
    price_series = pd.Series(dict(prices), dtype=float)
    before = {symbol: Decimal(qty) for symbol, qty in oms.positions().items()}
    if set(before) - prices.keys():
        raise OrderError("held positions require scenario reference prices")
    book = TargetBook(asof=observed.date(), weights=weights, sectors=pd.Series(dict(sectors)),
                      prices=price_series, adv=pd.Series(dtype=float), construction="quantile_equal_weight",
                      universe=len(values), diagnostics={"classification": data_classification})
    plan = build_trade_plan(weights, {s: float(q) for s, q in before.items()}, price_series, nav,
                            config=ReconcileConfig(no_trade_band=0, min_trade_notional=0))
    report = check(plan, book, settings, config=PreTradeConfig(require_market_open=True),
                   market_open=True, shortable={symbol: True for symbol in values.index}, now=at.date())
    if not 0 <= (at - observed).total_seconds() <= 60:
        return {**base, "state": "STALE" if observed <= at else "ERROR", "code": "INVALID_PRICE_TIME",
                "portfolio": weights.to_dict(), "risk": report.to_dict(), "orders": []}
    if report.blocked:
        return {**base, "state": "ERROR", "code": "RISK_BLOCKED", "portfolio": weights.to_dict(),
                "risk": report.to_dict(), "orders": []}
    expected = dict(before)
    orders = []
    with oms._transaction():
        # A pending marker survives an interrupted partial scenario. Replaying
        # that scenario requires explicit reconciliation and never sends again.
        if oms.connection.execute("SELECT 1 FROM simulation_runs WHERE run_id=?", (run_id,)).fetchone():
            raise OrderError("scenario is already admitted; reconcile or replay its result")
        oms.connection.execute("INSERT INTO simulation_runs VALUES(?,?,'RUNNING',NULL)",
                               (run_id, intent_sha256))
    with localcontext() as context:
        context.prec = 100
        notional_cap = Decimal(str(nav)) * Decimal(str(settings.max_name_weight))
        for trade in plan.trades:
            client_id = "sim-" + sha256(f"{run_id}:{trade.symbol}:{trade.side}".encode()).hexdigest()
            quantity, decimal_price = Decimal(str(trade.shares)), Decimal(str(trade.price))
            cap = (notional_cap / decimal_price).quantize(Decimal("0.00000001"), rounding=ROUND_DOWN)
            oms.create(client_id, trade.symbol, trade.side, quantity, limit_price=decimal_price, now=at)
            oms.approve(client_id, reference_price=decimal_price, price_asof=observed,
                        max_notional=notional_cap, max_position_quantity=cap, now=at)
            oms.acknowledge(client_id, now=at)
            order = oms.fill(client_id, f"{client_id}:fill", quantity, decimal_price, now=at)
            orders.append(order)
            expected[trade.symbol] = expected.get(trade.symbol, Decimal(0)) + (quantity if trade.side == "buy" else -quantity)
    reconciliation = oms.reconcile(expected)
    result = {**base, "state": "AVAILABLE", "portfolio": weights.to_dict(), "risk": report.to_dict(),
              "orders": orders, "positions": oms.positions(), "reconciliation": reconciliation,
              "intent_sha256": intent_sha256, "replayed": False}
    with oms._transaction():
        oms.connection.execute("UPDATE simulation_runs SET state='COMPLETED',result_json=? WHERE run_id=?",
                               (json.dumps(result, allow_nan=False, sort_keys=True), run_id))
    return result
