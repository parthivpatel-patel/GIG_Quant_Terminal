"""Executed-cost evidence that keeps estimated, simulated and observed costs apart.

Four evidence classes never share an aggregate:

* ``ESTIMATED``: pre-trade model output (spread + commission + square-root impact).
* ``SIMULATED``: offline simulator fills. Print corroboration never promotes them.
* ``OBSERVED_HISTORICAL``: caller-asserted historical executions. They enter the
  observed aggregate only when exactly one independent print corroborates them.
* ``BROKER_RECONCILED``: requires broker statements from an authorized account
  connection. Brokerage connectivity is disabled, so this class is rejected.

Fill decomposition is exact Decimal arithmetic against a causal arrival quote:
``side * (price - arrival_mid) * qty + fees == spread + slippage + fees``.
Market impact needs a post-trade midpoint observed after the fill; it is a
measured price move, not a causal attribution, and is UNKNOWN when absent.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from decimal import Decimal, InvalidOperation
from typing import Any

from gig.timestamps import aware_timestamp

VERSION = "executed-cost-evidence-v1"
ESTIMATED, SIMULATED, OBSERVED, BROKER = (
    "ESTIMATED", "SIMULATED", "OBSERVED_HISTORICAL", "BROKER_RECONCILED")
FILL_CLASSES = frozenset({SIMULATED, OBSERVED, BROKER})
SIMULATOR_SOURCES = frozenset({"offline-simulator", "offline-oms", "synthetic"})
_MAX_ROWS = 5000
_TOP = {"currency", "as_of", "estimates", "fills", "prints", "match_tolerance_seconds",
        "max_participation"}
_ESTIMATE = {"estimate_id", "instrument", "side", "quantity", "reference_price", "spread_bps",
             "commission_bps", "impact_eta", "adv_value"}
_FILL = {"fill_id", "instrument", "side", "quantity", "price", "fees", "executed_at",
         "knowledge_at", "evidence_class", "source", "arrival", "post_trade"}
_ARRIVAL = {"bid", "ask", "observed_at", "source"}
_POST = {"mid", "observed_at", "source"}
_PRINT = {"print_id", "instrument", "price", "size", "observed_at", "source"}


def _decimal(value: Any, name: str, *, positive: bool = False) -> Decimal:
    if isinstance(value, bool) or not isinstance(value, (int, float, str, Decimal)):
        raise ValueError(f"{name} must be a finite decimal")
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    try:
        number = Decimal(str(value))
    except InvalidOperation as exc:
        raise ValueError(f"{name} must be a finite decimal") from exc
    if not number.is_finite() or number < 0 or (positive and number == 0) or abs(number) > Decimal("1e15"):
        raise ValueError(f"{name} out of bounds")
    return number


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > 256:
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _side(value: Any) -> int:
    if value not in ("BUY", "SELL"):
        raise ValueError("side must be BUY or SELL")
    return 1 if value == "BUY" else -1


def _fields(row: Any, required: set[str], optional: frozenset[str] | set[str] = frozenset(),
            name: str = "row") -> Mapping[str, Any]:
    if not isinstance(row, Mapping) or not required <= set(row) or set(row) - required - optional:
        raise ValueError(f"{name} fields do not match the contract")
    return row


def _rows(value: Any, name: str) -> Sequence[Any]:
    if not isinstance(value, list) or len(value) > _MAX_ROWS:
        raise ValueError(f"{name} must be a bounded list")
    return value


def _hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     default=str).encode()).hexdigest()


def _estimate(row: Mapping[str, Any]) -> dict[str, Any]:
    side = _side(row["side"])
    quantity = _decimal(row["quantity"], "quantity", positive=True)
    price = _decimal(row["reference_price"], "reference_price", positive=True)
    adv = _decimal(row["adv_value"], "adv_value")
    notional = quantity * price
    spread_bps = _decimal(row["spread_bps"], "spread_bps")
    commission_bps = _decimal(row["commission_bps"], "commission_bps")
    eta = _decimal(row["impact_eta"], "impact_eta")
    # Square-root participation is irrational; Decimal.sqrt keeps it deterministic.
    impact_bps = None if adv == 0 else eta * (notional / adv).sqrt() * 10_000
    bp = Decimal("0.0001")
    return {
        "estimate_id": _text(row["estimate_id"], "estimate_id"),
        "instrument": _text(row["instrument"], "instrument"), "side": row["side"],
        "evidence_class": ESTIMATED, "notional": notional,
        "transaction_fees": notional * commission_bps * bp,
        "spread_cost": notional * spread_bps * bp / 2,
        "slippage": None,
        "market_impact": None if impact_bps is None else notional * impact_bps * bp,
        "market_impact_state": "UNKNOWN_NO_ADV" if impact_bps is None else "MODEL_ESTIMATE",
        "model": "half-spread + commission + square-root impact (TransactionCostModel parameters)",
        "direction": side,
    }


def _match_prints(fills: list[dict[str, Any]], prints: list[dict[str, Any]], tolerance_ns: int) -> None:
    claims: dict[str, list[str]] = {}
    for fill in fills:
        candidates = [p for p in prints
                      if p["instrument"] == fill["instrument"] and p["price"] == fill["price"]
                      and p["size"] == fill["quantity"]
                      and abs(p["observed_ns"] - fill["executed_ns"]) <= tolerance_ns]
        fill["_candidates"] = [p["print_id"] for p in candidates]
        for print_id in fill["_candidates"]:
            claims.setdefault(print_id, []).append(fill["fill_id"])
    by_id = {p["print_id"]: p for p in prints}
    for fill in fills:
        candidate_ids = fill.pop("_candidates")
        if not candidate_ids:
            status, matched = "UNMATCHED", None
        elif len(candidate_ids) > 1 or len(claims[candidate_ids[0]]) > 1:
            status, matched = "AMBIGUOUS", None
        else:
            matched = by_id[candidate_ids[0]]
            status = "MATCHED" if matched["source"] != fill["source"] else "SAME_SOURCE_UNVERIFIED"
        fill["print_corroboration"] = status
        fill["matched_print_id"] = None if matched is None else matched["print_id"]


def _fill(row: Mapping[str, Any], as_of_ns: int) -> dict[str, Any]:
    evidence_class = row["evidence_class"]
    if evidence_class not in FILL_CLASSES:
        raise ValueError("unsupported fill evidence class")
    if evidence_class == BROKER:
        raise ValueError("BROKER_RECONCILED evidence requires an authorized broker connection; brokerage is disabled")
    source = _text(row["source"], "source")
    if evidence_class == OBSERVED and source.lower() in SIMULATOR_SOURCES:
        raise ValueError("simulator output cannot be admitted as observed historical execution")
    executed = aware_timestamp(row["executed_at"])
    knowledge = aware_timestamp(row["knowledge_at"])
    if knowledge < executed or knowledge.value > as_of_ns:
        raise ValueError("fill knowledge must follow execution and precede the report cutoff")
    side = _side(row["side"])
    quantity = _decimal(row["quantity"], "quantity", positive=True)
    price = _decimal(row["price"], "price", positive=True)
    fees = _decimal(row["fees"], "fees")
    arrival = _fields(row["arrival"], _ARRIVAL, name="arrival")
    bid, ask = _decimal(arrival["bid"], "bid", positive=True), _decimal(arrival["ask"], "ask", positive=True)
    if bid > ask:
        raise ValueError("crossed arrival quote is not admitted")
    arrival_at = aware_timestamp(arrival["observed_at"])
    if arrival_at > executed:
        raise ValueError("arrival quote must be observed at or before execution")
    mid = (bid + ask) / 2
    touch = ask if side == 1 else bid
    spread = side * (touch - mid) * quantity
    slippage = side * (price - touch) * quantity
    shortfall = side * (price - mid) * quantity + fees
    if spread + slippage + fees != shortfall:  # pragma: no cover - exact Decimal identity
        raise ArithmeticError("cost decomposition does not conserve shortfall")
    impact = None
    impact_state = "UNKNOWN_NO_POST_TRADE_QUOTE"
    post = row.get("post_trade")
    if post is not None:
        post = _fields(post, _POST, name="post_trade")
        post_at = aware_timestamp(post["observed_at"])
        if post_at <= executed or post_at.value > as_of_ns:
            raise ValueError("post-trade midpoint must follow execution and precede the cutoff")
        impact = side * (_decimal(post["mid"], "post_trade_mid", positive=True) - mid) * quantity
        impact_state = "MEASURED_PRICE_MOVE_NOT_CAUSAL"
        _text(post["source"], "post_trade.source")
    notional = price * quantity
    return {
        "fill_id": _text(row["fill_id"], "fill_id"), "instrument": _text(row["instrument"], "instrument"),
        "side": row["side"], "evidence_class": evidence_class, "source": source,
        "arrival_source": _text(arrival["source"], "arrival.source"),
        "quantity": quantity, "price": price, "notional": notional, "arrival_mid": mid,
        "executed_at": executed.isoformat(), "executed_ns": executed.value,
        "arrival_at": arrival_at.isoformat(), "arrival_ns": arrival_at.value,
        "transaction_fees": fees, "spread_cost": spread, "slippage": slippage,
        "market_impact": impact, "market_impact_state": impact_state,
        "implementation_shortfall": shortfall,
        "shortfall_bps": shortfall / notional * 10_000,
    }


def _print(row: Mapping[str, Any], as_of_ns: int) -> dict[str, Any]:
    observed = aware_timestamp(row["observed_at"])
    if observed.value > as_of_ns:
        raise ValueError("print observed after the report cutoff")
    return {"print_id": _text(row["print_id"], "print_id"), "instrument": _text(row["instrument"], "instrument"),
            "price": _decimal(row["price"], "price", positive=True),
            "size": _decimal(row["size"], "size", positive=True),
            "observed_ns": observed.value, "source": _text(row["source"], "source")}


def _participation(fill: dict[str, Any], prints: list[dict[str, Any]], limit: Decimal | None) -> None:
    window = [p["size"] for p in prints if p["instrument"] == fill["instrument"]
              and fill["arrival_ns"] <= p["observed_ns"] <= fill["executed_ns"]
              and p["source"] != fill["source"]]
    volume = sum(window, Decimal(0))
    if volume == 0:
        fill["observed_interval_volume"] = None
        fill["participation"] = None
        fill["capacity_state"] = "UNKNOWN_NO_INDEPENDENT_PRINTS"
        return
    fill["observed_interval_volume"] = volume
    fill["participation"] = fill["quantity"] / volume
    if limit is None:
        fill["capacity_state"] = "NO_ASSUMPTION_SUPPLIED"
    else:
        fill["capacity_state"] = "WITHIN_ASSUMPTION" if fill["participation"] <= limit else "EXCEEDS_ASSUMPTION"


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    notional = sum((r["notional"] for r in rows), Decimal(0))
    out: dict[str, Any] = {"count": len(rows), "notional": notional}
    for key in ("transaction_fees", "spread_cost", "slippage", "market_impact"):
        known = [r[key] for r in rows if r.get(key) is not None]
        total = sum(known, Decimal(0)) if known else None
        out[key] = total
        out[f"{key}_known_count"] = len(known)
        out[f"{key}_bps"] = None if total is None or notional == 0 or len(known) != len(rows) else total / notional * 10_000
    return out


def cost_evidence_report(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Build a deterministic, class-separated executed-cost evidence report."""
    _fields(payload, {"currency", "as_of"}, _TOP - {"currency", "as_of"}, name="request")
    currency = _text(payload["currency"], "currency")
    if currency != "USD":
        raise ValueError("only USD cost evidence is supported")
    as_of = aware_timestamp(payload["as_of"])
    tolerance = payload.get("match_tolerance_seconds", 1)
    if isinstance(tolerance, bool) or not isinstance(tolerance, int) or not 0 <= tolerance <= 3600:
        raise ValueError("match_tolerance_seconds must be an integer in [0, 3600]")
    limit = payload.get("max_participation")
    limit = None if limit is None else _decimal(limit, "max_participation", positive=True)
    if limit is not None and limit > 1:
        raise ValueError("max_participation must be at most 1")
    estimates = [_estimate(_fields(r, _ESTIMATE, name="estimate"))
                 for r in _rows(payload.get("estimates", []), "estimates")]
    fills = [_fill(_fields(r, _FILL - {"post_trade"}, {"post_trade"}, name="fill"), as_of.value)
             for r in _rows(payload.get("fills", []), "fills")]
    prints = [_print(_fields(r, _PRINT, name="print"), as_of.value)
              for r in _rows(payload.get("prints", []), "prints")]
    for name, rows, key in (("estimate", estimates, "estimate_id"), ("fill", fills, "fill_id"),
                            ("print", prints, "print_id")):
        if len({r[key] for r in rows}) != len(rows):
            raise ValueError(f"duplicate {name} identity")
    _match_prints(fills, prints, tolerance * 1_000_000_000)
    for fill in fills:
        _participation(fill, prints, limit)
        corroborated = fill["print_corroboration"] == "MATCHED"
        if fill["evidence_class"] == SIMULATED:
            fill["evidence_state"] = "SIMULATED_NOT_MARKET_EXECUTION"
        else:
            fill["evidence_state"] = ("OBSERVED_PRINT_CORROBORATED" if corroborated
                                      else "CALLER_ASSERTED_UNCORROBORATED")
    observed = [f for f in fills if f["evidence_state"] == "OBSERVED_PRINT_CORROBORATED"]
    uncorroborated = [f for f in fills if f["evidence_state"] == "CALLER_ASSERTED_UNCORROBORATED"]
    simulated = [f for f in fills if f["evidence_class"] == SIMULATED]
    for fill in fills:
        fill.pop("executed_ns")
        fill.pop("arrival_ns")
    inputs_sha256 = _hash(payload)
    report = {
        "version": VERSION, "currency": currency, "as_of": as_of.isoformat(),
        "inputs_sha256": inputs_sha256,
        "classes": {
            ESTIMATED: _aggregate(estimates),
            SIMULATED: _aggregate(simulated),
            OBSERVED: _aggregate(observed),
            "OBSERVED_UNCORROBORATED_EXCLUDED": {"count": len(uncorroborated)},
            BROKER: {"state": "UNAVAILABLE", "reason": "BROKER_DISCONNECTED"},
        },
        "estimates": estimates, "fills": fills,
        "prints_admitted": len(prints),
        "claims": {
            "observed_executions_print_corroborated": bool(observed),
            "broker_reconciled": False,
            "simulated_promoted_by_prints": False,
            "market_impact_causal": False,
            "aggregates_cross_evidence_classes": False,
        },
        "limitations": [
            "Estimated, simulated and observed costs are never summed together.",
            "Print corroboration matches price, size, instrument and time only; it does not prove counterparty identity.",
            "Market impact is the post-trade midpoint move and is not a causal impact estimate.",
            "Capacity uses independent prints between arrival and execution only; absent prints leave it UNKNOWN.",
            "Broker reconciliation is unavailable because no broker account is connected.",
        ],
    }
    report["report_sha256"] = _hash({k: v for k, v in report.items()})
    return report
