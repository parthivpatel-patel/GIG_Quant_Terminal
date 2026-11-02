"""Exact FIFO accounting and explicit research risk for simulated equity fills."""

from __future__ import annotations

import math
from collections import defaultdict
from decimal import Decimal
from fractions import Fraction
from typing import Any

from gig.execution.oms import _decimal, _exact, _text
from gig.risk.analytics import historical_loss_risk, stress_positions
from gig.timestamps import aware_timestamp


def signed_decimal(value: Any) -> Decimal:
    """Signed exact values retain the OMS's bounded decimal admission."""
    if isinstance(value, (bool, float)):
        raise ValueError("an exact signed decimal string or integer is required")
    if isinstance(value, str) and len(value) > 128:
        raise ValueError("signed decimal exceeds bounds")
    try:
        result = Decimal(value)
    except (TypeError, ValueError, ArithmeticError):
        raise ValueError("invalid signed decimal") from None
    _decimal(result.copy_abs(), positive=False)
    return result


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def canonical_fills(executions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Identical event retries are idempotent; conflicting identities fail."""
    if not isinstance(executions, list) or len(executions) > 10000:
        raise ValueError("simulated accounting supports at most 10000 fills")
    found: dict[str, dict[str, Any]] = {}
    required = {"execution_id", "client_id", "symbol", "side", "quantity", "price",
                "fee", "timestamp", "classification"}
    for row in executions:
        if not isinstance(row, dict) or set(row) != required:
            raise ValueError("execution evidence has unsupported or missing fields")
        for key in ("execution_id", "client_id", "symbol"):
            _text(row[key], key)
        if row["side"] not in {"buy", "sell"} or row["classification"] != "SIMULATED":
            raise ValueError("only explicit simulated equity executions are admitted")
        checked = {**row, "quantity": str(_decimal(row["quantity"])),
                   "price": str(_decimal(row["price"])),
                   "fee": str(_decimal(row["fee"], positive=False)),
                   "timestamp": aware_timestamp(row["timestamp"]).isoformat()}
        previous = found.get(checked["execution_id"])
        if previous is not None and previous != checked:
            raise ValueError("execution identity conflicts with preserved evidence")
        found[checked["execution_id"]] = checked
    return sorted(found.values(), key=lambda row: (aware_timestamp(row["timestamp"]), row["execution_id"]))


@_exact
def reconcile_account(
    executions: list[dict[str, Any]], initial_cash: str, marks: dict[str, Any]
) -> dict[str, Any]:
    """Replay all fills in timestamp/identity order with exact FIFO lot costs.

    Fees are expensed at execution, separately from gross realized trading P&L.
    Mark sources are not authenticated here. No cash flows beyond initial cash,
    financing, borrow costs, FX, dividends or nonlinear derivatives are implied.
    """
    cash = initial = _decimal(initial_cash)
    fills = canonical_fills(executions)
    lots: dict[str, list[list[Decimal]]] = defaultdict(list)
    realized: dict[str, Decimal] = defaultdict(Decimal)
    fees: dict[str, Decimal] = defaultdict(Decimal)
    for fill in fills:
        symbol = fill["symbol"]
        quantity, price, fee = (Decimal(fill[key]) for key in ("quantity", "price", "fee"))
        signed = quantity if fill["side"] == "buy" else -quantity
        cash -= signed * price + fee
        fees[symbol] += fee
        remaining = signed
        while lots[symbol] and remaining and (lots[symbol][0][0] > 0) != (remaining > 0):
            lot_quantity, lot_price = lots[symbol][0]
            closed = min(abs(lot_quantity), abs(remaining))
            direction = Decimal(1) if lot_quantity > 0 else Decimal(-1)
            realized[symbol] += closed * (price - lot_price) * direction
            lots[symbol][0][0] -= closed * direction
            remaining += closed * direction
            if not lots[symbol][0][0]:
                lots[symbol].pop(0)
        if remaining:
            lots[symbol].append([remaining, price])
    rows = []
    total_market_value = total_unrealized = Decimal(0)
    for symbol in sorted(lots):
        quantity = sum((lot[0] for lot in lots[symbol]), Decimal(0))
        cost_basis = sum((lot[0] * lot[1] for lot in lots[symbol]), Decimal(0))
        if quantity:
            if symbol not in marks:
                raise ValueError("every nonzero position requires an explicit mark")
            mark = _decimal(marks[symbol])
            market_value = quantity * mark
            unrealized = market_value - cost_basis
        else:
            mark, market_value, unrealized = None, Decimal(0), Decimal(0)
        total_market_value += market_value
        total_unrealized += unrealized
        rows.append({"instrument_id": symbol, "quantity": str(quantity),
                     "mark": str(mark) if mark is not None else None,
                     "market_value": str(market_value), "cost_basis": str(cost_basis),
                     "realized_gross_pnl": str(realized[symbol]),
                     "unrealized_pnl": str(unrealized), "fees": str(fees[symbol]),
                     "open_lots": [{"quantity": str(q), "price": str(p)} for q, p in lots[symbol]],
                     "classification": "SIMULATED"})
    total_realized = sum(realized.values(), Decimal(0))
    total_fees = sum(fees.values(), Decimal(0))
    nav = cash + total_market_value
    pnl = total_realized + total_unrealized - total_fees
    if nav != initial + pnl:
        raise ValueError("cash/position/P&L accounting reconciliation failed")
    return {"rows": rows, "cash": str(cash), "initial_cash": str(initial),
            "market_value": str(total_market_value), "nav": str(nav),
            "realized_gross_pnl": str(total_realized), "unrealized_pnl": str(total_unrealized),
            "fees": str(total_fees), "total_pnl": str(pnl), "fill_count": len(fills),
            "cost_basis_method": "FIFO_EXECUTION_TIMESTAMP_THEN_ID",
            "fees_treatment": "EXPENSE_AT_EXECUTION",
            "classification": "SIMULATED", "currency": "USD",
            "excluded_costs": ["financing", "stock_borrow", "FX", "unreported_fees"],
            "broker_connected": False, "trading_approved": False}


def _ratio(value: Decimal, nav: Decimal) -> tuple[float, dict[str, str]]:
    exact = Fraction(value) / Fraction(nav)
    return float(exact), {"numerator": str(exact.numerator), "denominator": str(exact.denominator)}


@_exact
def risk_report(
    account: dict[str, Any], *, metadata: dict[str, Any], limits: dict[str, Any],
    nav_history: list[tuple[str, str]], now: str, confidence: float,
    scenarios: list[dict[str, Any]],
) -> dict[str, Any]:
    """Risk estimates and strict limit evidence; absent factors are never zero."""
    confidence = _finite(confidence, "confidence")
    if not 0 < confidence < 1:
        raise ValueError("confidence must lie strictly between zero and one")
    permitted = {"max_gross", "max_abs_net", "max_name_weight", "max_sector_gross",
                 "max_sector_net", "max_factor_exposure", "min_cash", "max_drawdown"}
    if not isinstance(limits, dict) or set(limits) - permitted:
        raise ValueError("unsupported portfolio risk limit")
    scalar_limits: dict[str, float] = {}
    for key, value in limits.items():
        if key not in {"max_factor_exposure", "min_cash"}:
            checked = _finite(value, key)
            if not 0 <= checked <= (1 if key == "max_drawdown" else 100):
                raise ValueError("portfolio risk limit exceeds bounds")
            scalar_limits[key] = checked
    if "min_cash" in limits:
        _decimal(limits["min_cash"], positive=False)
    factor_limits = limits.get("max_factor_exposure", {})
    if not isinstance(factor_limits, dict) or len(factor_limits) > 16:
        raise ValueError("factor limits require a bounded mapping")
    for key, value in factor_limits.items():
        _text(key, "factor")
        if not 0 <= _finite(value, "factor exposure limit") <= 100:
            raise ValueError("factor limit exceeds bounds")
    if not isinstance(metadata, dict) or len(metadata) > 64:
        raise ValueError("instrument metadata requires a bounded mapping")
    checked_meta = {}
    for symbol, row in metadata.items():
        _text(symbol, "instrument")
        if not isinstance(row, dict) or set(row) - {"sector", "factors"}:
            raise ValueError("invalid instrument metadata")
        sector = row.get("sector")
        if sector is not None:
            _text(sector, "sector")
        factors = row.get("factors", {})
        if not isinstance(factors, dict) or len(factors) > 16:
            raise ValueError("factor metadata exceeds bounds")
        for key, value in factors.items():
            _text(key, "factor")
            if abs(_finite(value, "factor loading")) > 100:
                raise ValueError("factor loading exceeds bounds")
        checked_meta[symbol] = {"sector": sector, "factors": factors}
    nav = Decimal(account["nav"])
    cash = Decimal(account["cash"])
    held = [row for row in account["rows"] if Decimal(row["quantity"])]
    violations: list[str] = []
    if nav <= 0:
        violations.append("NON_POSITIVE_NAV")
    notionals = {row["instrument_id"]: Decimal(row["market_value"]) for row in held}
    gross = sum(map(abs, notionals.values()), Decimal(0))
    net = sum(notionals.values(), Decimal(0))
    weights = {key: _ratio(value, nav)[0] for key, value in notionals.items()} if nav > 0 else {}
    exact_weights = {key: _ratio(value, nav)[1] for key, value in notionals.items()} if nav > 0 else {}
    sector_gross: dict[str, Decimal] = defaultdict(Decimal)
    sector_net: dict[str, Decimal] = defaultdict(Decimal)
    missing_sector = []
    for symbol, value in notionals.items():
        sector = checked_meta.get(symbol, {}).get("sector")
        if sector is None:
            missing_sector.append(symbol)
        else:
            sector_gross[sector] += abs(value)
            sector_net[sector] += value
    factors = set(factor_limits)
    for row in checked_meta.values():
        factors.update(row["factors"])
    factor_exposure = {}
    for factor in sorted(factors):
        missing = [symbol for symbol in notionals if factor not in checked_meta.get(symbol, {}).get("factors", {})]
        value = None
        if not missing and nav > 0:
            total = sum((Fraction(notional) * Fraction(str(checked_meta[symbol]["factors"][factor]))
                         for symbol, notional in notionals.items()), Fraction(0))
            value = float(total / Fraction(nav))
        factor_exposure[factor] = {"state": "UNKNOWN" if value is None else "AVAILABLE",
                                   "value": value, "missing_instruments": missing}
        if factor in factor_limits and value is None:
            violations.append(f"MISSING_FACTOR:{factor}")
        elif factor in factor_limits and value is not None and abs(value) > factor_limits[factor]:
            violations.append(f"FACTOR_LIMIT:{factor}")
    if missing_sector and ({"max_sector_gross", "max_sector_net"} & limits.keys()):
        violations.append("MISSING_SECTOR")
    if nav > 0:
        comparisons = {"max_gross": _ratio(gross, nav)[0],
                       "max_abs_net": abs(_ratio(net, nav)[0]),
                       "max_name_weight": max(map(abs, weights.values()), default=0.),
                       "max_sector_gross": max((_ratio(v, nav)[0] for v in sector_gross.values()), default=0.),
                       "max_sector_net": max((abs(_ratio(v, nav)[0]) for v in sector_net.values()), default=0.)}
        for key, value in comparisons.items():
            if key in scalar_limits and value > scalar_limits[key] + 1e-12:
                violations.append(key.upper())
    if "min_cash" in limits and cash < Decimal(limits["min_cash"]):
        violations.append("MIN_CASH")
    cutoff = aware_timestamp(now)
    history: dict[Any, Decimal] = {}
    for stamp, value in nav_history:
        observed = aware_timestamp(stamp)
        amount = signed_decimal(value)
        if observed > cutoff:
            continue
        key = observed.isoformat()
        if key in history and history[key] != amount:
            raise ValueError("conflicting NAV history at one observation time")
        history[key] = amount
    history[cutoff.isoformat()] = nav
    ordered = sorted(history.items(), key=lambda item: aware_timestamp(item[0]))
    peak = max([Decimal(account["initial_cash"]), *(value for _, value in ordered)])
    drawdown = float((Fraction(nav) - Fraction(peak)) / Fraction(peak))
    if "max_drawdown" in limits and -drawdown > limits["max_drawdown"] + 1e-12:
        violations.append("MAX_DRAWDOWN")
    nonpositive_bases = sum(previous <= 0 for (_, previous), _ in zip(ordered[:-1], ordered[1:], strict=True))
    returns = [float((Fraction(current) - Fraction(previous)) / Fraction(previous))
               for (_, previous), (_, current) in zip(ordered[:-1], ordered[1:], strict=True)
               if previous > 0]
    historical = ({"state": "AVAILABLE", **historical_loss_risk(returns, confidence)}
                  if len(returns) >= 20 and not nonpositive_bases else
                  {"state": "UNAVAILABLE", "var": None, "expected_shortfall": None,
                   "observations": len(returns), "minimum_observations": 20,
                   "nonpositive_return_bases": nonpositive_bases})
    historical["classification"] = "MODEL_ESTIMATE_FROM_SIMULATED_NAV"
    historical["horizon"] = "one saved simulation observation; intervals may be irregular"
    if not isinstance(scenarios, list) or len(scenarios) > 32:
        raise ValueError("scenario count exceeds bound")
    scenario_rows = []
    for scenario in scenarios:
        if not isinstance(scenario, dict) or set(scenario) != {"name", "shocks"}:
            raise ValueError("scenario requires name and explicit shocks")
        _text(scenario["name"], "scenario")
        shocks = scenario["shocks"]
        if not isinstance(shocks, dict) or set(shocks) != set(notionals):
            raise ValueError("every held instrument requires an explicit scenario shock")
        shock_values = [_finite(shocks[symbol], "shock") for symbol in notionals]
        if any(not -1 <= value <= 100 for value in shock_values):
            raise ValueError("equity scenario shocks exceed bounds")
        if nav > 0 and held:
            modeled = stress_positions([float(row["quantity"]) for row in held],
                                       [float(row["mark"]) for row in held], [shock_values], nav=float(nav))
            pnl, nav_return = modeled["pnl"][0], modeled["nav_return"][0]
        else:
            pnl, nav_return = (0., 0.) if nav > 0 else (None, None)
        scenario_rows.append({"name": scenario["name"], "pnl": pnl, "nav_return": nav_return,
                              "classification": "MODEL_ESTIMATE", "shocks": shocks})
    return {"gross_notional": str(gross), "net_notional": str(net),
            "gross_exposure": _ratio(gross, nav)[0] if nav > 0 else None,
            "net_exposure": _ratio(net, nav)[0] if nav > 0 else None,
            "weights": weights, "weight_exact": exact_weights,
            "sector_state": "UNKNOWN" if missing_sector else "AVAILABLE",
            "missing_sector_instruments": missing_sector,
            "sector_gross": {key: _ratio(value, nav)[0] for key, value in sector_gross.items()} if nav > 0 else {},
            "sector_net": {key: _ratio(value, nav)[0] for key, value in sector_net.items()} if nav > 0 else {},
            "factor_exposure": factor_exposure, "limits": limits,
            "limits_state": "AVAILABLE" if limits else "UNKNOWN", "violations": violations,
            "current_drawdown": drawdown, "peak_nav": str(peak),
            "historical_loss_risk": historical, "scenarios": scenario_rows,
            "scenario_loss_risk": historical_loss_risk([row["nav_return"] for row in scenario_rows], confidence, minimum=1)
            if scenario_rows and nav > 0 else {"state": "UNAVAILABLE", "var": None, "expected_shortfall": None},
            "classification": "MODEL_ESTIMATE", "trading_approved": False}
