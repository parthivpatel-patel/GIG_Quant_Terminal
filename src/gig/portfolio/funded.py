"""Exact supported-scope cash and lifecycle accounting on existing OMS fills."""
from __future__ import annotations

from collections import defaultdict
from decimal import Decimal
from typing import Any

from gig.execution.oms import TERMINAL, _decimal, _exact
from gig.timestamps import aware_timestamp


@_exact
def funded_account(initial_cash: str, instruments: dict[str, dict[str, Any]],
                   fills: list[dict[str, Any]], orders: list[dict[str, Any]],
                   settlements: list[dict[str, Any]], marks: dict[str, Any],
                   now: str, max_age_seconds: float, *, lifecycle_events=None,
                   trade_settlements=None) -> dict[str, Any]:
    """FIFO premiums use contract multipliers; settlement is not an OMS fill."""
    initial = _decimal(initial_cash)
    cash = initial
    lifecycle_events, trade_settlements = lifecycle_events or [], trade_settlements or []
    dividend_income = dividend_receivable = Decimal(0)
    adjustments: dict[str, Decimal] = defaultdict(Decimal)
    lots: dict[str, list[list[Decimal]]] = defaultdict(list)
    realized: dict[str, Decimal] = defaultdict(Decimal)
    fees: dict[str, Decimal] = defaultdict(Decimal)
    raw: dict[str, Decimal] = defaultdict(Decimal)
    settled: dict[str, Decimal] = defaultdict(Decimal)
    journal = []
    for fill in fills:
        terms = instruments[fill["symbol"]]
        quantity = Decimal(fill["quantity"])*(1 if fill["side"] == "buy" else -1)
        raw[fill["symbol"]] += quantity
        journal.append({"id": fill["execution_id"], "instrument_id": fill["symbol"],
            "quantity": quantity, "price": Decimal(fill["price"])*terms["multiplier"],
            "fee": Decimal(fill["fee"]), "timestamp": fill["timestamp"], "kind": "OMS_FILL"})
    for event in settlements:
        instrument = event["instrument_id"]
        quantity = Decimal(event["position_quantity"])
        settled[instrument] += quantity
        journal.append({"id": event["settlement_id"], "instrument_id": instrument,
            "quantity": -quantity, "price": Decimal(event["payoff"])*instruments[instrument]["multiplier"],
            "fee": Decimal(event["fee"]), "timestamp": event["available_at"], "kind": "CASH_SETTLEMENT"})
    for event in lifecycle_events:
        journal.append({**event, "id": event["event_id"], "timestamp": event["applied_at"]})
    for event in sorted(journal, key=lambda row: (aware_timestamp(row["timestamp"]), row["kind"] != "OMS_FILL", row["id"])):
        if event["kind"] in {"CASH_DIVIDEND", "DIVIDEND_ACCRUAL"}:
            amount = Decimal(event["cash_amount"])
            cash += amount
            dividend_income += Decimal(event.get("income_amount", event["cash_amount"]))
            dividend_receivable += Decimal(event.get("receivable_amount", "0"))
            continue
        if event["kind"] == "SPLIT":
            symbol = event["instrument_id"]
            before = sum((q for q, _ in lots[symbol]), Decimal(0))
            try:
                changed = [[q*event["ratio_new"]/event["ratio_old"], p*event["ratio_old"]/event["ratio_new"]] for q, p in lots[symbol]]
                for quantity, price in changed:
                    _decimal(quantity)
                    _decimal(price)
                    if quantity != quantity.to_integral_value():
                        raise ValueError("fractional split delivery or cash-in-lieu is unsupported")
            except ArithmeticError:
                raise ValueError("split lot basis cannot be represented exactly at supported precision") from None
            if sum((q*p for q, p in changed), Decimal(0)) != sum((q*p for q, p in lots[symbol]), Decimal(0)):
                raise ValueError("split must conserve exact lot cost basis")
            lots[symbol] = changed
            adjustments[symbol] += sum((q for q, _ in changed), Decimal(0))-before
            continue
        symbol, quantity, price, fee = (event[k] for k in ("instrument_id", "quantity", "price", "fee"))
        cash -= quantity*price+fee
        fees[symbol] += fee
        remaining = quantity
        while lots[symbol] and remaining and (lots[symbol][0][0] > 0) != (remaining > 0):
            lot_quantity, lot_price = lots[symbol][0]
            closed = min(abs(lot_quantity), abs(remaining))
            direction = Decimal(1) if lot_quantity > 0 else Decimal(-1)
            realized[symbol] += closed*(price-lot_price)*direction
            lots[symbol][0][0] -= closed*direction
            remaining += closed*direction
            if not lots[symbol][0][0]:
                lots[symbol].pop(0)
        if remaining:
            lots[symbol].append([remaining, price])
    pending_expiries, missing, stale, rows = [], [], [], []
    total_value = total_unrealized = collateral = Decimal(0)
    quantities = {}
    at = aware_timestamp(now)
    for symbol in sorted(set(raw) | set(settled)):
        terms = instruments[symbol]
        quantity = sum((lot[0] for lot in lots[symbol]), Decimal(0))
        if quantity != raw[symbol]-settled[symbol]+adjustments[symbol]:
            raise ValueError("funded lifecycle quantity reconciliation failed")
        quantities[symbol] = quantity
        if quantity < 0:
            raise ValueError("unsupported short liability in funded account")
        if terms["kind"] == "OPTION" and quantity and aware_timestamp(terms["expiry"]) <= at:
            pending_expiries.append(symbol)
        basis = sum((q*p for q, p in lots[symbol]), Decimal(0))
        price = value = unrealized = None
        if not quantity:
            value = unrealized = Decimal(0)
        elif symbol not in marks or symbol in pending_expiries:
            missing.append(symbol)
        else:
            mark = marks[symbol]
            if aware_timestamp(mark["as_of"]) > at:
                raise ValueError("funded mark lies after the decision clock")
            if (at.value-aware_timestamp(mark["as_of"]).value)/1e9 > max_age_seconds:
                stale.append(symbol)
            price = Decimal(mark["price"])
            current_value = quantity*price*terms["multiplier"]
            value = current_value
            current_unrealized = current_value-basis
            unrealized = current_unrealized
            total_value += current_value
            total_unrealized += current_unrealized
        rows.append({**terms, "quantity": str(quantity), "raw_oms_quantity": str(raw[symbol]),
            "settled_quantity": str(settled[symbol]), "mark": str(price) if price is not None else None,
            "mark_asof": marks.get(symbol, {}).get("as_of"), "market_value": str(value) if value is not None else None,
            "cost_basis": str(basis), "realized_gross_pnl": str(realized[symbol]),
            "unrealized_pnl": str(unrealized) if unrealized is not None else None,
            "fees": str(fees[symbol]), "classification": "SIMULATED"})
    from gig.portfolio.lifecycle import settlement_balances
    receivables, payables, settled_trade, pending_trades = settlement_balances(fills, instruments, trade_settlements, now)
    for symbol, adjustment in adjustments.items():
        settled_trade[symbol] = settled_trade.get(symbol, Decimal(0))+adjustment
    reservations = []
    buy_reserve = put_reserve = fee_reserve = Decimal(0)
    reserved_sales: dict[str, Decimal] = defaultdict(Decimal)
    for order in orders:
        if order["state"] in TERMINAL:
            continue
        terms = instruments[order["symbol"]]
        remaining = Decimal(order["quantity"])-Decimal(order["filled"])
        executed_fees = sum((Decimal(f["fee"]) for f in fills if f["client_id"] == order["client_id"]), Decimal(0))
        fee_left = Decimal(order["fee_budget"])-executed_fees
        if fee_left < 0:
            raise ValueError("executed fees exceed funded reservation")
        premium = remaining*Decimal(order["funded_limit_price"])*terms["multiplier"] if order["side"] == "buy" else Decimal(0)
        # This account does not open shorts, including short puts. Every sell
        # reserves owned inventory; conditional future buys do not offset it.
        secured = Decimal(0)
        buy_reserve += premium
        put_reserve += secured
        fee_reserve += fee_left
        if order["side"] == "sell" and not secured:
            reserved_sales[order["symbol"]] += remaining
        reservations.append({"client_id": order["client_id"], "instrument_id": order["symbol"],
            "remaining_quantity": str(remaining), "buy_cash": str(premium),
            "short_put_collateral": str(secured), "fee_cash": str(fee_left), "state": order["state"]})
    for symbol, quantity in reserved_sales.items():
        available_inventory = min(settled_trade.get(symbol, Decimal(0)), quantities.get(symbol, Decimal(0))) if instruments[symbol]["kind"] == "EQUITY" else quantities.get(symbol, Decimal(0))
        if quantity > max(available_inventory, Decimal(0)):
            raise ValueError("working sales exceed owned inventory")
    reserved_cash = buy_reserve+put_reserve+fee_reserve
    available = cash-receivables-collateral-reserved_cash
    funding_ok = cash >= 0 and available >= 0
    mark_state = "UNKNOWN" if missing else "STALE" if stale else "AVAILABLE"
    total_realized = sum(realized.values(), Decimal(0))
    total_fees = sum(fees.values(), Decimal(0))
    if dividend_receivable < 0:
        raise ValueError("unsupported negative dividend receivable")
    nav = cash+total_value+dividend_receivable if not missing else None
    pnl = total_realized+total_unrealized+dividend_income-total_fees if not missing else None
    if nav is not None and (pnl is None or nav != initial+pnl):
        raise ValueError("funded cash, liabilities and P&L conservation failed")
    return {"currency": "USD", "initial_cash": str(initial), "cash": str(cash),
        "nav": str(nav) if nav is not None else None, "total_pnl": str(pnl) if pnl is not None else None,
        "realized_gross_pnl": str(total_realized), "unrealized_pnl": str(total_unrealized) if not missing else None,
        "fees": str(total_fees), "rows": rows, "raw_oms_positions": {k: str(v) for k, v in sorted(raw.items())},
        "dividend_income": str(dividend_income), "dividend_receivable": str(dividend_receivable), "lifecycle_position_adjustments": {k: str(v) for k, v in adjustments.items() if v},
        "settled_cash": str(cash+payables-receivables), "unsettled_sale_receivables": str(receivables),
        "unsettled_buy_payables": str(payables), "pending_trade_settlements": pending_trades,
        "settled_trade_positions": {k: str(v) for k, v in sorted(settled_trade.items())},
        "effective_positions": {k: str(v) for k, v in sorted(quantities.items())},
        "settled_positions": {k: str(v) for k, v in sorted(settled.items())}, "pending_expiries": pending_expiries,
        "mark_state": mark_state, "missing_marks": missing, "stale_marks": stale,
        "reservations": reservations, "cash_secured_put_collateral": str(collateral),
        "reserved_cash": str(reserved_cash), "available_cash": str(available),
        "funding_state": "AVAILABLE" if funding_ok and not pending_expiries else "UNKNOWN" if pending_expiries else "ERROR",
        "reconciliation_state": "AVAILABLE" if mark_state == "AVAILABLE" and funding_ok and not pending_expiries else "UNKNOWN",
        "funding_scope": "USD_CASH_ACCOUNT_LONG_EQUITY_LONG_EUROPEAN_CASH_OPTIONS_ONLY",
        "fully_funded_certified": False, "external_counterparty_reconciliation": "UNKNOWN",
        "future_settlement_fee_state": "UNKNOWN_UNTIL_EXPLICIT_SETTLEMENT",
        "financing": "NO_BORROWING_OR_LEVERAGED_DEBIT_ALLOWED_WITHIN_SUPPORTED_SCOPE",
        "unsupported": ["all opening short positions", "short put assignment liabilities", "American exercise", "physical delivery", "FX", "borrowed cash", "unreported cash flows"],
        "settlement_policy": "EUROPEAN_EXPIRY_AUTOMATIC_CASH_INTRINSIC_EXERCISE_LONG_OPTIONS_ONLY",
        "cost_basis_method": "FIFO_OCCURRENCE_TIMESTAMP_THEN_EVENT_ID", "trading_approved": False}
