"""Supported equity cash-settlement and corporate-action arithmetic.

No stock loan, physical option delivery, cash-in-lieu guess or arbitrary capital
credit is introduced. Source event clocks and compensation records stay explicit.
"""
from __future__ import annotations

from decimal import Decimal

from gig.execution.oms import _exact
from gig.timestamps import aware_timestamp


@_exact
def settlement_balances(fills, instruments, settlement_rows, now):
    at = aware_timestamp(now)
    dates = {row["execution_id"]: aware_timestamp(row["settlement_at"]) for row in settlement_rows}
    receivables = payables = Decimal(0)
    settled = {}
    pending = []
    for fill in fills:
        symbol = fill["symbol"]
        quantity = Decimal(fill["quantity"])*(1 if fill["side"] == "buy" else -1)
        due = dates.get(fill["execution_id"], aware_timestamp(fill["timestamp"]))
        if due <= at:
            settled[symbol] = settled.get(symbol, Decimal(0))+quantity
        elif instruments[symbol]["kind"] == "EQUITY":
            gross = abs(quantity)*Decimal(fill["price"])
            if quantity > 0:
                payables += gross
            else:
                receivables += gross
            pending.append({"execution_id": fill["execution_id"], "instrument_id": symbol,
                            "side": fill["side"], "gross_cash": str(gross), "settlement_at": due.isoformat()})
    return receivables, payables, settled, pending


@_exact
def dividend_entitlement(fills, settlements, lifecycle, *, instrument_id, ex_at, record_at):
    """Trade before ex-time and settlement by record-time; no later buys qualify."""
    ex, record = aware_timestamp(ex_at), aware_timestamp(record_at)
    dates = {r["execution_id"]: aware_timestamp(r["settlement_at"]) for r in settlements}
    events = []
    for fill in fills:
        if fill["symbol"] != instrument_id or aware_timestamp(fill["timestamp"]) >= ex:
            continue
        if dates.get(fill["execution_id"], aware_timestamp(fill["timestamp"])) <= record:
            events.append((aware_timestamp(fill["timestamp"]), 0, fill["execution_id"],
                           Decimal(fill["quantity"])*(1 if fill["side"] == "buy" else -1)))
    for row in lifecycle:
        if row["kind"] == "SPLIT" and row["instrument_id"] == instrument_id and aware_timestamp(row["effective_at"]) < ex:
            if aware_timestamp(row["applied_at"]) > record:
                raise ValueError("late split knowledge prevents an unambiguous dividend entitlement")
            events.append((aware_timestamp(row["applied_at"]), 1, row["event_id"], row))
    quantity = Decimal(0)
    for _, kind, _, value in sorted(events, key=lambda x: x[:3]):
        quantity = quantity+value if kind == 0 else quantity*value["ratio_new"]/value["ratio_old"]
    if quantity < 0:
        raise ValueError("unsupported negative dividend entitlement")
    return quantity
