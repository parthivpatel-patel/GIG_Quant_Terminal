"""Bounded deterministic offline TWAP planning, with no clock waits or orders."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal

from gig.execution.oms import OrderError, _decimal, _exact, _text, _time


@dataclass(frozen=True, slots=True)
class TWAPSlice:
    client_id: str
    timestamp: datetime
    quantity: Decimal
    classification: str = "SIMULATED_PLAN"


@_exact
def plan_twap(
    client_id: str, quantity: str | int | Decimal, start: datetime,
    end: datetime, slices: int, *, lot_size: str | int | Decimal = "1",
) -> tuple[TWAPSlice, ...]:
    """Schedule lot-exact children over [start, end), distributing remainder.

    Timestamps have explicit Python datetime microsecond precision. No exchange
    calendar, participation curve, market-data feed, or broker is implied.
    """
    _text(client_id, "client_id")
    qty, lot = _decimal(quantity), _decimal(lot_size)
    start, end = _time(start), _time(end)
    if isinstance(slices, bool) or not isinstance(slices, int) or not 1 <= slices <= 10_000:
        raise OrderError("slices must be an integer between 1 and 10000")
    if end <= start:
        raise OrderError("end must be after start")
    lots, remainder = divmod(qty, lot)
    if remainder:
        raise OrderError("quantity must be an exact lot multiple")
    count = min(slices, int(lots))
    duration = end - start
    duration_us = (duration.days * 86400 + duration.seconds) * 1_000_000 + duration.microseconds
    if duration_us < count:
        raise OrderError("schedule is shorter than its timestamp precision")
    base, extra = divmod(int(lots), count)
    if len(client_id) > 116:
        raise OrderError("parent client_id too long for deterministic child IDs")
    return tuple(TWAPSlice(f"{client_id}:{i+1}", start + timedelta(microseconds=i * duration_us // count),
                          lot * (base + (1 if i < extra else 0))) for i in range(count))
