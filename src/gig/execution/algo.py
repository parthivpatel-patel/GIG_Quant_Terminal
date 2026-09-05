"""
Order placement mechanics: limit pricing and child-order slicing.

Two choices here are deliberate and worth stating, because both cost money if
made the other way.

**Marketable limits, not market orders.** A market order accepts any price the
book offers. In a name with a wide spread or a thin top of book that can be far
from the last print, and the resulting slippage is invisible in the account
until it shows up as unexplained drag against the backtest. A limit priced a
few basis points through the touch fills in normal conditions and simply does
not fill in abnormal ones, which is the correct behaviour for a strategy whose
edge is measured in basis points per name.

**Slicing by notional.** One parent order sized at the full target competes
with itself. Splitting into child orders bounded by notional keeps each print
small relative to the book, which is the condition under which the square-root
impact term in the cost model is a fair estimate rather than an underestimate.

This is not a scheduled participation algo. A desk running real size would work
orders against a volume curve over the session; here the slices go out back to
back, and that difference is a known limitation rather than a hidden one.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from gig.types import Side


@dataclass(frozen=True, slots=True)
class ExecutionConfig:
    use_limit: bool = True
    limit_offset_bps: float = 10.0     # how far through the touch to price
    max_slice_notional: float = 25_000.0
    slice_pause_s: float = 0.35
    tick: float = 0.01


def limit_price(side: Side, reference: float, offset_bps: float, tick: float = 0.01) -> float:
    """
    Price a marketable limit: above the reference to buy, below it to sell.

    Rounded to the tick *away* from the aggressive side so rounding never makes
    the order less likely to fill than the caller asked for.
    """
    if reference <= 0:
        raise ValueError("reference price must be positive")
    shift = reference * (offset_bps / 10_000.0)
    raw = reference + shift if side == "buy" else reference - shift
    if tick <= 0:
        return float(max(raw, 0.0))
    rounded = math.ceil(raw / tick) * tick if side == "buy" else math.floor(raw / tick) * tick
    return float(max(round(rounded, 4), tick))


def slice_shares(
    shares: float,
    price: float,
    max_slice_notional: float,
    allow_fractional: bool = False,
) -> list[float]:
    """
    Split a parent order into child orders bounded by notional.

    Whole-share slicing distributes the remainder across the first children
    rather than leaving a stub child of one or two shares, which would pay a
    full spread for almost no size.
    """
    if shares <= 0:
        return []
    if price <= 0 or max_slice_notional <= 0:
        return [float(shares)]

    per_slice = max_slice_notional / price
    if not allow_fractional:
        per_slice = math.floor(per_slice)
        if per_slice < 1:
            return [float(shares)]
        n = math.ceil(shares / per_slice)
        base, extra = divmod(int(shares), n)
        return [float(base + (1 if i < extra else 0)) for i in range(n) if base + (1 if i < extra else 0) > 0]

    n = math.ceil(shares / per_slice)
    return [float(shares / n)] * n
