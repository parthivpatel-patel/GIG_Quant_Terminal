"""Deterministic event-data analytics with explicit units and unknown signs.

Timestamps are integer UTC nanoseconds. Quote matching never uses a quote newer
than trade time minus lag. Neither quotes nor trades may be inferred from bars.
"""
from __future__ import annotations

import math
from bisect import bisect_right
from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Integral

import numpy as np


def _finite(value: float, name: str, *, positive: bool = False) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be numeric")
    try:
        out = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not math.isfinite(out) or (out <= 0 if positive else out < 0):
        raise ValueError(f"{name} must be finite and {'positive' if positive else 'nonnegative'}")
    return out


def _integer(value: int, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer")
    return int(value)


def _identity(instrument: str, source: str, data_kind: str) -> None:
    if any(not isinstance(x, str) or not x.strip() for x in (instrument, source, data_kind)):
        raise ValueError("instrument, source and data_kind must be nonempty strings")


@dataclass(frozen=True, slots=True)
class Quote:
    timestamp_ns: int
    instrument: str
    bid: float
    ask: float
    bid_size: float
    ask_size: float
    sequence: int = 0
    source: str = "unspecified"
    data_kind: str = "UNKNOWN"

    def __post_init__(self) -> None:
        object.__setattr__(self, "timestamp_ns", _integer(self.timestamp_ns, "timestamp_ns"))
        object.__setattr__(self, "sequence", _integer(self.sequence, "sequence"))
        _identity(self.instrument, self.source, self.data_kind)
        for name in ("bid", "ask", "bid_size", "ask_size"):
            object.__setattr__(self, name, _finite(getattr(self, name), name, positive=name in ("bid", "ask")))
        if self.bid > self.ask:
            raise ValueError("crossed quotes are not admitted")

    @property
    def midpoint(self) -> float:
        return self.bid / 2.0 + self.ask / 2.0


@dataclass(frozen=True, slots=True)
class Trade:
    timestamp_ns: int
    instrument: str
    price: float
    size: float
    sequence: int = 0
    source: str = "unspecified"
    data_kind: str = "UNKNOWN"

    def __post_init__(self) -> None:
        object.__setattr__(self, "timestamp_ns", _integer(self.timestamp_ns, "timestamp_ns"))
        object.__setattr__(self, "sequence", _integer(self.sequence, "sequence"))
        _identity(self.instrument, self.source, self.data_kind)
        object.__setattr__(self, "price", _finite(self.price, "price", positive=True))
        object.__setattr__(self, "size", _finite(self.size, "size", positive=True))


@dataclass(frozen=True, slots=True)
class MatchedTrade:
    trade: Trade
    quote: Quote | None
    quote_age_ns: int | None
    reason: str


@dataclass(frozen=True, slots=True)
class TradeClassification:
    match: MatchedTrade
    sign: int  # +1 buy initiated, -1 sell initiated, 0 unknown
    method: str


@dataclass(frozen=True, slots=True)
class KyleEstimate:
    coefficient: float | None  # price units per signed-volume unit
    intercept: float | None
    r_squared: float | None
    sample_count: int
    state: str


@dataclass(frozen=True, slots=True)
class VPINResult:
    bucket_volume: float
    imbalances: tuple[float, ...]  # abs(buy-sell)/bucket volume
    values: tuple[float, ...]  # rolling arithmetic average over complete buckets
    window_buckets: int
    completed_buckets: int
    incomplete_volume: float
    excluded_unknown_volume: float
    state: str


def _unique_events(events: Sequence[Quote] | Sequence[Trade]) -> None:
    if len(events) > 1_000_000:
        raise ValueError("maximum event count exceeded")
    identities = [(e.instrument, e.timestamp_ns, e.sequence) for e in events]
    if len(set(identities)) != len(identities):
        raise ValueError("duplicate event identity: instrument, timestamp_ns, sequence")


def match_quotes(
    trades: Sequence[Trade], quotes: Sequence[Quote], *, lag_ns: int = 0,
    max_age_ns: int | None = 5_000_000_000,
) -> list[MatchedTrade]:
    """As-of same-instrument match, preserving trade input order.

    Among quotes with equal nanoseconds, greatest sequence wins. A quote exactly
    at trade time is admissible at zero lag; callers needing strictly preceding
    events must set lag_ns >= 1. Age is measured from actual trade time.
    """
    lag_ns = _integer(lag_ns, "lag_ns")
    if lag_ns < 0:
        raise ValueError("lag_ns must be nonnegative")
    if max_age_ns is not None:
        max_age_ns = _integer(max_age_ns, "max_age_ns")
        if max_age_ns < 0:
            raise ValueError("max_age_ns must be nonnegative")
    _unique_events(trades)
    _unique_events(quotes)
    groups: dict[str, list[Quote]] = {}
    for q in sorted(quotes, key=lambda q: (q.timestamp_ns, q.sequence)):
        groups.setdefault(q.instrument, []).append(q)
    times = {s: [q.timestamp_ns for q in rows] for s, rows in groups.items()}
    out = []
    for trade in trades:
        rows = groups.get(trade.instrument, [])
        i = bisect_right(times.get(trade.instrument, []), trade.timestamp_ns - lag_ns) - 1
        if i < 0:
            out.append(MatchedTrade(trade, None, None, "NO_PRIOR_QUOTE"))
            continue
        q = rows[i]
        age = int(trade.timestamp_ns - q.timestamp_ns)
        if max_age_ns is not None and age > max_age_ns:
            out.append(MatchedTrade(trade, None, age, "STALE_QUOTE"))
        else:
            out.append(MatchedTrade(trade, q, age, "MATCHED"))
    return out


def lee_ready(matches: Sequence[MatchedTrade]) -> list[TradeClassification]:
    """Quote rule, then tick rule at midpoint or missing quote.

    A zero tick carries the last nonzero tick, separately for each instrument.
    If no prior nonzero tick exists, a midpoint/missing-quote print is UNKNOWN.
    This is a trade-sign estimate, not exchange aggressor-side observation.
    """
    _unique_events([m.trade for m in matches])
    result: list[TradeClassification | None] = [None] * len(matches)
    previous: dict[str, float] = {}
    last_tick: dict[str, int] = {}
    for index, match in sorted(enumerate(matches), key=lambda p: (p[1].trade.timestamp_ns, p[1].trade.sequence)):
        t = match.trade
        old = previous.get(t.instrument)
        tick = 0 if old is None or t.price == old else (1 if t.price > old else -1)
        if tick:
            last_tick[t.instrument] = tick
        q = match.quote
        if q is not None and (q.instrument != t.instrument or q.timestamp_ns > t.timestamp_ns):
            raise ValueError("invalid quote/trade match")
        if q is not None and t.price != q.midpoint:
            sign = 1 if t.price > q.midpoint else -1
            method = "QUOTE_ESTIMATE"
        else:
            sign = last_tick.get(t.instrument, 0)
            method = "TICK_ESTIMATE" if sign else "UNKNOWN"
        result[index] = TradeClassification(match, sign, method)
        previous[t.instrument] = t.price
    return [r for r in result if r is not None]


def order_flow_imbalance(quotes: Sequence[Quote]) -> np.ndarray:
    """Cont-style best-level OFI in displayed-size units; first event is zero.

    Inputs must contain one instrument, ordered by timestamp and sequence. No
    inference from OHLCV or snapshots with undocumented event ordering is made.
    """
    _unique_events(quotes)
    if len({q.instrument for q in quotes}) > 1:
        raise ValueError("OFI requires one instrument")
    keys = [(q.timestamp_ns, q.sequence) for q in quotes]
    if keys != sorted(keys):
        raise ValueError("quotes must be chronologically ordered")
    out: np.ndarray = np.zeros(len(quotes), dtype=float)
    for i in range(1, len(quotes)):
        old, new = quotes[i - 1], quotes[i]
        out[i] = (
            (new.bid_size if new.bid >= old.bid else 0.0)
            - (old.bid_size if new.bid <= old.bid else 0.0)
            - (new.ask_size if new.ask <= old.ask else 0.0)
            + (old.ask_size if new.ask >= old.ask else 0.0)
        )
    if not np.isfinite(out).all():
        raise ValueError("OFI arithmetic overflow")
    return out


def kyle_lambda(price_changes: Sequence[float], signed_volumes: Sequence[float]) -> KyleEstimate:
    """OLS delta-midprice = intercept + lambda * signed volume.

    Inputs must be contemporaneously aligned, non-overlapping intervals. The
    estimator does not claim causality, significance, or annualized impact.
    """
    y = np.asarray(price_changes, dtype=float)
    x = np.asarray(signed_volumes, dtype=float)
    if y.ndim != 1 or x.shape != y.shape or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("aligned finite one-dimensional arrays are required")
    n = len(x)
    if n < 3:
        return KyleEstimate(None, None, None, n, "UNAVAILABLE")
    # Scaling protects the fit from arbitrary volume units and large magnitudes.
    scale_x = max(float(np.max(np.abs(x))), 1.0)
    scale_y = max(float(np.max(np.abs(y))), 1.0)
    xs, ys = x / scale_x, y / scale_y
    xc, yc = xs - xs.mean(), ys - ys.mean()
    den = float(xc @ xc)
    if den <= np.finfo(float).eps * n:
        return KyleEstimate(None, None, None, n, "UNAVAILABLE")
    slope = float(xc @ yc) / den
    intercept = float(ys.mean() - slope * xs.mean())
    residual = yc - slope * xc
    tss = float(yc @ yc)
    r2 = None if tss == 0.0 else float(max(0.0, min(1.0, 1.0 - float(residual @ residual) / tss)))
    coefficient = slope * scale_y / scale_x
    intercept *= scale_y
    if not math.isfinite(coefficient) or not math.isfinite(intercept):
        raise ValueError("Kyle estimate arithmetic overflow")
    return KyleEstimate(coefficient, intercept, r2, n, "AVAILABLE")


def vpin(
    sizes: Sequence[float], signs: Sequence[int], *, bucket_volume: float,
    window_buckets: int = 50, max_buckets: int = 100_000,
) -> VPINResult:
    """Equal-volume bucket VPIN using caller-supplied trade-sign estimates.

    Unknown signs (0) are excluded and their volume is reported. Trades are
    split exactly across bucket boundaries; the unfinished final bucket is
    excluded. Values start only after a full rolling window. This is an
    event-sign VPIN estimate, not bulk-volume-classification VPIN.
    """
    bucket_volume = _finite(bucket_volume, "bucket_volume", positive=True)
    window_buckets = _integer(window_buckets, "window_buckets")
    max_buckets = _integer(max_buckets, "max_buckets")
    if (window_buckets < 1 or not 1 <= max_buckets <= 100_000
            or len(sizes) != len(signs) or len(sizes) > 1_000_000):
        raise ValueError("positive window/bucket limits and aligned inputs required")
    volumes = [_finite(v, "size", positive=True) for v in sizes]
    directions = [_integer(s, "sign") for s in signs]
    if any(s not in (-1, 0, 1) for s in directions):
        raise ValueError("sign must be -1, 0, or 1")
    try:
        total = math.fsum(v for v, s in zip(volumes, directions, strict=True) if s)
        unknown = math.fsum(v for v, s in zip(volumes, directions, strict=True) if not s)
    except OverflowError as exc:
        raise ValueError("aggregate trade volume overflow") from exc
    if not math.isfinite(total) or total / bucket_volume > max_buckets:
        raise ValueError("maximum bucket count exceeded")
    imbalance, filled = 0.0, 0.0
    buckets: list[float] = []
    for volume, sign in zip(volumes, directions, strict=True):
        if not sign:
            continue
        remaining = volume
        while remaining > 0:
            take = min(remaining, bucket_volume - filled)
            if take <= 0 or remaining - take == remaining:
                raise ValueError("bucket precision insufficient for volume range")
            imbalance += sign * take
            filled += take
            remaining -= take
            if filled >= bucket_volume:
                buckets.append(min(1.0, abs(imbalance) / bucket_volume))
                filled, imbalance = 0.0, 0.0
    # Avoid O(bucket_count*window_buckets) work on an otherwise admitted request.
    # A compensated three-term update and periodic exact rebase keep the mean
    # stable without allocating a window-sized temporary at every observation.
    values = []
    running = 0.0
    for i, imbalance in enumerate(buckets):
        running = math.fsum((running, imbalance,
                            -buckets[i-window_buckets] if i >= window_buckets else 0.0))
        if i >= window_buckets-1:
            if i % 1024 == 0:
                running = math.fsum(buckets[i-window_buckets+1:i+1])
            values.append(min(1.0, max(0.0, running/window_buckets)))
    return VPINResult(bucket_volume, tuple(buckets), tuple(values), window_buckets, len(buckets),
                      filled, unknown, "AVAILABLE" if values else "UNAVAILABLE")


def spread_metrics(quote: Quote) -> dict[str, float | str | int | None]:
    """Displayed top-of-book spread and queue imbalance, preserving attribution.

    Queue imbalance is (bid_size-ask_size)/(bid_size+ask_size); zero combined
    depth yields UNKNOWN, represented by None rather than zero imbalance.
    """
    spread = quote.ask - quote.bid
    depth = quote.bid_size + quote.ask_size
    if not math.isfinite(depth):
        raise ValueError("combined displayed depth overflow")
    return {"midpoint": quote.midpoint, "quoted_spread": spread,
            "quoted_spread_bps": spread / quote.midpoint * 10_000,
            "queue_imbalance": None if depth == 0 else (quote.bid_size - quote.ask_size) / depth,
            "queue_state": "UNKNOWN" if depth == 0 else "AVAILABLE",
            "source": quote.source, "data_kind": quote.data_kind,
            "instrument": quote.instrument, "timestamp_ns": quote.timestamp_ns}


def execution_costs(
    trade: Trade, *, arrival_mid: float, side: int, fees: float = 0.0,
    benchmark_price: float | None = None,
) -> dict[str, float | str | int | None]:
    """Implementation shortfall relative to caller-provided arrival benchmark.

    Positive cost is adverse for either side. Fees use total currency, quantity
    uses underlying units. This calculation does not manufacture an arrival
    quote or infer a trade's economic side from its sign classifier.
    """
    arrival_mid = _finite(arrival_mid, "arrival_mid", positive=True)
    fees = _finite(fees, "fees")
    side = _integer(side, "side")
    if side not in (-1, 1):
        raise ValueError("side must be +1 buy or -1 sell")
    signed_price_cost = side * (trade.price - arrival_mid)
    implementation_shortfall = signed_price_cost * trade.size + fees
    cost_bps = (signed_price_cost / arrival_mid + fees / trade.size / arrival_mid) * 10_000
    benchmark_cost = None
    if benchmark_price is not None:
        benchmark_price = _finite(benchmark_price, "benchmark_price", positive=True)
        benchmark_cost = side * (trade.price - benchmark_price) * trade.size + fees
    if not all(math.isfinite(v) for v in (implementation_shortfall, cost_bps)):
        raise ValueError("execution cost arithmetic overflow")
    if benchmark_cost is not None and not math.isfinite(benchmark_cost):
        raise ValueError("benchmark cost arithmetic overflow")
    return {"implementation_shortfall": implementation_shortfall, "cost_bps": cost_bps,
            "signed_price_cost_per_unit": signed_price_cost, "fees": fees,
            "benchmark_shortfall": benchmark_cost, "instrument": trade.instrument,
            "timestamp_ns": trade.timestamp_ns, "trade_source": trade.source,
            "data_kind": trade.data_kind, "benchmark_provenance": "CALLER_SUPPLIED"}
