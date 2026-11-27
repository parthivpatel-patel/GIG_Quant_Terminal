"""Bar construction from trade prints and a structural bar quality report.

Time, tick, volume and dollar bars are built from caller-supplied prints in
their given order; trades are never split, reordered or invented. The quality
report flags structural defects and statistical oddities without repairing
them and without consulting an exchange calendar.
"""
from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np

from gig import checks

MAX_TRADES = 20000
MAX_BARS = 5000
KINDS = ("TIME", "TICK", "VOLUME", "DOLLAR")
ERRORS = {"NONPOSITIVE_PRICE", "OHLC_INCONSISTENT", "NEGATIVE_VOLUME", "DUPLICATE_TIMESTAMP", "OUT_OF_ORDER"}


def _rows(value: Any, name: str, fields: tuple[str, ...], maximum: int) -> list[dict]:
    if not isinstance(value, list) or not 1 <= len(value) <= maximum:
        raise ValueError(f"{name} must list 1 to {maximum} rows")
    for row in value:
        if not isinstance(row, dict) or set(row) != set(fields):
            raise ValueError(f"each row of {name} needs exactly: {', '.join(fields)}")
    return value


def aggregate_trades(trades, *, kind: str, threshold) -> dict:
    """Group ordered prints into bars closed by elapsed time, count, volume or notional."""
    rows = _rows(trades, "trades", ("timestamp_ns", "price", "size"), MAX_TRADES)
    if kind not in KINDS:
        raise ValueError("kind must be TIME, TICK, VOLUME or DOLLAR")
    stamps = [checks.integer(row["timestamp_ns"], "timestamp_ns", 0, 2**63-1) for row in rows]
    prices = [checks.number(row["price"], "price", positive=True) for row in rows]
    sizes = [checks.number(row["size"], "size", positive=True) for row in rows]
    if any(later < earlier for earlier, later in zip(stamps, stamps[1:], strict=False)):
        raise ValueError("trades must be in nondecreasing timestamp order")
    if kind in ("TIME", "TICK"):
        limit: float = checks.integer(threshold, "threshold", 1, 2**62)
    else:
        limit = checks.number(threshold, "threshold", positive=True)
    bars: list[dict] = []
    members: list[int] = []

    def close(complete: bool) -> None:
        price, size = np.array([prices[i] for i in members]), np.array([sizes[i] for i in members])
        bar = {"start_ns": stamps[members[0]], "end_ns": stamps[members[-1]], "open": float(price[0]),
               "high": float(price.max()), "low": float(price.min()), "close": float(price[-1]),
               "volume": float(size.sum()), "notional": float(price @ size),
               "vwap": float(price @ size/size.sum()), "trades": len(members), "complete": complete}
        if kind == "TIME":
            bar["interval_start_ns"] = stamps[members[0]]//int(limit)*int(limit)
        bars.append(bar)
        members.clear()

    progress = 0.
    for index in range(len(rows)):
        if kind == "TIME" and members and stamps[index]//int(limit) != stamps[members[0]]//int(limit):
            close(True)
        members.append(index)
        if kind == "TIME":
            continue
        progress += {"TICK": 1., "VOLUME": sizes[index], "DOLLAR": prices[index]*sizes[index]}[kind]
        if progress >= limit:
            close(True)
            progress = 0.
    if members:
        # Nothing after the last print shows that its bar had finished forming.
        close(False)
    result = {"method": f"{kind}_BARS", "threshold": limit, "trades": len(rows), "bars": bars,
              "complete_bars": sum(bar["complete"] for bar in bars),
              "trade_handling": "PRINTS_NEVER_SPLIT_THRESHOLD_OVERSHOOT_STAYS_IN_ITS_BAR",
              "derivation": "DERIVED_FROM_CALLER_PRINTS_UNVERIFIED"}
    if kind == "TIME":
        spans = [bar["interval_start_ns"]//int(limit) for bar in bars]
        result["empty_intervals"] = spans[-1]-spans[0]+1-len(spans)
    return result


def quality_report(bars, *, outlier_threshold=8., stale_run=5, gap_multiple=4.) -> dict:
    """Structural errors and statistical warnings for one bar series."""
    rows = _rows(bars, "bars", ("timestamp_ns", "open", "high", "low", "close", "volume"), MAX_BARS)
    cut = checks.number(outlier_threshold, "outlier_threshold", low=3., high=100.)
    run = checks.integer(stale_run, "stale_run", 2, 1000)
    multiple = checks.number(gap_multiple, "gap_multiple", low=1.5, high=1000.)
    stamps = np.array([checks.integer(row["timestamp_ns"], "timestamp_ns", 0, 2**63-1) for row in rows], dtype=np.int64)
    values = np.array([[checks.number(row[key], key) for key in ("open", "high", "low", "close", "volume")]
                       for row in rows])
    prices, volume = values[:, :4], values[:, 4]
    findings: list[dict] = []

    def flag(index: int | np.integer, code: str, detail: str) -> None:
        findings.append({"index": int(index), "code": code, "severity": "ERROR" if code in ERRORS else "WARNING",
                         "detail": detail})

    bad = (prices <= 0).any(axis=1)
    for index in np.where(bad)[0]:
        flag(index, "NONPOSITIVE_PRICE", "a price is zero or negative")
    for index in np.where(~bad & ((prices[:, 1] < prices.max(axis=1)) | (prices[:, 2] > prices.min(axis=1))))[0]:
        flag(index, "OHLC_INCONSISTENT", "high is not the maximum or low is not the minimum")
    for index in np.where(volume < 0)[0]:
        flag(index, "NEGATIVE_VOLUME", "volume is negative")
    for index in np.where(volume == 0)[0]:
        flag(index, "ZERO_VOLUME", "no volume in the bar")
    step = np.diff(stamps)
    for index in np.where(step == 0)[0]+1:
        flag(index, "DUPLICATE_TIMESTAMP", "same timestamp as the previous bar")
    for index in np.where(step < 0)[0]+1:
        flag(index, "OUT_OF_ORDER", "timestamp precedes the previous bar")
    forward = step[step > 0]
    if len(forward):
        typical = float(np.median(forward))
        for index in np.where(step > multiple*typical)[0]+1:
            flag(index, "GAP", f"spacing is {step[index-1]/typical:.6g} times the median")
    close = prices[:, 3]
    outlier_state = "UNAVAILABLE_NONPOSITIVE_CLOSE"
    if (close > 0).all() and len(close) >= 3:
        change = np.diff(np.log(close))
        centre = float(np.median(change))
        spread = 1.4826*float(np.median(np.abs(change-centre)))
        outlier_state = "EVALUATED" if spread > 0 else "UNAVAILABLE_ZERO_MEDIAN_DEVIATION"
        if spread > 0:
            score = (change-centre)/spread
            for index in np.where(np.abs(score) > cut)[0]:
                flag(index+1, "RETURN_OUTLIER", f"robust z-score {score[index]:.6g}")
    start = 0
    for stop in range(1, len(close)+1):
        if stop == len(close) or close[stop] != close[start]:
            if stop-start >= run:
                flag(start, "STALE_CLOSE_RUN", f"{stop-start} consecutive identical closes")
            start = stop
    findings.sort(key=lambda row: (row["index"], row["code"]))
    counts = Counter(row["code"] for row in findings)
    errors = sum(row["severity"] == "ERROR" for row in findings)
    return {"method": "STRUCTURAL_BAR_QUALITY_REPORT", "bars": len(rows), "errors": errors,
            "warnings": len(findings)-errors, "structurally_usable": errors == 0,
            "counts": dict(sorted(counts.items())), "findings": findings[:200],
            "findings_truncated": len(findings) > 200, "outlier_state": outlier_state,
            "report_limits": "CALENDAR_UNAWARE_NO_REPAIR_NO_VENDOR_OR_CORPORATE_ACTION_VERIFICATION",
            "derivation": "DERIVED_FROM_CALLER_BARS_UNVERIFIED"}
