"""Bars and indicators for the market-charting workspace.

Synthetic mode serves deterministic, fictional OHLCV bars (labelled SYNTHETIC);
they are not market observations. Local mode reads only bars admitted to the
terminal workspace's point-in-time store, as known at the cutoff, and reports
their freshness against an explicit policy. Nothing is fetched, interpolated or
filled: without admitted bars the result is UNAVAILABLE.
"""
from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from gig.research import indicators as ind
from gig.service.contracts import Contract, State
from gig.service.serialization import public_error

TIMEFRAMES = {"1D": None, "1W": "W-FRI", "1M": "ME"}
PERIODS_PER_YEAR = {"1D": 252.0, "1W": 52.0, "1M": 12.0}
SYNTHETIC_SOURCE = "deterministic-synthetic-ohlcv-v1"
SYNTHETIC_ASOF = "2026-09-30T20:00:00+00:00"
MAX_BARS = 1000
PARAMETERS = {"sma": [20, 50], "ema": [20], "bollinger": {"n": 20, "k": 2.0}, "vwap": 20, "rsi": 14,
              "macd": {"fast": 12, "slow": 26, "signal": 9}, "atr": 14, "realized_volatility": 20}
_SYMBOL = re.compile(r"^[A-Za-z0-9._:\-]{1,64}$")


def _unavailable(scope: str, code: str = "DATA_UNAVAILABLE") -> Contract:
    return Contract.admit(State.UNAVAILABLE, None, source="local", classification="UNAVAILABLE",
                          scope=scope, error=public_error(code))


@lru_cache(maxsize=1)
def synthetic_bars() -> pd.DataFrame:
    """504 business days of fictional OHLCV for SYN001-SYN012, closing at 20:00 UTC."""
    rng = np.random.default_rng(20261104)
    days = pd.bdate_range(end="2026-09-30", periods=504)
    stamps = (days + pd.Timedelta(hours=20)).tz_localize("UTC")
    frames = []
    market = rng.normal(0.0002, 0.008, days.size)
    for i in range(12):
        drift, vol = rng.normal(0.0002, 0.0003), rng.uniform(0.009, 0.022)
        returns = drift + 0.6 * market + rng.normal(0, vol, days.size)
        close = (40 + 12 * i) * np.exp(np.cumsum(returns))
        gap = rng.normal(0, vol / 4, days.size)
        open_ = np.concatenate(([close[0] * np.exp(-returns[0])], close[:-1])) * np.exp(gap)
        high = np.maximum(open_, close) * np.exp(np.abs(rng.normal(0, vol / 2, days.size)))
        low = np.minimum(open_, close) * np.exp(-np.abs(rng.normal(0, vol / 2, days.size)))
        volume = np.round(rng.lognormal(13 - i * 0.08, 0.35, days.size) * (1 + 25 * np.abs(returns)))
        frames.append(pd.DataFrame({"symbol": f"SYN{i + 1:03d}", "timestamp": stamps, "open": open_.round(4),
                                    "high": high.round(4), "low": low.round(4), "close": close.round(4),
                                    "volume": volume.astype(np.int64)}))
    return pd.concat(frames, ignore_index=True)


def resample(frame: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """Aggregate bars: first open, highest high, lowest low, last close, summed volume.

    A period is stamped with the timestamp of its last contributing bar, so an
    aggregated bar is never dated before the observations it summarizes.
    """
    rule = TIMEFRAMES[timeframe]
    frame = frame.sort_values("timestamp")
    if rule is None:
        return frame.reset_index(drop=True)
    grouped = frame.assign(stamp=frame["timestamp"]).set_index("timestamp").resample(rule)
    out = pd.DataFrame({"open": grouped["open"].first(), "high": grouped["high"].max(), "low": grouped["low"].min(),
                        "close": grouped["close"].last(), "volume": grouped["volume"].sum(), "timestamp": grouped["stamp"].max()})
    return out.dropna(subset=["close"]).reset_index(drop=True)


def _clean(values: np.ndarray) -> list[float | None]:
    return [None if not np.isfinite(v) else round(float(v), 6) for v in values]


def compute(frame: pd.DataFrame, timeframe: str) -> dict[str, Any]:
    o, h, lo, c = (frame[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    v = frame["volume"].to_numpy(float)
    p = PARAMETERS
    bands = ind.bollinger(c, p["bollinger"]["n"], p["bollinger"]["k"])
    lines = ind.macd(c, **p["macd"])
    return {
        "sma20": _clean(ind.sma(c, 20)), "sma50": _clean(ind.sma(c, 50)), "ema20": _clean(ind.ema(c, 20)),
        "bb_upper": _clean(bands["upper"]), "bb_middle": _clean(bands["middle"]), "bb_lower": _clean(bands["lower"]),
        "vwap20": _clean(ind.vwap(h, lo, c, v, p["vwap"])), "rsi14": _clean(ind.rsi(c, p["rsi"])),
        "macd": _clean(lines["macd"]), "macd_signal": _clean(lines["signal"]), "macd_histogram": _clean(lines["histogram"]),
        "atr14": _clean(ind.atr(h, lo, c, p["atr"])),
        "rv20": _clean(ind.realized_volatility(c, p["realized_volatility"], PERIODS_PER_YEAR[timeframe])),
    }


def _series(frame: pd.DataFrame, symbol: str, timeframe: str, *, volume_unit: str) -> dict[str, Any]:
    bars = resample(frame, timeframe).tail(MAX_BARS).reset_index(drop=True)
    return {
        "symbol": symbol, "timeframe": timeframe, "timeframes": list(TIMEFRAMES), "volume_unit": volume_unit,
        "bars": {"t": [pd.Timestamp(t).isoformat() for t in bars["timestamp"]],
                 **{k: [round(float(x), 6) for x in bars[k]] for k in ("open", "high", "low", "close")},
                 "volume": [float(x) for x in bars["volume"]]},
        "indicators": compute(bars, timeframe), "parameters": PARAMETERS,
        "conventions": "EMA seeded with SMA; RSI/ATR Wilder; Bollinger population std; VWAP rolling 20-bar typical price; "
                       f"realized volatility annualized with {PERIODS_PER_YEAR[timeframe]:g} periods per year.",
    }


def _watchlist(frame: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []
    for symbol, group in frame.sort_values("timestamp").groupby("symbol", sort=True):
        close = group["close"].to_numpy(float)
        rows.append({"symbol": symbol, "last": round(float(close[-1]), 4),
                     "change": None if close.size < 2 else round(float(close[-1] / close[-2] - 1), 6),
                     "bars": int(close.size), "last_timestamp": pd.Timestamp(group["timestamp"].iloc[-1]).isoformat()})
    return rows


def synthetic_chart(symbol: str, timeframe: str) -> Contract:
    scope = "charts.bars"
    frame = synthetic_bars()
    if timeframe not in TIMEFRAMES or symbol not in set(frame["symbol"]):
        return _unavailable(scope, "INVALID_REQUEST")
    data = _series(frame[frame["symbol"] == symbol], symbol, timeframe, volume_unit="fictional units")
    data["watchlist"] = _watchlist(frame)
    data["notes"] = ["Fictional instruments and deterministic synthetic bars; not market observations."]
    return Contract.admit(State.AVAILABLE, data, source=SYNTHETIC_SOURCE, classification="SYNTHETIC",
                          as_of=SYNTHETIC_ASOF, scope=scope, evidence_state="RESEARCH_OUTPUT", readiness="RESEARCH_ONLY")


def local_chart(store: Path, symbol: str | None, timeframe: str, cutoff: pd.Timestamp,
                max_age_seconds: float | None) -> Contract:
    """Admitted bars from the workspace store as known at ``cutoff``.

    One source/classification/adjustment/currency combination is charted at a
    time, so differently adjusted or sourced bars are never spliced together.
    """
    scope = "charts.bars"
    if timeframe not in TIMEFRAMES or (symbol is not None and not _SYMBOL.match(symbol)):
        return _unavailable(scope, "INVALID_REQUEST")
    if max_age_seconds is None:
        return _unavailable(scope, "INVALID_REQUEST")
    if not store.is_file():
        return _unavailable(scope)
    from gig.data.validated import PointInTimeStore
    snapshot = PointInTimeStore(store, read_only=True).snapshot(cutoff, max_age_seconds=max_age_seconds)
    rows = snapshot["rows"]
    if not rows:
        return _unavailable(scope)
    frame = pd.DataFrame(rows)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, format="ISO8601")
    frame["volume"] = frame["volume"].astype("int64")
    frame = frame.rename(columns={"instrument_id": "symbol"})
    series_key = ["source", "classification", "corporate_action_policy", "currency"]
    symbol = symbol or sorted(frame["symbol"].unique())[0]
    chosen = frame[frame["symbol"] == symbol]
    if chosen.empty:
        return _unavailable(scope)
    latest = chosen.sort_values("timestamp").iloc[-1]
    key = tuple(latest[k] for k in series_key)
    chosen = chosen[(chosen[series_key] == pd.Series(key, index=series_key)).all(axis=1)]
    same = frame[(frame[series_key] == pd.Series(key, index=series_key)).all(axis=1)]
    observed = pd.Timestamp(chosen["timestamp"].max())
    age = (cutoff - observed).total_seconds()
    state = State.STALE if age > max_age_seconds else State.AVAILABLE
    data = _series(chosen[["symbol", "timestamp", "open", "high", "low", "close", "volume"]], symbol, timeframe,
                   volume_unit="as admitted")
    data["watchlist"] = _watchlist(same)
    data["admitted"] = {"source": key[0], "classification": key[1], "corporate_action_policy": key[2], "currency": key[3],
                        "snapshot_hashes": sorted(set(chosen["snapshot_hash"])), "age_seconds": age,
                        "freshness_policy_seconds": max_age_seconds, "cutoff": cutoff.isoformat()}
    data["notes"] = ["Admitted local bars as known at the cutoff; classification is caller-asserted at admission."]
    return Contract.admit(state, data, source=f"local-admitted:{key[0]}", classification=key[1],
                          as_of=observed.isoformat(), scope=scope, evidence_state="CALLER_SUPPLIED_NOT_VENDOR_VERIFIED",
                          readiness="RESEARCH_ONLY", source_identity=",".join(data["admitted"]["snapshot_hashes"])[:1024])
