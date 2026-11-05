"""Technical indicators on bar arrays, with explicit warm-up periods.

Every function takes plain float arrays ordered oldest first and returns arrays
of the same length. Positions without enough history are NaN; nothing is
back-filled or extrapolated, so a value at index ``i`` depends only on bars
``0..i`` (no look-ahead).

Conventions (stated because published formulas differ):

* EMA uses ``alpha = 2 / (n + 1)``, seeded with the simple mean of the first
  ``n`` values.
* RSI and ATR use Wilder smoothing (``alpha = 1 / n``), seeded with the simple
  mean of the first ``n`` changes or true ranges.
* True range starts at the second bar, because it needs the previous close.
* Bollinger bands use the population standard deviation (``ddof=0``).
* VWAP is a rolling ``n``-bar volume-weighted typical price
  (``(high + low + close) / 3``); it is not a session-anchored intraday VWAP.
* Realized volatility is the sample standard deviation (``ddof=1``) of ``n``
  log returns, annualized by ``sqrt(periods_per_year)``.
"""
from __future__ import annotations

import numpy as np

__all__ = ["atr", "bollinger", "ema", "macd", "realized_volatility", "rsi", "sma", "true_range", "vwap"]


def _series(values, name: str = "values") -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite")
    return array


def _window(n: int) -> int:
    if isinstance(n, bool) or not isinstance(n, int) or n < 1:
        raise ValueError("window must be a positive integer")
    return n


def sma(values, n: int) -> np.ndarray:
    x, n = _series(values), _window(n)
    out = np.full(x.size, np.nan)
    if x.size >= n:
        sums = np.cumsum(np.concatenate(([0.0], x)))
        out[n - 1:] = (sums[n:] - sums[:-n]) / n
    return out


def _smooth(x: np.ndarray, n: int, alpha: float, start: int = 0) -> np.ndarray:
    """Recursive smoothing of ``x[start:]`` seeded with the mean of its first ``n`` values."""
    out = np.full(x.size, np.nan)
    first = start + n - 1
    if x.size - start < n:
        return out
    out[first] = x[start:first + 1].mean()
    for i in range(first + 1, x.size):
        out[i] = out[i - 1] + alpha * (x[i] - out[i - 1])
    return out


def ema(values, n: int) -> np.ndarray:
    x, n = _series(values), _window(n)
    return _smooth(x, n, 2.0 / (n + 1))


def bollinger(values, n: int = 20, k: float = 2.0) -> dict[str, np.ndarray]:
    x, n = _series(values), _window(n)
    if not np.isfinite(k) or k <= 0:
        raise ValueError("band width must be positive")
    mid = sma(x, n)
    std = np.full(x.size, np.nan)
    for i in range(n - 1, x.size):
        std[i] = x[i - n + 1:i + 1].std(ddof=0)
    return {"upper": mid + k * std, "middle": mid, "lower": mid - k * std}


def rsi(close, n: int = 14) -> np.ndarray:
    x, n = _series(close, "close"), _window(n)
    out = np.full(x.size, np.nan)
    if x.size <= n:
        return out
    change = np.diff(x)
    gains = np.concatenate(([np.nan], np.clip(change, 0, None)))
    losses = np.concatenate(([np.nan], np.clip(-change, 0, None)))
    gain = _smooth(gains, n, 1.0 / n, start=1)
    loss = _smooth(losses, n, 1.0 / n, start=1)
    valid = ~np.isnan(gain)
    with np.errstate(divide="ignore", invalid="ignore"):
        value = 100 - 100 / (1 + gain / loss)
    value = np.where(loss == 0, np.where(gain == 0, 50.0, 100.0), value)
    out[valid] = value[valid]
    return out


def macd(close, fast: int = 12, slow: int = 26, signal: int = 9) -> dict[str, np.ndarray]:
    x = _series(close, "close")
    fast, slow, signal = _window(fast), _window(slow), _window(signal)
    if fast >= slow:
        raise ValueError("fast window must be shorter than slow window")
    line = ema(x, fast) - ema(x, slow)
    signal_line = np.full(x.size, np.nan)
    start = slow - 1
    if x.size - start >= signal:
        signal_line = _smooth(np.where(np.isnan(line), 0.0, line), signal, 2.0 / (signal + 1), start=start)
    return {"macd": line, "signal": signal_line, "histogram": line - signal_line}


def true_range(high, low, close) -> np.ndarray:
    h, lo, c = _series(high, "high"), _series(low, "low"), _series(close, "close")
    if not h.size == lo.size == c.size:
        raise ValueError("high, low and close must have the same length")
    if (h < lo).any():
        raise ValueError("high must not be below low")
    out = np.full(c.size, np.nan)
    if c.size > 1:
        prev = c[:-1]
        out[1:] = np.maximum(h[1:] - lo[1:], np.maximum(np.abs(h[1:] - prev), np.abs(lo[1:] - prev)))
    return out


def atr(high, low, close, n: int = 14) -> np.ndarray:
    n = _window(n)
    tr = true_range(high, low, close)
    return _smooth(np.where(np.isnan(tr), 0.0, tr), n, 1.0 / n, start=1)


def vwap(high, low, close, volume, n: int = 20) -> np.ndarray:
    h, lo, c = _series(high, "high"), _series(low, "low"), _series(close, "close")
    v, n = _series(volume, "volume"), _window(n)
    if not h.size == lo.size == c.size == v.size:
        raise ValueError("bar arrays must have the same length")
    if (v < 0).any():
        raise ValueError("volume must be nonnegative")
    typical = (h + lo + c) / 3
    weighted, total = sma(typical * v, n), sma(v, n)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(total > 0, weighted / total, np.nan)


def realized_volatility(close, n: int = 20, periods_per_year: float = 252.0) -> np.ndarray:
    x, n = _series(close, "close"), _window(n)
    if n < 2:
        raise ValueError("realized volatility needs at least two returns")
    if (x <= 0).any():
        raise ValueError("prices must be positive")
    if not np.isfinite(periods_per_year) or periods_per_year <= 0:
        raise ValueError("periods_per_year must be positive")
    out = np.full(x.size, np.nan)
    returns = np.diff(np.log(x))
    for i in range(n, x.size):
        out[i] = returns[i - n:i].std(ddof=1) * np.sqrt(periods_per_year)
    return out
