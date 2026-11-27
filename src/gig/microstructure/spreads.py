"""Spread estimates from caller-supplied prices.

Roll (1984) recovers a spread from the autocovariance of price changes.
Corwin and Schultz (2012) recover one from high-low ranges. Both are model
estimates. Neither reads a quote, a book or an order.
"""
from __future__ import annotations

import math

import numpy as np

from gig import checks

MAX_PRICES = 5000
# Corwin-Schultz denominator, 3 - 2*sqrt(2).
_HIGH_LOW = 3-2*math.sqrt(2)


def _prices(value, name: str) -> np.ndarray:
    prices = checks.array(value, name, dimensions=1, maximum=MAX_PRICES)
    if not 2 <= prices.shape[0] <= MAX_PRICES or np.any(prices <= 0):
        raise ValueError(f"{name} must hold 2 to 5000 positive values")
    return prices


def roll_spread(prices) -> dict:
    """Implied spread from the serial covariance of price changes."""
    series = _prices(prices, "prices")
    if series.shape[0] < 10:
        raise ValueError("prices must hold 10 to 5000 values")
    changes = np.diff(series)
    covariance = float(np.cov(changes[:-1], changes[1:], ddof=1)[0, 1])
    scale = float(changes @ changes)/int(changes.shape[0])
    identified = covariance < -1e-10*max(scale, 1e-18)
    spread = 2*math.sqrt(-covariance) if identified else None
    return {"method": "ROLL_1984", "observations": int(series.shape[0]), "price_changes": int(changes.shape[0]),
            "autocovariance": covariance, "covariance_divisor": "N_MINUS_ONE",
            "spread": spread, "spread_state": "IDENTIFIED" if identified else "NOT_IDENTIFIED_NONNEGATIVE_AUTOCORRELATION",
            "relative_spread": None if spread is None else spread/float(series.mean()),
            "estimator_limits": "CONSTANT_SPREAD_IID_TRADE_SIGNS_NO_ADVERSE_SELECTION_OR_OVERNIGHT_JUMPS",
            "classification": "MODEL_ESTIMATE"}


def corwin_schultz(high, low) -> dict:
    """High-low spread for every pair of consecutive bars."""
    highs, lows = _prices(high, "high"), _prices(low, "low")
    if highs.shape != lows.shape or highs.shape[0] < 2:
        raise ValueError("high and low must be the same series of at least 2 bars")
    if np.any(highs < lows):
        raise ValueError("each high must be at least its low")
    one_day = np.log(highs/lows)**2
    beta = one_day[:-1]+one_day[1:]
    gamma = np.log(np.maximum(highs[:-1], highs[1:])/np.minimum(lows[:-1], lows[1:]))**2
    alpha = (np.sqrt(2*beta)-np.sqrt(beta))/_HIGH_LOW-np.sqrt(gamma/_HIGH_LOW)
    raw = 2*(np.exp(alpha)-1)/(1+np.exp(alpha))
    negative = int(np.sum(raw < 0))
    return {"method": "CORWIN_SCHULTZ_2012", "bars": int(highs.shape[0]), "windows": int(raw.shape[0]),
            "windowing": "CONSECUTIVE_OVERLAPPING_PAIRS",
            "mean_spread": float(raw.mean()), "mean_spread_floored_at_zero": float(np.maximum(raw, 0).mean()),
            "median_spread": float(np.median(raw)), "negative_estimates": negative,
            "negative_share": negative/int(raw.shape[0]),
            "estimator_limits": "BROWNIAN_HIGH_LOW_RANGE_OVERLAPPING_WINDOWS_ARE_NOT_INDEPENDENT_SAMPLES",
            "classification": "MODEL_ESTIMATE"}
