"""Black–Scholes pricing, Newton implied vol, and a simple RV–IV carry signal."""

from __future__ import annotations

import math

import numpy as np
from scipy.stats import norm


def black_scholes(
    spot: float,
    strike: float,
    rate: float,
    div: float,
    vol: float,
    ttm: float,
    call: bool = True,
) -> dict[str, float]:
    """European BSM price and first-order Greeks. `ttm` in years, `vol` annualized."""
    if spot <= 0 or strike <= 0 or ttm <= 0 or vol <= 0:
        return {"price": float("nan"), "delta": float("nan"), "gamma": float("nan"), "vega": float("nan"), "theta": float("nan")}
    sqrt_t = math.sqrt(ttm)
    d1 = (math.log(spot / strike) + (rate - div + 0.5 * vol * vol) * ttm) / (vol * sqrt_t)
    d2 = d1 - vol * sqrt_t
    df_r = math.exp(-rate * ttm)
    df_q = math.exp(-div * ttm)
    if call:
        price = df_q * spot * norm.cdf(d1) - df_r * strike * norm.cdf(d2)
        delta = df_q * norm.cdf(d1)
        theta = (
            -df_q * spot * norm.pdf(d1) * vol / (2 * sqrt_t)
            - rate * df_r * strike * norm.cdf(d2)
            + div * df_q * spot * norm.cdf(d1)
        )
    else:
        price = df_r * strike * norm.cdf(-d2) - df_q * spot * norm.cdf(-d1)
        delta = -df_q * norm.cdf(-d1)
        theta = (
            -df_q * spot * norm.pdf(d1) * vol / (2 * sqrt_t)
            + rate * df_r * strike * norm.cdf(-d2)
            - div * df_q * spot * norm.cdf(-d1)
        )
    gamma = df_q * norm.pdf(d1) / (spot * vol * sqrt_t)
    vega = df_q * spot * norm.pdf(d1) * sqrt_t
    return {
        "price": float(price),
        "delta": float(delta),
        "gamma": float(gamma),
        "vega": float(vega / 100.0),  # per vol point
        "theta": float(theta / 365.0),  # per calendar day
        "d1": float(d1),
        "d2": float(d2),
    }


def implied_vol(
    price: float,
    spot: float,
    strike: float,
    rate: float,
    div: float,
    ttm: float,
    call: bool = True,
    tol: float = 1e-6,
    max_iter: int = 50,
) -> float:
    """Newton–Raphson implied volatility. Returns NaN if it fails to converge."""
    if price <= 0 or ttm <= 0:
        return float("nan")
    vol = 0.25
    for _ in range(max_iter):
        pr = black_scholes(spot, strike, rate, div, vol, ttm, call)
        if not np.isfinite(pr["price"]):
            return float("nan")
        diff = pr["price"] - price
        vega_raw = pr["vega"] * 100.0  # undo per-point scaling
        if abs(vega_raw) < 1e-12:
            return float("nan")
        vol -= diff / vega_raw
        if vol <= 1e-6 or vol > 5.0:
            vol = min(max(vol, 1e-4), 5.0)
        if abs(diff) < tol:
            return float(vol)
    return float("nan")


def rv_iv_signal(realized_vol: float, implied_vol: float) -> float:
    """
    Variance risk premium proxy: IV − RV.

    Positive → implied rich vs realized → short vol tilt.
    Negative → implied cheap → long vol tilt.
    """
    if not (np.isfinite(realized_vol) and np.isfinite(implied_vol)):
        return float("nan")
    return float(implied_vol - realized_vol)
