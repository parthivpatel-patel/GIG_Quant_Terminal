"""
=============================================================================
RENAISSANCE OPTIONS ENGINE
Professional-grade options mathematics used by quant funds.


MODELS:
  ├── Black-Scholes-Merton   — theoretical fair value for every contract
  ├── Newton-Raphson         — finds implied volatility from market price
  ├── Full Greeks            — Delta/Gamma/Theta/Vega/Rho (1st order)
  │                            Vanna/Volga/Charm/Speed/Zomma (2nd/3rd order)
  ├── Monte Carlo            — 10,000 GBM paths → true probability of profit
  ├── IV Rank & Percentile   — cheap vs expensive vol relative to history
  ├── Max Pain Theory        — where market makers want price to pin
  ├── Unusual Flow Detection — smart money radar on live chain
  └── 7-Strategy Auto-Select — optimal strategy from signal + vol environment


INSTALL:  pip install numpy scipy
=============================================================================
"""


import math
import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq
from datetime import datetime, date, timedelta
import logging


logger = logging.getLogger("OPTIONS")


# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
# RISK_FREE_RATE is no longer a hardcoded constant.
# It is fetched live from ^IRX (13-week T-Bill) via math_engine.get_risk_free_rate().
# Cached 24hr. Falls back to 0.05 if fetch fails.
def _get_rfr() -> float:
    try:
        from math_engine import get_risk_free_rate
        return get_risk_free_rate()
    except Exception:
        import yfinance as _yf
        try:
            fi = _yf.Ticker("^IRX").fast_info
            irx = getattr(fi, "last_price", None)
            if irx and irx > 0:
                return float(irx) / 100.0
        except Exception:
            pass
        return 0.05  # safe fallback

RISK_FREE_RATE   = property(lambda self: _get_rfr())  # type: ignore
TRADING_DAYS     = 252
MC_PATHS         = 10_000      # Monte Carlo paths
MC_STEPS_PER_DAY = 1           # steps per day in simulation
UNUSUAL_FLOW_VOL_MULT = 3.0    # volume > 3× OI = unusual
MIN_UNUSUAL_PREMIUM  = 10_000  # minimum notional to flag ($)




# ═════════════════════════════════════════════════════════════════════════════
# 1. BLACK-SCHOLES-MERTON CORE
# ═════════════════════════════════════════════════════════════════════════════


def bsm_d1_d2(S, K, T, r, sigma):
    """Compute d1 and d2 for BSM.
    S=spot, K=strike, T=time in years, r=risk-free, sigma=IV (decimal)
    """
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return None, None
    sqrtT = math.sqrt(T)
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT
    return d1, d2




def bsm_price(S, K, T, r, sigma, option_type="call"):
    """BSM theoretical price. Returns float."""
    d1, d2 = bsm_d1_d2(S, K, T, r, sigma)
    if d1 is None:
        return max(S - K, 0) if option_type == "call" else max(K - S, 0)
    disc = math.exp(-r * T)
    if option_type == "call":
        return S * norm.cdf(d1) - K * disc * norm.cdf(d2)
    else:
        return K * disc * norm.cdf(-d2) - S * norm.cdf(-d1)




def bsm_fair_value(S, K, T, r, sigma, option_type="call"):
    """Round to 2dp for display."""
    return round(bsm_price(S, K, T, r, sigma, option_type), 2)




# ═════════════════════════════════════════════════════════════════════════════
# 2. NEWTON-RAPHSON IMPLIED VOLATILITY SOLVER
# ═════════════════════════════════════════════════════════════════════════════


def implied_volatility(market_price, S, K, T, r, option_type="call",
                       tol=1e-6, max_iter=100):
    """
    Newton-Raphson IV solver.
    Finds sigma such that BSM(sigma) = market_price.
    Falls back to Brent's method if NR doesn't converge.
    """
    if T <= 0 or market_price <= 0:
        return None


    intrinsic = max(S - K, 0) if option_type == "call" else max(K - S, 0)
    if market_price < intrinsic - 0.01:
        return None


    # ── Newton-Raphson ────────────────────────────────────────────────────
    sigma = 0.3   # initial guess
    for _ in range(max_iter):
        d1, d2 = bsm_d1_d2(S, K, T, r, sigma)
        if d1 is None:
            break
        price = bsm_price(S, K, T, r, sigma, option_type)
        vega  = S * norm.pdf(d1) * math.sqrt(T)
        if abs(vega) < 1e-10:
            break
        sigma_new = sigma - (price - market_price) / vega
        if sigma_new <= 0:
            sigma = sigma / 2
            continue
        if abs(sigma_new - sigma) < tol:
            return round(sigma_new, 6)
        sigma = sigma_new


    # ── Brent fallback ────────────────────────────────────────────────────
    try:
        iv = brentq(
            lambda s: bsm_price(S, K, T, r, s, option_type) - market_price,
            1e-4, 20.0, xtol=tol, maxiter=200
        )
        return round(iv, 6)
    except Exception:
        return None




# ═════════════════════════════════════════════════════════════════════════════
# 3. GREEKS — FIRST ORDER
# ═════════════════════════════════════════════════════════════════════════════


def greeks_first_order(S, K, T, r, sigma, option_type="call"):
    """
    Returns dict with Delta, Gamma, Theta, Vega, Rho.
    Theta in per-day terms (divide annual by 365).
    """
    d1, d2 = bsm_d1_d2(S, K, T, r, sigma)
    if d1 is None:
        return {}


    sqrtT = math.sqrt(T)
    disc  = math.exp(-r * T)
    pdf1  = norm.pdf(d1)


    if option_type == "call":
        delta = norm.cdf(d1)
        rho   = K * T * disc * norm.cdf(d2) / 100
        theta = (- (S * pdf1 * sigma) / (2 * sqrtT)
                 - r * K * disc * norm.cdf(d2)) / 365
    else:
        delta = norm.cdf(d1) - 1
        rho   = -K * T * disc * norm.cdf(-d2) / 100
        theta = (- (S * pdf1 * sigma) / (2 * sqrtT)
                 + r * K * disc * norm.cdf(-d2)) / 365


    gamma = pdf1 / (S * sigma * sqrtT)
    vega  = S * pdf1 * sqrtT / 100   # per 1% move in vol


    return {
        "delta": round(delta, 5),
        "gamma": round(gamma, 6),
        "theta": round(theta, 5),   # $ per day
        "vega":  round(vega,  5),   # $ per 1% IV move
        "rho":   round(rho,   5),
    }




# ═════════════════════════════════════════════════════════════════════════════
# 4. GREEKS — SECOND & THIRD ORDER
# ═════════════════════════════════════════════════════════════════════════════


def greeks_higher_order(S, K, T, r, sigma, option_type="call"):
    """
    Second order: Vanna, Volga (Vomma), Charm, Veta
    Third order:  Speed, Zomma, Color, Ultima
    """
    d1, d2 = bsm_d1_d2(S, K, T, r, sigma)
    if d1 is None:
        return {}


    sqrtT = math.sqrt(T)
    pdf1  = norm.pdf(d1)


    # ── Second order ─────────────────────────────────────────────────────
    # Vanna: sensitivity of Delta to Vol (or Vega to Spot)
    vanna = -pdf1 * d2 / sigma


    # Volga (Vomma): sensitivity of Vega to Vol
    vega_raw = S * pdf1 * sqrtT
    volga    = vega_raw * d1 * d2 / sigma


    # Charm: rate of change of Delta with time (delta decay per day)
    if option_type == "call":
        charm = (-pdf1 * (2*r*T - d2*sigma*sqrtT) / (2*T*sigma*sqrtT)) / 365
    else:
        charm = (-pdf1 * (2*r*T - d2*sigma*sqrtT) / (2*T*sigma*sqrtT)) / 365


    # Veta: rate of change of Vega with time
    veta = vega_raw * (r*d1/(sigma*sqrtT) - (1 + d1*d2)/(2*T)) / 365


    # ── Third order ───────────────────────────────────────────────────────
    gamma_raw = pdf1 / (S * sigma * sqrtT)


    # Speed: rate of change of Gamma with Spot (third derivative of price)
    speed = -gamma_raw / S * (d1 / (sigma * sqrtT) + 1)


    # Zomma: rate of change of Gamma with Vol
    zomma = gamma_raw * (d1 * d2 - 1) / sigma


    # Color: rate of change of Gamma with Time
    color = (-pdf1 / (2 * S * T * sigma * sqrtT) *
             (2*r*T + 1 + d1 * (2*r*T - d2*sigma*sqrtT) / (sigma*sqrtT))) / 365


    # Ultima: third derivative with respect to Vol
    ultima = (-vega_raw / sigma**2 *
              (d1*d2*(1 - d1*d2) + d1**2 + d2**2))


    return {
        # 2nd order
        "vanna":  round(vanna,  6),
        "volga":  round(volga,  6),
        "charm":  round(charm,  7),
        "veta":   round(veta,   6),
        # 3rd order
        "speed":  round(speed,  8),
        "zomma":  round(zomma,  7),
        "color":  round(color,  7),
        "ultima": round(ultima, 6),
    }




def all_greeks(S, K, T, r, sigma, option_type="call"):
    """Combined first + higher order Greeks."""
    g  = greeks_first_order(S, K, T, r, sigma, option_type)
    ho = greeks_higher_order(S, K, T, r, sigma, option_type)
    return {**g, **ho}




# ═════════════════════════════════════════════════════════════════════════════
# 5. MONTE CARLO — 10,000 GBM PATHS → TRUE PROBABILITY OF PROFIT
# ═════════════════════════════════════════════════════════════════════════════


def monte_carlo_analysis(S, K, T, r, sigma, option_type="call", premium=None,
                         n_paths=MC_PATHS, strategy="long"):
    """
    Simulates 10,000 geometric Brownian motion paths.
    Returns:
      pop       — probability of profit
      ev        — expected value per contract
      percentiles — distribution of outcomes at expiry
      paths_sample — 50 sample paths for chart display
    """
    if T <= 0 or sigma <= 0:
        return {"pop": 0.5, "ev": 0, "percentiles": {}}


    n_steps = max(1, int(T * TRADING_DAYS))
    dt      = T / n_steps
    drift   = (r - 0.5 * sigma**2) * dt
    diffuse = sigma * math.sqrt(dt)


    rng      = np.random.default_rng(42)   # reproducible seed
    Z        = rng.standard_normal((n_paths, n_steps))
    log_ret  = drift + diffuse * Z
    log_path = np.cumsum(log_ret, axis=1)
    S_paths  = S * np.exp(np.hstack([np.zeros((n_paths, 1)), log_path]))
    S_final  = S_paths[:, -1]


    # ── Payoff at expiry ──────────────────────────────────────────────────
    if option_type == "call":
        intrinsic = np.maximum(S_final - K, 0)
    else:
        intrinsic = np.maximum(K - S_final, 0)


    pv_intrinsic = intrinsic * math.exp(-r * T)


    if premium is None:
        premium = bsm_price(S, K, T, r, sigma, option_type)


    cost    = premium * 100   # per contract
    payoffs = pv_intrinsic * 100 - cost


    pop = float(np.mean(payoffs > 0))
    ev  = float(np.mean(payoffs))


    # Percentile distribution of outcomes
    pcts = {}
    for p in [5, 10, 25, 50, 75, 90, 95]:
        pcts[f"p{p}"] = round(float(np.percentile(payoffs, p)), 2)


    # 50 sample paths for sparkline (downsampled to 20 points each)
    idx        = rng.choice(n_paths, 50, replace=False)
    sample_raw = S_paths[idx, :]
    step_keep  = max(1, n_steps // 20)
    sample_paths = sample_raw[:, ::step_keep].tolist()


    return {
        "pop":          round(pop * 100, 1),
        "ev":           round(ev, 2),
        "ev_per_dollar": round(ev / (cost + 1e-8), 4),
        "percentiles":  pcts,
        "sample_paths": sample_paths,
        "n_paths":      n_paths,
    }




# ═════════════════════════════════════════════════════════════════════════════
# 6. IV RANK & PERCENTILE
# ═════════════════════════════════════════════════════════════════════════════


def iv_rank_and_percentile(current_iv, iv_history):
    """
    IV Rank:       (current - 52wk_low) / (52wk_high - 52wk_low) × 100
    IV Percentile: % of days in past year where IV was BELOW current
    Both return 0-100.
    """
    if not iv_history or len(iv_history) < 2:
        return {"iv_rank": 50.0, "iv_percentile": 50.0,
                "iv_52wk_high": current_iv, "iv_52wk_low": current_iv,
                "iv_mean": current_iv, "iv_std": 0.0}


    arr  = np.array(iv_history, dtype=float)
    lo   = float(arr.min())
    hi   = float(arr.max())
    mu   = float(arr.mean())
    std  = float(arr.std())


    rank  = (current_iv - lo) / (hi - lo) * 100 if hi > lo else 50.0
    pctile = float(np.mean(arr < current_iv) * 100)


    return {
        "iv_rank":       round(np.clip(rank, 0, 100), 1),
        "iv_percentile": round(pctile, 1),
        "iv_52wk_high":  round(hi, 2),
        "iv_52wk_low":   round(lo, 2),
        "iv_mean":       round(mu, 2),
        "iv_std":        round(std, 2),
        "iv_z_score":    round((current_iv - mu) / (std + 1e-8), 2),
    }




# ═════════════════════════════════════════════════════════════════════════════
# 7. MAX PAIN THEORY
# ═════════════════════════════════════════════════════════════════════════════


def compute_max_pain(chain_calls, chain_puts, spot):
    """
    Max Pain = strike where total dollar loss to option buyers is maximized.
    (= where market makers are most profitable = where they try to pin price)


    chain_calls / chain_puts: DataFrames with columns [strike, openInterest]
    Returns: max_pain_price, pain_by_strike dict, distance from spot
    """
    try:
        import pandas as pd


        # Build unified strike list
        call_oi = {}
        put_oi  = {}


        for _, row in chain_calls.iterrows():
            k = float(row["strike"])
            oi = float(row.get("openInterest", 0) or 0)
            call_oi[k] = oi


        for _, row in chain_puts.iterrows():
            k = float(row["strike"])
            oi = float(row.get("openInterest", 0) or 0)
            put_oi[k] = oi


        strikes = sorted(set(list(call_oi.keys()) + list(put_oi.keys())))
        if not strikes:
            return {"max_pain": spot, "distance_pct": 0.0, "pain_by_strike": {}}


        pain = {}
        for test_price in strikes:
            # Loss to call holders: if expires above test_price
            call_loss = sum(
                max(test_price - k, 0) * oi
                for k, oi in call_oi.items()
            )
            # Loss to put holders: if expires below test_price
            put_loss = sum(
                max(k - test_price, 0) * oi
                for k, oi in put_oi.items()
            )
            pain[test_price] = call_loss + put_loss


        max_pain_strike = min(pain, key=pain.get)
        distance_pct    = (max_pain_strike - spot) / spot * 100


        # Return top 10 strikes by pain for chart
        sorted_pain = dict(sorted(pain.items(), key=lambda x: x[0]))


        return {
            "max_pain":         round(max_pain_strike, 2),
            "distance_pct":     round(distance_pct, 2),
            "pain_by_strike":   {str(k): round(v, 0) for k, v in sorted_pain.items()},
            "interpretation":   (
                "PINNING ↓" if distance_pct < -2 else
                "PINNING ↑" if distance_pct > 2  else
                "AT MAX PAIN"
            ),
        }
    except Exception as e:
        logger.warning(f"Max pain error: {e}")
        return {"max_pain": spot, "distance_pct": 0.0, "pain_by_strike": {}}




# ═════════════════════════════════════════════════════════════════════════════
# 8. UNUSUAL FLOW DETECTION — smart money radar
# ═════════════════════════════════════════════════════════════════════════════


def detect_unusual_flow(chain_calls, chain_puts, spot, current_iv):
    """
    Flags unusual options activity = potential smart money positioning.


    Criteria:
      1. Volume > UNUSUAL_FLOW_VOL_MULT × Open Interest (fresh positions)
      2. Premium > MIN_UNUSUAL_PREMIUM (not retail noise)
      3. OTM preferred (directional, not hedging)
      4. Short-dated (< 30 DTE) — urgent conviction
      5. IV of contract >> current IV (buyer paying up)
    """
    unusual = []


    def scan_chain(df, opt_type):
        if df is None or df.empty:
            return
        try:
            for _, row in df.iterrows():
                vol    = float(row.get("volume", 0) or 0)
                oi     = float(row.get("openInterest", 0) or 0)
                strike = float(row.get("strike", 0))
                last   = float(row.get("lastPrice", 0) or 0)
                bid    = float(row.get("bid", 0) or 0)
                ask    = float(row.get("ask", 0) or 0)
                iv_c   = float(row.get("impliedVolatility", 0) or 0) * 100
                itm    = bool(row.get("inTheMoney", False))


                if vol < 100 or last < 0.05:
                    continue


                notional = vol * last * 100
                vol_to_oi = vol / (oi + 1) if oi > 0 else vol


                score = 0
                flags = []


                # Fresh position (volume >> OI)
                if vol_to_oi >= UNUSUAL_FLOW_VOL_MULT:
                    score += 30
                    flags.append(f"VOL/OI={vol_to_oi:.1f}x")


                # Large premium
                if notional >= MIN_UNUSUAL_PREMIUM:
                    score += 25
                    flags.append(f"${notional/1000:.0f}K notional")


                # OTM preference = directional bet
                if not itm:
                    score += 15
                    flags.append("OTM")


                # IV premium (paying up)
                if current_iv > 0 and iv_c > current_iv * 1.15:
                    score += 20
                    flags.append(f"IV premium {iv_c:.0f}% vs avg {current_iv:.0f}%")


                # Big volume absolutely
                if vol > 10000:
                    score += 10
                    flags.append(f"VOL={vol:,.0f}")


                if score >= 50:
                    direction = "BULLISH" if opt_type == "call" else "BEARISH"
                    unusual.append({
                        "type":      opt_type.upper(),
                        "direction": direction,
                        "strike":    round(strike, 2),
                        "volume":    int(vol),
                        "oi":        int(oi),
                        "premium":   round(last, 2),
                        "notional":  round(notional, 0),
                        "vol_oi":    round(vol_to_oi, 1),
                        "iv":        round(iv_c, 1),
                        "flags":     flags,
                        "score":     score,
                    })
        except Exception as e:
            logger.debug(f"Flow scan error: {e}")


    scan_chain(chain_calls, "call")
    scan_chain(chain_puts,  "put")


    # Sort by score desc, return top 5
    unusual.sort(key=lambda x: x["score"], reverse=True)
    return unusual[:5]




# ═════════════════════════════════════════════════════════════════════════════
# 9. STRATEGY AUTO-SELECTION ENGINE — 7 strategies
# ═════════════════════════════════════════════════════════════════════════════


STRATEGIES = {
    "BUY_CALL":      "BUY CALL",
    "BUY_PUT":       "BUY PUT",
    "BULL_PUT_SPREAD": "BULL PUT SPREAD",
    "BEAR_CALL_SPREAD": "BEAR CALL SPREAD",
    "LONG_STRADDLE": "LONG STRADDLE",
    "IRON_CONDOR":   "IRON CONDOR",
    "COVERED_CALL":  "COVERED CALL",
}


def select_strategy(direction, iv_rank, iv_percentile, conviction,
                    hurst, expected_move_pct, days_to_earnings=999):
    """
    Auto-select optimal strategy from 7 options based on:
      direction     — BUY / SHORT / HOLD
      iv_rank       — 0-100 (cheap <35, expensive >65)
      iv_percentile — 0-100
      conviction    — 0-1
      hurst         — 0-1 (trending >0.55, mean-rev <0.45)
      expected_move_pct — expected % price move
      days_to_earnings  — days until next earnings


    Returns: {strategy, rationale, confidence}
    """
    bull    = direction == "BUY"
    bear    = direction == "SHORT"
    neutral = direction == "HOLD"
    cheap   = iv_rank < 35
    rich    = iv_rank > 65
    trending = hurst > 0.55
    mean_rev = hurst < 0.45
    high_conv = conviction > 0.6
    near_earnings = days_to_earnings <= 5


    strategy    = None
    rationale   = []
    confidence  = 0.5


    # ── Near earnings: volatility play ─────────────────────────────────
    if near_earnings and cheap:
        strategy  = "LONG_STRADDLE"
        rationale = [
            f"Earnings in {days_to_earnings}d",
            f"IV cheap (IVR={iv_rank:.0f}) → buy volatility",
            "Straddle profits from big move in either direction",
        ]
        confidence = 0.72


    # ── High IV + neutral → Iron Condor ────────────────────────────────
    elif neutral and rich and not near_earnings:
        strategy  = "IRON_CONDOR"
        rationale = [
            f"No directional edge (HOLD signal)",
            f"IV rich (IVR={iv_rank:.0f}) → sell premium both sides",
            "Iron Condor profits if price stays range-bound",
        ]
        confidence = 0.65


    # ── Bullish + high IV → Bull Put Spread (better risk/reward than BUY CALL) ──
    elif bull and rich and not trending:
        strategy  = "BULL_PUT_SPREAD"
        rationale = [
            f"Bullish signal (composite conviction {conviction:.0f}%)",
            f"IV expensive (IVR={iv_rank:.0f}) → sell premium not buy it",
            "Bull Put Spread: collect credit, profit if stock stays above sell strike",
        ]
        confidence = 0.70


    # ── Bearish + high IV → Bear Call Spread ───────────────────────────
    elif bear and rich and not trending:
        strategy  = "BEAR_CALL_SPREAD"
        rationale = [
            f"Bearish signal",
            f"IV expensive (IVR={iv_rank:.0f}) → sell calls",
            "Bear Call Spread: collect credit, profit if stock stays below sell strike",
        ]
        confidence = 0.68


    # ── Strong bullish + trending + cheap IV → Buy Call ─────────────────
    elif bull and (cheap or trending) and high_conv:
        strategy  = "BUY_CALL"
        rationale = [
            f"Strong bullish signal (conviction {conviction*100:.0f}%)",
            f"Hurst={hurst:.3f} → {'trending momentum' if trending else 'mean reversion'}",
            f"IV {'cheap' if cheap else 'moderate'} (IVR={iv_rank:.0f}) → buying vol makes sense",
        ]
        confidence = min(0.85, 0.60 + conviction * 0.4)


    # ── Strong bearish + cheap IV → Buy Put ─────────────────────────────
    elif bear and (cheap or trending) and high_conv:
        strategy  = "BUY_PUT"
        rationale = [
            f"Strong bearish signal (conviction {conviction*100:.0f}%)",
            f"IV {'cheap' if cheap else 'moderate'} (IVR={iv_rank:.0f}) → buying vol makes sense",
        ]
        confidence = min(0.80, 0.58 + conviction * 0.35)


    # ── Bullish + high IV + trending → Covered Call (if long stock) ──────
    elif bull and rich and trending:
        strategy  = "COVERED_CALL"
        rationale = [
            f"Bullish but IV rich (IVR={iv_rank:.0f})",
            "Covered Call: enhance yield by selling OTM call against stock position",
            "Keeps upside to sell strike, collects premium",
        ]
        confidence = 0.62


    # ── Default ──────────────────────────────────────────────────────────
    else:
        strategy  = "BUY_CALL" if bull else "BUY_PUT" if bear else "LONG_STRADDLE"
        rationale = [f"Default selection for {direction} signal, IVR={iv_rank:.0f}"]
        confidence = 0.50


    return {
        "strategy":   STRATEGIES.get(strategy, strategy),
        "strategy_id":strategy,
        "rationale":  rationale,
        "confidence": round(confidence, 3),
        "iv_environment": "CHEAP" if cheap else "RICH" if rich else "MODERATE",
        "regime":     "TRENDING" if trending else "MEAN_REVERTING" if mean_rev else "RANDOM",
    }




# ═════════════════════════════════════════════════════════════════════════════
# 10. FULL CONTRACT ANALYSIS — puts it all together for one contract
# ═════════════════════════════════════════════════════════════════════════════


def analyze_contract(S, K, T_days, option_type, market_price,
                     r=None, iv_history=None):
    """
    Full analysis for a single options contract.
    Returns all pricing, Greeks, IV, and Monte Carlo data.
    """
    if r is None: r = _get_rfr()
    T    = max(T_days / TRADING_DAYS, 1/TRADING_DAYS)
    iv_h = iv_history or []


    # 1. Newton-Raphson IV from market price
    iv = implied_volatility(market_price, S, K, T, r, option_type)
    if iv is None or iv <= 0:
        iv = 0.25   # fallback to 25% if unsolvable


    # 2. BSM fair value
    fair_val = bsm_fair_value(S, K, T, r, iv, option_type)
    edge     = round(fair_val - market_price, 2)  # +edge = we're buying cheap


    # 3. All Greeks
    g1 = greeks_first_order(S, K, T, r, iv, option_type)
    g2 = greeks_higher_order(S, K, T, r, iv, option_type)


    # 4. Monte Carlo
    mc = monte_carlo_analysis(S, K, T, r, iv, option_type, market_price)


    # 5. IV Rank & Percentile
    current_hv = iv * 100  # use IV as proxy if no history
    iv_stats   = iv_rank_and_percentile(current_hv, [v * 100 for v in iv_h] if iv_h else [current_hv])


    # 6. Breakeven
    if option_type == "call":
        be = round(K + market_price, 2)
        max_profit = "UNLIMITED"
        max_loss   = round(market_price * 100, 2)
    else:
        be = round(K - market_price, 2)
        max_profit = round((K - market_price) * 100, 2)
        max_loss   = round(market_price * 100, 2)


    return {
        # Core
        "iv":          round(iv * 100, 2),
        "fair_value":  fair_val,
        "market_price":round(market_price, 2),
        "edge":        edge,
        "breakeven":   be,
        "max_profit":  max_profit,
        "max_loss":    max_loss,
        # Monte Carlo
        "pop":         mc["pop"],
        "ev":          mc["ev"],
        "ev_per_dollar": mc["ev_per_dollar"],
        "mc_percentiles": mc["percentiles"],
        "sample_paths":   mc.get("sample_paths", []),
        # Greeks 1st order
        "delta":  g1.get("delta"),
        "gamma":  g1.get("gamma"),
        "theta":  g1.get("theta"),
        "vega":   g1.get("vega"),
        "rho":    g1.get("rho"),
        # Greeks 2nd order
        "vanna":  g2.get("vanna"),
        "volga":  g2.get("volga"),
        "charm":  g2.get("charm"),
        "veta":   g2.get("veta"),
        # Greeks 3rd order
        "speed":  g2.get("speed"),
        "zomma":  g2.get("zomma"),
        "color":  g2.get("color"),
        "ultima": g2.get("ultima"),
        # IV analytics
        "iv_rank":       iv_stats["iv_rank"],
        "iv_percentile": iv_stats["iv_percentile"],
        "iv_52wk_high":  iv_stats["iv_52wk_high"],
        "iv_52wk_low":   iv_stats["iv_52wk_low"],
        "iv_mean":       iv_stats["iv_mean"],
        "iv_z_score":    iv_stats["iv_z_score"],
    }




# ═════════════════════════════════════════════════════════════════════════════
# 11. MASTER PIPELINE — called by the server for each symbol
# ═════════════════════════════════════════════════════════════════════════════


def run_full_options_analysis(symbol, spot, direction, signals,
                               chain_calls, chain_puts,
                               expiry_str, exp_nice, dte,
                               hv_history=None):
    """
    Master function: given a live chain + signals, runs every model
    and returns a complete options analysis object.
    """
    hv20   = signals.get("hv20", 25) / 100
    garch  = signals.get("garch_vol", 25) / 100
    conv   = signals.get("conviction", 0.3)
    ivr    = signals.get("ivr", 50)
    hurst  = signals.get("hurst", 0.5)
    bull   = direction == "BUY"
    T      = max(dte, 1) / TRADING_DAYS
    r      = _get_rfr()


    # ── Best IV estimate from GARCH + HV ────────────────────────────────
    base_iv = max(hv20 * 1.08, garch * 1.05, 0.05)


    # ── 1. Strategy selection ────────────────────────────────────────────
    strat = select_strategy(
        direction       = direction,
        iv_rank         = ivr,
        iv_percentile   = ivr,   # using IVR as proxy
        conviction      = conv,
        hurst           = hurst,
        expected_move_pct = signals.get("atr_pct", 2.0),
    )


    # ── 2. Max Pain ──────────────────────────────────────────────────────
    max_pain_data = {}
    if chain_calls is not None and chain_puts is not None:
        max_pain_data = compute_max_pain(chain_calls, chain_puts, spot)


    # ── 3. Unusual flow ──────────────────────────────────────────────────
    flow = []
    if chain_calls is not None and chain_puts is not None:
        flow = detect_unusual_flow(chain_calls, chain_puts, spot, hv20 * 100)


    # ── 4. Pick contract from live chain ────────────────────────────────
    import math as _math


    def pick_best_row(df, is_call):
        if df is None or df.empty:
            return None
        df = df.copy()
        target = spot * (1.01 if is_call else 0.99)
        df["dist"] = (df["strike"] - target).abs()
        if "bid" in df.columns and "ask" in df.columns:
            mid = (df["bid"].astype(float) + df["ask"].astype(float)) / 2.0
            spread = (df["ask"].astype(float) - df["bid"].astype(float)).clip(lower=0)
            df["spread_pct"] = np.where(mid > 0, spread / mid * 100.0, 999.0)
        else:
            df["spread_pct"] = 999.0
        liq = df
        if "openInterest" in df.columns:
            liq = liq[liq["openInterest"].astype(float) >= 100]
        if "volume" in df.columns:
            liq = liq[liq["volume"].astype(float) >= 50]
        liq = liq[liq["spread_pct"] <= 12.0]
        candidates = liq if not liq.empty else (df[df["volume"] > 0] if "volume" in df.columns else df)
        if candidates.empty:
            candidates = df
        return candidates.sort_values("dist").iloc[0]


    row = pick_best_row(chain_calls if bull else chain_puts, bull)


    if row is None:
        # Pure BSM fallback
        K       = round(spot * (1.01 if bull else 0.99), 2)
        iv_used = base_iv
        prem    = bsm_price(spot, K, T, r, iv_used, "call" if bull else "put")
        prem    = max(round(prem, 2), 0.01)
        market_price = prem
    else:
        K            = float(row["strike"])
        market_price = float(row.get("lastPrice") or row.get("ask") or 0.5)
        iv_used      = base_iv


    opt_type = "call" if bull else "put"


    # ── 5. Full contract analysis ────────────────────────────────────────
    contract = analyze_contract(
        S            = spot,
        K            = K,
        T_days       = dte,
        option_type  = opt_type,
        market_price = market_price,
        r            = r,
        iv_history   = hv_history,
    )


    # ── 6. Build simple_signal for display ───────────────────────────────
    row_iv  = float((row.get("impliedVolatility", base_iv) if row is not None else base_iv))
    row_iv  = row_iv * 100 if row_iv < 5 else row_iv   # normalise
    row_bid = float((row.get("bid", 0) or 0)) if row is not None else round(market_price * 0.97, 2)
    row_ask = float((row.get("ask", 0) or 0)) if row is not None else round(market_price * 1.03, 2)
    row_vol = int((row.get("volume", 0) or 0)) if row is not None else 0
    row_oi  = int((row.get("openInterest", 0) or 0)) if row is not None else 0
    row_itm = bool((row.get("inTheMoney", False))) if row is not None else False


    if opt_type == "call":
        be = round(K + market_price, 2)
        mp = "UNLIMITED"
        ml = round(market_price * 100, 2)
    else:
        be = round(K - market_price, 2)
        mp = round((K - market_price) * 100, 2)
        ml = round(market_price * 100, 2)


    simple_signal = {
        "action": f"BUY {opt_type.upper()}",
        "legs":   [{
            "action":  "BUY",
            "type":    opt_type.upper(),
            "strike":  round(K, 2),
            "expiry":  exp_nice,
            "premium": round(market_price, 2),
            "cost":    ml,
            "volume":  row_vol,
            "oi":      row_oi,
            "iv":      round(row_iv, 1),
            "bid":     round(row_bid, 2),
            "ask":     round(row_ask, 2),
            "itm":     row_itm,
        }],
        "breakeven":  be,
        "max_profit": mp,
        "max_loss":   ml,
    }

    execution_quality = {
        "spread_pct": round(float((row_ask - row_bid) / max((row_ask + row_bid) / 2.0, 1e-6) * 100), 2),
        "volume": row_vol,
        "open_interest": row_oi,
        "liquidity_ok": bool(row_vol >= 50 and row_oi >= 100 and row_ask > row_bid),
    }


    # ── 7. Assemble final result ─────────────────────────────────────────
    return {
        # Trade signal
        "strategy":      strat["strategy"],
        "strategy_id":   strat["strategy_id"],
        "direction":     "BULLISH" if bull else "BEARISH",
        "iv_label":      f"IVR {round(ivr,0):.0f} — {strat['iv_environment']}",
        "ivr":           round(ivr, 1),
        "dte":           dte,
        "expiry":        expiry_str,
        "expiry_nice":   exp_nice,
        "recommended":   conv > 0.5,
        "source":        "LIVE CHAIN" if row is not None else "BSM COMPUTED",


        # Pricing
        "iv":            contract["iv"],
        "fair_value":    contract["fair_value"],
        "edge":          contract["edge"],
        "hv":            round(hv20 * 100, 1),
        "garch_vol":     round(garch * 100, 1),
        "vol_premium":   round((contract["iv"] / 100 - hv20) * 100, 1),
        "exp_move":      round(spot * base_iv * _math.sqrt(T), 2),
        "exp_move_pct":  round(base_iv * _math.sqrt(T) * 100, 1),
        "max_profit":    mp,
        "max_loss":      ml,
        "breakeven":     be,


        # Monte Carlo
        "pop":           contract["pop"],
        "ev":            contract["ev"],
        "ev_per_dollar": contract["ev_per_dollar"],
        "mc_percentiles":contract["mc_percentiles"],
        "mc_paths":      MC_PATHS,


        # Greeks — all orders
        "greeks": {
            "delta":  contract["delta"],
            "gamma":  contract["gamma"],
            "theta":  contract["theta"],
            "vega":   contract["vega"],
            "rho":    contract["rho"],
            "vanna":  contract["vanna"],
            "volga":  contract["volga"],
            "charm":  contract["charm"],
            "veta":   contract["veta"],
            "speed":  contract["speed"],
            "zomma":  contract["zomma"],
            "color":  contract["color"],
            "ultima": contract["ultima"],
        },


        # IV analytics
        "iv_rank":       contract["iv_rank"],
        "iv_percentile": contract["iv_percentile"],
        "iv_52wk_high":  contract["iv_52wk_high"],
        "iv_52wk_low":   contract["iv_52wk_low"],
        "iv_z_score":    contract["iv_z_score"],


        # Max Pain
        "max_pain":         max_pain_data.get("max_pain", spot),
        "max_pain_dist":    max_pain_data.get("distance_pct", 0.0),
        "max_pain_interp":  max_pain_data.get("interpretation", ""),
        "pain_by_strike":   max_pain_data.get("pain_by_strike", {}),


        # Unusual flow
        "unusual_flow":     flow,
        "smart_money_bull": sum(1 for f in flow if f["direction"] == "BULLISH"),
        "smart_money_bear": sum(1 for f in flow if f["direction"] == "BEARISH"),


        # Strategy selection
        "strategy_rationale":  strat["rationale"],
        "strategy_confidence": strat["confidence"],
        "iv_environment":      strat["iv_environment"],
        "regime":              strat["regime"],
        "execution_quality":   execution_quality,


        # Simple signal for React display
        "simple_signal": simple_signal,
        "legs":          [],
    }