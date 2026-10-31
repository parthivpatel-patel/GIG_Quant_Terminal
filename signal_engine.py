"""
=============================================================================
SIGNAL ENGINE — Pine Script v4.1 algorithms ported to Python
All models from: Option Signal Strategy – Quant v4.1

MODELS:
  ├── V-Reversal Detection          — swing flush + RSI + EMA slope flip
  ├── Kalman Filter Probability     — kalmanGain=0.25 smoothed bull/bear prob
  ├── Bayesian Probability Update   — bayesDecay=0.95 posterior
  ├── Confidence Score (1–100%)     — prediction confidence + dir confidence
  ├── Chandelier Trailing Stop      — 3.5x ATR, tighten at 2.5R, BE at 1.25R
  ├── Hybrid Forecast               — LR slope + momentum + ATR regime
  ├── Range Forecast (1σ/2σ)        — vol-boosted sigma horizons
  ├── Squeeze Detection             — ATR compression < 0.65 threshold
  ├── Breakout Engine               — range high/low + volume confirmation
  ├── Sector Correlation Detect     — rolling correlation vs sector ETFs
  ├── Ensemble Direction Score      — trend(40%)+regime(25%)+weekly(20%)+inst(15%)
  └── 100%+ Return Filter           — targets explosive setups only
=============================================================================
"""

import numpy as np
import math
import logging
from datetime import datetime, timedelta

logger = logging.getLogger("SIGNAL")

# ─── Constants matching Pine Script inputs ────────────────────────────────────
FAST_LEN       = 9        # fastLen
SLOW_LEN       = 21       # slowLen
ADX_LEN        = 14       # adxLen
ADX_SMOOTH     = 14       # adxSmooth
MIN_ADX_STRONG = 17.5     # minAdxStrong
ATR_STOP_MULT  = 2.8      # atrStopMult
ATR_TP_RR      = 3.5      # atrTpRR
KALMAN_GAIN    = 0.25     # kalmanGain
BAYES_DECAY    = 0.95     # bayesDecay
V_RSI_LEN      = 14       # vRsiLen
V_RSI_LOW      = 40       # vRsiLow
V_RSI_HIGH     = 60       # vRsiHigh
V_LOOKBACK     = 20       # vLookback
INST_VOL_FACTOR   = 1.4   # instVolFactor
INST_ATR_FACTOR   = 1.15  # instAtrFactor
INST_VOL_LOOKBACK = 60    # instVolLookback
BR_LOOKBACK    = 20       # brLookback
SQUEEZE_LOOK   = 50       # squeezeLookback
SQUEEZE_THRESH = 0.65     # squeezeThresh
CHAND_MULT     = 3.5      # chandMult
TIGHT_MULT     = 1.8      # tightMult
BE_AT_R        = 1.25     # beAtR
LOCK_AT_R      = 2.5      # lockAtR
FORECAST_BARS  = 24       # forecastBars
FORECAST_SIGMA = 1.0      # forecastSigma
MIN_ATR_PCT    = 0.12     # minAtrPct
MAX_ATR_PCT    = 4.5      # maxAtrPct
OTM_OFFSET_PCT = 1.5      # otmOffsetPct
EXPIRY_DAYS    = 5        # expiryDays
REGIME_FAST    = 21       # regimeMaFastLen
REGIME_SLOW    = 50       # regimeMaSlowLen
REGIME_LONG    = 200      # regimeMaLongLen

# Sector ETF domain map for detection
SECTOR_ETF_MAP = {
    "XLK": "Technology",
    "XLF": "Financials",
    "XLE": "Energy",
    "XLB": "Materials",
    "XLV": "Healthcare",
    "XLI": "Industrials",
    "XLC": "Communication",
    "XLY": "Consumer Disc",
    "GLD": "Gold/Metals",
    "SLV": "Silver/Metals",
    "DBC": "Commodities",
}


# ═════════════════════════════════════════════════════════════════════════════
# HELPER MATH
# ═════════════════════════════════════════════════════════════════════════════

def _ema(prices, period):
    """Exponential moving average."""
    arr = np.array(prices, dtype=float)
    if len(arr) < period:
        return arr
    k   = 2.0 / (period + 1)
    out = np.empty(len(arr))
    out[0] = arr[0]
    for i in range(1, len(arr)):
        out[i] = arr[i] * k + out[i-1] * (1 - k)
    return out


def _sma(prices, period):
    arr = np.array(prices, dtype=float)
    if len(arr) < period:
        return np.full(len(arr), arr[-1])
    result = np.convolve(arr, np.ones(period)/period, mode='full')[:len(arr)]
    return result


def _atr(high, low, close, period=14):
    h, l, c = np.array(high, dtype=float), np.array(low, dtype=float), np.array(close, dtype=float)
    if len(h) < 2:
        return float(h[-1] - l[-1]) if len(h) > 0 else 1.0
    tr = np.maximum(h[1:]-l[1:], np.maximum(np.abs(h[1:]-c[:-1]), np.abs(l[1:]-c[:-1])))
    if len(tr) < period:
        return float(np.mean(tr)) if len(tr) > 0 else 1.0
    return float(np.mean(tr[-period:]))


def _rsi(close, period=14):
    arr = np.array(close, dtype=float)
    if len(arr) < period + 1:
        return 50.0
    d = np.diff(arr)
    gains  = np.where(d > 0, d, 0.0)
    losses = np.where(d < 0, -d, 0.0)
    ag = np.mean(gains[-period:])
    al = np.mean(losses[-period:])
    if al == 0:
        return 100.0
    return float(100 - 100 / (1 + ag / al))


def _adx(high, low, close, period=14, smooth=14):
    """Full ADX calculation with +DI and -DI."""
    h = np.array(high,  dtype=float)
    l = np.array(low,   dtype=float)
    c = np.array(close, dtype=float)
    if len(c) < period + smooth + 5:
        return 25.0, 25.0, 25.0

    move_up   = h[1:] - h[:-1]
    move_down = l[:-1] - l[1:]
    pdm = np.where((move_up > move_down) & (move_up > 0), move_up, 0.0)
    ndm = np.where((move_down > move_up) & (move_down > 0), move_down, 0.0)
    tr  = np.maximum(h[1:]-l[1:], np.maximum(np.abs(h[1:]-c[:-1]), np.abs(l[1:]-c[:-1])))

    atr14 = np.convolve(tr,  np.ones(period)/period, 'valid')
    pdi14 = np.convolve(pdm, np.ones(period)/period, 'valid')
    ndi14 = np.convolve(ndm, np.ones(period)/period, 'valid')

    n = min(len(atr14), len(pdi14), len(ndi14))
    pdi = pdi14[-n:] / (atr14[-n:] + 1e-8) * 100
    ndi = ndi14[-n:] / (atr14[-n:] + 1e-8) * 100
    dx  = np.abs(pdi - ndi) / (pdi + ndi + 1e-8) * 100
    adx_val = float(np.mean(dx[-smooth:])) if len(dx) >= smooth else float(np.mean(dx))

    return float(pdi[-1]), float(ndi[-1]), adx_val


def _normal_cdf(x):
    """Fast Abramowitz & Stegun approximation (same as Pine Script)."""
    t    = 1.0 / (1.0 + 0.2316419 * abs(x))
    d    = 0.3989423 * math.exp(-0.5 * x * x)
    prob = d * t * (0.3193815 + t * (-0.3565638 + t * (1.781478 + t * (-1.821256 + t * 1.330274))))
    return (1.0 - prob) if x > 0 else prob


# ═════════════════════════════════════════════════════════════════════════════
# 1. V-REVERSAL DETECTION
# ═════════════════════════════════════════════════════════════════════════════

def detect_v_reversal(df, atr_val=None):
    """
    Detects V-reversal entries from Pine Script V-Reversal Engine.

    Pine rules:
      flushDown  = low <= swingLow + atr*0.10  OR  rsi < vRsiLow+5
      flushUp    = high >= swingHigh - atr*0.10  OR  rsi > vRsiHigh-5
      vBullRebound = close > open AND close > close[1]
      vBearRebound = close < open AND close < close[1]
      emaSlopeFlipUp   = ema_slope_now > 0 AND ema_slope_prev <= 0
      emaSlopeFlipDown = ema_slope_now < 0 AND ema_slope_prev >= 0
      reclaimFastEmaLong  = close > fastEma
      reclaimFastEmaShort = close < fastEma
    """
    if df is None or len(df) < V_LOOKBACK + 5:
        return False, False, {}

    close  = df["Close"].values.astype(float)
    high   = df["High"].values.astype(float)
    low    = df["Low"].values.astype(float)
    open_  = df["Open"].values.astype(float)

    if atr_val is None:
        atr_val = _atr(high, low, close)

    rsi_val   = _rsi(close, V_RSI_LEN)
    fast_ema  = _ema(close, FAST_LEN)

    swing_low  = float(np.min(low[-V_LOOKBACK-1:-1]))   # [1] shifted = exclude last bar
    swing_high = float(np.max(high[-V_LOOKBACK-1:-1]))

    # Flush conditions
    flush_down = (low[-1] <= swing_low + atr_val * 0.10) or (rsi_val < V_RSI_LOW + 5)
    flush_up   = (high[-1] >= swing_high - atr_val * 0.10) or (rsi_val > V_RSI_HIGH - 5)

    # Rebound candle
    v_bull_rebound = (close[-1] > open_[-1]) and (close[-1] > close[-2])
    v_bear_rebound = (close[-1] < open_[-1]) and (close[-1] < close[-2])

    # EMA slope flip
    ema_slope_now  = fast_ema[-1] - fast_ema[-2]
    ema_slope_prev = fast_ema[-2] - fast_ema[-3] if len(fast_ema) >= 3 else 0.0
    slope_flip_up   = (ema_slope_now > 0) and (ema_slope_prev <= 0)
    slope_flip_down = (ema_slope_now < 0) and (ema_slope_prev >= 0)

    # Reclaim EMA
    reclaim_long  = close[-1] > fast_ema[-1]
    reclaim_short = close[-1] < fast_ema[-1]

    v_call = flush_down and v_bull_rebound and reclaim_long  and slope_flip_up
    v_put  = flush_up   and v_bear_rebound and reclaim_short and slope_flip_down

    metadata = {
        "flush_down":      flush_down,
        "flush_up":        flush_up,
        "slope_flip_up":   slope_flip_up,
        "slope_flip_down": slope_flip_down,
        "rsi_at_flush":    rsi_val,
        "swing_low":       round(swing_low, 2),
        "swing_high":      round(swing_high, 2),
        "reclaim_long":    reclaim_long,
        "reclaim_short":   reclaim_short,
    }
    return v_call, v_put, metadata


# ═════════════════════════════════════════════════════════════════════════════
# 2. ENSEMBLE DIRECTION SCORE (Pine Script section 11)
# ═════════════════════════════════════════════════════════════════════════════

def compute_ensemble_direction(signals, macro_bias=0.0, sector_bias=0.0, sector_strength="Neutral"):
    """
    Pine Script ensemble:
      baseDirScore = 0.4*trendDir + 0.25*regimeDir + 0.2*weeklyDir + 0.15*instDir
      adjDirScore  = baseDirScore + 0.15*macroBias + 0.15*sectorBias + sectorBoost
    """
    trend_dir  = 1.0 if signals.get("direction") == "BUY"   else \
                -1.0 if signals.get("direction") == "SHORT" else 0.0

    ema_cross  = signals.get("ema_cross_9_21", 0)
    regime_dir = 1.0 if ema_cross > 0 else -1.0 if ema_cross < 0 else 0.0

    ema200_cross = signals.get("ema_cross_50_200", 0)
    weekly_dir   = 1.0 if ema200_cross > 0 else -1.0 if ema200_cross < 0 else 0.0

    vol_ratio    = signals.get("vol_ratio", 1.0)
    atr_ratio    = signals.get("atr_pct", 1.5) / 1.5   # normalized
    inst_active  = (vol_ratio >= INST_VOL_FACTOR) or (atr_ratio >= INST_ATR_FACTOR)
    inst_dir     = trend_dir if inst_active else 0.0

    base_dir = (0.40 * trend_dir + 0.25 * regime_dir +
                0.20 * weekly_dir + 0.15 * inst_dir)

    sector_boost = 0.15 if sector_strength == "Leading" else \
                  -0.15 if sector_strength == "Weak"    else 0.0

    adj_dir = base_dir + 0.15 * macro_bias + 0.15 * sector_bias + sector_boost
    adj_dir = float(np.clip(adj_dir, -1.0, 1.0))

    bull_prob_ens = ((adj_dir + 1.0) / 2.0) * 100.0
    bull_prob_ens = max(0, min(100, bull_prob_ens))

    return adj_dir, bull_prob_ens, inst_active


# ═════════════════════════════════════════════════════════════════════════════
# 3. BAYESIAN + KALMAN PROBABILITY (Pine Script section 12)
# ═════════════════════════════════════════════════════════════════════════════

# Persistent state for Bayesian/Kalman
_bayes_state = {}   # symbol → {bayes_prob, kalman_prob}


def kalman_bayes_prob(symbol, adj_dir_score, bull_prob_ensemble, decay=BAYES_DECAY, gain=KALMAN_GAIN):
    """
    Pine Script Bayesian + Kalman:
      likeBull = 0.5 + adjDirScore * 0.3   (clipped 0.1–0.9)
      posterior = prior*likeBull / (prior*likeBull + (1-prior)*likeBear)
      bayesProb = decay * bayesProb + (1-decay) * posterior * 100
      kalmanProb = kalmanProb[prev] + gain * (bayesProbRaw - kalmanProb[prev])
    """
    state = _bayes_state.get(symbol, {"bayes_prob": bull_prob_ensemble, "kalman_prob": bull_prob_ensemble})

    base_p = max(0.05, min(0.95, bull_prob_ensemble / 100.0))
    prior  = max(0.05, min(0.95, state["bayes_prob"] / 100.0))

    like_bull = max(0.1, min(0.9, 0.5 + adj_dir_score * 0.3))
    like_bear = 1.0 - like_bull

    denom     = prior * like_bull + (1.0 - prior) * like_bear
    post_bull = (prior * like_bull) / denom if denom > 0 else 0.5

    bayes_raw = decay * state["bayes_prob"] + (1.0 - decay) * (post_bull * 100.0)
    kalman    = state["kalman_prob"] + gain * (bayes_raw - state["kalman_prob"])

    bull_kf   = float(np.clip(kalman, 0.0, 100.0))
    bear_kf   = 100.0 - bull_kf

    _bayes_state[symbol] = {"bayes_prob": bayes_raw, "kalman_prob": kalman}
    return bull_kf, bear_kf


# ═════════════════════════════════════════════════════════════════════════════
# 4. CONFIDENCE SCORE (Pine Script ✅ CONFIDENCE section)
# ═════════════════════════════════════════════════════════════════════════════

def compute_confidence(signals, adj_dir_score, bull_prob_kf, is_call=True):
    """
    Pine Script confidence:
      trendStrength   = clip(adxVal, 10, 60)
      predictionConf  = trendStrength*0.35 + volState*15 + momentumState*0.25 +
                        instState*20 + (1-macroState)*15
      dirConf         = bullProbKF if call else (100-bullProbKF)
      finalConf       = 0.65*predictionConf + 0.35*dirConf
    """
    adx      = float(signals.get("adx", 25))
    atr_pct  = float(signals.get("atr_pct", 1.5))
    vol_ratio= float(signals.get("vol_ratio", 1.0))
    mom5     = float(abs(signals.get("mom_5d", 0)))
    hv20     = float(signals.get("hv20", 25))

    trend_strength = min(max(adx, 10.0), 60.0)

    # volState: compressed ATR = good entry
    atr_sma_proxy    = float(signals.get("atr", atr_pct * signals.get("price",100) / 100))
    atr_compression  = atr_pct / max(atr_pct, 0.01)   # self-relative (always 1 without history)
    # Use bb_width as squeeze proxy
    bb_width = float(signals.get("bb_width", 0.05))
    vol_state = 1.0 if bb_width < 0.04 else 0.6        # compressed = better

    momentum_state = min(mom5 * 2.5, 40.0)

    vol_ratio_val = vol_ratio
    atr_ratio_val = atr_pct / 1.5
    inst_active   = (vol_ratio_val >= INST_VOL_FACTOR) or (atr_ratio_val >= INST_ATR_FACTOR)
    inst_state    = 1.0 if inst_active else 0.6

    macro_state   = 1.0 - min(abs(adj_dir_score) * 0.3, 0.3)   # more extreme = less neutral

    prediction_conf = (trend_strength * 0.35 + vol_state * 15.0 +
                       momentum_state * 0.25 + inst_state * 20.0 +
                       (1.0 - macro_state) * 15.0)
    prediction_conf = float(np.clip(prediction_conf, 5.0, 100.0))

    dir_conf   = bull_prob_kf if is_call else (100.0 - bull_prob_kf)
    final_conf = 0.65 * prediction_conf + 0.35 * dir_conf
    final_conf = int(round(np.clip(final_conf, 1.0, 100.0)))

    return final_conf, int(round(prediction_conf))


# ═════════════════════════════════════════════════════════════════════════════
# 5. HYBRID FORECAST (Pine Script section 11)
# ═════════════════════════════════════════════════════════════════════════════

def compute_hybrid_forecast(close_arr, adj_dir_score, regime_sign, atr_val):
    """
    Pine Script:
      lrSlope = sma(close - close[1], 5)
      mom     = ema(close, 8) - ema(close, 21)
      hybridForecast = close + lrSlope*8 + lrSlope*20*0.4 + mom*2.0 +
                       atrNow*regimeSign*0.4 + atrNow*adjDirScore*atrMult*0.6
    """
    close = np.array(close_arr, dtype=float)
    if len(close) < 22:
        return float(close[-1]), float(close[-1])

    diffs    = close[-6:] - close[-7:-1]
    lr_slope = float(np.mean(diffs[-5:]))

    ema8  = _ema(close, 8)
    ema21 = _ema(close, 21)
    mom   = float(ema8[-1] - ema21[-1])

    atr_mult = min(atr_val / (close[-1] + 1e-8), 1.0)

    forecast = (close[-1] + lr_slope * 8 + lr_slope * 20 * 0.4 +
                mom * 2.0 + atr_val * regime_sign * 0.4 +
                atr_val * adj_dir_score * atr_mult * 0.6)

    return float(forecast), float(lr_slope)


# ═════════════════════════════════════════════════════════════════════════════
# 6. RANGE FORECAST (Pine Script section 15)
# ═════════════════════════════════════════════════════════════════════════════

def compute_range_forecast(close_arr, atr_val, adj_dir_score,
                           atr_compression=1.0, atr_inst_ratio=1.0,
                           hybrid_forecast=None, sigma_mult=FORECAST_SIGMA,
                           bars=FORECAST_BARS):
    """
    Pine Script aggressive range forecast:
      retVol = stdev(log(close/close[1]), 50)
      sigmaBarLR = close * retVol
      sigmaPerBarBase = 0.5*atr + 0.5*sigmaBarLR
      volBoost += max(0, atrCompression-1)*0.5 + max(0, atrInstRatio-1)*0.3
      volBoost *= 0.7 + 0.6*adjDirScore^2
      sigmaHorizon = sigmaPerBarBase * volBoost * sqrt(bars)
    """
    close = np.array(close_arr, dtype=float)
    if len(close) < 15:
        return None

    rets = np.log(close[-51:] / close[-52:-1] + 1e-10) if len(close) >= 52 else np.diff(np.log(close + 1e-10))
    ret_vol = float(np.std(rets)) if len(rets) > 2 else 0.01

    sigma_bar_lr       = close[-1] * ret_vol
    sigma_per_bar_base = 0.5 * atr_val + 0.5 * sigma_bar_lr

    vol_boost  = 1.0
    vol_boost += max(0.0, atr_compression - 1.0) * 0.5
    vol_boost += max(0.0, atr_inst_ratio   - 1.0) * 0.3
    vol_boost *= 0.7 + 0.6 * (adj_dir_score ** 2)

    sigma_horizon = sigma_per_bar_base * vol_boost * math.sqrt(bars)

    mid = (hybrid_forecast * 0.7 + close[-1] * 0.3) if hybrid_forecast else close[-1]

    return {
        "low_1s":   round(mid - sigma_mult * sigma_horizon, 3),
        "high_1s":  round(mid + sigma_mult * sigma_horizon, 3),
        "low_2s":   round(mid - 2.0 * sigma_horizon, 3),
        "high_2s":  round(mid + 2.0 * sigma_horizon, 3),
        "mid":      round(mid, 3),
        "sigma":    round(sigma_horizon, 3),
        "bars":     bars,
    }


# ═════════════════════════════════════════════════════════════════════════════
# 7. SQUEEZE + BREAKOUT DETECTION (Pine Script section 13)
# ═════════════════════════════════════════════════════════════════════════════

def detect_squeeze_and_breakout(df, signals):
    """
    Pine Script:
      atrCompression = atrRisk / sma(atrRisk, 50)
      isCompressed   = atrCompression <= 0.65
      upBreak  = close > brHigh[1] AND volConfirm AND callBias
      downBreak = close < brLow[1] AND volConfirm AND putBias
    """
    if df is None or len(df) < SQUEEZE_LOOK + 5:
        return {"squeeze": False, "breakout": "NONE", "compression": 1.0}

    high   = df["High"].values.astype(float)
    low    = df["Low"].values.astype(float)
    close  = df["Close"].values.astype(float)
    volume = df["Volume"].values.astype(float)

    # ATR history for compression
    atr_series = []
    for i in range(2, len(close)):
        tr = max(high[i]-low[i],
                 abs(high[i]-close[i-1]),
                 abs(low[i]-close[i-1]))
        atr_series.append(tr)
    atr_arr = np.array(atr_series[-SQUEEZE_LOOK-5:])
    atr_sma = float(np.mean(atr_arr[-SQUEEZE_LOOK:])) if len(atr_arr) >= SQUEEZE_LOOK else float(np.mean(atr_arr))
    atr_cur = float(atr_arr[-1]) if len(atr_arr) > 0 else 1.0
    compression = atr_cur / (atr_sma + 1e-8)
    is_compressed = compression <= SQUEEZE_THRESH

    # Breakout range
    br_high_prev = float(np.max(high[-BR_LOOKBACK-1:-1]))
    br_low_prev  = float(np.min(low[-BR_LOOKBACK-1:-1]))
    br_high_curr = float(np.max(high[-BR_LOOKBACK:]))
    br_low_curr  = float(np.min(low[-BR_LOOKBACK:]))

    # Volume confirmation
    vol_sma = float(np.mean(volume[-BR_LOOKBACK:])) if len(volume) >= BR_LOOKBACK else float(volume[-1])
    vol_confirm = (volume[-1] / (vol_sma + 1e-8)) >= 1.3   # brVolFactor=1.3

    is_call  = signals.get("direction") == "BUY"
    is_put   = signals.get("direction") == "SHORT"

    up_break   = (close[-1] > br_high_prev) and (close[-1] > br_high_curr) and vol_confirm and is_call
    down_break = (close[-1] < br_low_prev)  and (close[-1] < br_low_curr)  and vol_confirm and is_put

    breakout_state = (
        "EXPLOSIVE_UP"   if up_break   and is_compressed else
        "EXPLOSIVE_DOWN" if down_break and is_compressed else
        "BREAKOUT_UP"    if up_break   else
        "BREAKOUT_DOWN"  if down_break else
        "COILING"        if is_compressed else
        "NONE"
    )

    squeeze_state = "Compressed (coiled)" if is_compressed else \
                    "Expanding (in move)" if compression > 1.0 else "Normal"

    return {
        "squeeze":       is_compressed,
        "breakout":      breakout_state,
        "compression":   round(compression, 4),
        "squeeze_state": squeeze_state,
        "vol_confirm":   vol_confirm,
        "br_high":       round(br_high_curr, 3),
        "br_low":        round(br_low_curr, 3),
    }


# ═════════════════════════════════════════════════════════════════════════════
# 8. CHANDELIER TRAILING STOP (Pine Script Quant Exit Engine)
# ═════════════════════════════════════════════════════════════════════════════

def compute_chandelier_stop(entry_price, current_price, high_since_entry,
                             low_since_entry, atr_val, is_long=True,
                             mult=CHAND_MULT, tight_mult=TIGHT_MULT,
                             be_at_r=BE_AT_R, lock_at_r=LOCK_AT_R,
                             stop_dist_mult=ATR_STOP_MULT):
    """
    Pine Script Chandelier stop logic:
      stopDist   = atrRisk * atrStopMult (2.8)
      trailMultNow = tightMult if r >= lockAtR else chandMult
      chandLongStop  = hiSinceEntry - atr * trailMultNow
      longStop = max(baseLongStop, chandLongStop)
      if r >= beAtR: longStop = max(longStop, entryPrice)  # breakeven
    """
    stop_dist = atr_val * stop_dist_mult

    if is_long:
        r_now = (current_price - entry_price) / (stop_dist + 1e-8)
        trail_mult   = tight_mult if r_now >= lock_at_r else mult
        base_stop    = entry_price - stop_dist
        chand_stop   = high_since_entry - atr_val * trail_mult
        raw_stop     = max(base_stop, chand_stop)
        stop         = max(raw_stop, entry_price) if r_now >= be_at_r else raw_stop
        take_profit  = entry_price + stop_dist * ATR_TP_RR
    else:
        r_now = (entry_price - current_price) / (stop_dist + 1e-8)
        trail_mult   = tight_mult if r_now >= lock_at_r else mult
        base_stop    = entry_price + stop_dist
        chand_stop   = low_since_entry + atr_val * trail_mult
        raw_stop     = min(base_stop, chand_stop)
        stop         = min(raw_stop, entry_price) if r_now >= be_at_r else raw_stop
        take_profit  = entry_price - stop_dist * ATR_TP_RR

    return {
        "stop":         round(stop, 3),
        "take_profit":  round(take_profit, 3),
        "stop_dist":    round(stop_dist, 3),
        "r_multiple":   round(r_now, 2),
        "breakeven":    r_now >= be_at_r,
        "locked":       r_now >= lock_at_r,
        "trail_mult":   trail_mult,
    }


# ═════════════════════════════════════════════════════════════════════════════
# 9. INSTITUTIONAL DETECTION (Pine Script section 2)
# ═════════════════════════════════════════════════════════════════════════════

def detect_institutional(signals):
    """
    Pine Script:
      instActive = (volRatio >= instVolFactor) OR (atrInstRatio >= instAtrFactor)
      volText:   EXTREME/HIGH/Normal/LOW based on ratio
    """
    vol_ratio  = float(signals.get("vol_ratio", 1.0))
    atr_pct    = float(signals.get("atr_pct",   1.5))
    atr_ratio  = atr_pct / 1.5   # normalized to avg 1.5% ATR

    inst_active = (vol_ratio >= INST_VOL_FACTOR) or (atr_ratio >= INST_ATR_FACTOR)

    vol_text = (
        "EXTREME volume" if vol_ratio >= INST_VOL_FACTOR * 1.8 else
        "HIGH volume"    if vol_ratio >= INST_VOL_FACTOR       else
        "Normal volume"  if vol_ratio >= 0.8                   else
        "LOW volume"
    )
    atr_text = (
        "EXPLOSIVE vol" if atr_ratio >= INST_ATR_FACTOR * 1.8 else
        "HIGH vol"      if atr_ratio >= INST_ATR_FACTOR       else
        "Normal vol"    if atr_ratio >= 0.8                   else
        "Calm vol"
    )

    direction = signals.get("direction", "HOLD")
    inst_state = 1.0 if inst_active else 0.6

    inst_trend_text = (
        "Institutional ACTIVE (upside bias)"   if inst_active and direction == "BUY"  else
        "Institutional ACTIVE (downside bias)" if inst_active and direction == "SHORT" else
        "Institutional ACTIVE (no clear trend)" if inst_active else
        "Trend up, low inst. activity"          if direction == "BUY"  else
        "Trend down, low inst. activity"        if direction == "SHORT" else
        "Low inst. activity, no clear trend"
    )

    return {
        "inst_active":  inst_active,
        "inst_state":   inst_state,
        "vol_ratio":    round(vol_ratio, 3),
        "atr_ratio":    round(atr_ratio, 3),
        "vol_text":     vol_text,
        "atr_text":     atr_text,
        "trend_text":   inst_trend_text,
    }


# ═════════════════════════════════════════════════════════════════════════════
# 10. 100%+ RETURN FILTER — targets explosive options plays
# ═════════════════════════════════════════════════════════════════════════════

def is_100_pct_play(signals, confidence, squeeze_data, iv_rank, earnings_meta=None):
    """
    A signal qualifies as a 100%+ options play if it has:
    - High confidence (≥72%)
    - Strong direction signal
    - At least 2 explosive catalysts

    Catalysts: squeeze breakout, earnings play, institutional surge,
               short squeeze setup, low IV (cheap options), momentum surge.
    """
    catalysts    = []
    disqualifiers = []

    # 1. Squeeze breakout — coiled spring unwind = huge move
    if squeeze_data.get("breakout") in ("EXPLOSIVE_UP", "EXPLOSIVE_DOWN"):
        catalysts.append("⚡ SQUEEZE BREAKOUT — ATR compressed, volume surge")
    elif squeeze_data.get("squeeze") and squeeze_data.get("vol_confirm"):
        catalysts.append("⚡ COIL + VOLUME — breakout imminent")

    # 2. Institutional surge
    if signals.get("vol_ratio", 1.0) >= 2.5:
        catalysts.append(f"🏦 INSTITUTIONAL SURGE — vol {signals.get('vol_ratio',1):.1f}x avg")

    # 3. Low IV = cheap options (better leverage)
    if iv_rank is not None and iv_rank < 30:
        catalysts.append(f"💰 CHEAP OPTIONS — IVR {iv_rank:.0f}% → max leverage")
    elif iv_rank is not None and iv_rank < 45:
        catalysts.append(f"✓ Moderate IV — fair entry")
    elif iv_rank is not None and iv_rank > 75:
        disqualifiers.append(f"IV EXPENSIVE ({iv_rank:.0f}%) → crush risk post-move")

    # 4. Near earnings (with cheap IV)
    if earnings_meta:
        days = earnings_meta.get("days_away", 99)
        if days <= 3:
            catalysts.append(f"📅 EARNINGS IN {days}d — binary event catalyst")

    # 5. Short squeeze potential
    short_float = signals.get("short_float", 0) or 0
    if short_float > 15:
        catalysts.append(f"🔥 SHORT SQUEEZE — {short_float:.1f}% float short")

    # 6. Strong momentum burst
    mom_5d = abs(signals.get("mom_5d", 0))
    if mom_5d > 5:
        catalysts.append(f"📈 MOMENTUM BURST — {mom_5d:.1f}% in 5 days")

    # 7. Hurst trending (momentum continuation)
    hurst = signals.get("hurst", 0.5)
    if hurst > 0.65:
        catalysts.append(f"📊 HURST {hurst:.2f} — strong trending regime")

    # 8. ADX strong trend + conviction
    adx = signals.get("adx", 0)
    if adx > 30 and confidence >= 75:
        catalysts.append(f"💪 ADX {adx:.0f} + CONF {confidence}% — high conviction trend")

    # 9. OU mean reversion at extreme
    ou_z = signals.get("ou_zscore", 0)
    if abs(ou_z) > 2.5:
        catalysts.append(f"🎯 MEAN REVERSION — OU Z={ou_z:.2f} extreme")

    # Disqualifiers
    if signals.get("atr_pct", 0) > MAX_ATR_PCT:
        disqualifiers.append("ATR too wide — position sizing risk")
    if signals.get("atr_pct", 0) < MIN_ATR_PCT:
        disqualifiers.append("ATR too quiet — insufficient movement")

    n_cats = len(catalysts)
    is_100 = (confidence >= 72 and n_cats >= 2 and len(disqualifiers) == 0)

    # Estimate return potential
    if n_cats >= 4 and confidence >= 80:
        est_return = "300%–500%+ (exceptional setup)"
    elif n_cats >= 3 or (n_cats >= 2 and confidence >= 80):
        est_return = "150%–300% target"
    elif n_cats >= 2:
        est_return = "100%–150% target"
    else:
        est_return = "50%–100% (standard)"

    # Est hold time (Pine Script holdDays logic)
    adx_v    = signals.get("adx", 25)
    hold_score = (adx_v / 50.0) * 0.35 + min(signals.get("vol_ratio",1)/3,1) * 0.25 + (1.0 if signals.get("vol_ratio",1) >= INST_VOL_FACTOR else 0.6) * 0.2
    hold_days  = 3 if hold_score < 0.8 else 7 if hold_score < 1.2 else 14 if hold_score < 1.6 else 30
    hold_text  = "Scalp (Days)" if hold_days <= 5 else "Swing (1–2 Weeks)" if hold_days <= 14 else "Position (Weeks+)"

    return {
        "is_100_pct_play": is_100,
        "catalysts":       catalysts[:6],
        "disqualifiers":   disqualifiers,
        "estimated_return": est_return,
        "hold_days":       hold_days,
        "hold_text":       hold_text,
        "n_catalysts":     n_cats,
    }


# ═════════════════════════════════════════════════════════════════════════════
# 11. MASTER SIGNAL COMPUTATION — integrates all Pine Script models
# ═════════════════════════════════════════════════════════════════════════════

def run_pine_signals(symbol, df, signals, macro_bias=0.0, sector_bias=0.0,
                     sector_strength="Neutral", price_cache=None, iv_rank=None,
                     earnings_meta=None):
    """
    Master function: runs all Pine Script v4.1 algorithms.
    Adds to signals dict and returns enriched data.
    """
    if df is None or len(df) < 30:
        return signals

    close  = df["Close"].values.astype(float)
    high   = df["High"].values.astype(float)
    low    = df["Low"].values.astype(float)
    volume = df["Volume"].values.astype(float)
    atr_val= float(signals.get("atr", 1.0))

    # ── 1. Ensemble direction ───────────────────────────────────────────────
    adj_dir, bull_prob_ens, inst_active = compute_ensemble_direction(
        signals, macro_bias, sector_bias, sector_strength)

    # ── 2. Bayesian + Kalman ────────────────────────────────────────────────
    bull_kf, bear_kf = kalman_bayes_prob(symbol, adj_dir, bull_prob_ens)

    # ── 3. Confidence ───────────────────────────────────────────────────────
    is_call = signals.get("direction") == "BUY"
    conf, pred_conf = compute_confidence(signals, adj_dir, bull_kf, is_call)

    # ── 4. V-Reversal ──────────────────────────────────────────────────────
    v_call, v_put, v_meta = detect_v_reversal(df, atr_val)

    # ── 5. Institutional ───────────────────────────────────────────────────
    inst = detect_institutional(signals)

    # ── 6. Hybrid Forecast ─────────────────────────────────────────────────
    regime_sign = 1.0 if signals.get("ema_cross_50_200", 0) > 0 else \
                 -1.0 if signals.get("ema_cross_50_200", 0) < 0 else 0.0
    forecast, lr_slope = compute_hybrid_forecast(close.tolist(), adj_dir, regime_sign, atr_val)

    # ── 7. Range Forecast ──────────────────────────────────────────────────
    atr_ratio = float(signals.get("vol_ratio", 1.0))   # proxy for atrInstRatio
    rng_fcst  = compute_range_forecast(
        close.tolist(), atr_val, adj_dir,
        atr_compression=1.0, atr_inst_ratio=atr_ratio,
        hybrid_forecast=forecast)

    # ── 8. Squeeze + Breakout ──────────────────────────────────────────────
    sq_data = detect_squeeze_and_breakout(df, signals)

    # ── 9. 100%+ Return filter ─────────────────────────────────────────────
    play_100 = is_100_pct_play(signals, conf, sq_data, iv_rank, earnings_meta)

    # ── 10. Option strike suggestion (Pine logic) ──────────────────────────
    price      = float(signals.get("price", close[-1]))
    call_strike = round(price * (1 + OTM_OFFSET_PCT/100) * 2) / 2    # round to 0.5
    put_strike  = round(price * (1 - OTM_OFFSET_PCT/100) * 2) / 2
    expiry_date = (datetime.now() + timedelta(days=EXPIRY_DAYS)).strftime("%Y-%m-%d")

    # ── Compile all Pine outputs ───────────────────────────────────────────
    pine_out = {
        # Ensemble & probability
        "adj_dir_score":  round(adj_dir, 4),
        "bull_prob_ens":  round(bull_prob_ens, 2),
        "bull_prob_kf":   round(bull_kf, 2),
        "bear_prob_kf":   round(bear_kf, 2),

        # Confidence
        "confidence":     conf,
        "pred_confidence":pred_conf,

        # V-Reversal
        "v_call":         v_call,
        "v_put":          v_put,
        "v_reversal":     v_call or v_put,
        "v_meta":         v_meta,

        # Institutional
        "inst_active":    inst["inst_active"],
        "inst_text":      inst["trend_text"],
        "vol_text":       inst["vol_text"],
        "atr_text":       inst["atr_text"],

        # Forecasts
        "hybrid_forecast": round(forecast, 3),
        "lr_slope":        round(lr_slope, 5),
        "regime_sign":     regime_sign,
        "range_forecast":  rng_fcst,

        # Squeeze
        "squeeze":        sq_data,

        # 100%+ play
        "play_100":       play_100,
        "is_100_pct":     play_100["is_100_pct_play"],
        "confidence_grade": (
            "S+ (ELITE)"  if conf >= 85 else
            "A  (STRONG)" if conf >= 72 else
            "B  (GOOD)"   if conf >= 60 else
            "C  (FAIR)"   if conf >= 45 else
            "D  (WEAK)"
        ),

        # Option suggestion (Pine-style)
        "suggested_call_strike": round(call_strike, 2),
        "suggested_put_strike":  round(put_strike, 2),
        "suggested_expiry":      expiry_date,
        "hold_text":             play_100["hold_text"],
        "hold_days":             play_100["hold_days"],
    }

    # Enrich the signals dict
    signals.update(pine_out)
    return signals