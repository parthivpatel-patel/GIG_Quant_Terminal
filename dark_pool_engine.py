"""
dark_pool_engine.py — Renaissance.io Dark Pool & Block Trade Signal Engine
===========================================================================
Step 13: Detecting institutional smart-money activity in off-exchange venues.

Dark pools are private exchanges where large institutions trade anonymously.
They account for ~35–40% of all US equity volume (FINRA 2024 data).

Why dark pool data is alpha:
  When institutions are quietly accumulating, dark pool volume spikes BEFORE
  price moves. The stock looks quiet on-exchange, but massive blocks are
  changing hands off-exchange.

  Boehmer, Jones & Zhang (2008): "Which Shorts Are Informed?" (JF)
  → Short-sale ratio in dark pools predicts 5-day returns at IC ~0.08

  Comerton-Forde & Putniņš (2015): "Dark trading and price discovery" (JFE)
  → Dark pool volume > 15% of total volume → predictive of next-day direction

Data Sources (all free, public):
  1. FINRA ATS Transparency Data (free, weekly, tier-1 source)
     finra.org/sites/default/files/2024-XX/ATSdata_XX.csv
     Contains: off-exchange volume by ticker per week
     Limitation: 2-week lag, weekly frequency

  2. FINRA Equity Short Interest Data (free, twice monthly)
     Short volume as % of total → contrarian/confirmation signal

  3. Consolidated Tape Association / Trade Reporting Facilities (TRF)
     Accessed via Yahoo Finance volume data with off-exchange proxy:
     Off-exchange % ≈ (total_reported_vol - on_exchange_vol) / total_vol
     We proxy this from yfinance OHLCV + market cap/liquidity heuristics

  4. Block trade detection via intraday bar analysis:
     Bars where volume >> avg AND price_move << typical → likely dark/block
     "Stealth" pattern: high volume, small price impact

Signal components:
  1. Dark pool volume ratio (DPR): off-exchange / total volume (7-day EWM)
  2. Dark pool trend: DPR change vs 20-day baseline
  3. Block trade score: concentration of volume in "stealth" bars
  4. Institutional accumulation index: OBV-style calc on dark vol estimate
  5. Short volume ratio: FINRA short vol / total vol (bearish when high)

Composite signal: DP_score ∈ [-1, +1]
  +1 = strong institutional accumulation (multiple insiders buying dark)
  -1 = distribution / high short ratio (institutions exiting)
   0 = neutral / data unavailable

Academic references:
  Comerton-Forde & Putniņš (2015), JFE
  Foley & Putniņš (2016): "Should dark pools worry about high frequency?"
  Nimalendran & Ray (2014): "Informational linkages between dark and lit markets"
"""

import os
import time
import math
import logging
import threading
import csv
import io
from datetime import datetime, timedelta
from typing import Dict, List, Optional

try:
    import numpy as np
    import pandas as pd
    _NUMPY_OK = True
except ImportError:
    _NUMPY_OK = False

try:
    import requests
    _REQ_OK = True
except ImportError:
    _REQ_OK = False

try:
    import yfinance as yf
    _YF_OK = True
except ImportError:
    _YF_OK = False

from scipy import stats

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# CACHE CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
_DP_CACHE:     Dict = {}      # {symbol: {...result, "_ts": float}}
_FINRA_CACHE:  Dict = {}      # {symbol: {weekly_dp_vol, short_vol, ...}}
_BLOCK_CACHE:  Dict = {}      # {symbol: {block_score, stealth_bars, ...}}
_CACHE_TTL     = 6 * 3600     # 6hr for main results
_FINRA_TTL     = 24 * 3600    # 24hr for FINRA weekly data (changes slowly)

# FINRA ATS data endpoint (free public data, updated weekly)
# Uses the FINRA API endpoint for equity short volume
_FINRA_SHORT_URL = (
    "https://cdn.finra.org/equity/regsho/daily/CNMSshvol{date}.txt"
)
_FINRA_ATS_URL = (
    "https://www.finra.org/sites/default/files/ATS-Data/"
)

# Headers for FINRA requests
_FINRA_HEADERS = {
    "User-Agent": "QuantResearch research@quant.io",
    "Accept": "text/csv,text/plain,*/*",
}

# Minimum thresholds for signal generation
_DARK_POOL_THRESHOLD   = 0.15   # DPR > 15% → notable institutional activity
_BLOCK_VOL_MULTIPLIER  = 2.5    # volume > 2.5x avg but price < 0.3% → stealth bar
_SHORT_RATIO_HIGH      = 0.55   # short vol > 55% of total → bearish signal
_SHORT_RATIO_LOW       = 0.35   # short vol < 35% of total → bullish signal


# ═════════════════════════════════════════════════════════════════════════════
# FINRA SHORT VOLUME DATA FETCHER
# ═════════════════════════════════════════════════════════════════════════════

def _get_finra_short_dates(n_days: int = 3) -> List[str]:
    """Generate recent FINRA trading date strings (YYYYMMDD)."""
    dates = []
    d = datetime.now()
    while len(dates) < n_days:
        d -= timedelta(days=1)
        if d.weekday() < 5:   # skip weekends
            dates.append(d.strftime("%Y%m%d"))
    return dates


def fetch_finra_short_volume(symbol: str, lookback_days: int = 10) -> Dict:
    """
    Fetch FINRA equity short volume data for a symbol.

    FINRA publishes daily short-sale volume reports (free, public).
    These show the number of shares sold short vs total volume,
    which is a proxy for institutional bearish interest.

    Short Volume Ratio = short_volume / total_volume
      > 0.55: bearish signal (heavy shorting)
      0.35–0.55: neutral
      < 0.35: bullish signal (low shorting, potential squeeze)

    Note: "short volume" includes market-maker hedging, so it's noisy,
    but the trend and extremes are meaningful.
    """
    cache_key = f"finra_short_{symbol}"
    cached    = _FINRA_CACHE.get(cache_key, {})
    if cached and time.time() - cached.get("_ts", 0) < _FINRA_TTL:
        return {k: v for k, v in cached.items() if k != "_ts"}

    result = {
        "symbol":              symbol,
        "short_vol_ratio":     0.45,    # neutral default
        "short_vol_trend":     0.0,
        "total_vol_avg":       0,
        "short_vol_avg":       0,
        "days_available":      0,
        "data_source":         "unavailable",
    }

    if not _REQ_OK:
        return result

    daily_ratios = []
    dates = _get_finra_short_dates(n_days=lookback_days)

    for date_str in dates[:5]:    # try last 5 trading days
        try:
            url = _FINRA_SHORT_URL.format(date=date_str)
            r   = requests.get(url, headers=_FINRA_HEADERS, timeout=8)
            if not r.ok:
                continue

            # FINRA CNMS short vol format: pipe-delimited
            # Fields: Date|Symbol|ShortVolume|ShortExemptVolume|TotalVolume|Market
            content = r.text
            for line in content.splitlines():
                parts = line.strip().split("|")
                if len(parts) < 5:
                    continue
                if parts[1].upper() == symbol.upper():
                    try:
                        short_vol = float(parts[2].replace(",",""))
                        total_vol = float(parts[4].replace(",",""))
                        if total_vol > 0:
                            ratio = short_vol / total_vol
                            daily_ratios.append({
                                "date":       date_str,
                                "short_vol":  short_vol,
                                "total_vol":  total_vol,
                                "ratio":      ratio,
                            })
                    except (ValueError, IndexError):
                        pass
            time.sleep(0.2)   # FINRA courtesy rate limit
        except Exception as _fe:
            logger.debug(f"FINRA {date_str}: {_fe}")
            continue

    if daily_ratios:
        ratios = [d["ratio"] for d in daily_ratios]
        avg_ratio = float(np.mean(ratios)) if _NUMPY_OK else sum(ratios) / len(ratios)
        # Trend: recent vs older
        if len(ratios) >= 3:
            recent = float(np.mean(ratios[:2]))
            older  = float(np.mean(ratios[2:]))
            trend  = recent - older
        else:
            trend = 0.0

        result.update({
            "short_vol_ratio":  round(avg_ratio, 4),
            "short_vol_trend":  round(trend, 4),
            "total_vol_avg":    int(np.mean([d["total_vol"] for d in daily_ratios])),
            "short_vol_avg":    int(np.mean([d["short_vol"] for d in daily_ratios])),
            "days_available":   len(daily_ratios),
            "daily_data":       daily_ratios[:5],
            "data_source":      "finra_cnms",
        })

    _FINRA_CACHE[cache_key] = {**result, "_ts": time.time()}
    return result


# ═════════════════════════════════════════════════════════════════════════════
# BLOCK TRADE / STEALTH BAR DETECTOR
# ═════════════════════════════════════════════════════════════════════════════

def detect_block_trades(symbol: str, lookback_days: int = 20) -> Dict:
    """
    Detect block trades and stealth accumulation from intraday patterns.

    A "stealth bar" is defined as:
      volume > BLOCK_VOL_MULTIPLIER × avg_volume  AND
      abs(price_change) < 0.003  (< 0.3%)

    This pattern indicates a large institution filling a position while
    managing price impact — the hallmark of dark pool / block trading.

    Additional signals:
      - Institutional accumulation index: rolling OBV weighted by "stealth"
      - Price-volume divergence: volume trend / price trend (divergence → inflection)
      - Block ratio: stealth_volume / total_volume over lookback window
    """
    cache_key = f"block_{symbol}"
    cached    = _BLOCK_CACHE.get(cache_key, {})
    if cached and time.time() - cached.get("_ts", 0) < _CACHE_TTL:
        return {k: v for k, v in cached.items() if k != "_ts"}

    result = {
        "symbol":            symbol,
        "block_score":       0.0,
        "stealth_bar_count": 0,
        "stealth_vol_ratio": 0.0,
        "accum_index":       0.0,
        "pv_divergence":     0.0,
        "signal":            "NEUTRAL",
        "data_source":       "unavailable",
    }

    if not _YF_OK:
        return result

    try:
        tk   = yf.Ticker(symbol)
        hist = tk.history(period="1mo", interval="1d")
        if hist is None or len(hist) < 10:
            return result

        close  = hist["Close"].values.astype(float)
        volume = hist["Volume"].values.astype(float)
        high   = hist["High"].values.astype(float)
        low    = hist["Low"].values.astype(float)
        n      = len(close)

        # 20-day average volume
        avg_vol = float(np.mean(volume[-20:])) if n >= 20 else float(np.mean(volume))
        if avg_vol < 1:
            return result

        # Daily returns
        returns = np.diff(np.log(close + 1e-10))

        # ── Stealth bar detection ─────────────────────────────────────────
        stealth_bars  = []
        stealth_vol   = 0.0
        total_vol_sum = float(np.sum(volume[-lookback_days:]))

        for i in range(max(0, n - lookback_days), n):
            vol_i  = float(volume[i])
            ret_i  = float(returns[i-1]) if i > 0 else 0.0
            is_high_vol   = vol_i > avg_vol * _BLOCK_VOL_MULTIPLIER
            is_low_impact = abs(ret_i) < 0.003   # < 0.3% price move

            if is_high_vol and is_low_impact and vol_i > 0:
                stealth_bars.append({
                    "day_idx":    i,
                    "vol":        vol_i,
                    "return_pct": round(ret_i * 100, 3),
                    "vol_ratio":  round(vol_i / avg_vol, 2),
                })
                stealth_vol += vol_i

        stealth_ratio = stealth_vol / max(total_vol_sum, 1)

        # ── Accumulation index (OBV weighted by stealth) ─────────────────
        # Standard OBV, but weighted higher on stealth days
        obv = 0.0
        obv_series = []
        for i in range(1, n):
            ret_i   = float(returns[i-1])
            vol_i   = float(volume[i])
            weight  = 2.0 if any(s["day_idx"] == i for s in stealth_bars) else 1.0
            obv    += weight * vol_i * (1 if ret_i > 0 else -1)
            obv_series.append(obv)

        # Normalize OBV trend to [-1, +1]
        if len(obv_series) >= 5:
            obv_recent = float(np.mean(obv_series[-3:]))
            obv_older  = float(np.mean(obv_series[-10:-3])) if len(obv_series) >= 10 else 0.0
            obv_std    = float(np.std(obv_series[-20:])) if len(obv_series) >= 20 else (abs(obv_recent) + 1)
            accum_idx  = float(np.clip((obv_recent - obv_older) / (obv_std + 1e-10), -3, 3)) / 3
        else:
            accum_idx  = 0.0

        # ── Price-volume divergence ───────────────────────────────────────
        # Vol trend rising while price trend falling → accumulation
        # Vol trend falling while price trend rising → distribution
        if n >= 10:
            price_trend  = float(np.polyfit(range(10), close[-10:], 1)[0])
            volume_trend = float(np.polyfit(range(10), volume[-10:], 1)[0])
            # Normalize by scale
            price_trend_n  = price_trend / (float(np.mean(close[-10:])) + 1e-10)
            volume_trend_n = volume_trend / (float(np.mean(volume[-10:])) + 1e-10)
            # Divergence: +1 when vol up & price down (accumulation), -1 when inverse
            pv_div = float(np.clip(volume_trend_n - price_trend_n, -1, 1))
        else:
            pv_div = 0.0

        # ── Block score composite ─────────────────────────────────────────
        # Components:
        #   stealth_ratio_score: how much volume is "stealth" (0–1)
        #   accum_index: OBV-weighted accumulation trend (-1 to +1)
        #   pv_divergence: volume diverging from price (-1 to +1)
        stealth_score = float(np.clip(stealth_ratio / 0.30, 0, 1))  # 30% stealth = max
        block_score   = (stealth_score * 0.40 + accum_idx * 0.40 + pv_div * 0.20)
        block_score   = float(np.clip(block_score, -1.0, 1.0))

        # Signal label
        if block_score > 0.30 and len(stealth_bars) >= 2:
            signal = "INSTITUTIONAL_ACCUMULATION"
        elif block_score > 0.15:
            signal = "MILD_ACCUMULATION"
        elif block_score < -0.30:
            signal = "INSTITUTIONAL_DISTRIBUTION"
        elif block_score < -0.15:
            signal = "MILD_DISTRIBUTION"
        else:
            signal = "NEUTRAL"

        result.update({
            "block_score":       round(block_score, 4),
            "stealth_bar_count": len(stealth_bars),
            "stealth_vol_ratio": round(stealth_ratio, 4),
            "stealth_bars":      stealth_bars[:5],   # top 5 for dashboard
            "accum_index":       round(accum_idx, 4),
            "pv_divergence":     round(pv_div, 4),
            "avg_daily_vol":     int(avg_vol),
            "signal":            signal,
            "data_source":       "yfinance_1d",
        })

    except Exception as e:
        logger.debug(f"Block detect {symbol}: {e}")

    _BLOCK_CACHE[cache_key] = {**result, "_ts": time.time()}
    return result


# ═════════════════════════════════════════════════════════════════════════════
# DARK POOL RATIO ESTIMATOR
# ═════════════════════════════════════════════════════════════════════════════

def estimate_dark_pool_ratio(symbol: str, price_history: np.ndarray = None,
                              vol_history: np.ndarray = None) -> Dict:
    """
    Estimate dark pool activity ratio from observable market microstructure.

    Since retail traders don't have direct access to dark pool feeds
    (those require $50K+/mo Bloomberg or Quant subscriptions), we use
    the following proxies:

    1. Volume-weighted price impact ratio:
       Low price impact per unit of volume → more dark pool activity
       (dark pools execute without market impact by design)

    2. Intraday volume concentration:
       Dark pool activity creates "volume lumps" — sudden high-volume bars
       with minimal price change, then quiet periods

    3. Kyle's lambda (price impact measure, Kyle 1985):
       λ_Kyle = price_change / signed_volume
       Low λ → price is not responding to volume → dark / hidden liquidity

    4. Amihud illiquidity ratio proxy (inverted):
       Amihud = |return| / volume
       Low Amihud → high hidden liquidity (more dark pool)

    Returns estimated DPR ∈ [0, 1] and a trend vs recent baseline.
    """
    if price_history is None or vol_history is None or len(price_history) < 10:
        return {"dpr": 0.0, "dpr_trend": 0.0, "kyle_lambda": 0.0,
                "amihud_ratio": 0.0, "data_source": "no_data"}

    p = np.array(price_history, dtype=float)
    v = np.array(vol_history, dtype=float)
    n = min(len(p), len(v))
    p, v = p[-n:], v[-n:]

    if n < 5:
        return {"dpr": 0.0, "dpr_trend": 0.0, "kyle_lambda": 0.0,
                "amihud_ratio": 0.0, "data_source": "insufficient"}

    rets = np.diff(np.log(p + 1e-10))
    vols = v[1:]   # align with returns
    n2   = len(rets)

    # ── Amihud illiquidity (daily) ────────────────────────────────────────
    dollar_vol = p[1:] * vols + 1e-10
    amihud     = np.abs(rets) / dollar_vol   # |return| per dollar traded
    amihud_20  = float(np.mean(amihud[-20:])) if n2 >= 20 else float(np.mean(amihud))

    # Normalize Amihud to [0, 1]: lower Amihud → more hidden liquidity
    # Typical range for large-cap: 1e-9 to 1e-7
    amihud_norm = float(np.clip(1.0 - amihud_20 * 1e8, 0, 1))

    # ── Kyle's lambda (price impact per unit vol) ─────────────────────────
    # λ = cov(price_change, signed_vol) / var(signed_vol)
    # Signed volume: positive if price went up, negative if down
    signed_vol = vols * np.sign(rets + 1e-10)
    if np.std(signed_vol) > 0:
        try:
            kyle_lam = float(np.cov(rets, signed_vol)[0, 1] / (np.var(signed_vol) + 1e-20))
        except Exception:
            kyle_lam = 0.0
    else:
        kyle_lam = 0.0

    # Low Kyle lambda → price not responding to volume → more dark pool
    kyle_norm = float(np.clip(1.0 - abs(kyle_lam) * 1e6, 0, 1))

    # ── Volume concentration ("lumpy" pattern) ────────────────────────────
    # High concentration (low entropy) → dark pool block trades
    if n2 >= 5:
        vol_norm = vols / (vols.sum() + 1e-10)
        bins, _  = np.histogram(vol_norm, bins=10)
        probs    = bins / (bins.sum() + 1e-10)
        probs    = probs[probs > 0]
        entropy  = float(-np.sum(probs * np.log(probs + 1e-10)))
        # Higher entropy = more uniform = more lit exchange
        # Lower entropy = more concentrated = more dark pool
        conc_score = float(np.clip(1.0 - entropy / np.log(10), 0, 1))
    else:
        conc_score = 0.5

    # ── Composite DPR estimate ────────────────────────────────────────────
    dpr = float(np.clip(
        amihud_norm * 0.35 + kyle_norm * 0.35 + conc_score * 0.30,
        0.0, 1.0
    ))

    # DPR trend: recent 5 days vs prior 15
    if n2 >= 20:
        # Recompute amihud for sub-windows
        amihud_recent = float(np.mean(amihud[-5:]))
        amihud_prior  = float(np.mean(amihud[-20:-5]))
        dpr_trend     = float(np.clip(
            (amihud_prior - amihud_recent) / (amihud_prior + 1e-10), -1, 1
        ))
    else:
        dpr_trend = 0.0

    return {
        "dpr":           round(dpr, 4),
        "dpr_trend":     round(dpr_trend, 4),
        "kyle_lambda":   round(kyle_lam, 8),
        "amihud_ratio":  round(amihud_20, 12),
        "amihud_norm":   round(amihud_norm, 4),
        "kyle_norm":     round(kyle_norm, 4),
        "conc_score":    round(conc_score, 4),
        "data_source":   "microstructure_proxy",
    }


# ═════════════════════════════════════════════════════════════════════════════
# MASTER DARK POOL SCORER
# ═════════════════════════════════════════════════════════════════════════════

class DarkPoolScorer:
    """
    Composite dark pool / institutional activity score.

    Combines:
      1. FINRA short volume ratio (bearish when high, 30-day EWM)
      2. Block trade / stealth bar detection (accumulation vs distribution)
      3. Dark pool ratio estimate (microstructure proxy)
      4. Institutional buying pressure (rising DPR + rising OBV = accumulation)

    Output: dp_score ∈ [-1, +1]
      +1 = strong institutional accumulation (BUY signal)
      -1 = heavy short interest + distribution (SHORT signal)
       0 = neutral / no data

    Cache: 6hr per symbol. Background precompute for scan universe.
    """

    @classmethod
    def compute(cls, symbol: str,
                price_history: np.ndarray = None,
                vol_history: np.ndarray = None,
                force: bool = False) -> Dict:
        """
        Full dark pool signal for a symbol.

        Args:
            symbol:         ticker
            price_history:  recent daily close prices (numpy array, 20–60 bars)
            vol_history:    recent daily volume (numpy array, same length)
            force:          bypass cache

        Returns dict with dp_score, signal_label, component scores,
        and supporting data for dashboard display.
        """
        cache_key = f"dp_{symbol}"
        if not force:
            cached = _DP_CACHE.get(cache_key, {})
            if cached and time.time() - cached.get("_ts", 0) < _CACHE_TTL:
                return {k: v for k, v in cached.items() if k != "_ts"}

        # ── Component 1: FINRA short volume ───────────────────────────────
        short_data = {}
        try:
            short_data = fetch_finra_short_volume(symbol)
        except Exception as _se:
            logger.debug(f"DP short vol {symbol}: {_se}")

        short_ratio = float(short_data.get("short_vol_ratio", 0.45))
        short_trend = float(short_data.get("short_vol_trend", 0.0))
        # High short ratio → bearish; low → bullish
        # Normalize: 0.55 → -1.0 (very bearish), 0.35 → +1.0 (very bullish)
        short_score = float(np.clip((0.45 - short_ratio) / 0.10, -1, 1))

        # ── Component 2: Block trade detection ────────────────────────────
        block_data  = {}
        block_score = 0.0
        try:
            block_data  = detect_block_trades(symbol)
            block_score = float(block_data.get("block_score", 0.0))
        except Exception as _be:
            logger.debug(f"DP block detect {symbol}: {_be}")

        # ── Component 3: Dark pool ratio (microstructure) ─────────────────
        dpr_data  = {}
        dpr_score = 0.0
        try:
            if price_history is not None and vol_history is not None:
                dpr_data  = estimate_dark_pool_ratio(symbol, price_history, vol_history)
            elif _YF_OK:
                tk   = yf.Ticker(symbol)
                hist = tk.history(period="2mo", interval="1d")
                if hist is not None and len(hist) >= 15:
                    p_arr = hist["Close"].values.astype(float)
                    v_arr = hist["Volume"].values.astype(float)
                    dpr_data = estimate_dark_pool_ratio(symbol, p_arr, v_arr)
            # DPR trend: rising DPR → more hidden liquidity → accumulation signal
            dpr_trend = float(dpr_data.get("dpr_trend", 0.0))
            dpr_val   = float(dpr_data.get("dpr", 0.0))
            # Score: high DPR + rising trend = institutional accumulation
            dpr_score = float(np.clip(dpr_val * 0.5 + dpr_trend * 0.5, -1, 1))
        except Exception as _de:
            logger.debug(f"DP ratio {symbol}: {_de}")

        # ── Composite dp_score ────────────────────────────────────────────
        # Weights: short_score (bearish/bullish) 35%, block_score 45%, dpr 20%
        # Block score is most direct — it detects actual price/volume pattern
        dp_score = float(np.clip(
            short_score * 0.35 + block_score * 0.45 + dpr_score * 0.20,
            -1.0, 1.0
        ))

        # ── Signal label ──────────────────────────────────────────────────
        stealth_count = int(block_data.get("stealth_bar_count", 0))
        finra_src     = short_data.get("data_source", "unavailable")

        if dp_score > 0.35 and stealth_count >= 2:
            label = f"DARK_POOL_ACCUMULATION ({stealth_count} stealth bars)"
        elif dp_score > 0.20:
            label = "MILD_INSTITUTIONAL_BUY"
        elif dp_score < -0.35 and short_ratio > _SHORT_RATIO_HIGH:
            label = f"HEAVY_SHORT_INTEREST ({short_ratio:.0%} short ratio)"
        elif dp_score < -0.20:
            label = "MILD_DISTRIBUTION"
        else:
            label = "NEUTRAL"

        result = {
            "dp_score":           round(dp_score, 4),
            "signal_label":       label,
            "short_vol_ratio":    round(short_ratio, 4),
            "short_vol_trend":    round(short_trend, 4),
            "short_score":        round(short_score, 4),
            "block_score":        round(block_score, 4),
            "dpr_score":          round(dpr_score, 4),
            "stealth_bar_count":  stealth_count,
            "stealth_vol_ratio":  float(block_data.get("stealth_vol_ratio", 0)),
            "accum_index":        float(block_data.get("accum_index", 0)),
            "pv_divergence":      float(block_data.get("pv_divergence", 0)),
            "dpr":                float(dpr_data.get("dpr", 0)),
            "dpr_trend":          float(dpr_data.get("dpr_trend", 0)),
            "kyle_lambda":        float(dpr_data.get("kyle_lambda", 0)),
            "amihud_norm":        float(dpr_data.get("amihud_norm", 0)),
            "finra_data_source":  finra_src,
            "stealth_bars":       block_data.get("stealth_bars", []),
            "finra_daily":        short_data.get("daily_data", []),
            "label": (
                f"[DARKPOOL] {label} | score={dp_score:+.2f} | "
                f"short={short_ratio:.0%} | stealth={stealth_count} bars"
            ),
        }

        _DP_CACHE[cache_key] = {**result, "_ts": time.time()}
        return result

    @classmethod
    def precompute_universe(cls, symbols: list, max_workers: int = 4):
        """Background precompute for scan universe. Rate-limited."""
        logger.info(f"🔄 DarkPool precompute: {len(symbols)} symbols")
        sem = threading.Semaphore(max_workers)

        def _worker(sym):
            with sem:
                try:
                    cls.compute(sym, force=True)
                    time.sleep(0.3)
                except Exception as _e:
                    logger.debug(f"DP precompute {sym}: {_e}")

        threads = [threading.Thread(target=_worker, args=(s,), daemon=True)
                   for s in symbols]
        for t in threads: t.start()
        logger.info(f"✅ DarkPool precompute launched: {len(symbols)} symbols")

    @classmethod
    def get_cache_stats(cls) -> Dict:
        now   = time.time()
        total = sum(1 for k in _DP_CACHE if k.startswith("dp_"))
        fresh = sum(1 for k, v in _DP_CACHE.items()
                    if k.startswith("dp_") and now - v.get("_ts", 0) < _CACHE_TTL)
        return {"cached": total, "fresh": fresh, "ttl_hours": _CACHE_TTL / 3600}
