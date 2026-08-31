"""
=============================================================================
ORDER FLOW ENGINE  ·  Renaissance.io Institutional Grade
Real-time order flow imbalance, trade direction classification, and
market microstructure signals.

Signals:
  ├── OFI (Order Flow Imbalance) — signed trade pressure signal
  ├── Lee-Ready Algorithm         — trade direction classification
  ├── Kyle's Lambda               — price impact / informed trading coefficient
  ├── VPIN (Volume-Synchronized Probability of Informed Trading)
  └── Amihud Illiquidity          — return/volume ratio

All params from env vars — nothing hardcoded.
Academic basis:
  Lee & Ready (1991): "Inferring Trade Direction from Intraday Data"
  Kyle (1985): "Continuous Auctions and Insider Trading"
  Easley, Lopez de Prado & O'Hara (2012): "Flow Toxicity and Liquidity"
  Amihud (2002): "Illiquidity and Stock Returns"
=============================================================================
"""

import logging, os, math
import numpy as np
from typing import Dict, List, Optional
from collections import deque

logger = logging.getLogger("OrderFlow")

_KYLE_WINDOW   = int(os.environ.get("OFI_KYLE_WINDOW",   "20"))
_OFI_WINDOW    = int(os.environ.get("OFI_WINDOW",        "20"))
_VPIN_BUCKET   = int(os.environ.get("VPIN_BUCKET_VOL",   "50"))
_AMIHUD_WINDOW = int(os.environ.get("AMIHUD_WINDOW",     "20"))
_ILLIQ_SCALE   = float(os.environ.get("AMIHUD_SCALE",    "1e6"))


class OrderFlowSignals:
    """
    Computes real-time order flow signals from bar data and trade ticks.
    Thread-safe, rolling computation.
    """

    # ─── Lee-Ready trade direction ────────────────────────────────────────────

    @staticmethod
    def lee_ready_direction(
        price:     float,
        bid:       float,
        ask:       float,
        prev_price: float | None = None,
    ) -> int:
        """
        Lee-Ready (1991) trade direction classification.
        Returns: +1 (buy), -1 (sell), 0 (midquote/unknown)

        Rules:
          1. Quote rule: price above/below midquote → buy/sell
          2. Tick rule (if at midquote): compare to last trade price
        """
        mid = (bid + ask) / 2.0
        if price > mid + 1e-8:
            return +1   # buy — above midquote
        if price < mid - 1e-8:
            return -1   # sell — below midquote
        # At midquote — use tick rule
        if prev_price is not None:
            if price > prev_price + 1e-8: return +1
            if price < prev_price - 1e-8: return -1
        return 0

    @staticmethod
    def classify_bars_lee_ready(
        bars: List[dict],
    ) -> List[int]:
        """
        Classify trade direction for a sequence of bars using OHLCV proxy.
        Since we don't have tick data, we approximate:
          - If close > open: direction = +1 (net buyer aggression)
          - If close < open: direction = -1
          - Weighted by volume
        """
        directions = []
        for i, bar in enumerate(bars):
            o, h, l, c = bar.get("open",0), bar.get("high",0), bar.get("low",0), bar.get("close",0)
            # Approximation: use OHLC body direction as net flow proxy
            if c > o: directions.append(+1)
            elif c < o: directions.append(-1)
            else: directions.append(0)
        return directions

    # ─── Order Flow Imbalance ─────────────────────────────────────────────────

    @staticmethod
    def compute_ofi(
        bars:      List[dict],
        window:    int = _OFI_WINDOW,
    ) -> dict:
        """
        Order Flow Imbalance — signed volume-weighted trade pressure.
        OFI = Σ(buy_volume - sell_volume) over window.
        Normalised to [-1, +1].

        Uses Lee-Ready proxy for direction when quote data unavailable.
        Higher OFI → more net buying pressure → bullish.
        """
        if len(bars) < 2:
            return {"ofi": 0.0, "ofi_norm": 0.5, "buy_vol": 0.0, "sell_vol": 0.0}

        recent = bars[-window:]
        dirs   = OrderFlowSignals.classify_bars_lee_ready(recent)

        buy_vol  = sum(b.get("volume",0) * (d == 1)  for b, d in zip(recent, dirs))
        sell_vol = sum(b.get("volume",0) * (d == -1) for b, d in zip(recent, dirs))
        total    = buy_vol + sell_vol + 1e-8
        ofi      = (buy_vol - sell_vol) / total   # [-1, +1]
        ofi_norm = (ofi + 1.0) / 2.0              # [0, 1] for AladdinScorer

        return {
            "ofi":        round(float(ofi),      4),
            "ofi_norm":   round(float(ofi_norm), 4),
            "buy_vol":    round(float(buy_vol),   0),
            "sell_vol":   round(float(sell_vol),  0),
            "total_vol":  round(float(total),     0),
        }

    # ─── Kyle's Lambda (Price Impact) ────────────────────────────────────────

    @staticmethod
    def kyle_lambda(
        bars:   List[dict],
        window: int = _KYLE_WINDOW,
    ) -> dict:
        """
        Kyle's Lambda — price impact coefficient.
        λ = Cov(ΔP, signed_volume) / Var(signed_volume)

        High lambda → informed trading → wider spreads / adverse selection.
        Estimated from OHLCV using Lee-Ready direction proxy.

        Returns:
            lambda_est:  price impact per unit signed volume
            lambda_norm: normalised [0,1] — 0.5=average, higher=more informed
        """
        if len(bars) < 5:
            return {"lambda_est": 0.0, "lambda_norm": 0.5, "informed_flow": False}

        recent  = bars[-window:]
        dirs    = OrderFlowSignals.classify_bars_lee_ready(recent)

        delta_p = np.array([b.get("close",0) - b.get("open",0) for b in recent], dtype=float)
        s_vol   = np.array([b.get("volume",0) * d for b, d in zip(recent, dirs)], dtype=float)

        var_sv  = np.var(s_vol)
        if var_sv < 1e-10:
            return {"lambda_est": 0.0, "lambda_norm": 0.5, "informed_flow": False}

        cov_pv  = float(np.cov(delta_p, s_vol)[0, 1])
        lam     = cov_pv / var_sv

        # Normalise: lambda_norm above 0.65 = significant informed trading
        avg_price = np.mean([b.get("close",0) for b in recent]) + 1e-8
        lam_pct   = abs(lam) / avg_price * 1e4   # bps per 100 shares
        lam_norm  = float(np.clip(lam_pct / 5.0, 0.0, 1.0))   # 5bps/100sh = max

        informed = lam_norm > 0.65

        return {
            "lambda_est":    round(float(lam),    8),
            "lambda_norm":   round(lam_norm,       4),
            "lambda_bps":    round(lam_pct,        4),
            "informed_flow": informed,
        }

    # ─── VPIN (Volume-Synchronized Probability of Informed Trading) ───────────

    @staticmethod
    def vpin(
        bars:        List[dict],
        bucket_vol:  int = _VPIN_BUCKET,
    ) -> dict:
        """
        Simplified VPIN — volume-synchronised version of PIN model.
        VPIN ≈ |buy_volume - sell_volume| / total_volume per volume bucket.

        Range [0, 1]: 0 = no informed trading, 1 = all trades are informed.
        Threshold: VPIN > 0.5 historically precedes market stress.
        """
        if len(bars) < 5:
            return {"vpin": 0.5, "vpin_stress": False}

        dirs  = OrderFlowSignals.classify_bars_lee_ready(bars)
        vpin_vals = []
        bucket_buy = 0.0
        bucket_sell = 0.0
        bucket_running = 0.0

        for bar, d in zip(bars, dirs):
            v = float(bar.get("volume", 0))
            if d == 1:
                bucket_buy  += v
            elif d == -1:
                bucket_sell += v
            bucket_running += v

            if bucket_running >= bucket_vol:
                total_b = bucket_buy + bucket_sell + 1e-8
                vpin_vals.append(abs(bucket_buy - bucket_sell) / total_b)
                bucket_buy = bucket_sell = bucket_running = 0.0

        if not vpin_vals:
            # Not enough data for full buckets — use all bars as one bucket
            total_d = [b.get("volume",0) * d for b, d in zip(bars, dirs)]
            buy_v  = sum(x for x in total_d if x > 0)
            sell_v = sum(abs(x) for x in total_d if x < 0)
            total  = buy_v + sell_v + 1e-8
            vpin_vals = [abs(buy_v - sell_v) / total]

        vpin_val = float(np.mean(vpin_vals[-20:]))   # rolling average
        vpin_val = float(np.clip(vpin_val, 0.0, 1.0))

        return {
            "vpin":        round(vpin_val,       4),
            "vpin_stress": vpin_val > 0.5,
            "n_buckets":   len(vpin_vals),
        }

    # ─── Amihud Illiquidity ───────────────────────────────────────────────────

    @staticmethod
    def amihud_illiquidity(
        bars:   List[dict],
        window: int = _AMIHUD_WINDOW,
        scale:  float = _ILLIQ_SCALE,
    ) -> dict:
        """
        Amihud (2002) illiquidity ratio.
        ILLIQ_t = |r_t| / Volume_t

        Higher ILLIQ → stock moves more per unit volume → less liquid.
        Normalised against the window average.
        Low liquidity = higher adverse selection cost → lower conviction on entry.
        """
        if len(bars) < 5:
            return {"illiq": 0.0, "illiq_norm": 0.5, "liquid": True}

        recent = bars[-window:]
        illiq_vals = []
        for bar in recent:
            c   = float(bar.get("close", 0))
            o   = float(bar.get("open",  c))
            vol = float(bar.get("volume", 1))
            if vol < 1 or o <= 0:
                continue
            ret = abs((c - o) / (o + 1e-8))
            illiq_vals.append(ret / (vol * scale))

        if not illiq_vals:
            return {"illiq": 0.0, "illiq_norm": 0.5, "liquid": True}

        illiq_mean = float(np.mean(illiq_vals))
        # Normalise: compare to trailing window
        if len(illiq_vals) > 5:
            prev_mean = float(np.mean(illiq_vals[:-5]))
            illiq_rel = illiq_vals[-1] / (prev_mean + 1e-12)
        else:
            illiq_rel = 1.0

        illiq_norm = float(np.clip(1.0 / (1.0 + illiq_rel), 0.0, 1.0))   # high illiq → low norm

        return {
            "illiq":      round(illiq_mean * 1e6, 4),   # scaled for readability
            "illiq_norm": round(illiq_norm,         4),
            "liquid":     illiq_norm > 0.4,
        }

    # ─── Composite OFI Score ─────────────────────────────────────────────────

    @staticmethod
    def composite_orderflow_score(
        bars: List[dict],
        bid:  float | None = None,
        ask:  float | None = None,
    ) -> dict:
        """
        Composite order flow signal combining OFI + Kyle lambda + VPIN + Amihud.
        Returns a single score [0,1] for AladdinScorer integration.
        """
        if not bars or len(bars) < 3:
            return {"orderflow_score": 0.5, "orderflow_norm": 0.5,
                    "direction": "NEUTRAL", "components": {}}

        ofi   = OrderFlowSignals.compute_ofi(bars)
        kyle  = OrderFlowSignals.kyle_lambda(bars)
        vpin_ = OrderFlowSignals.vpin(bars)
        ilq   = OrderFlowSignals.amihud_illiquidity(bars)

        # Weights from env (default: OFI dominates, adjusted by liquidity)
        w_ofi  = float(os.environ.get("OFI_WEIGHT_OFI",   "0.40"))
        w_kyle = float(os.environ.get("OFI_WEIGHT_KYLE",  "0.20"))
        w_vpin = float(os.environ.get("OFI_WEIGHT_VPIN",  "0.20"))
        w_ilq  = float(os.environ.get("OFI_WEIGHT_ILLIQ", "0.20"))

        # VPIN stress = uncertainty → pull toward 0.5
        vpin_contrib = 0.5 if vpin_["vpin_stress"] else (1.0 - vpin_["vpin"])

        score = (w_ofi  * ofi["ofi_norm"]   +
                 w_kyle * (1.0 - kyle["lambda_norm"]) +   # high lambda → informed → bearish for retail
                 w_vpin * vpin_contrib +
                 w_ilq  * ilq["illiq_norm"])

        score = float(np.clip(score, 0.0, 1.0))
        direction = "BULLISH" if score > 0.55 else "BEARISH" if score < 0.45 else "NEUTRAL"

        return {
            "orderflow_score": round(score, 4),
            "orderflow_norm":  round(score, 4),
            "direction":       direction,
            "components": {
                "ofi":        ofi,
                "kyle":       kyle,
                "vpin":       vpin_,
                "amihud":     ilq,
            },
        }


# Module-level instance (stateless — all methods are static)
_OFI_ENGINE = OrderFlowSignals()

def get_orderflow_engine() -> OrderFlowSignals:
    return _OFI_ENGINE
