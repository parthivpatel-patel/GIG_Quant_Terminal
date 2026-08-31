"""
=============================================================================
RENAISSANCE ENSEMBLE ENGINE — ensemble_engine.py
=============================================================================
The Core of Medallion: Ensemble of Ensembles

Renaissance's edge isn't one model — it's layering hundreds of barely-profitable
strategies into a single consistent signal. Individual models are 50.5/49.5,
but the ensemble is 55/45 at massive scale with low correlation.

ARCHITECTURE:
  ├── Alpha Source Registry      — catalog of all signal generators
  ├── IC-Adaptive Weighting      — rolling information coefficient per source
  ├── Regime-Aware Blending      — shifts weights based on market regime
  ├── Correlation-Penalized      — decorrelated signals get higher weight
  ├── Meta-Ensemble Stacking     — combines sub-ensembles (ML, quant, NLP)
  ├── Signal Decay Tracker       — detects alpha decay and auto-downgrades
  └── Turnover Constraint        — limits portfolio churn from signal noise

SIGNAL CATEGORIES:
  1. Quantitative Math Signals  (math_engine: Hurst, OU, GARCH, Kalman, BSM)
  2. ML/DL Predictions          (ml_engine + deep_learning_engine)
  3. NLP Alpha                  (nlp_engine: FinBERT, news, social)
  4. Technical Indicators       (signal_engine: RSI, MACD, ADX, Squeeze)
  5. Options Flow Intelligence  (options_engine: unusual flow, MaxPain)
  6. Microstructure Signals     (orderflow_engine: OFI, VPIN, Kyle lambda)
  7. Alternative Data           (alt_data_engine: satellite, web traffic)
  8. Macro Regime               (math_engine.MacroScorer: VIX, yield curve)

Renaissance.io — Institutional Grade Quantitative Trading
=============================================================================
"""

import numpy as np
import logging
import threading
import time
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from collections import deque

logger = logging.getLogger("ENSEMBLE")

# ═══════════════════════════════════════════════════════════════════════════════
# 1. ALPHA SOURCE REGISTRY
# ═══════════════════════════════════════════════════════════════════════════════

class AlphaSource:
    """Represents one alpha signal generator."""
    __slots__ = ('name', 'category', 'weight', 'ic_history', 'ic_ewm',
                 'last_value', 'last_ts', 'decay_counter', 'active')

    def __init__(self, name: str, category: str, base_weight: float = 0.01):
        self.name = name
        self.category = category
        self.weight = base_weight
        self.ic_history: deque = deque(maxlen=252)  # 1 year rolling
        self.ic_ewm: float = 0.0
        self.last_value: float = 0.0
        self.last_ts: float = 0.0
        self.decay_counter: int = 0
        self.active: bool = True


class AlphaRegistry:
    """
    Central catalog of all alpha sources in the system.
    Each source is independently tracked for IC, decay, and regime performance.
    """

    # All registered alpha sources with categories and base weights
    SOURCES = {
        # ── Quantitative Math (math_engine) ─────────────────────────────
        "hurst_exponent":          ("quant_math", 0.04),
        "ou_halflife":             ("quant_math", 0.03),
        "ou_zscore":               ("quant_math", 0.04),
        "garch_vol":               ("quant_math", 0.03),
        "kalman_trend":            ("quant_math", 0.04),
        "hmm_regime":              ("quant_math", 0.03),
        "monte_carlo_pop":         ("quant_math", 0.02),
        "kelly_fraction":          ("quant_math", 0.02),
        "entropy_fusion":          ("quant_math", 0.02),
        "cusum_break":             ("quant_math", 0.02),
        "copula_tail":             ("quant_math", 0.01),

        # ── ML/DL Predictions ───────────────────────────────────────────
        "gb_score":                ("ml_dl", 0.05),
        "lstm_pattern":            ("ml_dl", 0.04),
        "ann_prediction":          ("ml_dl", 0.03),
        "bilstm_attention":        ("ml_dl", 0.04),
        "transformer_pred":        ("ml_dl", 0.03),
        "regime_conditioned":      ("ml_dl", 0.03),
        "neural_ensemble":         ("ml_dl", 0.05),
        "anomaly_score":           ("ml_dl", 0.02),

        # ── NLP Alpha ───────────────────────────────────────────────────
        "finbert_sentiment":       ("nlp", 0.04),
        "news_composite":          ("nlp", 0.03),
        "social_momentum":         ("nlp", 0.02),
        "earnings_tone":           ("nlp", 0.03),
        "sec_filing_signal":       ("nlp", 0.02),
        "fed_language":            ("nlp", 0.02),

        # ── Technical Indicators ────────────────────────────────────────
        "rsi_signal":              ("technical", 0.03),
        "macd_signal":             ("technical", 0.03),
        "adx_trend":               ("technical", 0.02),
        "bb_position":             ("technical", 0.02),
        "squeeze_detect":          ("technical", 0.02),
        "v_reversal":              ("technical", 0.02),
        "chandelier_stop":         ("technical", 0.01),
        "ema_cross":               ("technical", 0.02),
        "supertrend":              ("technical", 0.01),
        "ichimoku":                ("technical", 0.01),
        "obv_signal":              ("technical", 0.01),
        "mfi_signal":              ("technical", 0.01),

        # ── Options Flow ────────────────────────────────────────────────
        "options_unusual_flow":    ("options", 0.03),
        "max_pain_signal":         ("options", 0.02),
        "put_call_ratio":          ("options", 0.02),
        "iv_skew":                 ("options", 0.02),
        "gamma_exposure":          ("options", 0.02),

        # ── Microstructure ──────────────────────────────────────────────
        "ofi_signal":              ("microstructure", 0.02),
        "vpin_toxicity":           ("microstructure", 0.02),
        "kyle_lambda":             ("microstructure", 0.01),
        "l2_spread":               ("microstructure", 0.01),
        "l2_imbalance":            ("microstructure", 0.01),
        "dark_pool_signal":        ("microstructure", 0.02),

        # ── Alternative Data ────────────────────────────────────────────
        "insider_cluster":         ("alt_data", 0.02),
        "revision_momentum":       ("alt_data", 0.02),
        "catalyst_score":          ("alt_data", 0.02),
        "earnings_surprise":       ("alt_data", 0.02),

        # ── Macro Regime ────────────────────────────────────────────────
        "vix_regime":              ("macro", 0.02),
        "yield_curve":             ("macro", 0.01),
        "credit_spread":           ("macro", 0.01),
        "dxy_momentum":            ("macro", 0.01),
    }

    def __init__(self):
        self._sources: Dict[str, AlphaSource] = {}
        for name, (cat, weight) in self.SOURCES.items():
            self._sources[name] = AlphaSource(name, cat, weight)
        self._lock = threading.Lock()

    def update_signal(self, name: str, value: float, timestamp: float = None):
        """Update a signal value."""
        if name not in self._sources:
            return
        with self._lock:
            src = self._sources[name]
            src.last_value = float(np.clip(value, -1, 1))
            src.last_ts = timestamp or time.time()

    def update_ic(self, name: str, predicted: float, actual_return: float):
        """Update information coefficient for a signal."""
        if name not in self._sources:
            return
        with self._lock:
            src = self._sources[name]
            # IC = correlation between prediction and actual
            ic_contribution = np.sign(predicted) * np.sign(actual_return)
            src.ic_history.append(ic_contribution)
            # EWM update: alpha=0.05 (20-day half-life)
            src.ic_ewm = 0.95 * src.ic_ewm + 0.05 * ic_contribution

    def get_all_signals(self) -> Dict[str, float]:
        """Get current values of all active signals."""
        with self._lock:
            return {name: src.last_value
                    for name, src in self._sources.items()
                    if src.active and src.last_ts > 0}

    def get_source_stats(self) -> List[Dict]:
        """Get stats for all sources."""
        with self._lock:
            stats = []
            for name, src in self._sources.items():
                ic_arr = np.array(list(src.ic_history)) if src.ic_history else np.array([0])
                stats.append({
                    "name": name,
                    "category": src.category,
                    "weight": round(src.weight, 4),
                    "ic_ewm": round(src.ic_ewm, 4),
                    "ic_mean": round(float(ic_arr.mean()), 4),
                    "ic_sharpe": round(float(ic_arr.mean() / max(ic_arr.std(), 0.01)), 4),
                    "active": src.active,
                    "decay_counter": src.decay_counter,
                    "has_value": src.last_ts > 0
                })
            return sorted(stats, key=lambda x: abs(x["ic_ewm"]), reverse=True)


# ═══════════════════════════════════════════════════════════════════════════════
# 2. IC-ADAPTIVE WEIGHT OPTIMIZER
# ═══════════════════════════════════════════════════════════════════════════════

class ICWeightOptimizer:
    """
    Optimizes ensemble weights based on rolling IC.
    Renaissance principle: let the data decide which signals matter.
    """

    MIN_WEIGHT = 0.005   # No signal goes below 0.5%
    MAX_WEIGHT = 0.15    # No single signal above 15%
    IC_THRESHOLD = 0.02  # Minimum IC to keep full weight
    DECAY_PENALTY = 0.5  # Weight reduction per decay period

    @classmethod
    def optimize_weights(cls, registry: AlphaRegistry,
                         regime: str = "neutral") -> Dict[str, float]:
        """
        Compute IC-adaptive weights for all signals.

        Steps:
        1. Start from base weights
        2. Scale by rolling IC (signals with higher IC get more weight)
        3. Apply correlation penalty (redundant signals downweighted)
        4. Regime adjustment (some signals work better in specific regimes)
        5. Normalize to sum = 1.0
        """
        weights = {}
        stats = {s["name"]: s for s in registry.get_source_stats()}

        for name, src_info in stats.items():
            base = registry._sources[name].weight
            ic = abs(src_info["ic_ewm"])

            # IC scaling: higher IC → higher weight
            ic_multiplier = 1.0 + max(ic - cls.IC_THRESHOLD, 0) * 10.0

            # Decay penalty
            decay = src_info["decay_counter"]
            decay_mult = max(1.0 - decay * cls.DECAY_PENALTY, 0.1)

            # Regime adjustment
            regime_mult = cls._regime_multiplier(name, src_info["category"], regime)

            w = base * ic_multiplier * decay_mult * regime_mult
            weights[name] = np.clip(w, cls.MIN_WEIGHT, cls.MAX_WEIGHT)

        # Normalize
        total = sum(weights.values())
        if total > 0:
            weights = {k: v / total for k, v in weights.items()}

        return weights

    @classmethod
    def _regime_multiplier(cls, name: str, category: str, regime: str) -> float:
        """Regime-specific weight adjustments."""
        regime_lower = regime.lower()

        # In high-vol regimes: boost risk signals, reduce momentum
        if "bear" in regime_lower or "crisis" in regime_lower or "high_vol" in regime_lower:
            if category == "macro":
                return 1.5
            if category == "microstructure":
                return 1.3
            if "momentum" in name or "trend" in name:
                return 0.7
            if "mean_revert" in name or "ou" in name:
                return 1.2

        # In bull regimes: boost momentum, reduce mean-reversion
        if "bull" in regime_lower or "risk_on" in regime_lower:
            if "momentum" in name or "trend" in name:
                return 1.3
            if "ou" in name or "mean_revert" in name:
                return 0.8

        return 1.0


# ═══════════════════════════════════════════════════════════════════════════════
# 3. CORRELATION-PENALIZED COMBINER
# ═══════════════════════════════════════════════════════════════════════════════

class CorrelationPenalizer:
    """
    Penalizes highly correlated signals to improve diversification.
    If two signals have correlation > 0.7, the weaker one is downweighted.
    """

    def __init__(self):
        self._signal_history: Dict[str, deque] = {}
        self._lock = threading.Lock()

    def record(self, signals: Dict[str, float]):
        """Record a snapshot of all signals for correlation tracking."""
        with self._lock:
            for name, value in signals.items():
                if name not in self._signal_history:
                    self._signal_history[name] = deque(maxlen=60)
                self._signal_history[name].append(value)

    def get_correlation_matrix(self) -> Dict:
        """Compute pairwise correlations between signals."""
        with self._lock:
            names = [n for n, h in self._signal_history.items() if len(h) >= 20]
            if len(names) < 2:
                return {"names": names, "matrix": [], "highly_correlated_pairs": []}

            data = np.array([list(self._signal_history[n])[-20:] for n in names])

        # Correlation matrix
        corr = np.corrcoef(data)

        # Find highly correlated pairs
        pairs = []
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                if abs(corr[i, j]) > 0.7:
                    pairs.append({
                        "signal_a": names[i],
                        "signal_b": names[j],
                        "correlation": round(float(corr[i, j]), 4)
                    })

        return {
            "names": names,
            "matrix": corr.tolist(),
            "highly_correlated_pairs": pairs
        }

    def apply_penalty(self, weights: Dict[str, float],
                      ic_stats: Dict[str, float]) -> Dict[str, float]:
        """Penalize weights of correlated signals. Keep the stronger one."""
        corr_data = self.get_correlation_matrix()
        adjusted = dict(weights)

        for pair in corr_data.get("highly_correlated_pairs", []):
            a, b = pair["signal_a"], pair["signal_b"]
            corr = abs(pair["correlation"])

            # Keep the higher-IC signal, penalize the other
            ic_a = abs(ic_stats.get(a, 0))
            ic_b = abs(ic_stats.get(b, 0))

            penalty = 0.3 * (corr - 0.7) / 0.3  # Scale penalty by correlation strength

            if ic_a >= ic_b and b in adjusted:
                adjusted[b] *= (1 - penalty)
            elif a in adjusted:
                adjusted[a] *= (1 - penalty)

        # Re-normalize
        total = sum(adjusted.values())
        if total > 0:
            adjusted = {k: v / total for k, v in adjusted.items()}

        return adjusted


# ═══════════════════════════════════════════════════════════════════════════════
# 4. SIGNAL DECAY TRACKER
# ═══════════════════════════════════════════════════════════════════════════════

class SignalDecayTracker:
    """
    Detects alpha decay — when a signal stops being predictive.
    After N consecutive weeks of IC < threshold, signal is flagged for review.
    """

    IC_WEAK_THRESHOLD = 0.02
    CONSECUTIVE_WEEKS_TO_FLAG = 4

    def __init__(self):
        self._weekly_ic: Dict[str, deque] = {}
        self._flagged: Dict[str, datetime] = {}

    def record_weekly_ic(self, name: str, weekly_ic: float):
        """Record a weekly IC observation."""
        if name not in self._weekly_ic:
            self._weekly_ic[name] = deque(maxlen=52)
        self._weekly_ic[name].append(weekly_ic)

        # Check for decay
        history = list(self._weekly_ic[name])
        if len(history) >= self.CONSECUTIVE_WEEKS_TO_FLAG:
            recent = history[-self.CONSECUTIVE_WEEKS_TO_FLAG:]
            if all(abs(ic) < self.IC_WEAK_THRESHOLD for ic in recent):
                self._flagged[name] = datetime.now()
                logger.warning(f"⚠ Signal decay detected: {name} "
                               f"(IC < {self.IC_WEAK_THRESHOLD} for "
                               f"{self.CONSECUTIVE_WEEKS_TO_FLAG} weeks)")

    def get_decaying_signals(self) -> List[Dict]:
        """Get list of signals showing decay."""
        return [
            {"name": name, "flagged_at": ts.isoformat(),
             "recent_ic": list(self._weekly_ic.get(name, []))[-4:]}
            for name, ts in self._flagged.items()
        ]

    def is_decaying(self, name: str) -> bool:
        return name in self._flagged


# ═══════════════════════════════════════════════════════════════════════════════
# 5. META-ENSEMBLE — The Master Combiner
# ═══════════════════════════════════════════════════════════════════════════════

class MedalionEnsemble:
    """
    The master ensemble that combines ALL alpha sources.

    Architecture (3 layers):
    Layer 1: Individual signals (50+ alpha sources)
    Layer 2: Category sub-ensembles (8 categories)
    Layer 3: Meta-ensemble (final prediction)

    This 3-layer architecture is why Renaissance can extract
    signal from noise — each layer reduces variance.
    """

    CATEGORY_WEIGHTS = {
        "ml_dl":          0.25,
        "quant_math":     0.20,
        "nlp":            0.15,
        "technical":      0.12,
        "options":        0.10,
        "microstructure": 0.08,
        "alt_data":       0.05,
        "macro":          0.05,
    }

    def __init__(self):
        self.registry = AlphaRegistry()
        self.ic_optimizer = ICWeightOptimizer()
        self.corr_penalizer = CorrelationPenalizer()
        self.decay_tracker = SignalDecayTracker()
        self._trade_history: deque = deque(maxlen=1000)
        self._lock = threading.Lock()
        self._last_weights: Dict[str, float] = {}
        self._weights_file = "ensemble_weights.json"
        self._load_state()

    def ingest_signals(self, signals: Dict[str, float], regime: str = "neutral"):
        """
        Ingest a snapshot of all available signals.
        Call this with each analysis cycle.
        """
        for name, value in signals.items():
            self.registry.update_signal(name, value)
        self.corr_penalizer.record(signals)

    def compute_master_signal(self, signals: Dict[str, float],
                              regime: str = "neutral") -> Dict:
        """
        THE CORE: Compute the final Medallion ensemble prediction.

        Returns:
            {
                "master_signal": float [-1, 1],
                "direction": str,
                "confidence": float [0, 1],
                "category_scores": dict,
                "top_contributors": list,
                "signal_count": int,
                "regime": str,
                "decaying_signals": list,
                "correlation_warnings": list
            }
        """
        self.ingest_signals(signals, regime)

        # Layer 1: Get IC-adaptive weights
        weights = self.ic_optimizer.optimize_weights(self.registry, regime)

        # Apply correlation penalty
        ic_stats = {s["name"]: s["ic_ewm"]
                    for s in self.registry.get_source_stats()}
        weights = self.corr_penalizer.apply_penalty(weights, ic_stats)
        self._last_weights = weights

        # Layer 2: Category sub-ensemble scores
        category_scores = {}
        category_details = {}
        for cat in self.CATEGORY_WEIGHTS:
            cat_signals = {name: signals.get(name, 0)
                           for name, src in self.registry._sources.items()
                           if src.category == cat and name in signals}
            if cat_signals:
                cat_weights = {n: weights.get(n, 0) for n in cat_signals}
                total_w = sum(cat_weights.values())
                if total_w > 0:
                    score = sum(v * cat_weights[n] / total_w
                                for n, v in cat_signals.items())
                    category_scores[cat] = round(float(score), 4)
                    category_details[cat] = {
                        "score": round(float(score), 4),
                        "n_signals": len(cat_signals),
                        "avg_ic": round(float(np.mean([
                            abs(ic_stats.get(n, 0)) for n in cat_signals
                        ])), 4)
                    }

        # Layer 3: Meta-ensemble
        master = 0.0
        total_cat_weight = 0.0
        for cat, cat_weight in self.CATEGORY_WEIGHTS.items():
            if cat in category_scores:
                master += category_scores[cat] * cat_weight
                total_cat_weight += cat_weight

        if total_cat_weight > 0:
            master /= total_cat_weight

        master = float(np.clip(master, -1, 1))

        # Confidence: how many signals agree + signal magnitude
        signal_values = [v for v in signals.values() if v != 0]
        if signal_values:
            agreement = sum(1 for v in signal_values if np.sign(v) == np.sign(master))
            confidence = (agreement / len(signal_values)) * abs(master)
        else:
            confidence = 0.0

        # Direction
        if master > 0.05:
            direction = "LONG"
        elif master < -0.05:
            direction = "SHORT"
        else:
            direction = "FLAT"

        # Top contributors
        contributions = []
        for name, value in signals.items():
            w = weights.get(name, 0)
            contrib = value * w
            if abs(contrib) > 0.001:
                contributions.append({
                    "signal": name,
                    "value": round(value, 4),
                    "weight": round(w, 4),
                    "contribution": round(contrib, 6),
                    "ic": round(ic_stats.get(name, 0), 4)
                })
        contributions.sort(key=lambda x: abs(x["contribution"]), reverse=True)

        # Warnings
        corr_data = self.corr_penalizer.get_correlation_matrix()
        decaying = self.decay_tracker.get_decaying_signals()

        return {
            "master_signal": round(master, 4),
            "direction": direction,
            "confidence": round(float(confidence), 4),
            "category_scores": category_details,
            "top_contributors": contributions[:15],
            "signal_count": len([v for v in signals.values() if v != 0]),
            "total_registered": len(self.registry._sources),
            "regime": regime,
            "decaying_signals": decaying[:5],
            "correlation_warnings": corr_data.get("highly_correlated_pairs", [])[:5],
            "weights_snapshot": {k: round(v, 4)
                                 for k, v in sorted(weights.items(),
                                                    key=lambda x: x[1], reverse=True)[:20]}
        }

    def record_trade_result(self, signals: Dict[str, float],
                            actual_return: float):
        """
        Record a completed trade for IC updating.
        Call after each trade closes.
        """
        with self._lock:
            self._trade_history.append({
                "signals": signals,
                "actual_return": actual_return,
                "ts": time.time()
            })

        # Update IC for each signal
        for name, predicted in signals.items():
            self.registry.update_ic(name, predicted, actual_return)

        self._save_state()

    def get_ensemble_stats(self) -> Dict:
        """Get comprehensive ensemble statistics."""
        source_stats = self.registry.get_source_stats()
        categories = {}
        for s in source_stats:
            cat = s["category"]
            if cat not in categories:
                categories[cat] = {"count": 0, "active": 0, "avg_ic": []}
            categories[cat]["count"] += 1
            if s["active"]:
                categories[cat]["active"] += 1
            categories[cat]["avg_ic"].append(abs(s["ic_ewm"]))

        for cat in categories:
            ics = categories[cat]["avg_ic"]
            categories[cat]["avg_ic"] = round(float(np.mean(ics)), 4) if ics else 0

        return {
            "total_sources": len(self.registry._sources),
            "active_sources": sum(1 for s in source_stats if s["active"]),
            "categories": categories,
            "top_signals_by_ic": source_stats[:10],
            "decaying_signals": self.decay_tracker.get_decaying_signals(),
            "trade_count": len(self._trade_history),
            "last_weights": {k: round(v, 4)
                             for k, v in sorted(self._last_weights.items(),
                                                key=lambda x: x[1], reverse=True)[:20]}
        }

    def _save_state(self):
        try:
            data = {
                "weights": self._last_weights,
                "updated": datetime.now().isoformat()
            }
            with open(self._weights_file, "w") as f:
                json.dump(data, f, indent=2)
        except Exception:
            pass

    def _load_state(self):
        try:
            if os.path.exists(self._weights_file):
                with open(self._weights_file) as f:
                    data = json.load(f)
                self._last_weights = data.get("weights", {})
        except Exception:
            pass


# ═══════════════════════════════════════════════════════════════════════════════
# 6. TURNOVER CONSTRAINT
# ═══════════════════════════════════════════════════════════════════════════════

class TurnoverConstraint:
    """
    Limits signal-driven portfolio churn.
    Smooths signal transitions to reduce transaction costs.
    """

    def __init__(self, max_daily_turnover: float = 0.20, smoothing: float = 0.3):
        self._prev_signal: Dict[str, float] = {}
        self.max_daily_turnover = max_daily_turnover
        self.smoothing = smoothing

    def apply(self, symbol: str, new_signal: float) -> float:
        """Apply turnover constraint to signal."""
        prev = self._prev_signal.get(symbol, 0.0)
        delta = new_signal - prev

        # Smooth: EWM blend with previous
        smoothed = prev + self.smoothing * delta

        # Cap maximum change
        if abs(delta) > self.max_daily_turnover:
            smoothed = prev + np.sign(delta) * self.max_daily_turnover

        self._prev_signal[symbol] = smoothed
        return smoothed


# ═══════════════════════════════════════════════════════════════════════════════
# MODULE SINGLETON
# ═══════════════════════════════════════════════════════════════════════════════

_medallion: Optional[MedalionEnsemble] = None
_medallion_lock = threading.Lock()

def get_medallion_ensemble() -> MedalionEnsemble:
    global _medallion
    if _medallion is None:
        with _medallion_lock:
            if _medallion is None:
                _medallion = MedalionEnsemble()
                logger.info(f"✅ Medallion Ensemble initialized: "
                            f"{len(_medallion.registry._sources)} alpha sources")
    return _medallion


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    ens = get_medallion_ensemble()

    # Simulate signals
    test_signals = {
        "hurst_exponent": 0.35,
        "ou_zscore": -0.8,
        "gb_score": 0.62,
        "lstm_pattern": 0.55,
        "finbert_sentiment": 0.4,
        "rsi_signal": -0.2,
        "macd_signal": 0.3,
        "options_unusual_flow": 0.5,
    }
    result = ens.compute_master_signal(test_signals, regime="bull")
    print(json.dumps(result, indent=2, default=str))
    print("\n✅ Ensemble Engine operational")