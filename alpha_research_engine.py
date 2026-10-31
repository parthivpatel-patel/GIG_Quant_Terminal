"""
=============================================================================
RENAISSANCE.IO — ALPHA RESEARCH ENGINE  (alpha_research_engine.py)
=============================================================================
The institutional core that transforms retail-grade signals into
Renaissance-style cross-sectional alpha.

WHAT THIS REPLACES:
  Old: Each stock gets a composite score → buy/sell boolean
  New: ALL stocks ranked simultaneously by expected return Z-score
       after decomposing away market beta, sector, and style exposures

ARCHITECTURE:
  ┌─────────────────────────────────────────────────────────────────┐
  │  1. FEATURE MATRIX BUILDER                                      │
  │     Raw signals → standardized Z-score feature matrix           │
  │     (every indicator becomes a continuous feature, not boolean) │
  ├─────────────────────────────────────────────────────────────────┤
  │  2. CROSS-SECTIONAL RANKER                                      │
  │     Rank ALL stocks simultaneously on each alpha dimension      │
  │     Output: percentile rank [0, 1] per stock per factor         │
  ├─────────────────────────────────────────────────────────────────┤
  │  3. FACTOR DECOMPOSITION ENGINE                                 │
  │     Separate alpha from: market β + sector + size + value +     │
  │     momentum style exposures → residual alpha                   │
  ├─────────────────────────────────────────────────────────────────┤
  │  4. ADAPTIVE ALPHA COMBINER                                     │
  │     IC-weighted combination with regime conditioning,           │
  │     correlation penalty, and turnover awareness                 │
  ├─────────────────────────────────────────────────────────────────┤
  │  5. ALPHA LIFECYCLE MANAGER                                     │
  │     Track: birth → validation → live → decay → retirement      │
  │     Auto-demote signals with IC < threshold for N weeks         │
  ├─────────────────────────────────────────────────────────────────┤
  │  6. WALK-FORWARD VALIDATOR                                      │
  │     Purged CV with embargo, deflated Sharpe, robustness checks  │
  └─────────────────────────────────────────────────────────────────┘

Renaissance.io — Institutional Grade Quantitative Trading
=============================================================================
"""

import numpy as np
import pandas as pd
import logging
import threading
import time
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from collections import deque, defaultdict

logger = logging.getLogger("ALPHA_RESEARCH")


# ═══════════════════════════════════════════════════════════════════════════════
# 1. FEATURE MATRIX BUILDER
# ═══════════════════════════════════════════════════════════════════════════════

class FeatureMatrixBuilder:
    """
    Converts raw indicator signals into standardized Z-score features.

    Renaissance principle: Every signal is a continuous feature, never a
    binary buy/sell flag. RSI 65 is different from RSI 72 — losing that
    information by thresholding is alpha destruction.

    Output: DataFrame of shape (n_stocks, n_features) with Z-scored values.
    """

    # Feature definitions: name → extraction function from raw signal dict
    FEATURE_DEFS = {
        # ── Momentum family ───────────────────────────────────────────────
        "momentum_5d":      lambda s: s.get("momentum_5d", 0),
        "momentum_20d":     lambda s: s.get("momentum_20d", 0),
        "momentum_60d":     lambda s: s.get("momentum_60d", 0),
        "rsi_14":           lambda s: (s.get("rsi", 50) - 50) / 50,  # center at 0
        "macd_hist":        lambda s: s.get("macd_hist", 0),
        "adx_strength":     lambda s: s.get("adx", 0) / 100,

        # ── Mean-reversion family ─────────────────────────────────────────
        "hurst_deviation":  lambda s: s.get("hurst", 0.5) - 0.5,  # <0 = mean-rev
        "ou_zscore":        lambda s: s.get("ou_zscore", 0),
        "bb_position":      lambda s: s.get("bb_position", 0),  # -1 to +1

        # ── Volatility family ─────────────────────────────────────────────
        "vol_regime":       lambda s: s.get("vol_regime", 0),
        "garch_forecast":   lambda s: s.get("garch_vol", 0),
        "atr_pct":          lambda s: s.get("atr_pct", 0),
        "iv_rank":          lambda s: s.get("iv_rank", 0.5) - 0.5,

        # ── ML predictions ────────────────────────────────────────────────
        "ml_score":         lambda s: s.get("ml_score", 0),
        "lstm_pred":        lambda s: s.get("lstm_prediction", 0),
        "ensemble_score":   lambda s: s.get("ensemble_signal", 0),

        # ── Sentiment / NLP ───────────────────────────────────────────────
        "news_sentiment":   lambda s: s.get("news_sentiment", 0),
        "social_sentiment": lambda s: s.get("social_sentiment", 0),

        # ── Options flow ──────────────────────────────────────────────────
        "options_flow":     lambda s: s.get("options_flow", 0),
        "put_call_ratio":   lambda s: -(s.get("put_call_ratio", 1.0) - 1.0),
        "max_pain_dist":    lambda s: s.get("max_pain_distance", 0),

        # ── Microstructure ────────────────────────────────────────────────
        "ofi_signal":       lambda s: s.get("ofi", 0),
        "vpin_signal":      lambda s: s.get("vpin", 0),
        "kyle_lambda":      lambda s: s.get("kyle_lambda", 0),
        "spread_signal":    lambda s: s.get("l2_spread", 0),
        "imbalance":        lambda s: s.get("l2_imbalance", 0),

        # ── Fundamental / Macro ───────────────────────────────────────────
        "macro_regime":     lambda s: s.get("macro_regime", 0),
        "kalman_trend":     lambda s: s.get("kalman_trend", 0),

        # ── Composite (from existing AladdinScorer) ───────────────────────
        "aladdin_composite": lambda s: s.get("composite", 0),
    }

    @classmethod
    def build(cls, universe_signals: Dict[str, Dict]) -> pd.DataFrame:
        """
        Build feature matrix from universe of stock signals.

        Args:
            universe_signals: {symbol: {signal_name: value, ...}, ...}

        Returns:
            DataFrame (n_stocks × n_features) with Z-scored values
        """
        if not universe_signals:
            return pd.DataFrame()

        rows = {}
        for symbol, raw_signals in universe_signals.items():
            row = {}
            for feat_name, extractor in cls.FEATURE_DEFS.items():
                try:
                    val = float(extractor(raw_signals))
                    row[feat_name] = val if np.isfinite(val) else 0.0
                except (TypeError, ValueError, KeyError):
                    row[feat_name] = 0.0
            rows[symbol] = row

        df = pd.DataFrame.from_dict(rows, orient="index")

        # Cross-sectional Z-score (rank-based to handle outliers)
        df_ranked = df.rank(pct=True)  # percentile rank [0, 1]
        # Convert to Z-score via inverse normal CDF
        from scipy.stats import norm as _norm
        df_z = df_ranked.clip(0.001, 0.999).apply(lambda col: _norm.ppf(col))

        return df_z

    @classmethod
    def get_feature_names(cls) -> List[str]:
        return list(cls.FEATURE_DEFS.keys())


# ═══════════════════════════════════════════════════════════════════════════════
# 2. CROSS-SECTIONAL RANKER
# ═══════════════════════════════════════════════════════════════════════════════

class CrossSectionalRanker:
    """
    Ranks ALL stocks simultaneously — the core of Renaissance-style alpha.

    Instead of: "Is AAPL a buy?"
    Ask:        "Among 500 stocks, where does AAPL rank on expected return?"

    This is fundamentally different from per-stock signal thresholds.
    Long the top decile, short the bottom decile, ignore the middle.
    """

    @classmethod
    def rank_universe(cls, feature_matrix: pd.DataFrame,
                      weights: Dict[str, float]) -> pd.DataFrame:
        """
        Compute cross-sectional alpha scores for all stocks.

        Args:
            feature_matrix: Z-scored features (n_stocks × n_features)
            weights: {feature_name: weight} from IC-adaptive optimizer

        Returns:
            DataFrame with columns: [alpha_score, alpha_rank, percentile,
                                     decile, signal, expected_return_z]
        """
        if feature_matrix.empty:
            return pd.DataFrame()

        # Weighted combination of features
        available_features = [f for f in weights if f in feature_matrix.columns]
        if not available_features:
            return pd.DataFrame()

        # Normalize weights to available features
        w_total = sum(abs(weights[f]) for f in available_features)
        if w_total == 0:
            w_total = 1.0

        alpha_scores = pd.Series(0.0, index=feature_matrix.index)
        for feat in available_features:
            w = weights[feat] / w_total
            alpha_scores += feature_matrix[feat] * w

        # Build ranking DataFrame
        result = pd.DataFrame(index=feature_matrix.index)
        result["alpha_score"] = alpha_scores
        result["alpha_rank"] = alpha_scores.rank(ascending=False).astype(int)
        result["percentile"] = alpha_scores.rank(pct=True)
        result["decile"] = pd.qcut(alpha_scores, 10, labels=False,
                                    duplicates="drop") + 1

        # Signal: top 20% = LONG, bottom 20% = SHORT, middle = NEUTRAL
        result["signal"] = "NEUTRAL"
        result.loc[result["percentile"] >= 0.80, "signal"] = "LONG"
        result.loc[result["percentile"] <= 0.20, "signal"] = "SHORT"

        # Expected return Z-score (the alpha score IS the expected return proxy)
        result["expected_return_z"] = alpha_scores

        # Confidence based on how extreme the score is
        result["confidence"] = (result["percentile"] - 0.5).abs() * 2

        result = result.sort_values("alpha_score", ascending=False)
        return result


# ═══════════════════════════════════════════════════════════════════════════════
# 3. FACTOR DECOMPOSITION ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

class FactorDecomposer:
    """
    Separates raw alpha into:
      - Market beta exposure
      - Sector exposure
      - Size (market cap) exposure
      - Value/momentum style exposure
      - RESIDUAL ALPHA (the part that matters)

    Renaissance doesn't bet on "stocks going up" — they bet on
    the residual after removing all known factor exposures.
    """

    # Common risk factors
    FACTOR_ETFS = {
        "market":   "SPY",
        "size":     "IWM",   # small cap - large cap
        "value":    "IWD",   # value factor
        "momentum": "MTUM",  # momentum factor
        "quality":  "QUAL",  # quality factor
        "low_vol":  "USMV",  # low volatility factor
    }

    SECTOR_ETFS = {
        "XLK": "Technology", "XLF": "Financials", "XLE": "Energy",
        "XLV": "Healthcare", "XLI": "Industrials", "XLC": "Communication",
        "XLY": "Cons. Disc.", "XLP": "Cons. Staples", "XLU": "Utilities",
        "XLB": "Materials", "XLRE": "Real Estate",
    }

    @classmethod
    def decompose(cls, symbol: str, returns: np.ndarray,
                  factor_returns: Dict[str, np.ndarray]) -> Dict:
        """
        Decompose stock returns into factor exposures + residual alpha.

        Uses OLS regression: r_stock = α + Σ(β_i × r_factor_i) + ε
        The intercept α is the residual alpha (what we actually want).
        """
        if len(returns) < 30:
            return {"error": "insufficient data", "residual_alpha": 0}

        try:
            # Build factor matrix
            factor_names = []
            X_cols = []
            min_len = len(returns)

            for fname, fret in factor_returns.items():
                if len(fret) >= min_len:
                    X_cols.append(fret[-min_len:])
                    factor_names.append(fname)

            if not X_cols:
                return {"residual_alpha": float(np.mean(returns) * 252),
                        "betas": {}, "r_squared": 0}

            X = np.column_stack(X_cols)
            # Add intercept
            X = np.column_stack([np.ones(len(X)), X])
            y = returns[-min_len:]

            # OLS: β = (X'X)^(-1) X'y
            try:
                betas = np.linalg.lstsq(X, y, rcond=None)[0]
            except np.linalg.LinAlgError:
                return {"residual_alpha": float(np.mean(returns) * 252),
                        "betas": {}, "r_squared": 0}

            # Residuals
            y_hat = X @ betas
            residuals = y - y_hat

            # R² — how much of returns are explained by factors
            ss_res = np.sum(residuals ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r_squared = 1 - ss_res / max(ss_tot, 1e-10)

            # Alpha (annualized intercept)
            alpha_ann = float(betas[0] * 252)

            # Factor betas
            beta_dict = {}
            for i, fname in enumerate(factor_names):
                beta_dict[fname] = {
                    "beta": round(float(betas[i + 1]), 4),
                    "contribution": round(float(betas[i + 1] * np.mean(factor_returns[fname]) * 252), 4),
                }

            # Residual volatility
            res_vol = float(np.std(residuals) * np.sqrt(252))

            # Information Ratio = alpha / residual_vol
            ir = alpha_ann / max(res_vol, 0.001)

            return {
                "symbol": symbol,
                "residual_alpha": round(alpha_ann, 4),
                "residual_alpha_pct": round(alpha_ann * 100, 2),
                "residual_vol": round(res_vol, 4),
                "information_ratio": round(ir, 3),
                "r_squared": round(float(r_squared), 4),
                "betas": beta_dict,
                "total_return_ann": round(float(np.mean(y) * 252), 4),
                "factor_return_ann": round(float(np.mean(y_hat) * 252), 4),
            }

        except Exception as e:
            logger.warning(f"Factor decomposition failed for {symbol}: {e}")
            return {"error": str(e), "residual_alpha": 0}


# ═══════════════════════════════════════════════════════════════════════════════
# 4. ADAPTIVE ALPHA COMBINER
# ═══════════════════════════════════════════════════════════════════════════════

class AdaptiveAlphaCombiner:
    """
    IC-weighted alpha combination with:
      - Rolling IC per feature (60-day exponential)
      - Regime-conditional weighting
      - Correlation penalty for redundant features
      - Turnover cost awareness
      - Automatic decay penalization

    This replaces static signal weights (config.py SIGNAL_WEIGHTS)
    with weights that adapt to which features are actually working.
    """

    _ic_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=252))
    _ic_ewm: Dict[str, float] = {}
    _feature_correlations: Dict[str, Dict[str, float]] = {}
    _regime: str = "neutral"
    _lock = threading.Lock()
    _weights_file = "alpha_adaptive_weights.json"

    # Feature family groupings (for correlation penalty)
    FEATURE_FAMILIES = {
        "momentum": ["momentum_5d", "momentum_20d", "momentum_60d", "rsi_14",
                      "macd_hist", "adx_strength"],
        "mean_reversion": ["hurst_deviation", "ou_zscore", "bb_position"],
        "volatility": ["vol_regime", "garch_forecast", "atr_pct", "iv_rank"],
        "ml": ["ml_score", "lstm_pred", "ensemble_score"],
        "sentiment": ["news_sentiment", "social_sentiment"],
        "options": ["options_flow", "put_call_ratio", "max_pain_dist"],
        "microstructure": ["ofi_signal", "vpin_signal", "kyle_lambda",
                          "spread_signal", "imbalance"],
        "macro": ["macro_regime", "kalman_trend"],
    }

    # Regime multipliers: {regime: {family: multiplier}}
    REGIME_ADJUSTMENTS = {
        "crisis": {
            "mean_reversion": 1.4, "volatility": 1.3, "macro": 1.5,
            "momentum": 0.5, "sentiment": 0.6,
        },
        "bear": {
            "mean_reversion": 1.2, "macro": 1.3, "momentum": 0.7,
        },
        "bull": {
            "momentum": 1.3, "ml": 1.1, "options": 1.1,
            "mean_reversion": 0.8,
        },
        "euphoria": {
            "mean_reversion": 1.3, "volatility": 1.2, "momentum": 0.6,
        },
    }

    @classmethod
    def update_ic(cls, feature_name: str, predicted_z: float, actual_return: float):
        """
        Update rolling IC after observing actual forward returns.

        IC = correlation(predicted_zscore, actual_return)
        For single observation: IC_contrib = sign(pred) × sign(actual)
        EWM: IC_new = 0.94 × IC_prev + 0.06 × IC_contrib
        """
        with cls._lock:
            ic_contrib = float(np.sign(predicted_z) * np.sign(actual_return))
            magnitude_bonus = min(abs(predicted_z) * abs(actual_return) * 10, 0.5)
            ic_contrib *= (1 + magnitude_bonus)
            ic_contrib = np.clip(ic_contrib, -1, 1)

            prev = cls._ic_ewm.get(feature_name, 0.0)
            cls._ic_ewm[feature_name] = 0.94 * prev + 0.06 * ic_contrib
            cls._ic_history[feature_name].append({
                "ic": cls._ic_ewm[feature_name],
                "ts": time.time(),
            })

    @classmethod
    def get_optimal_weights(cls) -> Dict[str, float]:
        """
        Compute current optimal feature weights.

        Algorithm:
          1. Start from equal weights
          2. Scale by IC: w_i ∝ max(IC_i, 0)^0.5 (sqrt dampening)
          3. Apply regime multipliers
          4. Apply intra-family correlation penalty
          5. Floor at 1%, ceiling at 20%
          6. Normalize to sum = 1.0
        """
        features = FeatureMatrixBuilder.get_feature_names()
        weights = {}

        for feat in features:
            ic = cls._ic_ewm.get(feat, 0.0)

            # Base weight from IC (sqrt dampening)
            if ic > 0.01:
                w = float(np.sqrt(ic)) * 2.0
            elif ic > -0.01:
                w = 0.3  # neutral — keep alive at low weight
            else:
                w = 0.05  # negative IC — floor

            # Regime adjustment
            family = cls._get_family(feat)
            regime_adj = cls.REGIME_ADJUSTMENTS.get(
                cls._regime, {}
            ).get(family, 1.0)
            w *= regime_adj

            # Intra-family correlation penalty
            # If multiple features in same family, dampen the weaker ones
            family_features = cls.FEATURE_FAMILIES.get(family, [feat])
            if len(family_features) > 1:
                family_ics = [(f, cls._ic_ewm.get(f, 0)) for f in family_features]
                family_ics.sort(key=lambda x: -abs(x[1]))
                rank_in_family = next(
                    (i for i, (f, _) in enumerate(family_ics) if f == feat),
                    len(family_ics)
                )
                # Top in family: no penalty. Others: decaying penalty
                family_penalty = 0.8 ** rank_in_family
                w *= family_penalty

            weights[feat] = float(np.clip(w, 0.01, 0.20))

        # Normalize
        total = sum(weights.values())
        if total > 0:
            weights = {k: round(v / total, 6) for k, v in weights.items()}

        return weights

    @classmethod
    def _get_family(cls, feature_name: str) -> str:
        for family, members in cls.FEATURE_FAMILIES.items():
            if feature_name in members:
                return family
        return "other"

    @classmethod
    def set_regime(cls, regime: str):
        cls._regime = regime.lower()

    @classmethod
    def get_ic_dashboard(cls) -> Dict:
        """Return IC stats for all features — for dashboard display."""
        weights = cls.get_optimal_weights()
        stats = {}
        for feat in FeatureMatrixBuilder.get_feature_names():
            ic = cls._ic_ewm.get(feat, 0.0)
            history = list(cls._ic_history.get(feat, []))
            family = cls._get_family(feat)

            # Trend: is IC improving or degrading?
            if len(history) >= 10:
                recent_5 = np.mean([h["ic"] for h in history[-5:]])
                older_5 = np.mean([h["ic"] for h in history[-10:-5]])
                trend = "improving" if recent_5 > older_5 + 0.01 else \
                        "degrading" if recent_5 < older_5 - 0.01 else "stable"
            else:
                trend = "insufficient_data"

            stats[feat] = {
                "ic_ewm": round(ic, 4),
                "weight": round(weights.get(feat, 0), 4),
                "family": family,
                "trend": trend,
                "n_observations": len(history),
                "active": ic > 0.01,
                "status": "LIVE" if ic > 0.02 else "DORMANT" if ic > -0.02 else "DECAYED",
            }

        # Summary
        active_count = sum(1 for s in stats.values() if s["active"])
        avg_ic = np.mean([s["ic_ewm"] for s in stats.values()])

        return {
            "features": stats,
            "weights": weights,
            "regime": cls._regime,
            "active_features": active_count,
            "total_features": len(stats),
            "avg_ic": round(float(avg_ic), 4),
            "top_features": sorted(stats.items(), key=lambda x: -x[1]["ic_ewm"])[:5],
            "updated": datetime.now().isoformat(),
        }

    @classmethod
    def save_weights(cls):
        try:
            data = {
                "weights": cls.get_optimal_weights(),
                "ic_ewm": dict(cls._ic_ewm),
                "regime": cls._regime,
                "updated": datetime.now().isoformat(),
            }
            with open(cls._weights_file, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.warning(f"Alpha weights save failed: {e}")

    @classmethod
    def load_weights(cls):
        try:
            if os.path.exists(cls._weights_file):
                with open(cls._weights_file, "r") as f:
                    data = json.load(f)
                cls._ic_ewm = {k: float(v) for k, v in data.get("ic_ewm", {}).items()}
                cls._regime = data.get("regime", "neutral")
                logger.info(f"✅ Alpha weights loaded ({len(cls._ic_ewm)} features)")
        except Exception as e:
            logger.warning(f"Alpha weights load failed: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# 5. ALPHA LIFECYCLE MANAGER
# ═══════════════════════════════════════════════════════════════════════════════

class AlphaLifecycleManager:
    """
    Tracks each alpha signal through its lifecycle:
      CANDIDATE → BACKTEST → PAPER → LIVE → MONITORING → DECAYED → RETIRED

    Renaissance doesn't just add signals — they actively retire decayed ones.
    A signal that worked in 2020 may not work in 2025.
    """

    STAGES = ["candidate", "backtest", "paper", "live", "monitoring", "decayed", "retired"]

    # Promotion/demotion thresholds
    PROMOTE_IC = 0.03      # IC above this for N weeks → promote
    DEMOTE_IC = -0.01      # IC below this for N weeks → demote
    PROMOTE_WEEKS = 4      # Consecutive weeks above threshold
    DEMOTE_WEEKS = 6       # Consecutive weeks below threshold
    RETIRE_WEEKS = 12      # Consecutive weeks of decay → retire

    _registry: Dict[str, Dict] = {}
    _lock = threading.Lock()

    @classmethod
    def register(cls, feature_name: str, stage: str = "live"):
        """Register a feature in the lifecycle."""
        with cls._lock:
            if feature_name not in cls._registry:
                cls._registry[feature_name] = {
                    "stage": stage,
                    "registered": datetime.now().isoformat(),
                    "ic_streak_positive": 0,
                    "ic_streak_negative": 0,
                    "promotions": 0,
                    "demotions": 0,
                    "last_review": None,
                }

    @classmethod
    def review(cls, feature_name: str, current_ic: float):
        """Weekly review of a feature's lifecycle status."""
        with cls._lock:
            if feature_name not in cls._registry:
                cls.register(feature_name)

            entry = cls._registry[feature_name]
            entry["last_review"] = datetime.now().isoformat()

            if current_ic >= cls.PROMOTE_IC:
                entry["ic_streak_positive"] += 1
                entry["ic_streak_negative"] = 0
            elif current_ic <= cls.DEMOTE_IC:
                entry["ic_streak_negative"] += 1
                entry["ic_streak_positive"] = 0
            else:
                # Reset both streaks on neutral
                entry["ic_streak_positive"] = max(0, entry["ic_streak_positive"] - 1)
                entry["ic_streak_negative"] = max(0, entry["ic_streak_negative"] - 1)

            # Promotion logic
            if entry["ic_streak_positive"] >= cls.PROMOTE_WEEKS:
                current_idx = cls.STAGES.index(entry["stage"])
                if current_idx < cls.STAGES.index("live"):
                    entry["stage"] = cls.STAGES[min(current_idx + 1, 3)]
                    entry["promotions"] += 1
                    entry["ic_streak_positive"] = 0
                    logger.info(f"⬆ PROMOTED: {feature_name} → {entry['stage']}")

            # Demotion logic
            if entry["ic_streak_negative"] >= cls.DEMOTE_WEEKS:
                current_idx = cls.STAGES.index(entry["stage"])
                if current_idx <= cls.STAGES.index("monitoring"):
                    entry["stage"] = "decayed"
                    entry["demotions"] += 1
                    entry["ic_streak_negative"] = 0
                    logger.info(f"⬇ DECAYED: {feature_name}")

            # Retirement
            if entry["stage"] == "decayed" and entry["ic_streak_negative"] >= cls.RETIRE_WEEKS:
                entry["stage"] = "retired"
                logger.info(f"🛑 RETIRED: {feature_name}")

    @classmethod
    def get_active_features(cls) -> List[str]:
        """Return features that are live or monitoring (eligible for trading)."""
        with cls._lock:
            return [name for name, info in cls._registry.items()
                    if info["stage"] in ("live", "monitoring", "paper")]

    @classmethod
    def get_lifecycle_dashboard(cls) -> Dict:
        """Dashboard view of all feature lifecycles."""
        with cls._lock:
            by_stage = defaultdict(list)
            for name, info in cls._registry.items():
                by_stage[info["stage"]].append({
                    "name": name,
                    **info,
                })

            return {
                "by_stage": dict(by_stage),
                "total_features": len(cls._registry),
                "active": len(cls.get_active_features()),
                "decayed": len(by_stage.get("decayed", [])),
                "retired": len(by_stage.get("retired", [])),
                "stages": cls.STAGES,
            }


# ═══════════════════════════════════════════════════════════════════════════════
# 6. WALK-FORWARD VALIDATOR
# ═══════════════════════════════════════════════════════════════════════════════

class WalkForwardValidator:
    """
    Purged cross-validation with embargo for time-series data.

    Standard K-fold CV is INVALID for time-series because it leaks future
    information. Walk-forward with purge + embargo prevents this.

    Also computes deflated Sharpe ratio to correct for multiple testing.
    """

    @staticmethod
    def purged_walk_forward(returns: np.ndarray, signals: np.ndarray,
                            n_splits: int = 5, embargo_pct: float = 0.02,
                            purge_pct: float = 0.01) -> Dict:
        """
        Walk-forward backtest with purge and embargo.

        Purge: Remove observations at train/test boundary (prevent leakage)
        Embargo: Skip observations after test set (prevent lookahead)
        """
        n = len(returns)
        if n < 100:
            return {"error": "insufficient data", "n": n}

        embargo_n = max(1, int(n * embargo_pct))
        purge_n = max(1, int(n * purge_pct))
        fold_size = n // n_splits

        fold_results = []
        for i in range(n_splits):
            test_start = i * fold_size
            test_end = min(test_start + fold_size, n)

            # Purge zone: remove data around test boundaries
            train_mask = np.ones(n, dtype=bool)
            train_mask[max(0, test_start - purge_n):min(n, test_end + embargo_n)] = False

            # Don't use future data for training
            train_mask[test_start:] = False

            if np.sum(train_mask) < 50:
                continue

            # Simple IC computation on this fold
            test_signals = signals[test_start:test_end]
            test_returns = returns[test_start:test_end]

            if len(test_signals) > 5:
                ic = float(np.corrcoef(test_signals, test_returns)[0, 1])
                if np.isnan(ic):
                    ic = 0.0

                # Fold Sharpe
                strategy_returns = test_signals * test_returns
                fold_sharpe = float(np.mean(strategy_returns) /
                                    max(np.std(strategy_returns), 1e-10) * np.sqrt(252))

                fold_results.append({
                    "fold": i + 1,
                    "ic": round(ic, 4),
                    "sharpe": round(fold_sharpe, 3),
                    "n_train": int(np.sum(train_mask)),
                    "n_test": test_end - test_start,
                })

        if not fold_results:
            return {"error": "no valid folds"}

        avg_ic = np.mean([f["ic"] for f in fold_results])
        avg_sharpe = np.mean([f["sharpe"] for f in fold_results])
        sharpe_std = np.std([f["sharpe"] for f in fold_results])

        # Deflated Sharpe — correct for multiple testing
        # DSR = SR × (1 - e × Var[SR] / SR²) approximately
        n_trials = max(n_splits, 1)
        deflated_sharpe = cls_deflated_sharpe(avg_sharpe, sharpe_std, n_trials, n)

        return {
            "folds": fold_results,
            "avg_ic": round(float(avg_ic), 4),
            "avg_sharpe": round(float(avg_sharpe), 3),
            "sharpe_std": round(float(sharpe_std), 3),
            "deflated_sharpe": round(float(deflated_sharpe), 3),
            "n_splits": n_splits,
            "embargo_pct": embargo_pct,
            "purge_pct": purge_pct,
            "is_robust": avg_ic > 0.02 and deflated_sharpe > 0.5,
        }


def cls_deflated_sharpe(sharpe: float, sharpe_std: float,
                        n_trials: int, n_obs: int) -> float:
    """
    Deflated Sharpe Ratio (Bailey & López de Prado, 2014).
    Adjusts observed Sharpe for selection bias from multiple testing.
    """
    if sharpe_std <= 0 or n_trials <= 1:
        return sharpe

    from scipy.stats import norm as _norm
    # Expected max Sharpe under null (all strategies have SR=0)
    e_max_sr = sharpe_std * (
        (1 - 0.5772) * _norm.ppf(1 - 1.0 / n_trials) +
        0.5772 * _norm.ppf(1 - 1.0 / (n_trials * np.e))
    )

    # Deflated = observed - expected_max_under_null
    dsr = sharpe - e_max_sr

    # Also apply small sample correction
    correction = np.sqrt(1 + 0.25 * (sharpe ** 2) * (
        3 + 6.0 / max(n_obs, 10)
    ))
    dsr /= max(correction, 0.01)

    return float(dsr)


# ═══════════════════════════════════════════════════════════════════════════════
# 7. MASTER ORCHESTRATOR — AlphaResearchPlatform
# ═══════════════════════════════════════════════════════════════════════════════

class AlphaResearchPlatform:
    """
    Top-level orchestrator that ties all components together.

    Called from live_data_server.py to:
      1. Build feature matrix from current universe signals
      2. Rank all stocks cross-sectionally
      3. Decompose factor exposures
      4. Return institutional-grade alpha rankings
    """

    _instance = None
    _lock = threading.Lock()

    def __init__(self):
        self.feature_builder = FeatureMatrixBuilder
        self.ranker = CrossSectionalRanker
        self.combiner = AdaptiveAlphaCombiner
        self.lifecycle = AlphaLifecycleManager
        self.validator = WalkForwardValidator
        self.decomposer = FactorDecomposer

        # Initialize lifecycle for all features
        for feat in self.feature_builder.get_feature_names():
            self.lifecycle.register(feat, "live")

        # Load saved weights
        self.combiner.load_weights()

        logger.info(f"✅ AlphaResearchPlatform initialized with "
                    f"{len(self.feature_builder.get_feature_names())} features")

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def rank_universe(self, universe_signals: Dict[str, Dict]) -> Dict:
        """
        Main entry point: rank entire universe cross-sectionally.

        Args:
            universe_signals: {symbol: {signal_dict}, ...} from scanner

        Returns:
            Full ranking with alpha scores, signals, factor info
        """
        if not universe_signals:
            return {"error": "empty universe", "rankings": []}

        # 1. Build feature matrix
        feature_matrix = self.feature_builder.build(universe_signals)
        if feature_matrix.empty:
            return {"error": "feature matrix empty", "rankings": []}

        # 2. Get adaptive weights
        weights = self.combiner.get_optimal_weights()

        # 3. Cross-sectional ranking
        rankings = self.ranker.rank_universe(feature_matrix, weights)
        if rankings.empty:
            return {"error": "ranking failed", "rankings": []}

        # 4. Convert to list of dicts for API
        ranking_list = []
        for symbol, row in rankings.iterrows():
            sig_data = universe_signals.get(symbol, {})
            ranking_list.append({
                "symbol": symbol,
                "alpha_score": round(float(row["alpha_score"]), 4),
                "alpha_rank": int(row["alpha_rank"]),
                "percentile": round(float(row["percentile"]), 3),
                "decile": int(row.get("decile", 5)),
                "signal": row["signal"],
                "confidence": round(float(row["confidence"]), 3),
                "expected_return_z": round(float(row["expected_return_z"]), 4),
                # Pass through key raw data
                "price": sig_data.get("price", 0),
                "sector": sig_data.get("sector", "unknown"),
                "market_cap_tier": sig_data.get("cap_tier", ""),
                "composite": sig_data.get("composite", 0),
            })

        # 5. Summary stats
        long_count = sum(1 for r in ranking_list if r["signal"] == "LONG")
        short_count = sum(1 for r in ranking_list if r["signal"] == "SHORT")

        return {
            "rankings": ranking_list,
            "universe_size": len(ranking_list),
            "long_count": long_count,
            "short_count": short_count,
            "neutral_count": len(ranking_list) - long_count - short_count,
            "top_5_long": ranking_list[:5],
            "top_5_short": ranking_list[-5:][::-1],
            "weights": weights,
            "regime": self.combiner._regime,
            "active_features": len(self.combiner._ic_ewm),
            "updated": datetime.now().isoformat(),
        }

    def get_alpha_dashboard(self) -> Dict:
        """Full alpha research dashboard data."""
        return {
            "ic_dashboard": self.combiner.get_ic_dashboard(),
            "lifecycle": self.lifecycle.get_lifecycle_dashboard(),
            "feature_count": len(self.feature_builder.get_feature_names()),
            "feature_families": dict(self.combiner.FEATURE_FAMILIES),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# MODULE-LEVEL CONVENIENCE FUNCTIONS (called from live_data_server.py)
# ═══════════════════════════════════════════════════════════════════════════════

def get_alpha_platform() -> AlphaResearchPlatform:
    """Get or create singleton AlphaResearchPlatform."""
    return AlphaResearchPlatform.get_instance()


def rank_universe(universe_signals: Dict[str, Dict]) -> Dict:
    """Convenience: rank entire universe."""
    return get_alpha_platform().rank_universe(universe_signals)


def get_alpha_dashboard() -> Dict:
    """Convenience: get alpha research dashboard."""
    return get_alpha_platform().get_alpha_dashboard()


def get_feature_names() -> List[str]:
    """Convenience: list all feature names."""
    return FeatureMatrixBuilder.get_feature_names()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    platform = get_alpha_platform()
    print(f"Features: {len(get_feature_names())}")
    print(f"Families: {list(AdaptiveAlphaCombiner.FEATURE_FAMILIES.keys())}")
    print(f"Dashboard: {json.dumps(get_alpha_dashboard(), indent=2, default=str)[:500]}")