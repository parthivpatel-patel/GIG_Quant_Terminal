"""
=============================================================================
RENAISSANCE CROSS-ASSET ENGINE — cross_asset_engine.py
=============================================================================
Cross-Market Intelligence + Lead-Lag Detection

Renaissance doesn't trade in a vacuum. The Medallion Fund famously
exploited cross-asset relationships that academics hadn't published.

MODELS:
  ├── Granger Causality Scanner   — does X predict Y? (VAR-based F-test)
  ├── Lead-Lag Detector           — cross-correlation with time shifts
  ├── Correlation Regime Shifts   — DCC-style dynamic correlation
  ├── Contagion Scorer            — how fast do shocks propagate?
  ├── Risk-On/Risk-Off Indicator  — cross-asset momentum composite
  ├── Sector Rotation Signal      — relative strength across sectors
  └── Cross-Asset Alpha Signal    — unified signal from all components

TRACKED ASSETS:
  Equities : SPY, QQQ, IWM, DIA
  Bonds    : TLT, IEF, HYG, LQD
  Commodities: GLD, SLV, USO, DBA
  Currencies : UUP (DXY proxy), FXE, FXY
  Volatility : ^VIX, ^VIX9D, ^VIX3M

Renaissance.io — Institutional Grade Quantitative Trading
=============================================================================
"""

import numpy as np
import logging
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from collections import deque
from scipy import stats

logger = logging.getLogger("CROSS_ASSET")


# ═══════════════════════════════════════════════════════════════════════════════
# ASSET UNIVERSE
# ═══════════════════════════════════════════════════════════════════════════════

CROSS_ASSET_UNIVERSE = {
    "equities":     ["SPY", "QQQ", "IWM", "DIA"],
    "bonds":        ["TLT", "IEF", "HYG", "LQD"],
    "commodities":  ["GLD", "SLV", "USO", "DBA"],
    "currencies":   ["UUP", "FXE", "FXY"],
    "volatility":   ["^VIX"],
}

ALL_CROSS_ASSETS = [sym for group in CROSS_ASSET_UNIVERSE.values() for sym in group]


# ═══════════════════════════════════════════════════════════════════════════════
# 1. GRANGER CAUSALITY SCANNER
# ═══════════════════════════════════════════════════════════════════════════════

class GrangerCausalityScanner:
    """
    Tests whether returns of asset X Granger-cause returns of asset Y.
    Uses VAR model: Y_t = a + Σ(b_i * Y_{t-i}) + Σ(c_i * X_{t-i}) + e
    F-test on c_i coefficients.

    Practical: If gold returns predict tech returns 2 days ahead,
    that's actionable alpha.
    """

    @staticmethod
    def test(x_returns: np.ndarray, y_returns: np.ndarray,
             max_lag: int = 5) -> Dict:
        """
        Test if x Granger-causes y.

        Returns:
            {
                "granger_causes": bool,
                "best_lag": int,
                "f_statistic": float,
                "p_value": float,
                "direction": str,
                "predictive_power": float [0, 1]
            }
        """
        n = min(len(x_returns), len(y_returns))
        if n < max_lag * 3 + 10:
            return {"granger_causes": False, "error": "insufficient_data"}

        x = x_returns[:n]
        y = y_returns[:n]

        best_f = 0
        best_p = 1.0
        best_lag = 1

        for lag in range(1, max_lag + 1):
            # Restricted model: y ~ y_lags only
            Y = y[lag:]
            T = len(Y)

            # Build lagged matrices
            y_lags = np.column_stack([y[lag - i - 1:T + lag - i - 1] for i in range(lag)])
            x_lags = np.column_stack([x[lag - i - 1:T + lag - i - 1] for i in range(lag)])

            # Restricted: y ~ y_lags
            X_r = np.column_stack([np.ones(T), y_lags])
            try:
                beta_r = np.linalg.lstsq(X_r, Y, rcond=None)[0]
                resid_r = Y - X_r @ beta_r
                ssr_r = float(resid_r @ resid_r)
            except Exception:
                continue

            # Unrestricted: y ~ y_lags + x_lags
            X_u = np.column_stack([np.ones(T), y_lags, x_lags])
            try:
                beta_u = np.linalg.lstsq(X_u, Y, rcond=None)[0]
                resid_u = Y - X_u @ beta_u
                ssr_u = float(resid_u @ resid_u)
            except Exception:
                continue

            # F-test
            df1 = lag  # number of restrictions
            df2 = T - 2 * lag - 1
            if df2 <= 0 or ssr_u <= 0:
                continue

            f_stat = ((ssr_r - ssr_u) / df1) / (ssr_u / df2)
            try:
                p_value = 1.0 - stats.f.cdf(f_stat, df1, df2)
            except Exception:
                p_value = 1.0

            if f_stat > best_f:
                best_f = f_stat
                best_p = p_value
                best_lag = lag

        granger = best_p < 0.05
        predictive_power = min(best_f / 10.0, 1.0)

        # Direction: does positive x predict positive or negative y?
        direction = "unknown"
        if granger and best_lag <= len(x) and best_lag <= len(y):
            corr = np.corrcoef(x[:-best_lag], y[best_lag:])[0, 1]
            direction = "positive" if corr > 0 else "negative"

        return {
            "granger_causes": granger,
            "best_lag": best_lag,
            "f_statistic": round(float(best_f), 4),
            "p_value": round(float(best_p), 6),
            "direction": direction,
            "predictive_power": round(float(predictive_power), 4)
        }


# ═══════════════════════════════════════════════════════════════════════════════
# 2. LEAD-LAG DETECTOR
# ═══════════════════════════════════════════════════════════════════════════════

class LeadLagDetector:
    """
    Cross-correlation analysis to find which asset leads which.
    """

    @staticmethod
    def compute(x_returns: np.ndarray, y_returns: np.ndarray,
                max_lag: int = 10) -> Dict:
        """
        Compute cross-correlations at multiple lags.

        Positive best_lag: X leads Y by best_lag days.
        Negative best_lag: Y leads X.
        """
        n = min(len(x_returns), len(y_returns))
        if n < max_lag * 2:
            return {"error": "insufficient_data"}

        x = (x_returns[:n] - np.mean(x_returns[:n])) / max(np.std(x_returns[:n]), 1e-8)
        y = (y_returns[:n] - np.mean(y_returns[:n])) / max(np.std(y_returns[:n]), 1e-8)

        correlations = {}
        for lag in range(-max_lag, max_lag + 1):
            if lag >= 0:
                corr = float(np.corrcoef(x[:n - lag], y[lag:n])[0, 1])
            else:
                corr = float(np.corrcoef(x[-lag:n], y[:n + lag])[0, 1])
            correlations[lag] = round(corr, 4)

        best_lag = max(correlations, key=lambda k: abs(correlations[k]))
        best_corr = correlations[best_lag]

        if best_lag > 0:
            relationship = "x_leads_y"
        elif best_lag < 0:
            relationship = "y_leads_x"
        else:
            relationship = "contemporaneous"

        return {
            "best_lag": best_lag,
            "best_correlation": best_corr,
            "relationship": relationship,
            "lag_correlations": correlations,
            "signal_strength": round(abs(best_corr), 4),
            "is_significant": abs(best_corr) > 2 / np.sqrt(n)
        }


# ═══════════════════════════════════════════════════════════════════════════════
# 3. DYNAMIC CORRELATION TRACKER
# ═══════════════════════════════════════════════════════════════════════════════

class DynamicCorrelationTracker:
    """
    Tracks rolling correlations between asset pairs.
    Detects correlation regime shifts (breakdowns = crisis, new highs = crowded trade).
    """

    def __init__(self, window: int = 60):
        self.window = window
        self._history: Dict[str, deque] = {}  # "A_B" → deque of correlations
        self._lock = threading.Lock()

    def update(self, pair_key: str, x_returns: np.ndarray,
               y_returns: np.ndarray) -> Dict:
        """Update rolling correlation for a pair."""
        n = min(len(x_returns), len(y_returns), self.window)
        if n < 10:
            return {"correlation": 0, "regime": "unknown"}

        current_corr = float(np.corrcoef(x_returns[-n:], y_returns[-n:])[0, 1])

        with self._lock:
            if pair_key not in self._history:
                self._history[pair_key] = deque(maxlen=252)
            self._history[pair_key].append(current_corr)
            history = list(self._history[pair_key])

        if len(history) < 20:
            return {"correlation": round(current_corr, 4), "regime": "insufficient_history"}

        mean_corr = float(np.mean(history))
        std_corr = float(np.std(history))
        z_score = (current_corr - mean_corr) / max(std_corr, 0.01)

        # Regime detection
        if z_score > 2:
            regime = "correlation_spike"
        elif z_score < -2:
            regime = "correlation_breakdown"
        elif abs(current_corr) > 0.8:
            regime = "highly_correlated"
        elif abs(current_corr) < 0.2:
            regime = "decorrelated"
        else:
            regime = "normal"

        return {
            "correlation": round(current_corr, 4),
            "mean_correlation": round(mean_corr, 4),
            "z_score": round(z_score, 4),
            "regime": regime,
            "percentile": round(float(stats.percentileofscore(history, current_corr)), 1)
        }


# ═══════════════════════════════════════════════════════════════════════════════
# 4. RISK-ON / RISK-OFF INDICATOR
# ═══════════════════════════════════════════════════════════════════════════════

class RiskOnOffIndicator:
    """
    Cross-asset momentum composite that indicates overall market risk appetite.

    Risk-On signals: SPY↑, HYG↑, GLD↓, TLT↓, VIX↓, UUP↓
    Risk-Off signals: SPY↓, HYG↓, GLD↑, TLT↑, VIX↑, UUP↑
    """

    RISK_ON_WEIGHTS = {
        "SPY": 0.25,    # Equities up = risk on
        "QQQ": 0.15,
        "HYG": 0.15,    # High yield up = risk on
        "IWM": 0.10,    # Small caps up = risk on
    }

    RISK_OFF_WEIGHTS = {
        "TLT": 0.15,    # Treasuries up = risk off
        "GLD": 0.10,    # Gold up = risk off
        "UUP": 0.05,    # Dollar up = risk off
    }

    @classmethod
    def compute(cls, returns_5d: Dict[str, float]) -> Dict:
        """
        Compute Risk-On/Risk-Off score.

        Args:
            returns_5d: dict of symbol → 5-day return

        Returns:
            {
                "roro_score": float [-1, 1],  # +1 = full risk-on, -1 = risk-off
                "regime": str,
                "contributing": dict,
                "signal": float
            }
        """
        score = 0.0
        total_weight = 0.0
        contributing = {}

        for sym, weight in cls.RISK_ON_WEIGHTS.items():
            ret = returns_5d.get(sym, 0)
            contribution = np.sign(ret) * min(abs(ret) / 0.05, 1.0) * weight
            score += contribution
            total_weight += weight
            contributing[sym] = {"return_5d": round(ret * 100, 2),
                                  "contribution": round(float(contribution), 4),
                                  "direction": "risk_on"}

        for sym, weight in cls.RISK_OFF_WEIGHTS.items():
            ret = returns_5d.get(sym, 0)
            # Risk-off assets contribute negatively when up
            contribution = -np.sign(ret) * min(abs(ret) / 0.05, 1.0) * weight
            score += contribution
            total_weight += weight
            contributing[sym] = {"return_5d": round(ret * 100, 2),
                                  "contribution": round(float(contribution), 4),
                                  "direction": "risk_off"}

        roro = float(np.clip(score / max(total_weight, 0.01), -1, 1))

        if roro > 0.3:
            regime = "risk_on"
        elif roro < -0.3:
            regime = "risk_off"
        else:
            regime = "neutral"

        return {
            "roro_score": round(roro, 4),
            "regime": regime,
            "contributing": contributing,
            "signal": round(roro, 4)
        }


# ═══════════════════════════════════════════════════════════════════════════════
# 5. SECTOR ROTATION SIGNAL
# ═══════════════════════════════════════════════════════════════════════════════

class SectorRotationSignal:
    """
    Relative strength across sectors to detect rotation.
    When money flows from defensive to cyclical = bullish rotation.
    """

    SECTORS = {
        "XLK": "technology",
        "XLF": "financials",
        "XLV": "healthcare",
        "XLE": "energy",
        "XLI": "industrials",
        "XLP": "consumer_staples",
        "XLY": "consumer_discretionary",
        "XLU": "utilities",
        "XLRE": "real_estate",
        "XLB": "materials",
        "XLC": "communication",
    }

    CYCLICAL = {"XLK", "XLF", "XLE", "XLI", "XLY", "XLB", "XLC"}
    DEFENSIVE = {"XLV", "XLP", "XLU", "XLRE"}

    @classmethod
    def compute(cls, sector_returns_20d: Dict[str, float]) -> Dict:
        """
        Compute sector rotation signal.

        Args:
            sector_returns_20d: dict of sector ETF → 20-day return
        """
        cyclical_avg = np.mean([
            sector_returns_20d.get(s, 0) for s in cls.CYCLICAL
            if s in sector_returns_20d
        ]) if any(s in sector_returns_20d for s in cls.CYCLICAL) else 0

        defensive_avg = np.mean([
            sector_returns_20d.get(s, 0) for s in cls.DEFENSIVE
            if s in sector_returns_20d
        ]) if any(s in sector_returns_20d for s in cls.DEFENSIVE) else 0

        rotation = cyclical_avg - defensive_avg

        # Rankings
        ranked = sorted(sector_returns_20d.items(), key=lambda x: x[1], reverse=True)

        if rotation > 0.02:
            regime = "cyclical_leadership"
        elif rotation < -0.02:
            regime = "defensive_rotation"
        else:
            regime = "balanced"

        return {
            "rotation_signal": round(float(np.clip(rotation * 10, -1, 1)), 4),
            "cyclical_avg": round(float(cyclical_avg) * 100, 2),
            "defensive_avg": round(float(defensive_avg) * 100, 2),
            "spread": round(float(rotation) * 100, 2),
            "regime": regime,
            "sector_rankings": [
                {"sector": cls.SECTORS.get(sym, sym),
                 "symbol": sym,
                 "return_20d": round(ret * 100, 2)}
                for sym, ret in ranked
            ]
        }


# ═══════════════════════════════════════════════════════════════════════════════
# 6. MASTER CROSS-ASSET ANALYZER
# ═══════════════════════════════════════════════════════════════════════════════

class CrossAssetAnalyzer:
    """Unified cross-asset intelligence."""

    def __init__(self):
        self.corr_tracker = DynamicCorrelationTracker(window=60)
        self._cache: Dict = {}
        self._lock = threading.Lock()

    def full_analysis(self, returns_data: Dict[str, np.ndarray],
                      target_symbol: str = "SPY") -> Dict:
        """
        Run complete cross-asset analysis.

        Args:
            returns_data: dict of symbol → returns array
            target_symbol: the asset we want to predict
        """
        results = {}

        target_rets = returns_data.get(target_symbol)
        if target_rets is None or len(target_rets) < 20:
            return {"error": "target_symbol_missing"}

        # Granger causality scan
        granger_results = {}
        for sym, rets in returns_data.items():
            if sym == target_symbol or len(rets) < 30:
                continue
            gc = GrangerCausalityScanner.test(rets, target_rets, max_lag=5)
            if gc.get("granger_causes"):
                granger_results[sym] = gc
        results["granger_causes"] = granger_results

        # Lead-lag
        lead_lag_results = {}
        for sym, rets in returns_data.items():
            if sym == target_symbol or len(rets) < 20:
                continue
            ll = LeadLagDetector.compute(rets, target_rets, max_lag=5)
            if ll.get("is_significant"):
                lead_lag_results[sym] = ll
        results["lead_lag"] = lead_lag_results

        # Risk-On/Risk-Off
        returns_5d = {}
        for sym, rets in returns_data.items():
            if len(rets) >= 5:
                returns_5d[sym] = float(np.sum(rets[-5:]))
        results["risk_on_off"] = RiskOnOffIndicator.compute(returns_5d)

        # Sector Rotation
        returns_20d = {}
        for sym, rets in returns_data.items():
            if len(rets) >= 20:
                returns_20d[sym] = float(np.sum(rets[-20:]))
        results["sector_rotation"] = SectorRotationSignal.compute(returns_20d)

        # Composite cross-asset signal
        signals = []
        if "roro_score" in results.get("risk_on_off", {}):
            signals.append(results["risk_on_off"]["roro_score"] * 0.4)
        if "rotation_signal" in results.get("sector_rotation", {}):
            signals.append(results["sector_rotation"]["rotation_signal"] * 0.3)
        # Granger-based prediction
        for sym, gc in granger_results.items():
            if gc.get("direction") == "positive":
                last_ret = returns_data[sym][-1] if len(returns_data[sym]) > 0 else 0
                signals.append(np.sign(last_ret) * gc["predictive_power"] * 0.3)

        composite = float(np.clip(np.mean(signals), -1, 1)) if signals else 0.0

        results["cross_asset_alpha"] = {
            "signal": round(composite, 4),
            "confidence": round(min(len(signals) / 3.0, 1.0), 4),
            "n_significant_leads": len(granger_results),
            "n_significant_lags": len(lead_lag_results)
        }

        with self._lock:
            self._cache[target_symbol] = results

        return results


# ═══════════════════════════════════════════════════════════════════════════════
# MODULE SINGLETON
# ═══════════════════════════════════════════════════════════════════════════════

_analyzer: Optional[CrossAssetAnalyzer] = None

def get_cross_asset_analyzer() -> CrossAssetAnalyzer:
    global _analyzer
    if _analyzer is None:
        _analyzer = CrossAssetAnalyzer()
    return _analyzer


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    np.random.seed(42)
    data = {sym: np.random.randn(200) * 0.01 for sym in ALL_CROSS_ASSETS}
    analyzer = get_cross_asset_analyzer()
    result = analyzer.full_analysis(data, "SPY")
    print(f"Risk regime: {result.get('risk_on_off', {}).get('regime')}")
    print(f"Cross-asset alpha: {result.get('cross_asset_alpha', {}).get('signal')}")
    print("✅ Cross-Asset Engine operational")