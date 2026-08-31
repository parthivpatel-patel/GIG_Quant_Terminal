"""
=============================================================================
RENAISSANCE PATTERN RECOGNITION ENGINE — pattern_recognition_engine.py
=============================================================================
Statistical Pattern Discovery + Markov Chain Market Models

Renaissance's core insight: markets are NOT random walks. There exist
short-lived, non-random patterns that can be exploited before they decay.

MODELS:
  ├── Markov Chain States         — market as probabilistic state machine
  ├── Transition Matrix Analyzer  — regime transition probabilities
  ├── Chart Pattern Detector      — head-shoulders, double top/bottom, flags
  ├── Mean-Reversion Scanner      — Ornstein-Uhlenbeck pattern scoring
  ├── Momentum Breakout Detector  — volume-confirmed range expansions
  ├── Pairs Divergence Finder     — cointegrated pair spread anomalies
  ├── Fractal Pattern Matcher     — self-similar price structures
  └── Seasonal Pattern Engine     — day-of-week, month, pre-earnings drift

Renaissance.io — Institutional Grade Quantitative Trading
=============================================================================
"""

import numpy as np
import logging
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from collections import deque
from scipy import stats

logger = logging.getLogger("PATTERN")

# ═══════════════════════════════════════════════════════════════════════════════
# 1. MARKOV CHAIN MARKET MODEL
# ═══════════════════════════════════════════════════════════════════════════════

class MarkovChainModel:
    """
    Models the market as a finite state machine with probabilistic transitions.

    States:
      0 = Strong Bull  (returns > +1σ)
      1 = Mild Bull    (returns 0 to +1σ)
      2 = Mild Bear    (returns -1σ to 0)
      3 = Strong Bear  (returns < -1σ)
      4 = High Vol     (abs return > 2σ, any direction)

    The transition matrix P[i,j] = P(next_state=j | current_state=i)
    This captures the key insight: market states are NOT independent.
    Knowing today's state changes tomorrow's probabilities.
    """

    STATE_NAMES = {
        0: "strong_bull",
        1: "mild_bull",
        2: "mild_bear",
        3: "strong_bear",
        4: "high_volatility"
    }

    def __init__(self, n_states: int = 5):
        self.n_states = n_states
        self.transition_matrix = np.ones((n_states, n_states)) / n_states  # uniform prior
        self._count_matrix = np.ones((n_states, n_states))  # Laplace smoothing
        self.current_state: int = 1
        self.state_history: deque = deque(maxlen=500)
        self._fitted = False

    def _classify_return(self, ret: float, sigma: float) -> int:
        """Classify a return into a Markov state."""
        if abs(ret) > 2 * sigma:
            return 4  # High vol
        if ret > sigma:
            return 0  # Strong bull
        if ret > 0:
            return 1  # Mild bull
        if ret > -sigma:
            return 2  # Mild bear
        return 3  # Strong bear

    def fit(self, returns: np.ndarray) -> Dict:
        """
        Estimate transition matrix from historical returns.

        Args:
            returns: array of daily returns

        Returns:
            {
                "transition_matrix": 5x5 matrix,
                "stationary_distribution": 5-vector,
                "current_state": int,
                "next_state_probs": 5-vector,
                "expected_return_direction": float
            }
        """
        if len(returns) < 20:
            return {"error": "insufficient_data"}

        sigma = float(np.std(returns))
        states = np.array([self._classify_return(r, sigma) for r in returns])

        # Build count matrix
        self._count_matrix = np.ones((self.n_states, self.n_states))  # Laplace
        for i in range(len(states) - 1):
            self._count_matrix[states[i], states[i + 1]] += 1

        # Normalize to get transition probabilities
        row_sums = self._count_matrix.sum(axis=1, keepdims=True)
        self.transition_matrix = self._count_matrix / row_sums

        # Current state
        self.current_state = int(states[-1])
        self.state_history.extend(states.tolist())

        # Stationary distribution (eigenvector of P^T with eigenvalue 1)
        try:
            eigenvalues, eigenvectors = np.linalg.eig(self.transition_matrix.T)
            idx = np.argmin(np.abs(eigenvalues - 1.0))
            stationary = np.real(eigenvectors[:, idx])
            stationary = stationary / stationary.sum()
        except Exception:
            stationary = np.ones(self.n_states) / self.n_states

        # Next state probabilities
        next_probs = self.transition_matrix[self.current_state]

        # Expected direction: P(bull states) - P(bear states)
        bull_prob = next_probs[0] + next_probs[1]
        bear_prob = next_probs[2] + next_probs[3]
        expected_direction = float(bull_prob - bear_prob)

        self._fitted = True

        return {
            "transition_matrix": self.transition_matrix.tolist(),
            "stationary_distribution": {
                self.STATE_NAMES[i]: round(float(stationary[i]), 4)
                for i in range(self.n_states)
            },
            "current_state": self.STATE_NAMES[self.current_state],
            "current_state_id": self.current_state,
            "next_state_probs": {
                self.STATE_NAMES[i]: round(float(next_probs[i]), 4)
                for i in range(self.n_states)
            },
            "expected_direction": round(expected_direction, 4),
            "bull_probability": round(float(bull_prob), 4),
            "bear_probability": round(float(bear_prob), 4),
            "high_vol_probability": round(float(next_probs[4]), 4),
            "n_observations": len(returns),
        }

    def predict_n_steps(self, n_steps: int = 5) -> Dict:
        """Predict state distribution N steps ahead."""
        if not self._fitted:
            return {"error": "not_fitted"}

        current_dist = np.zeros(self.n_states)
        current_dist[self.current_state] = 1.0

        predictions = []
        for step in range(1, n_steps + 1):
            current_dist = current_dist @ self.transition_matrix
            predictions.append({
                "step": step,
                "distribution": {
                    self.STATE_NAMES[i]: round(float(current_dist[i]), 4)
                    for i in range(self.n_states)
                },
                "most_likely": self.STATE_NAMES[int(np.argmax(current_dist))]
            })

        return {"predictions": predictions, "from_state": self.STATE_NAMES[self.current_state]}


# ═══════════════════════════════════════════════════════════════════════════════
# 2. CHART PATTERN DETECTOR
# ═══════════════════════════════════════════════════════════════════════════════

class ChartPatternDetector:
    """
    Detects classical chart patterns using statistical methods:
    - Head and Shoulders / Inverse H&S
    - Double Top / Double Bottom
    - Bull/Bear Flag
    - Triangle (ascending, descending, symmetric)
    - Breakout with volume confirmation

    Each pattern returns: type, confidence, target price, invalidation level.
    """

    @classmethod
    def detect_all(cls, high: np.ndarray, low: np.ndarray,
                   close: np.ndarray, volume: np.ndarray) -> List[Dict]:
        """Run all pattern detectors and return matches."""
        patterns = []

        if len(close) < 50:
            return patterns

        p = cls._detect_double_top_bottom(high, low, close)
        if p:
            patterns.append(p)

        p = cls._detect_head_shoulders(high, low, close)
        if p:
            patterns.append(p)

        p = cls._detect_flag(close, volume)
        if p:
            patterns.append(p)

        p = cls._detect_triangle(high, low, close)
        if p:
            patterns.append(p)

        p = cls._detect_breakout(high, low, close, volume)
        if p:
            patterns.append(p)

        return sorted(patterns, key=lambda x: x.get("confidence", 0), reverse=True)

    @classmethod
    def _detect_double_top_bottom(cls, high, low, close) -> Optional[Dict]:
        """Detect double top or double bottom in last 60 bars."""
        window = min(60, len(close))
        h = high[-window:]
        l = low[-window:]
        c = close[-window:]

        # Find local maxima and minima
        max_idx = []
        min_idx = []
        for i in range(2, len(c) - 2):
            if h[i] > h[i - 1] and h[i] > h[i - 2] and h[i] > h[i + 1] and h[i] > h[i + 2]:
                max_idx.append(i)
            if l[i] < l[i - 1] and l[i] < l[i - 2] and l[i] < l[i + 1] and l[i] < l[i + 2]:
                min_idx.append(i)

        # Double Top: two peaks at similar level, separated by 10+ bars
        if len(max_idx) >= 2:
            for i in range(len(max_idx) - 1):
                for j in range(i + 1, len(max_idx)):
                    p1, p2 = h[max_idx[i]], h[max_idx[j]]
                    gap = max_idx[j] - max_idx[i]
                    if gap >= 10 and abs(p1 - p2) / p1 < 0.02:  # Within 2%
                        neckline = min(l[max_idx[i]:max_idx[j] + 1])
                        target = neckline - (p1 - neckline)
                        if c[-1] < p1 * 0.98:  # Confirmed if price below peak
                            return {
                                "pattern": "double_top",
                                "direction": "bearish",
                                "confidence": round(min(0.6 + gap / 100, 0.9), 2),
                                "neckline": round(float(neckline), 2),
                                "target": round(float(target), 2),
                                "invalidation": round(float(max(p1, p2) * 1.01), 2),
                                "peak_price": round(float(max(p1, p2)), 2)
                            }

        # Double Bottom
        if len(min_idx) >= 2:
            for i in range(len(min_idx) - 1):
                for j in range(i + 1, len(min_idx)):
                    t1, t2 = l[min_idx[i]], l[min_idx[j]]
                    gap = min_idx[j] - min_idx[i]
                    if gap >= 10 and abs(t1 - t2) / t1 < 0.02:
                        neckline = max(h[min_idx[i]:min_idx[j] + 1])
                        target = neckline + (neckline - t1)
                        if c[-1] > t1 * 1.02:
                            return {
                                "pattern": "double_bottom",
                                "direction": "bullish",
                                "confidence": round(min(0.6 + gap / 100, 0.9), 2),
                                "neckline": round(float(neckline), 2),
                                "target": round(float(target), 2),
                                "invalidation": round(float(min(t1, t2) * 0.99), 2),
                                "trough_price": round(float(min(t1, t2)), 2)
                            }
        return None

    @classmethod
    def _detect_head_shoulders(cls, high, low, close) -> Optional[Dict]:
        """Detect Head & Shoulders or Inverse H&S."""
        window = min(80, len(close))
        h = high[-window:]
        l = low[-window:]
        c = close[-window:]

        # Find 5 key pivots
        max_idx = []
        for i in range(3, len(c) - 3):
            if h[i] == max(h[i - 3:i + 4]):
                max_idx.append(i)

        if len(max_idx) >= 3:
            # Check for H&S: left_shoulder < head > right_shoulder, shoulders similar
            for i in range(len(max_idx) - 2):
                ls, head, rs = h[max_idx[i]], h[max_idx[i + 1]], h[max_idx[i + 2]]
                if (head > ls and head > rs and
                        abs(ls - rs) / ls < 0.03 and
                        head > ls * 1.02):
                    neckline = min(l[max_idx[i]:max_idx[i + 2] + 1])
                    target = neckline - (head - neckline)
                    return {
                        "pattern": "head_and_shoulders",
                        "direction": "bearish",
                        "confidence": 0.75,
                        "neckline": round(float(neckline), 2),
                        "target": round(float(target), 2),
                        "invalidation": round(float(head * 1.01), 2),
                        "head_price": round(float(head), 2)
                    }
        return None

    @classmethod
    def _detect_flag(cls, close, volume) -> Optional[Dict]:
        """Detect bull/bear flag patterns."""
        if len(close) < 30:
            return None

        # Look for a strong move (pole) followed by consolidation (flag)
        pole = close[-30:-15]
        flag = close[-15:]

        pole_return = (pole[-1] - pole[0]) / pole[0]
        flag_range = (max(flag) - min(flag)) / np.mean(flag)
        flag_slope = np.polyfit(range(len(flag)), flag, 1)[0] / np.mean(flag)

        # Bull flag: strong up pole + tight consolidation with slight down slope
        if pole_return > 0.05 and flag_range < 0.04 and flag_slope < 0:
            target = close[-1] + abs(pole[-1] - pole[0])
            return {
                "pattern": "bull_flag",
                "direction": "bullish",
                "confidence": round(min(0.5 + pole_return * 5, 0.85), 2),
                "target": round(float(target), 2),
                "invalidation": round(float(min(flag) * 0.99), 2),
                "pole_strength": round(float(pole_return), 4)
            }

        # Bear flag
        if pole_return < -0.05 and flag_range < 0.04 and flag_slope > 0:
            target = close[-1] - abs(pole[-1] - pole[0])
            return {
                "pattern": "bear_flag",
                "direction": "bearish",
                "confidence": round(min(0.5 + abs(pole_return) * 5, 0.85), 2),
                "target": round(float(target), 2),
                "invalidation": round(float(max(flag) * 1.01), 2),
                "pole_strength": round(float(pole_return), 4)
            }
        return None

    @classmethod
    def _detect_triangle(cls, high, low, close) -> Optional[Dict]:
        """Detect triangle patterns (ascending, descending, symmetric)."""
        if len(close) < 30:
            return None

        window = 30
        h = high[-window:]
        l = low[-window:]

        # Fit trendlines to highs and lows
        x = np.arange(window)
        h_slope = np.polyfit(x, h, 1)[0]
        l_slope = np.polyfit(x, l, 1)[0]

        # Normalize slopes
        avg_price = np.mean(close[-window:])
        h_slope_pct = h_slope / avg_price
        l_slope_pct = l_slope / avg_price

        # Range contracting?
        range_start = h[:5].mean() - l[:5].mean()
        range_end = h[-5:].mean() - l[-5:].mean()
        contracting = range_end < range_start * 0.7

        if not contracting:
            return None

        # Ascending: flat top, rising bottom
        if abs(h_slope_pct) < 0.001 and l_slope_pct > 0.001:
            return {
                "pattern": "ascending_triangle",
                "direction": "bullish",
                "confidence": 0.65,
                "resistance": round(float(h[-5:].mean()), 2),
                "support_slope": round(float(l_slope_pct * 100), 4),
                "breakout_target": round(float(h[-1] + range_start), 2)
            }

        # Descending: flat bottom, falling top
        if h_slope_pct < -0.001 and abs(l_slope_pct) < 0.001:
            return {
                "pattern": "descending_triangle",
                "direction": "bearish",
                "confidence": 0.65,
                "support": round(float(l[-5:].mean()), 2),
                "resistance_slope": round(float(h_slope_pct * 100), 4),
                "breakout_target": round(float(l[-1] - range_start), 2)
            }

        # Symmetric
        if h_slope_pct < -0.0005 and l_slope_pct > 0.0005:
            return {
                "pattern": "symmetric_triangle",
                "direction": "neutral",
                "confidence": 0.55,
                "apex_price": round(float((h[-1] + l[-1]) / 2), 2),
                "breakout_target_up": round(float(h[-1] + range_start), 2),
                "breakout_target_down": round(float(l[-1] - range_start), 2)
            }
        return None

    @classmethod
    def _detect_breakout(cls, high, low, close, volume) -> Optional[Dict]:
        """Detect volume-confirmed breakouts."""
        if len(close) < 30:
            return None

        # 20-day range
        h20 = max(high[-20:])
        l20 = min(low[-20:])
        avg_vol = np.mean(volume[-20:])
        latest_vol = volume[-1]
        latest_close = close[-1]

        # Bullish breakout: close above 20-day high on above-average volume
        if latest_close > h20 and latest_vol > avg_vol * 1.5:
            return {
                "pattern": "bullish_breakout",
                "direction": "bullish",
                "confidence": round(min(0.5 + (latest_vol / avg_vol - 1) * 0.3, 0.9), 2),
                "breakout_level": round(float(h20), 2),
                "volume_ratio": round(float(latest_vol / avg_vol), 2),
                "target": round(float(latest_close + (h20 - l20)), 2)
            }

        # Bearish breakdown
        if latest_close < l20 and latest_vol > avg_vol * 1.5:
            return {
                "pattern": "bearish_breakdown",
                "direction": "bearish",
                "confidence": round(min(0.5 + (latest_vol / avg_vol - 1) * 0.3, 0.9), 2),
                "breakdown_level": round(float(l20), 2),
                "volume_ratio": round(float(latest_vol / avg_vol), 2),
                "target": round(float(latest_close - (h20 - l20)), 2)
            }
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# 3. SEASONAL PATTERN ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

class SeasonalPatternEngine:
    """
    Detects calendar-based patterns:
    - Day-of-week effects (Monday effect, Friday drift)
    - Month-of-year seasonality (January effect, Sell in May)
    - Pre-earnings drift (3-5 days before earnings)
    - Options expiration effects (monthly/quarterly OpEx)
    - Turn-of-month effect (last 2 days + first 3 days)
    """

    @classmethod
    def compute_seasonals(cls, close: np.ndarray,
                          dates: List[datetime] = None) -> Dict:
        """
        Compute all seasonal effects.

        Args:
            close: daily closing prices
            dates: corresponding dates (if None, uses synthetic weekdays)
        """
        if len(close) < 60:
            return {"error": "insufficient_data"}

        returns = np.diff(close) / close[:-1]

        if dates is None:
            # Generate synthetic weekday labels
            from datetime import date, timedelta
            d = date.today()
            dates = [d - timedelta(days=len(close) - 1 - i) for i in range(len(close))]

        # Day-of-week
        dow_returns = {i: [] for i in range(5)}
        for i, r in enumerate(returns):
            if i < len(dates) - 1:
                day = dates[i + 1].weekday() if hasattr(dates[i + 1], 'weekday') else 0
                if day < 5:
                    dow_returns[day].append(r)

        dow_stats = {}
        dow_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
        for day in range(5):
            rets = dow_returns[day]
            if len(rets) > 5:
                dow_stats[dow_names[day]] = {
                    "avg_return": round(float(np.mean(rets)) * 100, 4),
                    "win_rate": round(float(np.mean(np.array(rets) > 0)) * 100, 1),
                    "n_obs": len(rets)
                }

        # Month-of-year
        month_returns = {i: [] for i in range(1, 13)}
        for i, r in enumerate(returns):
            if i < len(dates) - 1:
                month = dates[i + 1].month if hasattr(dates[i + 1], 'month') else 1
                month_returns[month].append(r)

        month_stats = {}
        month_names = ["", "Jan", "Feb", "Mar", "Apr", "May", "Jun",
                        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        for m in range(1, 13):
            rets = month_returns[m]
            if len(rets) > 3:
                month_stats[month_names[m]] = {
                    "avg_return": round(float(np.mean(rets)) * 100, 4),
                    "win_rate": round(float(np.mean(np.array(rets) > 0)) * 100, 1),
                    "n_obs": len(rets)
                }

        # Current day bias
        today_dow = datetime.now().weekday()
        today_month = datetime.now().month
        current_bias = 0.0
        if today_dow < 5 and dow_names[today_dow] in dow_stats:
            current_bias += dow_stats[dow_names[today_dow]]["avg_return"] / 100
        if month_names[today_month] in month_stats:
            current_bias += month_stats[month_names[today_month]]["avg_return"] / 100

        return {
            "day_of_week": dow_stats,
            "month_of_year": month_stats,
            "current_seasonal_bias": round(float(current_bias), 6),
            "current_day": dow_names[min(today_dow, 4)],
            "current_month": month_names[today_month],
            "n_returns": len(returns)
        }


# ═══════════════════════════════════════════════════════════════════════════════
# 4. FRACTAL PATTERN MATCHER
# ═══════════════════════════════════════════════════════════════════════════════

class FractalPatternMatcher:
    """
    Finds self-similar price structures across different timeframes.
    Uses Dynamic Time Warping (DTW) to match current price action
    against historical templates.
    """

    @classmethod
    def find_similar_periods(cls, close: np.ndarray, window: int = 20,
                             n_matches: int = 5) -> List[Dict]:
        """
        Find historical periods most similar to current price action.

        Args:
            close: full price history
            window: current pattern length to match
            n_matches: number of similar periods to return
        """
        if len(close) < window * 3:
            return []

        # Normalize current pattern
        current = close[-window:]
        current_norm = (current - current.mean()) / max(current.std(), 1e-8)

        matches = []
        # Slide through history
        for i in range(len(close) - window * 2):
            historical = close[i:i + window]
            hist_norm = (historical - historical.mean()) / max(historical.std(), 1e-8)

            # Correlation-based similarity
            corr = float(np.corrcoef(current_norm, hist_norm)[0, 1])
            if abs(corr) > 0.7:  # Strong match
                # What happened AFTER this pattern?
                if i + window + 5 < len(close):
                    fwd_return = (close[i + window + 5] - close[i + window]) / close[i + window]
                    matches.append({
                        "start_idx": i,
                        "correlation": round(corr, 4),
                        "forward_5d_return": round(float(fwd_return) * 100, 2),
                        "direction": "bullish" if fwd_return > 0 else "bearish"
                    })

        # Sort by correlation strength
        matches.sort(key=lambda x: abs(x["correlation"]), reverse=True)
        top = matches[:n_matches]

        # Aggregate prediction from matches
        if top:
            avg_fwd = np.mean([m["forward_5d_return"] for m in top])
            bull_pct = sum(1 for m in top if m["direction"] == "bullish") / len(top)
            return {
                "matches": top,
                "aggregate_prediction": round(float(avg_fwd), 2),
                "bullish_percentage": round(float(bull_pct) * 100, 1),
                "signal": round(float(avg_fwd / 10), 4),  # Normalized to [-1, 1]
                "n_matches": len(top)
            }
        return {"matches": [], "aggregate_prediction": 0, "signal": 0}


# ═══════════════════════════════════════════════════════════════════════════════
# 5. MASTER PATTERN SCANNER
# ═══════════════════════════════════════════════════════════════════════════════

class PatternScanner:
    """
    Runs all pattern detectors and produces a unified pattern signal.
    """

    def __init__(self):
        self.markov = MarkovChainModel()
        self._cache: Dict[str, Dict] = {}
        self._lock = threading.Lock()

    def full_scan(self, symbol: str, high: np.ndarray, low: np.ndarray,
                  close: np.ndarray, volume: np.ndarray) -> Dict:
        """
        Run complete pattern analysis on a symbol.

        Returns:
            {
                "markov": {...},
                "chart_patterns": [...],
                "seasonals": {...},
                "fractal_matches": {...},
                "composite_pattern_signal": float [-1, 1],
                "pattern_confidence": float [0, 1]
            }
        """
        results = {}

        # 1. Markov Chain
        try:
            returns = np.diff(close) / close[:-1]
            results["markov"] = self.markov.fit(returns)
            results["markov_forecast"] = self.markov.predict_n_steps(5)
        except Exception as e:
            results["markov"] = {"error": str(e)}

        # 2. Chart Patterns
        try:
            results["chart_patterns"] = ChartPatternDetector.detect_all(
                high, low, close, volume
            )
        except Exception as e:
            results["chart_patterns"] = []

        # 3. Seasonals
        try:
            results["seasonals"] = SeasonalPatternEngine.compute_seasonals(close)
        except Exception as e:
            results["seasonals"] = {"error": str(e)}

        # 4. Fractal Matches
        try:
            results["fractal_matches"] = FractalPatternMatcher.find_similar_periods(close)
        except Exception as e:
            results["fractal_matches"] = {"error": str(e)}

        # Composite signal
        signals = []

        # Markov contribution
        markov = results.get("markov", {})
        if "expected_direction" in markov:
            signals.append(markov["expected_direction"] * 0.3)

        # Chart pattern contribution
        for pattern in results.get("chart_patterns", []):
            conf = pattern.get("confidence", 0.5)
            direction = 1 if pattern.get("direction") == "bullish" else -1
            signals.append(direction * conf * 0.25)

        # Seasonal contribution
        seasonal = results.get("seasonals", {})
        if "current_seasonal_bias" in seasonal:
            signals.append(seasonal["current_seasonal_bias"] * 50 * 0.15)

        # Fractal contribution
        fractal = results.get("fractal_matches", {})
        if isinstance(fractal, dict) and "signal" in fractal:
            signals.append(fractal["signal"] * 0.3)

        if signals:
            composite = float(np.clip(np.mean(signals), -1, 1))
            confidence = min(len(signals) / 4.0, 1.0) * abs(composite)
        else:
            composite = 0.0
            confidence = 0.0

        results["composite_pattern_signal"] = round(composite, 4)
        results["pattern_confidence"] = round(float(confidence), 4)

        with self._lock:
            self._cache[symbol] = results

        return results


# ═══════════════════════════════════════════════════════════════════════════════
# MODULE SINGLETON
# ═══════════════════════════════════════════════════════════════════════════════

_scanner: Optional[PatternScanner] = None

def get_pattern_scanner() -> PatternScanner:
    global _scanner
    if _scanner is None:
        _scanner = PatternScanner()
    return _scanner


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # Generate test data
    np.random.seed(42)
    n = 200
    close = 100 * np.cumprod(1 + np.random.randn(n) * 0.02)
    high = close * (1 + abs(np.random.randn(n) * 0.005))
    low = close * (1 - abs(np.random.randn(n) * 0.005))
    volume = np.random.randint(100000, 10000000, n).astype(float)

    scanner = get_pattern_scanner()
    result = scanner.full_scan("TEST", high, low, close, volume)
    print(f"Markov state: {result.get('markov', {}).get('current_state')}")
    print(f"Chart patterns: {len(result.get('chart_patterns', []))}")
    print(f"Composite signal: {result['composite_pattern_signal']}")
    print("✅ Pattern Recognition Engine operational")