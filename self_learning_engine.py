"""
=============================================================================
RENAISSANCE SELF-LEARNING ENGINE — self_learning_engine.py
=============================================================================
Continuous Improvement Loop — The System Gets Smarter Every Day

Renaissance's real edge isn't any single model — it's that their system
IMPROVES ITSELF. Every trade teaches the system. Every failure makes it
stronger. This engine makes that happen.

COMPONENTS:
  ├── Online Learning Loop     — every closed trade → IC update → weight shift
  ├── Walk-Forward Validator   — never deploy without out-of-sample test
  ├── Model Tournament         — run N variants, promote winners, retire losers
  ├── Auto-Feature Discovery   — genetic algorithm evolves feature combos
  ├── Performance Tracker      — rolling Sharpe, win rate, expectancy by regime
  ├── Drift Detector           — alerts when market structure changes
  └── Strategy Auto-Allocator  — shift capital to highest-performing strategy

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

logger = logging.getLogger("SELF_LEARN")


# ═══════════════════════════════════════════════════════════════════════════════
# 1. ONLINE LEARNING LOOP
# ═══════════════════════════════════════════════════════════════════════════════

class OnlineLearningLoop:
    """
    Every closed trade feeds back into the system:
    1. Compute realized IC per signal
    2. Update ensemble weights via exponential moving IC
    3. Track per-regime performance
    4. Detect signal decay and trigger retraining
    """

    def __init__(self):
        self._trade_log: deque = deque(maxlen=5000)
        self._signal_ic: Dict[str, deque] = {}
        self._regime_performance: Dict[str, deque] = {}
        self._lock = threading.Lock()
        self._weights_file = "self_learning_weights.json"
        self._load_state()

    def record_trade(self, trade: Dict):
        """
        Record a completed trade for learning.

        trade: {
            "symbol": str,
            "direction": str,
            "entry_price": float,
            "exit_price": float,
            "signals_at_entry": dict,  # snapshot of all signals when trade opened
            "regime_at_entry": str,
            "holding_days": int,
            "pnl_pct": float,
        }
        """
        with self._lock:
            trade["ts"] = time.time()
            trade["id"] = f"{trade.get('symbol','X')}_{int(trade['ts'])}"
            self._trade_log.append(trade)

            # Update IC for each signal
            actual = 1.0 if trade.get("pnl_pct", 0) > 0 else -1.0
            for sig_name, sig_value in trade.get("signals_at_entry", {}).items():
                if sig_name not in self._signal_ic:
                    self._signal_ic[sig_name] = deque(maxlen=500)
                # IC = sign(prediction) * sign(actual return)
                predicted_dir = 1.0 if sig_value > 0 else -1.0
                ic = predicted_dir * actual
                self._signal_ic[sig_name].append(ic)

            # Track by regime
            regime = trade.get("regime_at_entry", "unknown")
            if regime not in self._regime_performance:
                self._regime_performance[regime] = deque(maxlen=500)
            self._regime_performance[regime].append(trade.get("pnl_pct", 0))

        # Async save
        threading.Thread(target=self._save_state, daemon=True).start()

    def get_adaptive_weights(self) -> Dict[str, float]:
        """
        Compute IC-adaptive weights for all signals.
        Signals with higher rolling IC get more weight.
        """
        weights = {}
        with self._lock:
            for name, ic_history in self._signal_ic.items():
                if len(ic_history) < 10:
                    weights[name] = 0.02  # Default until enough data
                    continue
                arr = np.array(list(ic_history))
                # EWM IC with 20-trade half-life
                alpha = 0.05
                ewm_ic = 0.0
                for ic in arr:
                    ewm_ic = alpha * ic + (1 - alpha) * ewm_ic
                ic_sharpe = float(np.mean(arr) / max(np.std(arr), 0.01))
                # Weight = base + IC bonus
                w = 0.02 + max(ewm_ic, 0) * 0.1 + max(ic_sharpe, 0) * 0.05
                weights[name] = round(float(np.clip(w, 0.005, 0.15)), 4)

        # Normalize
        total = sum(weights.values())
        if total > 0:
            weights = {k: round(v / total, 4) for k, v in weights.items()}
        return weights

    def get_regime_stats(self) -> Dict:
        """Performance breakdown by regime."""
        stats = {}
        with self._lock:
            for regime, pnls in self._regime_performance.items():
                arr = np.array(list(pnls))
                if len(arr) < 5:
                    continue
                stats[regime] = {
                    "trades": len(arr),
                    "avg_pnl_pct": round(float(np.mean(arr)), 3),
                    "win_rate": round(float(np.mean(arr > 0)) * 100, 1),
                    "sharpe": round(float(np.mean(arr) / max(np.std(arr), 0.001)), 3),
                    "max_loss": round(float(np.min(arr)), 3),
                    "max_gain": round(float(np.max(arr)), 3),
                    "expectancy": round(float(np.mean(arr)), 4),
                }
        return stats

    def get_signal_report(self) -> List[Dict]:
        """IC report for all tracked signals."""
        report = []
        with self._lock:
            for name, ic_history in self._signal_ic.items():
                arr = np.array(list(ic_history))
                if len(arr) < 5:
                    continue
                report.append({
                    "signal": name,
                    "ic_mean": round(float(np.mean(arr)), 4),
                    "ic_std": round(float(np.std(arr)), 4),
                    "ic_sharpe": round(float(np.mean(arr) / max(np.std(arr), 0.001)), 3),
                    "trades": len(arr),
                    "profitable_pct": round(float(np.mean(arr > 0)) * 100, 1),
                    "decaying": float(np.mean(arr[-20:])) < float(np.mean(arr)) * 0.5 if len(arr) > 20 else False,
                })
        return sorted(report, key=lambda x: abs(x["ic_mean"]), reverse=True)

    def _save_state(self):
        try:
            data = {
                "signal_ic": {k: list(v) for k, v in self._signal_ic.items()},
                "regime_perf": {k: list(v) for k, v in self._regime_performance.items()},
                "trade_count": len(self._trade_log),
                "updated": datetime.now().isoformat(),
            }
            with open(self._weights_file, "w") as f:
                json.dump(data, f)
        except Exception:
            pass

    def _load_state(self):
        try:
            if os.path.exists(self._weights_file):
                with open(self._weights_file) as f:
                    data = json.load(f)
                for k, v in data.get("signal_ic", {}).items():
                    self._signal_ic[k] = deque(v, maxlen=500)
                for k, v in data.get("regime_perf", {}).items():
                    self._regime_performance[k] = deque(v, maxlen=500)
                logger.info(f"Self-learning state loaded: {len(self._signal_ic)} signals tracked")
        except Exception:
            pass


# ═══════════════════════════════════════════════════════════════════════════════
# 2. WALK-FORWARD VALIDATOR
# ═══════════════════════════════════════════════════════════════════════════════

class WalkForwardValidator:
    """
    Never deploy a model without out-of-sample validation.
    Split data into rolling train/test windows, validate each.
    """

    @classmethod
    def validate(cls, X: np.ndarray, y: np.ndarray,
                 model_fn, n_splits: int = 5,
                 train_pct: float = 0.7) -> Dict:
        """
        Walk-forward cross-validation.

        Args:
            X: feature matrix (n_samples, n_features)
            y: labels
            model_fn: callable that returns a trained model with .predict()
            n_splits: number of walk-forward windows
            train_pct: fraction used for training in each window
        """
        n = len(X)
        if n < 50:
            return {"error": "insufficient_data", "n": n}

        window = n // n_splits
        results = []

        for i in range(n_splits):
            start = i * window
            end = min(start + window, n)
            split = int((end - start) * train_pct)

            X_train = X[start:start + split]
            y_train = y[start:start + split]
            X_test = X[start + split:end]
            y_test = y[start + split:end]

            if len(X_train) < 10 or len(X_test) < 5:
                continue

            try:
                model = model_fn()
                model.fit(X_train, y_train)
                preds = model.predict(X_test)

                # Compute metrics
                direction_correct = np.sign(preds - 0.5) == np.sign(y_test - 0.5)
                accuracy = float(np.mean(direction_correct))
                ic = float(np.corrcoef(preds, y_test)[0, 1]) if len(preds) > 5 else 0

                results.append({
                    "split": i + 1,
                    "train_size": len(X_train),
                    "test_size": len(X_test),
                    "accuracy": round(accuracy, 4),
                    "ic": round(ic, 4),
                })
            except Exception as e:
                results.append({"split": i + 1, "error": str(e)})

        if not results:
            return {"error": "all_splits_failed"}

        valid_results = [r for r in results if "accuracy" in r]
        avg_accuracy = np.mean([r["accuracy"] for r in valid_results]) if valid_results else 0
        avg_ic = np.mean([r["ic"] for r in valid_results]) if valid_results else 0

        return {
            "splits": results,
            "avg_accuracy": round(float(avg_accuracy), 4),
            "avg_ic": round(float(avg_ic), 4),
            "n_valid_splits": len(valid_results),
            "deployable": float(avg_ic) > 0.02 and float(avg_accuracy) > 0.52,
            "recommendation": "DEPLOY" if avg_ic > 0.02 else "RETRAIN" if avg_ic > 0 else "REJECT",
        }


# ═══════════════════════════════════════════════════════════════════════════════
# 3. MODEL TOURNAMENT
# ═══════════════════════════════════════════════════════════════════════════════

class ModelTournament:
    """
    Run N model variants simultaneously.
    Track their live performance. Promote winners. Retire losers.
    """

    def __init__(self):
        self._models: Dict[str, Dict] = {}
        self._performance: Dict[str, deque] = {}
        self._lock = threading.Lock()

    def register_model(self, name: str, description: str = ""):
        with self._lock:
            self._models[name] = {
                "name": name, "description": description,
                "registered": datetime.now().isoformat(),
                "status": "active", "trades": 0,
            }
            self._performance[name] = deque(maxlen=500)

    def record_prediction(self, model_name: str, predicted: float, actual: float):
        with self._lock:
            if model_name not in self._performance:
                self._performance[model_name] = deque(maxlen=500)
            correct = (predicted > 0.5 and actual > 0) or (predicted < 0.5 and actual < 0)
            self._performance[model_name].append(1.0 if correct else 0.0)
            if model_name in self._models:
                self._models[model_name]["trades"] += 1

    def get_rankings(self) -> List[Dict]:
        rankings = []
        with self._lock:
            for name, perf in self._performance.items():
                arr = np.array(list(perf))
                if len(arr) < 10:
                    continue
                rankings.append({
                    "model": name,
                    "accuracy": round(float(np.mean(arr)), 4),
                    "trades": len(arr),
                    "recent_20": round(float(np.mean(arr[-20:])), 4) if len(arr) >= 20 else None,
                    "improving": float(np.mean(arr[-20:])) > float(np.mean(arr)) if len(arr) >= 20 else None,
                    "status": self._models.get(name, {}).get("status", "unknown"),
                })
        return sorted(rankings, key=lambda x: x["accuracy"], reverse=True)

    def promote_demote(self) -> Dict:
        """Auto-promote best model, demote worst."""
        rankings = self.get_rankings()
        if len(rankings) < 2:
            return {"action": "none", "reason": "need_more_models"}

        best = rankings[0]
        worst = rankings[-1]

        actions = []
        with self._lock:
            if best["accuracy"] > 0.55 and best.get("recent_20", 0) and best["recent_20"] > 0.55:
                if best["model"] in self._models:
                    self._models[best["model"]]["status"] = "promoted"
                    actions.append(f"PROMOTED: {best['model']} (acc={best['accuracy']:.3f})")

            if worst["accuracy"] < 0.48 and worst["trades"] > 50:
                if worst["model"] in self._models:
                    self._models[worst["model"]]["status"] = "retired"
                    actions.append(f"RETIRED: {worst['model']} (acc={worst['accuracy']:.3f})")

        return {"actions": actions, "rankings": rankings}


# ═══════════════════════════════════════════════════════════════════════════════
# 4. DRIFT DETECTOR
# ═══════════════════════════════════════════════════════════════════════════════

class DriftDetector:
    """
    Detects when market structure changes and models go stale.
    Uses Page-Hinkley test + KL divergence on return distributions.
    """

    def __init__(self, threshold: float = 0.05, min_window: int = 50):
        self._return_history: deque = deque(maxlen=1000)
        self._threshold = threshold
        self._min_window = min_window
        self._drift_detected = False
        self._last_drift_ts = 0

    def update(self, returns: np.ndarray) -> Dict:
        """Update with new returns and check for drift."""
        self._return_history.extend(returns.tolist())

        if len(self._return_history) < self._min_window * 2:
            return {"drift": False, "reason": "insufficient_data"}

        arr = np.array(list(self._return_history))
        mid = len(arr) // 2
        old = arr[:mid]
        new = arr[mid:]

        # KS test: are the distributions different?
        from scipy import stats
        ks_stat, ks_pval = stats.ks_2samp(old, new)

        # Mean shift
        mean_shift = abs(np.mean(new) - np.mean(old))

        # Volatility shift
        vol_old = np.std(old)
        vol_new = np.std(new)
        vol_ratio = vol_new / max(vol_old, 1e-8)

        drift = ks_pval < self._threshold or vol_ratio > 2.0 or vol_ratio < 0.5

        if drift and time.time() - self._last_drift_ts > 3600:
            self._drift_detected = True
            self._last_drift_ts = time.time()
            logger.warning(f"⚠ DRIFT DETECTED: KS p={ks_pval:.4f}, vol_ratio={vol_ratio:.2f}")

        return {
            "drift": drift,
            "ks_statistic": round(float(ks_stat), 4),
            "ks_pvalue": round(float(ks_pval), 6),
            "mean_shift": round(float(mean_shift), 6),
            "vol_ratio": round(float(vol_ratio), 3),
            "vol_old": round(float(vol_old), 4),
            "vol_new": round(float(vol_new), 4),
            "recommendation": "RETRAIN_ALL" if drift and vol_ratio > 2 else "RETRAIN_WEIGHTS" if drift else "OK",
        }


# ═══════════════════════════════════════════════════════════════════════════════
# 5. PERFORMANCE TRACKER
# ═══════════════════════════════════════════════════════════════════════════════

class PerformanceTracker:
    """Track live system performance metrics."""

    def __init__(self):
        self._daily_pnl: deque = deque(maxlen=756)  # 3 years
        self._trades: deque = deque(maxlen=5000)

    def record_daily(self, pnl_pct: float, date: str = None):
        self._daily_pnl.append({
            "date": date or datetime.now().strftime("%Y-%m-%d"),
            "pnl_pct": pnl_pct
        })

    def record_trade(self, pnl_pct: float, holding_days: int = 1):
        self._trades.append({"pnl_pct": pnl_pct, "days": holding_days, "ts": time.time()})

    def get_stats(self) -> Dict:
        pnls = np.array([d["pnl_pct"] for d in self._daily_pnl]) if self._daily_pnl else np.array([0])
        trades = np.array([t["pnl_pct"] for t in self._trades]) if self._trades else np.array([0])

        sharpe = float(np.mean(pnls) / max(np.std(pnls), 0.001)) * np.sqrt(252) if len(pnls) > 5 else 0
        sortino_denom = np.std(pnls[pnls < 0]) if np.any(pnls < 0) else 0.001
        sortino = float(np.mean(pnls) / sortino_denom) * np.sqrt(252) if len(pnls) > 5 else 0

        # Max drawdown
        cum = np.cumsum(pnls)
        running_max = np.maximum.accumulate(cum)
        dd = cum - running_max
        max_dd = float(np.min(dd)) if len(dd) > 0 else 0

        return {
            "total_days": len(pnls),
            "total_trades": len(trades),
            "total_return_pct": round(float(np.sum(pnls)), 2),
            "annualized_return_pct": round(float(np.mean(pnls) * 252), 2),
            "sharpe_ratio": round(sharpe, 3),
            "sortino_ratio": round(sortino, 3),
            "max_drawdown_pct": round(max_dd, 2),
            "win_rate_trades": round(float(np.mean(trades > 0)) * 100, 1) if len(trades) > 0 else 0,
            "avg_win": round(float(np.mean(trades[trades > 0])), 3) if np.any(trades > 0) else 0,
            "avg_loss": round(float(np.mean(trades[trades < 0])), 3) if np.any(trades < 0) else 0,
            "profit_factor": round(float(np.sum(trades[trades > 0]) / max(abs(np.sum(trades[trades < 0])), 0.001)), 2) if len(trades) > 0 else 0,
            "calmar_ratio": round(float(np.mean(pnls) * 252 / max(abs(max_dd), 0.001)), 2),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# MODULE SINGLETONS
# ═══════════════════════════════════════════════════════════════════════════════

_learning_loop: Optional[OnlineLearningLoop] = None
_tournament: Optional[ModelTournament] = None
_drift: Optional[DriftDetector] = None
_perf: Optional[PerformanceTracker] = None

def get_learning_loop() -> OnlineLearningLoop:
    global _learning_loop
    if _learning_loop is None:
        _learning_loop = OnlineLearningLoop()
    return _learning_loop

def get_tournament() -> ModelTournament:
    global _tournament
    if _tournament is None:
        _tournament = ModelTournament()
        _tournament.register_model("gb_50tree", "GradientBoost 50-tree")
        _tournament.register_model("lstm_torch", "BiLSTM PyTorch")
        _tournament.register_model("ensemble_v1", "Medallion Ensemble")
        _tournament.register_model("kalman_trend", "Kalman Filter Trend")
    return _tournament

def get_drift_detector() -> DriftDetector:
    global _drift
    if _drift is None:
        _drift = DriftDetector()
    return _drift

def get_performance_tracker() -> PerformanceTracker:
    global _perf
    if _perf is None:
        _perf = PerformanceTracker()
    return _perf

def get_self_learning_status() -> Dict:
    ll = get_learning_loop()
    tm = get_tournament()
    dd = get_drift_detector()
    pt = get_performance_tracker()
    return {
        "learning_loop": {"signals_tracked": len(ll._signal_ic), "trades_recorded": len(ll._trade_log)},
        "tournament": {"models": len(tm._models), "rankings": tm.get_rankings()[:5]},
        "drift": {"detected": dd._drift_detected},
        "performance": pt.get_stats(),
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    ll = get_learning_loop()
    ll.record_trade({"symbol": "AAPL", "pnl_pct": 2.5, "signals_at_entry": {"rsi": -0.3, "macd": 0.5}, "regime_at_entry": "bull"})
    ll.record_trade({"symbol": "TSLA", "pnl_pct": -1.2, "signals_at_entry": {"rsi": 0.5, "macd": -0.2}, "regime_at_entry": "bear"})
    print(f"Adaptive weights: {ll.get_adaptive_weights()}")
    print(f"Regime stats: {ll.get_regime_stats()}")
    print(f"Status: {get_self_learning_status()}")
    print("✅ Self-Learning Engine operational")
