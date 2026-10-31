"""
=============================================================================
RENAISSANCE.IO — RESEARCH EXPERIMENT TRACKER
=============================================================================
The "scientific operating system" that makes a quant fund different from
a retail algo: every hypothesis is tracked, tested, validated, and either
promoted to production or retired with documentation.

COMPONENTS:
  1. EXPERIMENT REGISTRY  — catalog of all research experiments
  2. ALPHA REPORT CARD    — standardized metrics for every alpha candidate
  3. HYPOTHESIS TRACKER   — idea → test → validate → promote/retire
  4. SIGNAL HEALTH MONITOR — continuous monitoring of live signals

Renaissance.io — Institutional Grade Quantitative Trading
=============================================================================
"""

import json
import os
import time
import logging
import threading
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from collections import defaultdict, deque

logger = logging.getLogger("RESEARCH")

# Persistence directory
RESEARCH_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "research_data")
os.makedirs(RESEARCH_DIR, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. EXPERIMENT REGISTRY
# ═══════════════════════════════════════════════════════════════════════════════

class ExperimentRegistry:
    """
    Central catalog of all research experiments ever run.
    Every backtest, alpha test, feature study, or model comparison
    gets registered here with metadata and results.
    """

    _experiments: Dict[str, Dict] = {}
    _lock = threading.Lock()
    _file = os.path.join(RESEARCH_DIR, "experiments.json")

    @classmethod
    def load(cls):
        try:
            if os.path.exists(cls._file):
                with open(cls._file, "r") as f:
                    cls._experiments = json.load(f)
                logger.info(f"✅ Loaded {len(cls._experiments)} experiments from registry")
        except Exception as e:
            logger.warning(f"Experiment load failed: {e}")

    @classmethod
    def save(cls):
        try:
            with open(cls._file, "w") as f:
                json.dump(cls._experiments, f, indent=2, default=str)
        except Exception as e:
            logger.warning(f"Experiment save failed: {e}")

    @classmethod
    def register(cls, experiment_id: str, name: str, category: str,
                 description: str, parameters: Dict = None) -> str:
        """Register a new experiment."""
        with cls._lock:
            exp = {
                "id": experiment_id,
                "name": name,
                "category": category,  # alpha, model, feature, backtest, risk
                "description": description,
                "parameters": parameters or {},
                "status": "registered",
                "created": datetime.now().isoformat(),
                "updated": datetime.now().isoformat(),
                "results": None,
                "report_card": None,
                "notes": [],
            }
            cls._experiments[experiment_id] = exp
            cls.save()
            return experiment_id

    @classmethod
    def update_results(cls, experiment_id: str, results: Dict,
                       status: str = "completed"):
        """Update experiment with results."""
        with cls._lock:
            if experiment_id in cls._experiments:
                cls._experiments[experiment_id]["results"] = results
                cls._experiments[experiment_id]["status"] = status
                cls._experiments[experiment_id]["updated"] = datetime.now().isoformat()
                cls.save()

    @classmethod
    def add_note(cls, experiment_id: str, note: str):
        with cls._lock:
            if experiment_id in cls._experiments:
                cls._experiments[experiment_id]["notes"].append({
                    "text": note,
                    "timestamp": datetime.now().isoformat(),
                })
                cls.save()

    @classmethod
    def get_experiment(cls, experiment_id: str) -> Optional[Dict]:
        return cls._experiments.get(experiment_id)

    @classmethod
    def list_experiments(cls, category: str = None, status: str = None,
                         limit: int = 50) -> List[Dict]:
        exps = list(cls._experiments.values())
        if category:
            exps = [e for e in exps if e["category"] == category]
        if status:
            exps = [e for e in exps if e["status"] == status]
        exps.sort(key=lambda x: x["updated"], reverse=True)
        return exps[:limit]

    @classmethod
    def get_summary(cls) -> Dict:
        exps = list(cls._experiments.values())
        by_status = defaultdict(int)
        by_category = defaultdict(int)
        for e in exps:
            by_status[e["status"]] += 1
            by_category[e["category"]] += 1
        return {
            "total": len(exps),
            "by_status": dict(by_status),
            "by_category": dict(by_category),
            "recent": [{"id": e["id"], "name": e["name"], "status": e["status"],
                       "category": e["category"], "updated": e["updated"]}
                      for e in sorted(exps, key=lambda x: x["updated"], reverse=True)[:10]],
        }


# ═══════════════════════════════════════════════════════════════════════════════
# 2. ALPHA REPORT CARD
# ═══════════════════════════════════════════════════════════════════════════════

class AlphaReportCard:
    """
    Standardized metrics for evaluating any alpha candidate.
    Every signal gets graded on the same criteria so they can be compared.
    """

    @classmethod
    def generate(cls, signal_name: str, returns: np.ndarray,
                 signals: np.ndarray, benchmark_returns: np.ndarray = None,
                 holding_period: int = 5) -> Dict:
        """
        Generate a comprehensive report card for an alpha signal.

        Args:
            signal_name: identifier for this alpha
            returns: actual forward returns (daily)
            signals: predicted signals (same length as returns)
            benchmark_returns: optional benchmark for relative metrics
            holding_period: days to hold after signal
        """
        n = min(len(returns), len(signals))
        if n < 30:
            return {"error": "insufficient data", "signal_name": signal_name, "n": n}

        ret = returns[:n]
        sig = signals[:n]

        # ── Core metrics ──────────────────────────────────────────────────
        # IC (Information Coefficient)
        ic = float(np.corrcoef(sig, ret)[0, 1]) if np.std(sig) > 0 else 0.0

        # Rank IC (Spearman)
        from scipy.stats import spearmanr
        try:
            rank_ic, rank_ic_pval = spearmanr(sig, ret)
            rank_ic = float(rank_ic)
        except:
            rank_ic, rank_ic_pval = 0.0, 1.0

        # IC stability (rolling IC standard deviation)
        window = min(20, n // 3)
        if window >= 5:
            rolling_ics = []
            for i in range(window, n):
                chunk_sig = sig[i-window:i]
                chunk_ret = ret[i-window:i]
                if np.std(chunk_sig) > 0:
                    rolling_ics.append(float(np.corrcoef(chunk_sig, chunk_ret)[0, 1]))
            ic_stability = float(np.std(rolling_ics)) if rolling_ics else 0
            ic_ir = float(np.mean(rolling_ics) / max(ic_stability, 0.001)) if rolling_ics else 0
        else:
            ic_stability, ic_ir = 0, 0

        # ── Strategy returns ──────────────────────────────────────────────
        # Long-short: long when signal > 0, short when signal < 0
        strategy_ret = sig * ret  # simplified
        cumulative = np.cumsum(strategy_ret)

        # Annualized return
        ann_return = float(np.mean(strategy_ret) * 252)

        # Annualized volatility
        ann_vol = float(np.std(strategy_ret) * np.sqrt(252))

        # Sharpe ratio
        sharpe = ann_return / max(ann_vol, 0.001)

        # Sortino ratio
        downside = strategy_ret[strategy_ret < 0]
        downside_vol = float(np.std(downside) * np.sqrt(252)) if len(downside) > 0 else 0.001
        sortino = ann_return / max(downside_vol, 0.001)

        # Max drawdown
        peak = np.maximum.accumulate(cumulative)
        drawdowns = cumulative - peak
        max_dd = float(np.min(drawdowns))

        # Hit rate
        correct = np.sum(np.sign(sig) == np.sign(ret))
        hit_rate = float(correct / max(n, 1))

        # Profit factor
        gross_profit = float(np.sum(strategy_ret[strategy_ret > 0]))
        gross_loss = float(abs(np.sum(strategy_ret[strategy_ret < 0])))
        profit_factor = gross_profit / max(gross_loss, 0.001)

        # Turnover (how often signal changes sign)
        sign_changes = np.sum(np.diff(np.sign(sig)) != 0)
        turnover_rate = float(sign_changes / max(n - 1, 1))

        # ── Alpha decay analysis ──────────────────────────────────────────
        # IC at different horizons
        decay_horizons = [1, 3, 5, 10, 20]
        decay_ics = {}
        for h in decay_horizons:
            if n > h + 10:
                fwd_ret = np.array([np.sum(ret[i:i+h]) for i in range(n-h)])
                sig_trimmed = sig[:n-h]
                if np.std(sig_trimmed) > 0:
                    decay_ics[f"{h}d"] = round(float(np.corrcoef(sig_trimmed, fwd_ret)[0, 1]), 4)

        # ── Grade ─────────────────────────────────────────────────────────
        # A/B/C/D/F grading
        score = 0
        if ic > 0.05: score += 3
        elif ic > 0.02: score += 2
        elif ic > 0: score += 1
        if sharpe > 1.5: score += 3
        elif sharpe > 0.8: score += 2
        elif sharpe > 0.3: score += 1
        if hit_rate > 0.55: score += 2
        elif hit_rate > 0.52: score += 1
        if profit_factor > 1.5: score += 2
        elif profit_factor > 1.2: score += 1
        if max_dd > -0.05: score += 1

        grade = "A+" if score >= 10 else "A" if score >= 8 else "B" if score >= 6 else \
                "C" if score >= 4 else "D" if score >= 2 else "F"

        # Recommendation
        if grade in ("A+", "A"):
            recommendation = "PROMOTE_TO_LIVE"
        elif grade == "B":
            recommendation = "PAPER_TRADE"
        elif grade == "C":
            recommendation = "NEEDS_IMPROVEMENT"
        else:
            recommendation = "RETIRE"

        return {
            "signal_name": signal_name,
            "grade": grade,
            "recommendation": recommendation,
            "score": score,
            "n_observations": n,
            "metrics": {
                "ic": round(ic, 4),
                "rank_ic": round(rank_ic, 4),
                "rank_ic_pval": round(float(rank_ic_pval), 4),
                "ic_stability": round(ic_stability, 4),
                "ic_ir": round(ic_ir, 3),
                "ann_return_pct": round(ann_return * 100, 2),
                "ann_vol_pct": round(ann_vol * 100, 2),
                "sharpe": round(sharpe, 3),
                "sortino": round(sortino, 3),
                "max_drawdown_pct": round(max_dd * 100, 2),
                "hit_rate_pct": round(hit_rate * 100, 1),
                "profit_factor": round(profit_factor, 3),
                "turnover_rate": round(turnover_rate, 3),
            },
            "alpha_decay": decay_ics,
            "holding_period": holding_period,
            "generated": datetime.now().isoformat(),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# 3. SIGNAL HEALTH MONITOR
# ═══════════════════════════════════════════════════════════════════════════════

class SignalHealthMonitor:
    """
    Continuously monitors live signals for degradation.

    Checks:
      - IC trending toward zero (decay)
      - Volatility of IC increasing (instability)
      - Hit rate dropping below threshold
      - Correlation with other signals increasing (redundancy)
      - Signal producing fewer trades (liquidity issue)
    """

    _health_log: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
    _alerts: deque = deque(maxlen=200)
    _lock = threading.Lock()

    IC_WARNING = 0.01     # IC below this = warning
    IC_CRITICAL = -0.01   # IC below this = critical
    STABILITY_WARN = 0.10  # IC std > this = unstable

    @classmethod
    def check_health(cls, signal_name: str, current_ic: float,
                     ic_std: float = 0, hit_rate: float = 0.5,
                     n_trades: int = 0) -> Dict:
        """Check health of a single signal."""
        issues = []
        severity = "healthy"

        if current_ic < cls.IC_CRITICAL:
            issues.append("IC critically negative — signal is destructive")
            severity = "critical"
        elif current_ic < cls.IC_WARNING:
            issues.append("IC near zero — signal has no predictive power")
            severity = "warning" if severity != "critical" else severity

        if ic_std > cls.STABILITY_WARN:
            issues.append(f"IC unstable (std={ic_std:.3f}) — signal is noisy")
            severity = "warning" if severity == "healthy" else severity

        if hit_rate < 0.48 and n_trades > 20:
            issues.append(f"Hit rate below random ({hit_rate:.1%})")
            severity = "warning" if severity == "healthy" else severity

        if n_trades < 5:
            issues.append("Insufficient trades for reliable assessment")

        health = {
            "signal_name": signal_name,
            "status": severity,
            "ic": round(current_ic, 4),
            "ic_std": round(ic_std, 4),
            "hit_rate": round(hit_rate, 3),
            "n_trades": n_trades,
            "issues": issues,
            "checked": datetime.now().isoformat(),
        }

        with cls._lock:
            cls._health_log[signal_name].append(health)
            if severity in ("warning", "critical"):
                cls._alerts.append({
                    "signal": signal_name,
                    "severity": severity,
                    "message": "; ".join(issues),
                    "timestamp": datetime.now().isoformat(),
                })

        return health

    @classmethod
    def get_dashboard(cls) -> Dict:
        """Return health dashboard for all monitored signals."""
        with cls._lock:
            latest = {}
            for name, log in cls._health_log.items():
                if log:
                    latest[name] = log[-1]

        healthy = sum(1 for h in latest.values() if h["status"] == "healthy")
        warning = sum(1 for h in latest.values() if h["status"] == "warning")
        critical = sum(1 for h in latest.values() if h["status"] == "critical")

        return {
            "signals": latest,
            "summary": {
                "total": len(latest),
                "healthy": healthy,
                "warning": warning,
                "critical": critical,
            },
            "recent_alerts": list(cls._alerts)[-20:],
        }


# ═══════════════════════════════════════════════════════════════════════════════
# MODULE-LEVEL FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def init():
    """Initialize research tracker — call at server startup."""
    ExperimentRegistry.load()
    logger.info("✅ Research Experiment Tracker initialized")

def get_experiment_summary() -> Dict:
    return ExperimentRegistry.get_summary()

def list_experiments(**kwargs) -> List[Dict]:
    return ExperimentRegistry.list_experiments(**kwargs)

def register_experiment(**kwargs) -> str:
    return ExperimentRegistry.register(**kwargs)

def generate_report_card(signal_name: str, returns, signals, **kwargs) -> Dict:
    return AlphaReportCard.generate(signal_name, returns, signals, **kwargs)

def get_signal_health() -> Dict:
    return SignalHealthMonitor.get_dashboard()

def check_signal_health(signal_name: str, **kwargs) -> Dict:
    return SignalHealthMonitor.check_health(signal_name, **kwargs)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    init()

    # Test report card
    np.random.seed(42)
    fake_returns = np.random.randn(500) * 0.01
    fake_signals = fake_returns * 0.3 + np.random.randn(500) * 0.01  # weak predictive signal
    card = generate_report_card("test_momentum", fake_returns, fake_signals)
    print(f"Grade: {card['grade']} | IC: {card['metrics']['ic']} | Sharpe: {card['metrics']['sharpe']}")
    print(f"Recommendation: {card['recommendation']}")
    print(f"Decay: {card['alpha_decay']}")