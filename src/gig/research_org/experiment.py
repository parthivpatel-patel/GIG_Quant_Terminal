"""One causal strategy experiment and the skeptic's checks.

The series is authored and deterministic. A result on it is research evidence
about the code, never a market result, and the verdict is never positive.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from gig.backtest.metrics import performance_metrics
from gig.research.experiment import config_hash
from gig.research.multiple_testing import deflated_sharpe

HYPOTHESIS_ID = "lagged-sign-momentum-synthetic-v1"
EXPERIMENT_VERSION = 1
COST_BPS = 1.0
N_OBS = 240
SEED = 11
PLACEBO_DRAWS = 16
PLACEBO_SEED = 19


def authored_returns(n: int = N_OBS, seed: int = SEED) -> np.ndarray:
    """Deterministic fictional returns. Not a market sample."""
    rng = np.random.default_rng(seed)
    out = np.zeros(n)
    for i in range(1, n):
        out[i] = 0.05 * out[i - 1] + float(rng.normal(0.0, 0.01))
    return out


def causal_position(returns: np.ndarray) -> np.ndarray:
    """Position at t is the sign of the return that ended at t-1."""
    position = np.zeros(len(returns))
    position[1:] = np.sign(returns[:-1])
    return position


def lookahead_position(returns: np.ndarray) -> np.ndarray:
    """Same-bar sign. The skeptic must reject a result that used this."""
    return np.sign(returns)


def net_returns(returns: np.ndarray, position: np.ndarray, cost_bps: float) -> np.ndarray:
    turnover = np.abs(np.diff(position, prepend=0.0))
    return position * returns - turnover * (cost_bps / 10_000.0)


def _sharpe(returns: np.ndarray) -> float | None:
    metrics = performance_metrics(pd.Series(returns))
    value = metrics.get("sharpe")
    if value is None or not math.isfinite(float(value)):
        return None
    return float(value)


def _num(value: float | None) -> float | None:
    if value is None or not math.isfinite(value):
        return None
    return value


def uses_same_bar(returns: np.ndarray, position: np.ndarray) -> bool:
    return bool(np.all(np.isclose(position, np.sign(returns))))


def evaluate(returns: np.ndarray, position: np.ndarray, *, classification: str) -> dict[str, Any]:
    """Score one position path and let the skeptic accept or reject it."""
    net = net_returns(returns, position, COST_BPS)
    metrics = performance_metrics(pd.Series(net))
    sharpe = _sharpe(net)
    lookahead = uses_same_bar(returns, position)
    rng = np.random.default_rng(PLACEBO_SEED)
    placebo = []
    for _ in range(PLACEBO_DRAWS):
        shuffled = rng.permutation(returns)
        placebo.append(_sharpe(net_returns(shuffled, causal_position(shuffled), COST_BPS)))
    finite = [v for v in placebo if v is not None]
    dsr = deflated_sharpe(sharpe or 0.0, n_trials=1, n_obs=len(returns),
                          skew=float(metrics.get("skew") or 0.0), kurtosis=float(metrics.get("kurtosis") or 3.0))
    if lookahead:
        verdict = "REJECTED_LOOKAHEAD"
    elif classification != "OBSERVED":
        verdict = "INCONCLUSIVE"
    else:
        verdict = "INCONCLUSIVE"
    return {
        "classification": classification,
        "verdict": verdict,
        "lookahead_detected": lookahead,
        "sharpe_after_cost": sharpe,
        "total_return": _num(float(metrics.get("total_return", math.nan))),
        "ann_vol": _num(float(metrics.get("ann_vol", math.nan))),
        "max_drawdown": _num(float(metrics.get("max_drawdown", math.nan))),
        "ann_turnover": _num(float(metrics.get("ann_turnover", math.nan))),
        "placebo_sharpe_median": _num(float(np.median(finite))) if finite else None,
        "placebo_draws": PLACEBO_DRAWS,
        "deflated_sharpe": _num(dsr["dsr"]),
        "n_trials": 1,
        "cost_bps": COST_BPS,
        "n_obs": int(len(returns)),
        "economic_claim": "NONE",
        "interpretation": (
            "Rejected: the position uses the same bar it is scored on."
            if lookahead else
            "Inconclusive: the series is authored and synthetic, so the numbers are not a market result. "
            "A positive verdict is not available for this classification."
        ),
    }


def run_seeded() -> tuple[str, dict[str, Any]]:
    """Run the one approved hypothesis. Returns ``(config_hash, report)``."""
    config = {"hypothesis_id": HYPOTHESIS_ID, "version": EXPERIMENT_VERSION, "n": N_OBS,
              "seed": SEED, "cost_bps": COST_BPS, "placebo_draws": PLACEBO_DRAWS,
              "classification": "SYNTHETIC", "signal": "sign of the previous return"}
    report = evaluate(authored_returns(), causal_position(authored_returns()), classification="SYNTHETIC")
    report["hypothesis_id"] = HYPOTHESIS_ID
    report["question"] = (
        "On an authored synthetic series, does a one-day lagged sign have an after-cost "
        "result that survives a lookahead check? The organization may not call it a market edge."
    )
    report["dataset"] = "authored-deterministic-returns-v1"
    report["method"] = "Lagged sign, held one bar, cost of 1 bp of notional per position change."
    report["orders_submitted"] = 0
    report["models_promoted"] = 0
    report["code_rewritten"] = False
    report["follow_up"] = "Repeat only on admitted observations with a point-in-time cutoff. Not authorized yet."
    return config_hash(config), report
