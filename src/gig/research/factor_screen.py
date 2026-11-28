"""Cross-sectional factor ranks from a saved return panel.

Builds uncalibrated centered ranks from point-in-time ``return_1`` panels.
No ML fit, no IC calibration, and no trading approval.
"""
from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pandas as pd

from gig.timestamps import aware_timestamp

FACTOR_SCHEMA = "cs-factor-rank-v1"
PREDICTION_KIND = "CROSS_SECTIONAL_FACTOR_CENTERED_RANK_SCORE"


def return_panel(panel_rows: list[dict[str, Any]], *, now: str) -> pd.DataFrame:
    """Pivot finite ``return_1`` observations with timestamps at or before ``now``."""
    if not isinstance(panel_rows, list) or not panel_rows:
        raise ValueError("panel_rows must be a non-empty list")
    cutoff = aware_timestamp(now)
    frame = pd.DataFrame(panel_rows)
    if not {"timestamp", "instrument_id", "return_1"}.issubset(frame.columns):
        raise ValueError("panel_rows must include timestamp, instrument_id, and return_1")
    frame = frame.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    frame = frame.loc[frame["timestamp"] <= cutoff]
    frame["return_1"] = pd.to_numeric(frame["return_1"], errors="coerce")
    frame = frame.dropna(subset=["return_1"])
    if frame.empty:
        raise ValueError("no finite return_1 observations at or before now")
    panel = frame.pivot_table(index="timestamp", columns="instrument_id",
                              values="return_1", aggfunc="last").sort_index()
    panel = panel.dropna(how="any")
    if panel.shape[0] < 5 or panel.shape[1] < 2:
        raise ValueError("aligned return panel is too short or too narrow")
    return panel


def market_neutral_residuals(panel: pd.DataFrame) -> pd.DataFrame:
    """Equal-weight cross-sectional residual returns (market-neutral research panel)."""
    market = panel.mean(axis=1)
    residual = panel.sub(market, axis=0)
    if residual.empty or not np.isfinite(residual.to_numpy()).all():
        raise ValueError("market-neutral residual panel is unavailable")
    return residual


def cross_section_factor(
    panel: pd.DataFrame,
    *,
    mode: Literal["momentum", "mean_reversion", "blend", "market_neutral_momentum"] = "blend",
    momentum_lookback: int = 20,
    reversion_lookback: int = 5,
    momentum_weight: float = 0.5,
    reversion_weight: float = 0.5,
) -> pd.Series:
    """Raw cross-sectional factor scores at the latest complete panel date."""
    if mode not in {"momentum", "mean_reversion", "blend", "market_neutral_momentum"}:
        raise ValueError("mode must be momentum, mean_reversion, blend, or market_neutral_momentum")
    if not 2 <= momentum_lookback <= 252 or not 2 <= reversion_lookback <= 60:
        raise ValueError("factor lookbacks are outside the research bound")
    need = max(momentum_lookback if mode != "mean_reversion" else 0,
               reversion_lookback if mode != "momentum" and mode != "market_neutral_momentum" else 0)
    if mode == "market_neutral_momentum":
        need = momentum_lookback
    if len(panel) < need:
        raise ValueError("aligned return panel is shorter than the factor lookback")
    working = market_neutral_residuals(panel) if mode == "market_neutral_momentum" else panel
    scores = pd.Series(0.0, index=working.columns, dtype=float)
    if mode in {"momentum", "blend", "market_neutral_momentum"}:
        momentum = working.tail(momentum_lookback).sum(axis=0)
        weight = momentum_weight if mode == "blend" else 1.0
        scores = scores.add(weight * momentum, fill_value=0.0)
    if mode in {"mean_reversion", "blend"}:
        reversion = -working.tail(reversion_lookback).sum(axis=0)
        weight = reversion_weight if mode == "blend" else 1.0
        scores = scores.add(weight * reversion, fill_value=0.0)
    if mode == "blend":
        if not (0.0 <= momentum_weight <= 1.0 and 0.0 <= reversion_weight <= 1.0):
            raise ValueError("factor weights must lie in [0, 1]")
        if abs(momentum_weight + reversion_weight - 1.0) > 1e-12:
            raise ValueError("blend weights must sum to one")
    if not np.isfinite(scores.to_numpy()).all() or scores.nunique(dropna=False) < 2:
        raise ValueError("factor scores lack cross-sectional variation")
    return scores


def centered_rank_rows(scores: pd.Series, *, decision: str) -> list[dict[str, Any]]:
    rank = scores.rank(method="average")
    centered = (rank - rank.mean()) / max(1.0, float(rank.max() - rank.min()))
    return [{
        "instrument_id": symbol,
        "timestamp": decision,
        "prediction": float(centered[symbol]),
        "prediction_kind": PREDICTION_KIND,
        "trading_approved": False,
        "confidence": {"state": "UNKNOWN", "value": None},
        "uncertainty": {"state": "NOT_CALIBRATED", "value": None},
    } for symbol in sorted(centered.index)]


def score_factor_dataset(dataset: dict[str, Any], request: dict[str, Any]) -> dict[str, Any]:
    """Build a SignalWorkspace-compatible body from a saved dataset return panel."""
    now = aware_timestamp(request["now"]).isoformat()
    panel = return_panel(dataset["panel_rows"], now=now)
    scores = cross_section_factor(
        panel,
        mode=request["mode"],
        momentum_lookback=request["momentum_lookback"],
        reversion_lookback=request["reversion_lookback"],
        momentum_weight=request["momentum_weight"],
        reversion_weight=request["reversion_weight"],
    )
    decision = panel.index[-1]
    decision_stamp = decision.isoformat() if hasattr(decision, "isoformat") else str(decision)
    rows = centered_rank_rows(scores, decision=decision_stamp)
    usable = bool(rows and max(r["prediction"] for r in rows) > min(r["prediction"] for r in rows))
    return {
        "schema_version": FACTOR_SCHEMA,
        "request": request,
        "as_of": now,
        "dataset_id": dataset["dataset_id"],
        "source": dataset["source"],
        "input_classification": dataset["classification"],
        "signal_as_of": decision_stamp,
        "run_ids": [],
        "model_ids": [],
        "rows": rows,
        "diagnostics": [{
            "factor": request["mode"],
            "momentum_lookback": request["momentum_lookback"],
            "reversion_lookback": request["reversion_lookback"],
            "observations": int(len(panel)),
            "instruments": int(panel.shape[1]),
            "confidence": {"state": "UNKNOWN", "value": None,
                           "method": "NO_PROBABILITY_CALIBRATION"},
        }],
        "scoring_state": "AVAILABLE" if usable else "UNKNOWN",
        "usable_for_portfolio": usable,
        "trading_approved": False,
        "economically_validated": False,
        "metrics": {"eligible_models": 0, "instruments": len(rows), "trading_approved": False},
        "method": (
            "MARKET_NEUTRAL_RESIDUAL_MOMENTUM_CENTERED_RANK_V1"
            if request["mode"] == "market_neutral_momentum"
            else "CROSS_SECTIONAL_RETURN_PANEL_CENTERED_RANK_V1"
        ),
        "library_versions": {"numpy": np.__version__, "pandas": pd.__version__},
        "notes": [
            "Factor ranks use the aligned saved return_1 panel with no imputation.",
            "Scores are uncalibrated relative ranks, not IC-weighted model ensembles.",
            "market_neutral_momentum subtracts the equal-weight cross-section mean each day before momentum.",
            "No broker order or trading approval is established.",
        ],
    }
