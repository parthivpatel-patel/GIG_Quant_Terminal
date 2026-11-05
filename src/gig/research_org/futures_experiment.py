"""Futures time-series momentum on an authored excess-return series.

The strategy class is the existing ``FuturesTSMOM``. The series is generated,
not observed. A loss or a gain on it is evidence about this code path, not
about a futures market.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from gig.backtest.costs import TransactionCostModel
from gig.backtest.metrics import performance_metrics
from gig.research.experiment import config_hash
from gig.research.multiple_testing import deflated_sharpe
from gig.strategies.futures_tsmom import FuturesTSMOM

HYPOTHESIS_ID = "futures-tsmom-authored-excess-v1"
DATASET_ID = "authored-futures-excess-v1"
EXPERIMENT_VERSION = 1
SEED = 23
N_OBS = 600
START = "2016-01-04"
LOC = 0.0
SCALE = 0.01
HOLDOUT = 63
PLACEBO_DRAWS = 16
PLACEBO_SEED = 29
LOOKBACK = 252
VOL_LOOKBACK = 60
WIDER_SPREAD_BPS = 20.0


def excess_returns() -> pd.Series:
    """Deterministic fictional excess returns. Not a traded contract."""
    index = pd.bdate_range(START, periods=N_OBS)
    values = np.random.default_rng(SEED).normal(LOC, SCALE, N_OBS)
    return pd.Series(values, index=index, name="AUTHORED_EXCESS")


def _num(value: object) -> float | None:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def _pack(metrics: dict[str, float]) -> dict[str, float | None]:
    keys = ("n_obs", "total_return", "ann_return", "ann_vol", "sharpe", "sortino",
            "max_drawdown", "ann_turnover")
    return {key: _num(metrics.get(key)) for key in keys}


def _sharpe(metrics: dict[str, float | None]) -> float | None:
    return _num(metrics.get("sharpe"))


def _leakage(series: pd.Series) -> bool:
    """True when changing the last return moves an earlier weight."""
    original = FuturesTSMOM().run(series).weights.iloc[:-1]
    altered = series.copy()
    altered.iloc[-1] = float(altered.iloc[-1]) + 1.0
    shifted = FuturesTSMOM().run(altered).weights.iloc[:-1]
    if not original.index.equals(shifted.index) or list(original.columns) != list(shifted.columns):
        return True
    return not np.allclose(original.to_numpy(dtype=float), shifted.to_numpy(dtype=float), equal_nan=True)


def _curve(equity: pd.Series) -> list[dict[str, float | str]]:
    step = max(1, len(equity) // 60)
    sampled = equity.iloc[::step]
    if sampled.index[-1] != equity.index[-1]:
        sampled = pd.concat([sampled, equity.iloc[[-1]]])
    return [{"date": stamp.date().isoformat(), "equity": float(value)} for stamp, value in sampled.items()]


def run_futures() -> tuple[str, dict[str, Any]]:
    """Run the default strategy and the skeptic's checks. Returns ``(config_hash, report)``."""
    series = excess_returns()
    config = {
        "hypothesis_id": HYPOTHESIS_ID, "version": EXPERIMENT_VERSION, "dataset": DATASET_ID,
        "seed": SEED, "n": N_OBS, "loc": LOC, "scale": SCALE, "start": START,
        "strategy": "FuturesTSMOM", "lookback": LOOKBACK, "vol_lookback": VOL_LOOKBACK,
        "holdout_sessions": HOLDOUT, "classification": "SYNTHETIC",
        "cost": "TransactionCostModel defaults",
    }
    result = FuturesTSMOM(lookback=LOOKBACK, vol_lookback=VOL_LOOKBACK).run(series)
    metrics = _pack(result.metrics)
    if metrics["n_obs"] is None or metrics["n_obs"] < HOLDOUT:
        raise RuntimeError("futures experiment produced too few scored sessions")
    holdout = result.returns.iloc[-HOLDOUT:]
    hold_metrics = _pack(performance_metrics(holdout, turnover=result.turnover.reindex(holdout.index)))
    benchmark = _pack(performance_metrics(series.reindex(result.returns.index)))
    wider = _pack(FuturesTSMOM(lookback=LOOKBACK, vol_lookback=VOL_LOOKBACK,
                               cost_model=TransactionCostModel(spread_bps=WIDER_SPREAD_BPS)).run(series).metrics)
    shorter = _pack(FuturesTSMOM(lookback=126, vol_lookback=VOL_LOOKBACK).run(series).metrics)
    leakage = _leakage(series)
    rng = np.random.default_rng(PLACEBO_SEED)
    placebo: list[float] = []
    for _ in range(PLACEBO_DRAWS):
        shuffled = pd.Series(rng.permutation(series.to_numpy()), index=series.index)
        placebo_sharpe = _sharpe(_pack(FuturesTSMOM().run(shuffled).metrics))
        if placebo_sharpe is not None:
            placebo.append(placebo_sharpe)
    sharpe = _sharpe(metrics)
    dsr = deflated_sharpe(sharpe or 0.0, n_trials=1, n_obs=int(metrics["n_obs"] or 0),
                          skew=0.0, kurtosis=3.0)
    primary_sign = 0 if sharpe is None or sharpe == 0 else (1 if sharpe > 0 else -1)
    alt = _sharpe(shorter)
    alt_sign = 0 if alt is None or alt == 0 else (1 if alt > 0 else -1)
    sign_flip = primary_sign != 0 and alt_sign != 0 and primary_sign != alt_sign
    total = metrics["total_return"]
    if leakage:
        verdict = "REJECTED_LOOKAHEAD"
        interpretation = "Rejected: changing the last return moved an earlier weight."
    else:
        verdict = "INCONCLUSIVE"
        direction = "negative" if total is not None and total < 0 else "not negative"
        interpretation = (
            f"Inconclusive. The after-cost total return on this authored series is {direction}. "
            "The series is synthetic noise, so the number is not a futures-market result and "
            "cannot be a positive verdict."
        )
    failures = [
        "Classification is SYNTHETIC. A market claim is not available.",
        "Parameters are the module defaults. They were not fit on this series.",
    ]
    if total is not None and total < 0:
        failures.append("After-cost total return on the full scored sample is negative.")
    if sign_flip:
        failures.append("The 126-session lookback does not share the sign of the default 252-session Sharpe.")
    report: dict[str, Any] = {
        "hypothesis_id": HYPOTHESIS_ID,
        "question": (
            "On an authored zero-mean excess-return series, what does the existing futures "
            "time-series momentum strategy produce after its default costs, and does a held-out "
            "slice or a lookahead probe change that reading?"
        ),
        "strategy": "gig.strategies.futures_tsmom.FuturesTSMOM",
        "experiment_version": EXPERIMENT_VERSION,
        "classification": "SYNTHETIC",
        "evidence_class": "INCONCLUSIVE" if verdict == "INCONCLUSIVE" else "REJECTED",
        "verdict": verdict,
        "interpretation": interpretation,
        "economic_claim": "NONE",
        "lookahead_detected": leakage,
        "leakage_detected": leakage,
        "dataset": {
            "id": DATASET_ID, "provider": "none", "classification": "SYNTHETIC",
            "generator": "numpy Generator.normal", "seed": SEED, "loc": LOC, "scale": SCALE,
            "sessions": N_OBS, "start": START, "frequency": "business day",
            "instrument": "none", "currency": "none",
            "point_in_time": (
                "The signal shifts one session before the strategy earns the next session's return. "
                "No observation after the decision is used."
            ),
        },
        "period": {"start": series.index[0].date().isoformat(), "end": series.index[-1].date().isoformat(),
                   "scored_start": result.returns.index[0].date().isoformat(),
                   "scored_end": result.returns.index[-1].date().isoformat()},
        "training_boundary": (
            "No parameter was estimated. The first sessions are indicator warm-up inside the "
            "strategy, not a fitted training sample."
        ),
        "holdout": {**hold_metrics, "start": holdout.index[0].date().isoformat(),
                    "end": holdout.index[-1].date().isoformat(),
                    "rule": "Last 63 scored sessions. The strategy was not refit on the earlier part."},
        "benchmark": {"name": "Weight 1 in the same authored excess returns, no cost, same scored dates",
                      **benchmark},
        "cost_bps_model": "spread 4, commission 1, impact eta 0.10 at 1 percent participation",
        "cost_sensitivity": {"default_spread_bps": 4.0, "wider_spread_bps": WIDER_SPREAD_BPS,
                             "default_sharpe": sharpe, "wider_spread_sharpe": _sharpe(wider)},
        "parameter_stability": {"lookback_252_sharpe": sharpe, "lookback_126_sharpe": alt,
                                "sign_flip": sign_flip},
        "placebo_draws": PLACEBO_DRAWS,
        "placebo_sharpe_median": _num(float(np.median(placebo))) if placebo else None,
        "deflated_sharpe": _num(dsr["dsr"]),
        "n_trials": 1,
        "sharpe_after_cost": sharpe,
        "sortino": metrics["sortino"],
        "total_return": total,
        "ann_return": metrics["ann_return"],
        "ann_vol": metrics["ann_vol"],
        "max_drawdown": metrics["max_drawdown"],
        "ann_turnover": metrics["ann_turnover"],
        "n_obs": int(metrics["n_obs"]),
        "annualization": "252 sessions. The scored sample is shorter than a long market study.",
        "equity_curve": _curve(result.equity),
        "failure_conditions": failures,
        "orders_submitted": 0,
        "models_promoted": 0,
        "code_rewritten": False,
        "method": (
            "FuturesTSMOM defaults: 252-session sign of trailing excess return, vol scaled, "
            "position lagged one session, module default transaction-cost model."
        ),
        "reproduce": "python -m gig.research_org --compute futures-tsmom-authored-excess-v1",
        "follow_up": "Repeat only if an admitted futures excess-return series is in the store. None is.",
    }
    return config_hash(config), report
