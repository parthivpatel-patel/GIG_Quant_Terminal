"""Fit research models on a return series and compare GARCH with EWMA."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from gig.models.garch import ewma_variance, fit_garch11, qlike
from gig.models.hmm import fit_gaussian_hmm
from gig.risk.deleverage import exposure_scale


def diagnostics_from_returns(returns: np.ndarray, asof: str | None = None) -> dict[str, Any]:
    """Fit HMM and GARCH. Both stay RESEARCH regardless of the in-sample comparison."""
    series = np.asarray(returns, dtype=float)
    series = series[np.isfinite(series)]
    if series.size < 80:
        return {
            "available": False,
            "reason": "DATA UNAVAILABLE",
            "detail": "need at least 80 finite returns",
            "n": int(series.size),
        }
    hmm = fit_gaussian_hmm(series, n_states=2, n_iter=25)
    garch = fit_garch11(series)
    ewma = ewma_variance(series)
    # One-step variance forecast versus the next squared return, on the last fifth.
    split = int(series.size * 0.8)
    hold_r = series[split:]
    garch_var = garch.conditional_variance[split:]
    ewma_var = ewma[split:]
    garch_score = qlike(hold_r**2, garch_var)
    ewma_score = qlike(hold_r**2, ewma_var)
    if abs(garch_score - ewma_score) < 1e-4:
        winner = "inconclusive"
    elif garch_score < ewma_score:
        winner = "garch11"
    else:
        winner = "ewma"
    vol = float(np.std(series) * np.sqrt(252))
    scale = exposure_scale(drawdown=0.0, realized_vol=vol, vol_target=0.10)
    return {
        "available": True,
        "asof": asof,
        "n": int(series.size),
        "annualized_realized_vol": vol,
        "hmm": hmm.as_dict(),
        "garch": garch.as_dict(),
        "ewma": {
            "model": "EWMA",
            "lambda": 0.94,
            "next_vol_annualized": float(np.sqrt(ewma[-1] * 252)),
            "note": "RiskMetrics decay. Not a maximum-likelihood GARCH.",
        },
        "variance_forecast_comparison": {
            "metric": "QLIKE",
            "lower_is_better": True,
            "holdout": "last 20 percent, in-sample conditional variance (not a purged backtest)",
            "garch11": garch_score,
            "ewma": ewma_score,
            "lower_score": winner,
            "promotion": "none — in-sample QLIKE does not change registry status",
        },
        "suggested_exposure_scale": {
            "value": scale,
            "applied_to_orders": False,
            "inputs": "zero drawdown and full-sample realized vol versus a 10% target",
        },
    }


def equal_weight_returns(close: pd.DataFrame) -> pd.Series:
    """Cross-sectional mean of simple returns. Names with a missing close are skipped that day."""
    rets = close.pct_change()
    return rets.mean(axis=1, skipna=True).dropna()


def diagnostics_from_lake() -> dict[str, Any]:
    """Fit on the equal-weight return of the stored liquid panel. Empty lake is explicit."""
    from gig.data.lake import db_lock, load_liquid_panel

    with db_lock():
        panel, _store = load_liquid_panel(limit=40, lookback_days=800)
    if panel is None or panel.close is None or panel.close.empty:
        return {"available": False, "reason": "DATA UNAVAILABLE", "detail": "no bars in the lake"}
    series = equal_weight_returns(panel.close)
    asof = str(pd.Timestamp(series.index[-1]).date()) if len(series) else None
    payload = diagnostics_from_returns(series.to_numpy(dtype=float), asof=asof)
    payload["universe"] = "equal-weight mean of up to 40 most-liquid stored names"
    payload["source"] = "duckdb bars"
    return payload
