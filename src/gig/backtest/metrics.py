"""Portfolio performance statistics. Annualization assumes 252 sessions."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

TRADING_DAYS = 252


def performance_metrics(
    returns: pd.Series,
    turnover: pd.Series | None = None,
    rf_daily: float = 0.0,
) -> dict[str, float]:
    r = returns.dropna().astype(float)
    if len(r) < 2:
        return {"n_obs": float(len(r))}
    excess = r - rf_daily
    mu = float(excess.mean())
    vol = float(excess.std(ddof=1))
    sharpe = float(mu / vol * np.sqrt(TRADING_DAYS)) if vol > 0 else float("nan")
    downside = excess[excess < 0]
    dvol = float(downside.std(ddof=1)) if len(downside) > 1 else float("nan")
    sortino = float(mu / dvol * np.sqrt(TRADING_DAYS)) if dvol and dvol > 0 else float("nan")

    equity = (1 + r).cumprod()
    peak = equity.cummax()
    dd = equity / peak - 1.0
    max_dd = float(dd.min())
    calmar = float(mu * TRADING_DAYS / abs(max_dd)) if max_dd < 0 else float("nan")

    win = float((r > 0).mean())
    skew = float(stats.skew(r, bias=False))
    kurt = float(stats.kurtosis(r, fisher=False, bias=False))  # Pearson kurtosis

    out: dict[str, float] = {
        "n_obs": float(len(r)),
        "ann_return": float(mu * TRADING_DAYS),
        "ann_vol": float(vol * np.sqrt(TRADING_DAYS)),
        "sharpe": sharpe,
        "sortino": sortino,
        "calmar": calmar,
        "max_drawdown": max_dd,
        "win_rate": win,
        "skew": skew,
        "kurtosis": kurt,
        "total_return": float(equity.iloc[-1] - 1.0),
    }
    if turnover is not None:
        t = turnover.reindex(r.index).fillna(0)
        out["ann_turnover"] = float(t.mean() * TRADING_DAYS)
        out["avg_turnover"] = float(t.mean())
    return out
