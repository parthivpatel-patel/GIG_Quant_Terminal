"""Walk-forward gradient boosting on cross-sectional factors. OOS scores only."""

from __future__ import annotations

import numpy as np
import pandas as pd

from gig.research.walkforward import WalkForwardSplit


def factor_frame(factors: dict[str, pd.DataFrame], forward_returns: pd.DataFrame) -> pd.DataFrame:
    """Stack date×symbol rows. y is the *next* session return already shifted by caller."""
    parts = []
    for name, fac in factors.items():
        try:
            s = fac.stack(future_stack=True)
        except TypeError:
            s = fac.stack()
        s.name = name
        parts.append(s)
    x = pd.concat(parts, axis=1)
    try:
        y = forward_returns.stack(future_stack=True).rename("y")
    except TypeError:
        y = forward_returns.stack().rename("y")
    return x.join(y, how="inner").dropna(subset=["y"])


def walk_forward_scores(
    factors: dict[str, pd.DataFrame],
    forward_returns: pd.DataFrame,
    min_train: int = 252,
    test_size: int = 63,
    embargo: int = 1,
) -> pd.DataFrame:
    """
    Fit a histogram GBDT on each expanding train window; write predictions
    only on the embargoed test block. In-sample fitted values never enter
    the backtest — that is the leak desks reject in interviews.
    """
    try:
        from sklearn.ensemble import HistGradientBoostingRegressor
    except ImportError as exc:
        raise ImportError("Install scikit-learn for the walk-forward ranker") from exc

    frame = factor_frame(factors, forward_returns)
    feature_cols = [c for c in frame.columns if c != "y"]
    dates = pd.DatetimeIndex(sorted(frame.index.get_level_values(0).unique()))
    pred = pd.Series(np.nan, index=frame.index, dtype=float)

    for fold in WalkForwardSplit(min_train, test_size, embargo).split(dates):
        train_d = set(fold.train_index)
        test_d = set(fold.test_index)
        lvl = frame.index.get_level_values(0)
        tr = lvl.isin(train_d)
        te = lvl.isin(test_d)
        x_tr = frame.loc[tr, feature_cols]
        y_tr = frame.loc[tr, "y"]
        x_te = frame.loc[te, feature_cols]
        if len(x_tr) < 50 or x_te.empty:
            continue
        model = HistGradientBoostingRegressor(
            max_depth=3,
            max_iter=80,
            learning_rate=0.05,
            min_samples_leaf=20,
            random_state=42,
        )
        model.fit(x_tr, y_tr)
        pred.loc[te] = model.predict(x_te)

    out = pred.unstack()
    return out.reindex(index=forward_returns.index, columns=forward_returns.columns)
