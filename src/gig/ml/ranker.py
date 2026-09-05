"""
Walk-forward cross-sectional ranking. OOS scores only.

Two estimators share the same embargoed split:

* LightGBM ``lambdarank`` when ``lightgbm`` is installed — the ranking objective
  desks actually use for CS alpha, with per-date query groups so the loss is
  about ordering within a day, not about predicting return levels.
* HistGradientBoostingRegressor as the always-available fallback.

In-sample fitted values never enter the backtest. That is the leak desks reject.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

from gig.research.walkforward import WalkForwardSplit

RankerBackend = Literal["auto", "lightgbm", "sklearn"]


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


def _relevance_labels(y: pd.Series, n_bins: int = 5) -> np.ndarray:
    """
    Map continuous forward returns to integer relevance for lambdarank.

    Labels are assigned *within each date* so a quiet day does not get the same
    absolute cut-points as a volatile one. Empty or single-name days fall back
    to zeros.
    """
    out = np.zeros(len(y), dtype=np.int32)
    # Work on a contiguous positional index so group masks align with `out`.
    ordered = y.sort_index(level=0)
    pos = 0
    for _, vals in ordered.groupby(level=0, sort=False):
        n = len(vals)
        if n >= 2:
            ranks = vals.rank(method="first")
            bins = np.ceil(ranks / n * n_bins).clip(1, n_bins).astype(int) - 1
            out[pos : pos + n] = bins.to_numpy()
        pos += n
    # Reorder back to the caller's row order.
    return pd.Series(out, index=ordered.index).reindex(y.index).fillna(0).astype(np.int32).to_numpy()


def _group_sizes(index: pd.Index) -> list[int]:
    """LightGBM group vector: number of names per date, in row order."""
    dates = index.get_level_values(0)
    # value_counts does not preserve order; walk the sorted unique dates as they appear.
    sizes: list[int] = []
    last = None
    count = 0
    for dt in dates:
        if last is None:
            last = dt
            count = 1
        elif dt == last:
            count += 1
        else:
            sizes.append(count)
            last = dt
            count = 1
    if count:
        sizes.append(count)
    return sizes


def _fit_lightgbm(x_tr: pd.DataFrame, y_tr: pd.Series, x_te: pd.DataFrame) -> np.ndarray | None:
    try:
        import lightgbm as lgb
    except ImportError:
        return None

    # Sort by date so group sizes line up with consecutive rows.
    x_tr = x_tr.sort_index(level=0)
    y_tr = y_tr.reindex(x_tr.index)
    labels = _relevance_labels(y_tr)
    groups = _group_sizes(x_tr.index)
    if sum(groups) != len(x_tr) or min(groups) < 1:
        return None

    train = lgb.Dataset(
        x_tr.to_numpy(dtype=float),
        label=labels,
        group=groups,
        feature_name=list(x_tr.columns),
        free_raw_data=False,
    )
    params = {
        "objective": "lambdarank",
        "metric": "ndcg",
        "ndcg_eval_at": [5, 10],
        "learning_rate": 0.05,
        "num_leaves": 15,
        "min_data_in_leaf": 20,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "verbosity": -1,
        "seed": 42,
        "label_gain": list(range(5)),
    }
    booster = lgb.train(params, train, num_boost_round=80)
    return booster.predict(x_te.to_numpy(dtype=float))


def _fit_sklearn(x_tr: pd.DataFrame, y_tr: pd.Series, x_te: pd.DataFrame) -> np.ndarray:
    from sklearn.ensemble import HistGradientBoostingRegressor

    model = HistGradientBoostingRegressor(
        max_depth=3,
        max_iter=80,
        learning_rate=0.05,
        min_samples_leaf=20,
        random_state=42,
    )
    model.fit(x_tr, y_tr)
    return model.predict(x_te)


def available_backends() -> dict[str, bool]:
    out = {"sklearn": True, "lightgbm": False}
    try:
        import lightgbm  # noqa: F401

        out["lightgbm"] = True
    except ImportError:
        pass
    try:
        import sklearn  # noqa: F401
    except ImportError:
        out["sklearn"] = False
    return out


def walk_forward_scores(
    factors: dict[str, pd.DataFrame],
    forward_returns: pd.DataFrame,
    min_train: int = 252,
    test_size: int = 63,
    embargo: int = 1,
    backend: RankerBackend = "auto",
) -> pd.DataFrame:
    """
    Fit a ranker on each expanding train window; write predictions only on the
    embargoed test block.

    ``backend='auto'`` prefers LightGBM lambdarank, then sklearn GBDT.
    """
    backends = available_backends()
    if backend == "lightgbm" and not backends["lightgbm"]:
        raise ImportError('LightGBM is not installed. Run: pip install -e ".[ml]"')
    if backend == "sklearn" and not backends["sklearn"]:
        raise ImportError("Install scikit-learn for the walk-forward ranker")
    if backend == "auto" and not backends["sklearn"] and not backends["lightgbm"]:
        raise ImportError("Install scikit-learn or lightgbm for the walk-forward ranker")

    prefer_lgb = backend in {"auto", "lightgbm"} and backends["lightgbm"]

    frame = factor_frame(factors, forward_returns)
    feature_cols = [c for c in frame.columns if c != "y"]
    dates = pd.DatetimeIndex(sorted(frame.index.get_level_values(0).unique()))
    pred = pd.Series(np.nan, index=frame.index, dtype=float)
    used = "none"

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

        scores = None
        if prefer_lgb:
            scores = _fit_lightgbm(x_tr, y_tr, x_te)
            if scores is not None:
                used = "lightgbm"
        if scores is None:
            scores = _fit_sklearn(x_tr, y_tr, x_te)
            used = "sklearn"
        pred.loc[te] = scores

    out = pred.unstack()
    out.attrs["ranker_backend"] = used
    return out.reindex(index=forward_returns.index, columns=forward_returns.columns)
