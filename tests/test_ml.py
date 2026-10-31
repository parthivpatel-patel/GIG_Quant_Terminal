from __future__ import annotations

import numpy as np
import pandas as pd

from gig.ml.ranker import (
    _relevance_labels,
    available_backends,
    factor_frame,
    walk_forward_scores,
)
from gig.research.walkforward import WalkForwardSplit


def test_ml_scores_are_empty_in_train_window():
    dates = pd.bdate_range("2018-01-02", periods=500)
    names = [f"S{i}" for i in range(12)]
    rng = np.random.default_rng(0)
    fac = pd.DataFrame(rng.normal(size=(500, 12)), index=dates, columns=names)
    fwd = pd.DataFrame(rng.normal(0, 0.01, size=(500, 12)), index=dates, columns=names)
    scores = walk_forward_scores(
        {"f": fac}, fwd, min_train=252, test_size=63, embargo=5, backend="sklearn"
    )
    first = next(WalkForwardSplit(252, 63, 5).split(dates))
    train_block = scores.loc[first.train_index]
    assert train_block.isna().all().all()
    assert scores.notna().any().any()


def test_relevance_labels_are_within_date_bins():
    dates = pd.to_datetime(["2020-01-02"] * 10 + ["2020-01-03"] * 10)
    names = [f"S{i}" for i in range(10)] * 2
    idx = pd.MultiIndex.from_arrays([dates, names], names=["dt", "symbol"])
    y = pd.Series(np.linspace(-0.05, 0.05, 20), index=idx)
    labels = _relevance_labels(y, n_bins=5)
    assert labels.min() >= 0
    assert labels.max() <= 4
    assert len(labels) == 20
    # Best name on day 1 should outrank the worst on that same day.
    assert labels[9] > labels[0]


def test_available_backends_reports_sklearn():
    backends = available_backends()
    assert backends["sklearn"] is True


def test_lightgbm_backend_when_installed():
    if not available_backends()["lightgbm"]:
        return
    dates = pd.bdate_range("2018-01-02", periods=420)
    names = [f"S{i}" for i in range(16)]
    rng = np.random.default_rng(1)
    # Plant a weak ranking signal so lambdarank has something to fit.
    noise = rng.normal(size=(420, 16))
    signal = rng.normal(size=(420, 16))
    fac = pd.DataFrame(signal, index=dates, columns=names)
    fwd = pd.DataFrame(0.02 * signal + 0.01 * noise, index=dates, columns=names)
    scores = walk_forward_scores(
        {"f": fac}, fwd, min_train=200, test_size=40, embargo=2, backend="lightgbm"
    )
    assert scores.attrs.get("ranker_backend") == "lightgbm"
    assert scores.notna().any().any()
    first = next(WalkForwardSplit(200, 40, 2).split(dates))
    assert scores.loc[first.train_index].isna().all().all()


def test_factor_frame_aligns_features_and_y():
    dates = pd.bdate_range("2020-01-02", periods=5)
    names = ["A", "B"]
    fac = pd.DataFrame([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]], index=dates, columns=names)
    fwd = fac / 100.0
    frame = factor_frame({"mom": fac}, fwd)
    assert "mom" in frame.columns
    assert "y" in frame.columns
    assert len(frame) == 10
