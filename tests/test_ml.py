from __future__ import annotations

import numpy as np
import pandas as pd

from gig.ml.ranker import walk_forward_scores
from gig.research.walkforward import WalkForwardSplit


def test_ml_scores_are_empty_in_train_window():
    dates = pd.bdate_range("2018-01-02", periods=500)
    names = [f"S{i}" for i in range(12)]
    rng = np.random.default_rng(0)
    fac = pd.DataFrame(rng.normal(size=(500, 12)), index=dates, columns=names)
    fwd = pd.DataFrame(rng.normal(0, 0.01, size=(500, 12)), index=dates, columns=names)
    scores = walk_forward_scores({"f": fac}, fwd, min_train=252, test_size=63, embargo=5)
    first = next(WalkForwardSplit(252, 63, 5).split(dates))
    train_block = scores.loc[first.train_index]
    # OOS only: the initial train window must not carry fitted values
    assert train_block.isna().all().all()
    assert scores.notna().any().any()
