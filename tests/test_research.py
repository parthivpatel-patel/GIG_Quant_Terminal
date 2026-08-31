from __future__ import annotations

import pandas as pd

from gig.research.multiple_testing import bonferroni_pvalue, deflated_sharpe
from gig.research.walkforward import WalkForwardSplit


def test_walkforward_embargo_gap():
    idx = pd.bdate_range("2018-01-02", periods=600)
    folds = list(WalkForwardSplit(min_train=252, test_size=63, embargo=5).split(idx))
    assert folds
    f0 = folds[0]
    gap = idx.get_loc(f0.test_start) - idx.get_loc(f0.train_end)
    assert gap >= 6  # 1 step after train_end plus 5 embargo
    assert f0.train_end not in f0.test_index
    assert len(f0.test_index) == 63


def test_deflated_sharpe_penalizes_many_trials():
        one = deflated_sharpe(0.5, n_trials=1, n_obs=252)
        many = deflated_sharpe(0.5, n_trials=200, n_obs=252)
        assert many["dsr"] < one["dsr"]


def test_bonferroni():
    adj = bonferroni_pvalue([0.01, 0.04])
    assert adj[0] == 0.02
    assert adj[1] == 0.08
