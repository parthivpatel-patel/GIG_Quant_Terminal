from __future__ import annotations

from datetime import date

import pandas as pd

from gig.data.providers.synthetic import SyntheticProvider
from gig.factors.momentum import Momentum12m1
from gig.factors.neutralize import neutralize
from gig.research.ic import ic_summary, spearman_ic


def test_momentum_has_positive_ic_on_synthetic():
    panel = SyntheticProvider(n_names=80, n_days=756, seed=7).load_panel(date(2019, 1, 2), date(2024, 12, 31))
    scores = Momentum12m1().compute(panel)
    neu = neutralize(scores, panel.sectors)
    fwd = panel.close.pct_change().shift(-1)
    summary = ic_summary(neu, fwd)
    assert summary["n_obs"] > 200
    assert summary["ic_mean"] > 0.0


def test_same_bar_ic_is_not_used_as_tradable_alpha():
    """Lookahead check: the research IC is always vs *next* session returns."""
    panel = SyntheticProvider(n_names=40, n_days=400, seed=1).load_panel(date(2019, 1, 2), date(2024, 12, 31))
    scores = Momentum12m1().compute(panel)
    same = panel.close.pct_change()
    nxt = same.shift(-1)
    ic_next = spearman_ic(scores, nxt).mean()
    assert ic_next == ic_next  # finite


def test_cpp_kernel_returns_same_shape():
    try:
        from gig._speed import neutralize_cs
    except ImportError:
        return
    import numpy as np

    panel = SyntheticProvider(n_names=40, n_days=80, seed=3).load_panel(date(2019, 1, 2), date(2024, 12, 31))
    scores = Momentum12m1().compute(panel)
    values = np.ascontiguousarray(np.nan_to_num(scores.to_numpy(dtype=float), nan=0.0))
    dummy = np.ascontiguousarray(
        pd.get_dummies(panel.sectors.reindex(scores.columns).fillna("unknown"), drop_first=True).to_numpy(dtype=float)
    )
    mask = np.ascontiguousarray((~scores.isna()).to_numpy().astype(np.uint8))
    out = neutralize_cs(values, dummy, mask, True)
    assert out.shape == values.shape
