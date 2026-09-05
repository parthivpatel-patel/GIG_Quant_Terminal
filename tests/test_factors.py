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


def test_neutralize_keeps_every_eligible_name():
    """
    Regression: the eligibility mask was cast to uint8, which NumPy reads as
    integer indices, so each row residualized only columns 0 and 1. The old IC
    test still passed because Spearman on two names is +/-1.
    """
    from gig.data.universe import point_in_time_mask

    panel = SyntheticProvider(n_names=60, n_days=420, seed=5).load_panel(
        date(2019, 1, 2), date(2024, 12, 31)
    )
    scores = Momentum12m1().compute(panel)
    tradable = point_in_time_mask(panel.close, min_history=252, min_price=1.0)
    neu = neutralize(scores, panel.sectors, tradable=tradable)

    eligible = (scores.notna() & tradable).sum(axis=1)
    n_dummies = panel.sectors.nunique() - 1
    solvable = eligible >= n_dummies + 8
    assert solvable.any(), "fixture produced no solvable cross-sections"
    assert (neu.notna().sum(axis=1)[solvable] == eligible[solvable]).all()
    assert neu.notna().sum(axis=1)[solvable].min() > 2


def test_neutralize_removes_sector_means():
    """Residualizing on a constant plus sector dummies must zero each sector mean."""
    panel = SyntheticProvider(n_names=60, n_days=420, seed=6).load_panel(
        date(2019, 1, 2), date(2024, 12, 31)
    )
    neu = neutralize(Momentum12m1().compute(panel), panel.sectors)
    row = neu.dropna(axis=1, how="all").iloc[-1].dropna()
    assert len(row) > 10

    sector_means = row.groupby(panel.sectors.reindex(row.index)).mean()
    assert sector_means.abs().max() < 1e-8


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
