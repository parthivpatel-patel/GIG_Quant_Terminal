from __future__ import annotations

from gig.pipeline import run_equity_research


def test_full_equity_pipeline_runs():
    result = run_equity_research(source="synthetic", seed=42, persist=False, use_ml=True, use_news=False)
    assert "sharpe" in result.metrics
    assert result.metrics["n_obs"] > 200
    assert result.weights.abs().sum(axis=1).max() <= 2.0 + 1e-6
    assert "momentum_12_1" in result.factor_ic

