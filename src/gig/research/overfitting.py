"""Backtest overfitting diagnostics on caller-supplied strategy returns.

Combinatorially symmetric cross-validation (Bailey, Borwein, Lopez de Prado
and Zhu) and the minimum track record length. These quantify selection risk;
they cannot certify that a selected strategy is profitable.
"""
from __future__ import annotations

import math
from itertools import combinations

import numpy as np
from scipy.stats import norm, rankdata

from gig import checks

MAX_OBSERVATIONS = 5000
MAX_STRATEGIES = 200


def probability_of_backtest_overfitting(performance, *, partitions: int = 8) -> dict:
    """Share of in-sample/out-of-sample splits where the in-sample winner ranks in the bottom half.

    ``performance`` has one row per period and one column per strategy trial.
    Rows are cut into ``partitions`` contiguous blocks; every way of choosing
    half of them is one in-sample set, scored by the per-period Sharpe ratio.
    """
    m = checks.array(performance, "performance", dimensions=2, maximum=MAX_OBSERVATIONS*MAX_STRATEGIES)
    s = checks.integer(partitions, "partitions", 2, 16)
    t, n = m.shape
    if s % 2 or not 2 <= n <= MAX_STRATEGIES or not 2*s <= t <= MAX_OBSERVATIONS:
        raise ValueError("need an even partition count, 2 to 200 strategies and at least 2 rows per partition")
    blocks = np.array_split(np.arange(t), s)
    counts = np.array([len(block) for block in blocks], dtype=float)
    sums = np.array([m[block].sum(axis=0) for block in blocks])
    squares = np.array([(m[block]**2).sum(axis=0) for block in blocks])

    def sharpe(chosen: list[int]) -> np.ndarray:
        size = counts[chosen].sum()
        mean = sums[chosen].sum(axis=0)/size
        variance = (squares[chosen].sum(axis=0)-size*mean**2)/(size-1)
        if np.any(variance <= 1e-18*max(1., float(np.abs(m).max())**2)):
            raise ValueError("a strategy has no variation in one split; its Sharpe ratio is undefined")
        return mean/np.sqrt(variance)

    logits, pairs = [], []
    for inside in combinations(range(s), s//2):
        outside = [index for index in range(s) if index not in inside]
        train, test = sharpe(list(inside)), sharpe(outside)
        winner = int(np.argmax(train))
        relative = float(rankdata(test)[winner]/(n+1))
        logits.append(math.log(relative/(1-relative)))
        pairs.append((float(train[winner]), float(test[winner])))
    logit, pair = np.array(logits), np.array(pairs)
    slope, intercept = (np.polyfit(pair[:, 0], pair[:, 1], 1) if np.ptp(pair[:, 0]) > 0 else (float("nan"), float("nan")))
    return {"method": "COMBINATORIALLY_SYMMETRIC_CROSS_VALIDATION", "observations": t, "strategies": n,
            "partitions": s, "splits": len(logit),
            "probability_of_backtest_overfitting": float(np.mean(logit <= 0)),
            "median_logit": float(np.median(logit)),
            "out_of_sample_loss_probability": float(np.mean(pair[:, 1] < 0)),
            "mean_in_sample_sharpe": float(pair[:, 0].mean()), "mean_out_of_sample_sharpe": float(pair[:, 1].mean()),
            "degradation_slope": None if math.isnan(slope) else float(slope),
            "degradation_intercept": None if math.isnan(intercept) else float(intercept),
            "sharpe_unit": "PER_PERIOD_NOT_ANNUALIZED",
            "interpretation_limits": "SELECTION_RISK_AMONG_SUPPLIED_TRIALS_ONLY_NOT_PROFITABILITY_EVIDENCE",
            "classification": "MODEL_ESTIMATE"}


def minimum_track_record_length(sharpe, *, benchmark: float = 0., skew: float = 0., kurtosis: float = 3.,
                                confidence: float = .95) -> dict:
    """Observations needed for a per-period Sharpe ratio to exceed a benchmark at a confidence."""
    sr, floor = checks.number(sharpe, "sharpe", low=-100, high=100), checks.number(benchmark, "benchmark", low=-100, high=100)
    g3, g4 = checks.number(skew, "skew", low=-100, high=100), checks.number(kurtosis, "kurtosis", low=1., high=1e4)
    level = checks.confidence(confidence)
    if sr <= floor:
        raise ValueError("sharpe must exceed the benchmark; no track record length can establish it otherwise")
    dispersion = 1-g3*sr+(g4-1)/4*sr*sr
    if dispersion <= 0:
        raise ValueError("skew and kurtosis are inconsistent with a positive Sharpe-ratio variance")
    length = 1+dispersion*(float(norm.ppf(level))/(sr-floor))**2
    return {"method": "MINIMUM_TRACK_RECORD_LENGTH", "sharpe": sr, "benchmark": floor, "skew": g3,
            "kurtosis": g4, "confidence": level, "observations": length, "whole_observations": math.ceil(length),
            "sharpe_unit": "PER_PERIOD_NOT_ANNUALIZED",
            "interpretation_limits": "ASSUMES_STATIONARY_RETURNS_AND_A_SINGLE_UNSELECTED_TRIAL",
            "classification": "MODEL_ESTIMATE"}
