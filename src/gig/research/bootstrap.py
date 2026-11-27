"""Stationary bootstrap of a caller-supplied series.

Politis and Romano (1994): blocks have geometric length and wrap around the
series. The result is a percentile interval for the mean and the Sharpe ratio.
It is a resampling of the series you supply, not a test that a strategy worked.
"""
from __future__ import annotations

import numpy as np

from gig import checks

MAX_OBSERVATIONS = 2000
MAX_REPLICATIONS = 5000


def _segment(starts: np.ndarray, lengths: np.ndarray, prefix: np.ndarray, total: float, count: int) -> np.ndarray:
    """Sums of circular slices. ``prefix`` holds the cumulative sum of two concatenated copies."""
    cycles = lengths // count
    remainder = lengths % count
    return cycles*total+prefix[starts+remainder]-prefix[starts]


def stationary_bootstrap(series, *, mean_block, replications=1000, seed=0, confidence=.95) -> dict:
    """Percentile intervals from the stationary bootstrap."""
    values = checks.array(series, "series", dimensions=1, maximum=MAX_OBSERVATIONS)
    count = int(values.shape[0])
    if count < 10:
        raise ValueError("series must hold 10 to 2000 values")
    block = checks.number(mean_block, "mean_block", low=1., high=count)
    draws = checks.integer(replications, "replications", 100, MAX_REPLICATIONS)
    seed = checks.integer(seed, "seed", 0, 2**31-1)
    level = checks.confidence(confidence)
    centered = values-values.mean()
    if float(centered @ centered) == 0:
        raise ValueError("series must vary")
    probability = 1/block
    generator = np.random.default_rng(seed)
    # A geometric draw is at least one, so one draw per observation always covers the series.
    lengths = generator.geometric(probability, size=(draws, count))
    starts = generator.integers(0, count, size=(draws, count))
    ends = np.cumsum(lengths, axis=1)
    cross = np.argmax(ends >= count, axis=1)
    row = np.arange(draws)
    previous = np.zeros(draws, dtype=int)
    carried = cross > 0
    previous[carried] = ends[row[carried], cross[carried]-1]
    prefix = np.empty(2*count+1)
    square_prefix = np.empty(2*count+1)
    prefix[0] = square_prefix[0] = 0.
    prefix[1:count+1] = np.cumsum(values)
    prefix[count+1:] = prefix[count]+np.cumsum(values)
    squares = values*values
    square_prefix[1:count+1] = np.cumsum(squares)
    square_prefix[count+1:] = square_prefix[count]+np.cumsum(squares)
    total, total_squares = float(values.sum()), float(squares.sum())
    complete = np.arange(count)[None, :] < cross[:, None]
    sample_sum = (_segment(starts, lengths, prefix, total, count)*complete).sum(axis=1)
    sample_squares = (_segment(starts, lengths, square_prefix, total_squares, count)*complete).sum(axis=1)
    partial = count-previous
    sample_sum += _segment(starts[row, cross], partial, prefix, total, count)
    sample_squares += _segment(starts[row, cross], partial, square_prefix, total_squares, count)
    means = sample_sum/count
    variances = (sample_squares-count*means**2)/(count-1)
    sharpes = np.divide(means, np.sqrt(variances), out=np.full(draws, np.nan), where=variances > 0)
    degenerate = int(np.isnan(sharpes).sum())
    if degenerate == draws:
        raise ValueError("every bootstrap sample was constant; increase the series or the mean block")
    tail = (1-level)/2
    mean_interval = [float(x) for x in np.quantile(means, [tail, 1-tail])]
    sharpe_interval = [float(x) for x in np.nanquantile(sharpes, [tail, 1-tail])]
    point_mean = float(values.mean())
    point_sharpe = point_mean/float(values.std(ddof=1))
    return {"method": "POLITIS_ROMANO_1994_STATIONARY_BOOTSTRAP", "observations": count, "mean_block": block,
            "replications": draws, "seed": seed, "confidence": level, "sharpe_divisor": "N_MINUS_ONE",
            "wrap": "CIRCULAR", "block_distribution": "GEOMETRIC",
            "mean": point_mean, "sharpe": point_sharpe,
            "mean_interval": mean_interval, "sharpe_interval": sharpe_interval,
            "mean_interval_contains_zero": mean_interval[0] <= 0 <= mean_interval[1],
            "sharpe_interval_contains_zero": sharpe_interval[0] <= 0 <= sharpe_interval[1],
            "degenerate_sharpe_replications": degenerate,
            "resample_limits": "PERCENTILE_INTERVAL_OF_THE_SUPPLIED_SERIES_NOT_EVIDENCE_A_STRATEGY_WORKED",
            "classification": "MODEL_ESTIMATE"}
