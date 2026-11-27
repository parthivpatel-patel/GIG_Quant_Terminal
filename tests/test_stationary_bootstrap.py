"""Stationary bootstrap intervals against the block definition and the iid variance formula."""
from __future__ import annotations

import unittest

import numpy as np

from gig.research.bootstrap import stationary_bootstrap


def _defined_draws(series, mean_block, replications, seed):
    """The stationary bootstrap written as one circular block at a time."""
    values = np.asarray(series, float)
    count = len(values)
    generator = np.random.default_rng(seed)
    lengths = generator.geometric(1/mean_block, size=(replications, count))
    starts = generator.integers(0, count, size=(replications, count))
    means, sharpes = [], []
    for replication in range(replications):
        sample = []
        for block, length in enumerate(lengths[replication]):
            for step in range(int(length)):
                sample.append(values[(int(starts[replication, block])+step) % count])
                if len(sample) == count:
                    break
            if len(sample) == count:
                break
        drawn = np.asarray(sample)
        means.append(drawn.mean())
        sharpes.append(drawn.mean()/drawn.std(ddof=1))
    return np.asarray(means), np.asarray(sharpes)


class StationaryBootstrapTests(unittest.TestCase):
    def test_intervals_match_the_block_definition(self):
        series = np.sin(np.linspace(0, 8, 30))+np.array([.01]*30)
        result = stationary_bootstrap(series.tolist(), mean_block=4, replications=200, seed=11, confidence=.9)
        means, sharpes = _defined_draws(series, 4, 200, 11)
        tail = .05
        for got, expected in zip(result["mean_interval"], np.quantile(means, [tail, 1-tail]), strict=True):
            self.assertAlmostEqual(got, float(expected), places=12)
        for got, expected in zip(result["sharpe_interval"], np.quantile(sharpes, [tail, 1-tail]), strict=True):
            self.assertAlmostEqual(got, float(expected), places=12)
        self.assertEqual(result["mean"], float(series.mean()))
        self.assertEqual(result["sharpe"], float(series.mean()/series.std(ddof=1)))
        self.assertEqual(result["degenerate_sharpe_replications"], 0)
        self.assertEqual(result["method"], "POLITIS_ROMANO_1994_STATIONARY_BOOTSTRAP")
        again = stationary_bootstrap(series.tolist(), mean_block=4, replications=200, seed=11, confidence=.9)
        self.assertEqual(again["mean_interval"], result["mean_interval"])

    def test_unit_blocks_recover_the_variance_of_an_iid_resample(self):
        series = np.arange(40, dtype=float)
        result = stationary_bootstrap(series.tolist(), mean_block=1, replications=5000, seed=3)
        means, _ = _defined_draws(series, 1, 5000, 3)
        for got, expected in zip(result["mean_interval"], np.quantile(means, [.025, .975]), strict=True):
            self.assertAlmostEqual(got, float(expected), places=12)
        empirical = np.mean((series-series.mean())**2)
        self.assertAlmostEqual(float(means.var(ddof=1)), empirical/len(series), delta=empirical/len(series)*.08)

    def test_inputs_that_are_not_a_varying_series_are_refused(self):
        series = list(range(12))
        for bad in ({"series": [1]*12, "mean_block": 2},
                    {"series": series, "mean_block": 0},
                    {"series": series, "mean_block": 100},
                    {"series": series[:9], "mean_block": 1},
                    {"series": [True]*12, "mean_block": 1},
                    {"series": series, "mean_block": 2, "replications": 10},
                    {"series": series, "mean_block": 2, "seed": -1},
                    {"series": series, "mean_block": 2, "confidence": .1}):
            with self.assertRaises(ValueError):
                stationary_bootstrap(**bad)


if __name__ == "__main__":
    unittest.main()
