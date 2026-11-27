"""Roll and Corwin-Schultz spreads against their definitions and a known bid-ask bounce."""
from __future__ import annotations

import math
import unittest

import numpy as np

from gig.microstructure.spreads import corwin_schultz, roll_spread


class RollTests(unittest.TestCase):
    def test_the_spread_is_twice_the_root_of_a_negative_autocovariance(self):
        prices = [10, 10.4, 9.7, 10.2, 9.5, 10.6, 9.4, 10.1, 9.8, 10.5, 9.6, 10.3]
        changes = np.diff(prices)
        covariance = float(np.cov(changes[:-1], changes[1:], ddof=1)[0, 1])
        result = roll_spread(prices)
        self.assertLess(covariance, 0)
        self.assertEqual(result["spread_state"], "IDENTIFIED")
        self.assertAlmostEqual(result["autocovariance"], covariance)
        self.assertAlmostEqual(result["spread"], 2*math.sqrt(-covariance))
        self.assertAlmostEqual(result["relative_spread"], result["spread"]/float(np.mean(prices)))

    def test_a_known_bounce_is_recovered_and_a_trend_is_not_called_a_spread(self):
        generator = np.random.default_rng(2)
        mid = 50+np.cumsum(generator.normal(0, .02, 4000))
        prices = mid+.1*generator.choice([-1., 1.], 4000)
        result = roll_spread(prices.tolist())
        self.assertAlmostEqual(result["spread"], .2, delta=.025)
        trend = roll_spread(list(np.linspace(10, 20, 30)))
        self.assertEqual((trend["spread"], trend["spread_state"]), (None, "NOT_IDENTIFIED_NONNEGATIVE_AUTOCORRELATION"))
        for bad in ([1]*9, [10, -1, 10, 11, 12, 13, 14, 15, 16, 17], [True]*12):
            with self.assertRaises(ValueError):
                roll_spread(bad)


class CorwinSchultzTests(unittest.TestCase):
    def test_each_window_matches_the_published_expression(self):
        high, low = [10., 10.8, 10.4, 11.], [9.6, 10.1, 9.9, 10.5]
        scale = 3-2*math.sqrt(2)
        expected = []
        for left in range(3):
            beta = math.log(high[left]/low[left])**2+math.log(high[left+1]/low[left+1])**2
            gamma = math.log(max(high[left], high[left+1])/min(low[left], low[left+1]))**2
            alpha = (math.sqrt(2*beta)-math.sqrt(beta))/scale-math.sqrt(gamma/scale)
            expected.append(2*(math.exp(alpha)-1)/(1+math.exp(alpha)))
        result = corwin_schultz(high, low)
        self.assertAlmostEqual(result["mean_spread"], float(np.mean(expected)))
        self.assertEqual(result["negative_estimates"], sum(value < 0 for value in expected))
        self.assertAlmostEqual(result["mean_spread_floored_at_zero"], float(np.mean([max(value, 0) for value in expected])))
        flat = corwin_schultz([10, 10, 10], [10, 10, 10])
        self.assertEqual((flat["mean_spread"], flat["negative_estimates"]), (0., 0))
        gapped = corwin_schultz([10, 20], [9, 19])
        self.assertGreater(gapped["negative_estimates"], 0)
        with self.assertRaises(ValueError):
            corwin_schultz([10, 11], [9])
        with self.assertRaises(ValueError):
            corwin_schultz([10, 9], [11, 8])


if __name__ == "__main__":
    unittest.main()
