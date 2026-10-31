"""Causality, identity and deterministic pair-model references."""
from __future__ import annotations

import importlib.util
import math
import unittest

import numpy as np
import pandas as pd

from gig.research.stat_arb import (
    fit_pair, evaluate_pair, mean_reversion_diagnostics, cointegration_test,
    adf_test, select_pairs, kalman_hedge, walk_forward_pair, backtest_pair,
)


class PairResearchTests(unittest.TestCase):
    def test_exact_level_regression_and_zero_spread_unknown(self):
        x = np.arange(1., 31.)
        model = fit_pair(3 + 2*x, x, log_prices=False)
        self.assertAlmostEqual(model.alpha, 3)
        self.assertAlmostEqual(model.beta, 2)
        result = evaluate_pair(model, [5], [1])
        self.assertEqual(result["state"], "UNKNOWN")
        self.assertIsNone(result["zscore"])
        self.assertFalse(model.trading_approved)

    def test_log_regression_hedge_units(self):
        x = np.arange(1., 31.)
        y = np.exp(.4)*x**1.5
        model = fit_pair(y, x)
        self.assertAlmostEqual(model.alpha, .4, places=12)
        self.assertAlmostEqual(model.beta, 1.5, places=12)
        self.assertEqual(evaluate_pair(model, y, x)["beta_units"], "log_price_elasticity")

    def test_training_prefix_params_and_digest_ignore_future_mutation(self):
        x = np.arange(1., 51.)
        y = 2*x + 5 + np.sin(x)
        first = fit_pair(y, x, training_rows=30, log_prices=False)
        y[30:] = float("nan")
        x[30:] = -1
        second = fit_pair(y, x, training_rows=30, log_prices=False)
        self.assertEqual(first, second)

    def test_model_fingerprint_preserves_instrument_identity(self):
        x = np.arange(1., 31.)
        first = fit_pair(x*2+np.sin(x), x, dependent="A", independent="B")
        second = fit_pair(x*2+np.sin(x), x, dependent="C", independent="B")
        self.assertNotEqual(first.training_digest, second.training_digest)

    def test_input_identifiability_and_finite_admission(self):
        for y, x in (([1, 2, 3], [1, 1, 1]), ([1, float("nan"), 3], [1, 2, 3]), ([1, 2, -3], [1, 2, 3])):
            with self.assertRaises(ValueError):
                fit_pair(y, x)
        with self.assertRaises(ValueError):
            fit_pair([1, 2, 3], [1, 2, 3], dependent="X", independent="X")

    def test_ar1_half_life_exact_reference(self):
        # Deterministic AR1 trajectory x[t]=2+.8*x[t-1].
        values = [0.]
        for _ in range(30):
            values.append(2 + .8*values[-1])
        result = mean_reversion_diagnostics(values)
        self.assertAlmostEqual(result["phi"], .8, places=12)
        self.assertAlmostEqual(result["long_run_mean"], 10, places=11)
        self.assertAlmostEqual(result["half_life_steps"], math.log(.5)/math.log(.8), places=10)
        self.assertFalse(result["stationarity_tested"])

    def test_explosive_ar1_has_no_ou_half_life(self):
        result = mean_reversion_diagnostics(1.1**np.arange(30.))
        self.assertGreater(result["phi"], 1)
        self.assertIsNone(result["half_life_steps"])
        self.assertFalse(result["ou_compatible"])
        self.assertEqual(mean_reversion_diagnostics(np.ones(30))["state"], "UNKNOWN")

    def test_optional_significance_has_honest_unavailable_state(self):
        x = 100 + np.arange(60.)
        y = x + np.sin(np.arange(60.))
        coint = cointegration_test(y, x)
        adf = adf_test(y-x)
        if importlib.util.find_spec("statsmodels") is None:
            self.assertEqual(coint["state"], "UNAVAILABLE")
            self.assertIsNone(coint["pvalue"])
            self.assertEqual(adf["state"], "UNAVAILABLE")
            self.assertIsNone(adf["pvalue"])
        else:
            self.assertIn(coint["state"], ("AVAILABLE", "ERROR"))
            self.assertIn(adf["state"], ("AVAILABLE", "ERROR"))

    def test_pair_screening_never_selects_without_significance(self):
        x = 100+np.arange(60.)
        frame = pd.DataFrame({"A": x, "B": x+np.sin(x), "C": 2*x+np.cos(x)})
        result = select_pairs(frame, training_rows=40)
        self.assertEqual(result["family_size"], 3)
        self.assertEqual(result["correction"], "BONFERRONI_ALL_SCREENED_PAIRS")
        if importlib.util.find_spec("statsmodels") is None:
            self.assertTrue(all(not r["selected"] for r in result["pairs"]))
            self.assertEqual(result["state"], "UNAVAILABLE")
        with self.assertRaises(ValueError):
            select_pairs(frame.iloc[::-1], training_rows=40)

    def test_causal_kalman_prefix_unchanged_by_future_data(self):
        x = 100 + np.arange(60.)
        y = .5 + 1.1*x + np.sin(x)
        first = kalman_hedge(y[:30], x[:30], log_prices=False)
        second = kalman_hedge(y, x, log_prices=False)
        np.testing.assert_allclose(first["beta"], second["beta"][:30], atol=0, rtol=0)
        np.testing.assert_allclose(first["innovation"], second["innovation"][:30], atol=0, rtol=0)
        self.assertFalse(second["uses_future_observations"])
        self.assertTrue(np.all(second["innovation_variance"] > 0))

    def test_walk_forward_fit_test_boundaries_and_prefix_causality(self):
        x = 100 + np.arange(70.)
        y = 3 + 1.2*x + np.sin(x)
        result = walk_forward_pair(y, x, training_window=30, test_window=10, log_prices=False)
        self.assertEqual(len(result["spread"]), 40)
        self.assertEqual(result["folds"][0]["training_end_exclusive"], 30)
        self.assertEqual(result["folds"][0]["test_start"], 30)
        altered = y.copy()
        altered[50:] += 1000
        second = walk_forward_pair(altered, x, training_window=30, test_window=10, log_prices=False)
        np.testing.assert_allclose(result["spread"][:20], second["spread"][:20], rtol=0, atol=0)
        self.assertFalse(result["economic_validation"])
        self.assertFalse(result["transaction_costs_applied"])

    def test_backtest_next_close_lag_cost_and_drift_references(self):
        x = 100+np.arange(90.)
        y = 3+1.2*x+3*np.sin(np.arange(90.)/2)
        report = backtest_pair(y, x, training_window=30, test_window=10,
                               entry_z=.5, exit_z=.1, cost_bps=13, log_prices=False)
        previous = np.zeros(2)
        prior_returns = np.zeros(2)
        matrix = np.column_stack((y, x))
        self.assertTrue(any(np.abs(row["weights"]).sum() for row in report["rows"]))
        for row in report["rows"]:
            self.assertEqual(row["execution_row"], row["decision_row"]+1)
            self.assertEqual(row["return_end_row"], row["execution_row"]+1)
            self.assertLessEqual(row["training_end_exclusive"], row["decision_row"])
            weights = np.array(row["weights"])
            returns = matrix[row["return_end_row"]]/matrix[row["execution_row"]]-1
            expected_turnover = np.abs(weights-previous*(1+prior_returns)).sum()
            self.assertAlmostEqual(row["turnover"], expected_turnover)
            self.assertAlmostEqual(row["gross_return"], weights @ returns)
            self.assertAlmostEqual(row["cost"], expected_turnover*13/10000)
            self.assertAlmostEqual(row["net_return"], weights @ returns-row["cost"])
            self.assertLessEqual(np.abs(weights).sum(), 1+1e-12)
            previous, prior_returns = weights, returns
        self.assertFalse(report["terminal_liquidation_applied"])
        self.assertFalse(report["economic_validation"])

    def test_backtest_future_mutation_preserves_prior_decisions(self):
        x = 100+np.arange(90.)
        y = 3+1.2*x+3*np.sin(np.arange(90.)/2)
        first = backtest_pair(y, x, training_window=30, entry_z=.5, exit_z=.1)
        changed = y.copy()
        changed[70:] *= 2
        second = backtest_pair(changed, x, training_window=30, entry_z=.5, exit_z=.1)
        old = [row for row in first["rows"] if row["return_end_row"] < 70]
        new = [row for row in second["rows"] if row["return_end_row"] < 70]
        self.assertEqual(old, new)

    def test_backtest_ineligible_mask_blocks_exposure_and_types_are_explicit(self):
        x = 100+np.arange(60.)
        y = 3+1.2*x+np.sin(x)
        result = backtest_pair(y, x, training_window=30, eligibility=np.zeros(60, dtype=bool))
        self.assertTrue(all(row["net_return"] == 0 for row in result["rows"]))
        for kwargs in ({"eligibility": [1]*60}, {"entry_z": True}, {"exit_z": 3}, {"cost_bps": -1}):
            with self.assertRaises(ValueError):
                backtest_pair(y, x, training_window=30, **kwargs)


if __name__ == "__main__":
    unittest.main()
