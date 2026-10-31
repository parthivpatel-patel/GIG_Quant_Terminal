"""Independent numerical references for risk and safeguarded allocation."""
import unittest

import numpy as np

from gig.risk.analytics import drawdown_report, historical_loss_risk, portfolio_risk, stress_positions
from gig.portfolio.allocation import black_litterman, fractional_kelly, risk_parity


class RiskAnalyticsTests(unittest.TestCase):
    def test_fractional_empirical_tail_reference(self):
        report = historical_loss_risk([-0.10, -0.05, 0, 0.05], confidence=0.625, minimum=1)
        self.assertAlmostEqual(report["var"], 0.05)
        self.assertAlmostEqual(report["expected_shortfall"], (0.10+0.5*0.05)/1.5)
        self.assertEqual(report["observations"], 4)

    def test_gain_only_tail_is_not_clipped_to_zero(self):
        report = historical_loss_risk([0.1, 0.2, 0.3], confidence=0.99, minimum=1)
        self.assertAlmostEqual(report["var"], -0.1)
        self.assertAlmostEqual(report["expected_shortfall"], -0.1)

    def test_representable_tail_mean_does_not_overflow_its_sum(self):
        report = historical_loss_risk([-1e308]*20, confidence=0.5)
        self.assertTrue(np.isfinite(report["expected_shortfall"]))
        self.assertAlmostEqual(report["expected_shortfall"]/1e308, 1)

    def test_stress_nav_return_overflow_rejected(self):
        with self.assertRaises(ValueError):
            stress_positions([1], [1], [[1]], nav=1e-309)

    def test_portfolio_asset_resource_bound(self):
        with self.assertRaises(ValueError):
            portfolio_risk(np.zeros((20, 513)), np.ones(513)/513)

    def test_drawdown_starts_at_initial_nav_and_tracks_duration(self):
        report = drawdown_report([-0.1, 1/9, 0.2, -0.25])
        np.testing.assert_allclose(report["wealth"], [0.9, 1, 1.2, 0.9])
        self.assertAlmostEqual(report["max_drawdown"], -0.25)
        self.assertEqual(report["max_duration_observations"], 1)
        self.assertEqual(report["current_duration_observations"], 1)

    def test_total_loss_does_not_recover(self):
        report = drawdown_report([0.1, -1, 0.5])
        self.assertEqual(report["wealth"][-1], 0)
        self.assertEqual(report["current_drawdown"], -1)

    def test_linear_stress_short_and_long(self):
        report = stress_positions([10, -20], [100, 50], [[-0.1, -0.2], [0.2, -0.1]], nav=10000)
        np.testing.assert_allclose(report["pnl"], [100, 300])
        np.testing.assert_allclose(report["nav_return"], [0.01, 0.03])

    def test_portfolio_variance_contributions_reconcile(self):
        data = np.random.default_rng(91).normal(0, 0.02, (300, 3))
        weights = np.array([0.4, 0.3, 0.3])
        report = portfolio_risk(data, weights)
        independent = float(weights @ (np.cov(data.T)*252) @ weights)
        self.assertAlmostEqual(sum(report["variance_contributions"]), independent)
        self.assertAlmostEqual(sum(report["risk_fractions"]), 1)
        self.assertAlmostEqual(report["annualized_volatility"]**2, independent)

    def test_missing_invalid_or_insufficient_values_rejected(self):
        for values in [[0, np.nan]*20, [0, np.inf]*20, [0, 0]]:
            with self.assertRaises(ValueError):
                historical_loss_risk(values)
        with self.assertRaises(ValueError):
            historical_loss_risk([0]*20, confidence=1)
        with self.assertRaises(ValueError):
            drawdown_report([-1.01])
        with self.assertRaises(ValueError):
            stress_positions([1], [1], [[-1.1]], nav=1)


class AllocationTests(unittest.TestCase):
    def test_uncorrelated_risk_parity_is_inverse_volatility(self):
        report = risk_parity([[0.04, 0], [0, 0.16]])
        np.testing.assert_allclose(report["weights"], [2/3, 1/3], atol=2e-6)
        np.testing.assert_allclose(report["risk_fractions"], [0.5, 0.5], atol=2e-6)
        self.assertEqual(report["state"], "RISK_BUDGET_MATCHED")

    def test_binding_cap_reports_budget_mismatch(self):
        report = risk_parity([[0.04, 0], [0, 0.16]], max_weight=0.5)
        np.testing.assert_allclose(report["weights"], [0.5, 0.5], atol=1e-10)
        self.assertEqual(report["state"], "CONSTRAINED_APPROXIMATION")

    def test_black_litterman_no_surprise_views_preserve_prior(self):
        covariance = np.array([[0.04, 0.01], [0.01, 0.09]])
        market = np.array([0.6, 0.4])
        prior = 2.5*covariance @ market
        report = black_litterman(covariance, market, [[1, -1]], [prior[0]-prior[1]])
        np.testing.assert_allclose(report["expected_excess_returns"], prior, atol=1e-12)
        self.assertTrue(np.linalg.eigvalsh(report["predictive_covariance"]).min() > 0)

    def test_black_litterman_scalar_gaussian_reference(self):
        # Prior mean=.1, uncertainty=.002; observation .2 with noise .002
        report = black_litterman([[0.04]], [1], [[1]], [0.2], tau=0.05,
                                 risk_aversion=2.5, view_covariance=[[0.002]])
        self.assertAlmostEqual(report["expected_excess_returns"][0], 0.15)
        self.assertAlmostEqual(report["mean_uncertainty"][0][0], 0.001)
        self.assertAlmostEqual(report["predictive_covariance"][0][0], 0.041)

    def test_fractional_kelly_formula_and_caps(self):
        report = fractional_kelly([0.08, -0.04], [[0.04, 0], [0, 0.04]],
                                  fraction=0.25, max_name=1)
        np.testing.assert_allclose(report["weights"], [0.5, 0])
        capped = fractional_kelly([0.8, 0.8], [[0.04, 0], [0, 0.04]], max_name=0.4, max_gross=0.5)
        np.testing.assert_allclose(capped["weights"], [0.25, 0.25])
        self.assertFalse(capped["trading_approved"])

    def test_kelly_does_not_hide_overflow_by_clipping(self):
        with self.assertRaises(ValueError):
            fractional_kelly([1e308], [[1e-308]])

    def test_black_litterman_rejects_finite_inputs_with_overflowing_prior(self):
        with self.assertRaises(ValueError):
            black_litterman([[1e308]], [1], [[1]], [0.1], tau=1e308)

    def test_covariance_resource_bound_precedes_eigendecomposition(self):
        with self.assertRaises(ValueError):
            risk_parity(np.eye(513))

    def test_invalid_covariance_and_infeasible_policy_rejected(self):
        for covariance in [[[1, 2], [2, 1]], [[1, 1], [1, 1]], [[1, np.nan], [0, 1]]]:
            with self.assertRaises(ValueError):
                risk_parity(covariance)
        with self.assertRaises(ValueError):
            risk_parity(np.eye(2), max_weight=0.49)
        with self.assertRaises(ValueError):
            fractional_kelly([0.1], [[1]], fraction=2)
        with self.assertRaises(ValueError):
            fractional_kelly([0.1], [[1]], allow_short="false")


if __name__ == "__main__":
    unittest.main()
