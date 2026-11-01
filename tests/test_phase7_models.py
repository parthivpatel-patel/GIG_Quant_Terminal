"""Independent numerical references; runnable without optional pytest/DuckDB."""

from __future__ import annotations

import itertools
import math
import unittest

import numpy as np

from gig.ml.regimes import GaussianHMM, fit_gaussian_hmm, state_duration_diagnostics
from gig.ml.state_space import kalman_filter, rts_smooth
from gig.ml.volatility import GARCH11, fit_garch11


class KalmanReferences(unittest.TestCase):
    @staticmethod
    def filter(y):
        return kalman_filter(np.asarray(y, dtype=float)[:, None], [[1]], [[1]], [[1]], [[1]], [0], [[1]])

    def test_scalar_hand_calculation(self):
        result = self.filter([1, 2, 3])
        np.testing.assert_allclose(result.filtered_mean[:, 0], [.5, 1.4, 31 / 13])
        np.testing.assert_allclose(result.filtered_covariance[:, 0, 0], [.5, .6, 8 / 13])
        np.testing.assert_allclose(result.innovations[:, 0], [1, 1.5, 1.6])
        s = np.array([2, 2.5, 2.6])
        v = np.array([1, 1.5, 1.6])
        reference_loglik = -.5 * np.sum(np.log(2 * np.pi * s) + v * v / s)
        self.assertAlmostEqual(result.log_likelihood, reference_loglik, places=12)

    def test_rts_matches_independent_joint_gaussian_precision(self):
        # Prior N(0,1), random-walk variance 1, observation variance 1.
        # Assemble the joint posterior precision directly, without filtering.
        precision = np.array([[3., -1., 0.], [-1., 3., -1.], [0., -1., 2.]])
        expected_covariance = np.linalg.inv(precision)
        expected_mean = np.linalg.solve(precision, [1, 2, 3])
        result = self.filter([1, 2, 3])
        smooth = rts_smooth(result)
        np.testing.assert_allclose(smooth.mean[:, 0], expected_mean, atol=1e-14)
        np.testing.assert_allclose(smooth.covariance[:, 0, 0], expected_covariance.diagonal(), atol=1e-14)
        self.assertTrue(smooth.uses_future_observations)
        self.assertTrue((smooth.covariance[:, 0, 0] <= result.filtered_covariance[:, 0, 0] + 1e-14).all())

    def test_filter_is_prefix_causal(self):
        prefix = self.filter([1, 2])
        full = self.filter([1, 2, 300])
        np.testing.assert_array_equal(prefix.filtered_mean, full.filtered_mean[:2])
        np.testing.assert_array_equal(prefix.filtered_covariance, full.filtered_covariance[:2])
        self.assertNotEqual(rts_smooth(full).mean[0, 0], prefix.filtered_mean[0, 0])

    def test_multivariate_time_varying_design_covariances(self):
        rng = np.random.default_rng(19)
        observation = rng.normal(size=(40, 1, 2))
        result = kalman_filter(rng.normal(size=(40, 1)), np.eye(2), observation,
                               np.eye(2) * .001, [[.2]], [0, 0], np.eye(2))
        smooth = rts_smooth(result)
        for covariance in np.concatenate([result.filtered_covariance, smooth.covariance]):
            np.testing.assert_allclose(covariance, covariance.T, atol=1e-13)
            self.assertGreaterEqual(np.linalg.eigvalsh(covariance).min(), -1e-12)

    def test_input_and_result_mutation_protection(self):
        transition = np.ones((1, 1))
        result = kalman_filter([[1], [2]], transition, [[1]], [[1]], [[1]], [0], [[1]])
        transition[0, 0] = 7
        self.assertEqual(result.transition[0, 0], 1)
        with self.assertRaises(ValueError):
            result.filtered_mean[0, 0] = 99
        with self.assertRaises(ValueError):
            result.filtered_mean.setflags(write=True)

    def test_invalid_numerics_and_shapes(self):
        for value in [float("nan"), float("inf"), -float("inf")]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.filter([value])
        for covariance in [[[-1]], [[0]], [[float("nan")]]]:
            with self.subTest(covariance=covariance), self.assertRaises(ValueError):
                kalman_filter([[1]], [[1]], [[1]], [[1]], covariance, [0], [[1]])
        with self.assertRaises(ValueError):
            kalman_filter([[1]], np.eye(2), [[1]], [[1]], [[1]], [0], [[1]])


class HMMReferences(unittest.TestCase):
    @staticmethod
    def model():
        return GaussianHMM([.6, .4], [[.8, .2], [.3, .7]], [[-1], [1]], [[.5], [.8]])

    @staticmethod
    def enumerate_reference(model, values):
        paths, probabilities = [], []
        for path in itertools.product(range(2), repeat=len(values)):
            probability = model.initial[path[0]]
            for t, state in enumerate(path):
                variance = model.variances[state, 0]
                probability *= math.exp(-.5 * (values[t] - model.means[state, 0]) ** 2 / variance) / math.sqrt(2 * math.pi * variance)
                if t:
                    probability *= model.transition[path[t - 1], state]
            paths.append(path)
            probabilities.append(probability)
        total = sum(probabilities)
        posterior = np.zeros((len(values), 2))
        transitions = np.zeros((2, 2))
        for path, probability in zip(paths, probabilities, strict=True):
            for t, state in enumerate(path):
                posterior[t, state] += probability / total
                if t:
                    transitions[path[t - 1], state] += probability / total
        best = paths[int(np.argmax(probabilities))]
        return math.log(total), posterior, transitions, best

    def test_forward_backward_and_viterbi_against_enumerated_paths(self):
        model = self.model()
        values = [-.7, .2, 1.3]
        expected_loglik, expected_probabilities, expected_counts, expected_path = self.enumerate_reference(model, values)
        result = model.infer(values)
        self.assertAlmostEqual(result.log_likelihood, expected_loglik, places=12)
        np.testing.assert_allclose(result.smoothed_probabilities, expected_probabilities, atol=1e-14)
        np.testing.assert_allclose(result.transition_counts, expected_counts, atol=1e-14)
        np.testing.assert_array_equal(model.decode(values), expected_path)
        for t in range(len(values)):
            prefix = self.enumerate_reference(model, values[:t + 1])[1][-1]
            np.testing.assert_allclose(result.filtered_probabilities[t], prefix, atol=1e-14)

    def test_zero_transition_probabilities_supported(self):
        model = GaussianHMM([1, 0], np.eye(2), [[0], [1]], [[1], [1]])
        result = model.infer([0, 1, 2])
        np.testing.assert_allclose(result.smoothed_probabilities, [[1, 0]] * 3)
        np.testing.assert_array_equal(model.decode([0, 1, 2]), [0, 0, 0])

    def test_baum_welch_monotone_and_separated_regimes(self):
        rng = np.random.default_rng(71)
        values = np.r_[rng.normal(-2, .3, 80), rng.normal(2, .3, 80), rng.normal(-2, .3, 80)]
        fit = fit_gaussian_hmm(values, max_iterations=50)
        self.assertTrue(fit.converged, fit.stop_reason)
        self.assertTrue((np.diff(fit.likelihood_history) >= -1e-10).all())
        np.testing.assert_allclose(fit.model.means[:, 0], [-2, 2], atol=.1)
        decoded = fit.model.decode(values)
        np.testing.assert_array_equal(decoded, [0] * 80 + [1] * 80 + [0] * 80)
        np.testing.assert_allclose(fit.model.transition.sum(axis=1), 1)

    def test_single_state_mle(self):
        values = np.array([1., 2., 4., 5.])
        fit = fit_gaussian_hmm(values, states=1)
        self.assertAlmostEqual(fit.model.means[0, 0], values.mean())
        self.assertAlmostEqual(fit.model.variances[0, 0], values.var())
        np.testing.assert_allclose(fit.model.infer(values).filtered_probabilities, 1)

    def test_duration_censoring_and_absorbing_state(self):
        diagnostics = state_duration_diagnostics([0, 0, 1, 1, 0, 1, 1, 1], [[.8, .2], [.3, .7]])
        self.assertEqual(diagnostics[0]["complete_run_mean"], 1)
        self.assertEqual(diagnostics[1]["complete_run_mean"], 2)
        self.assertAlmostEqual(diagnostics[0]["expected_duration"], 5)
        self.assertTrue(diagnostics[0]["runs"][0]["left_censored"])
        self.assertTrue(diagnostics[1]["runs"][-1]["right_censored"])
        absorbing = state_duration_diagnostics([0, 0, 0], [[1]])[0]
        self.assertIsNone(absorbing["expected_duration"])
        self.assertEqual(absorbing["expected_duration_state"], "ABSORBING")
        self.assertIsNone(absorbing["complete_run_mean"])

    def test_parameter_admission_and_mutation(self):
        means = np.array([[-1.], [1.]])
        model = GaussianHMM([.5, .5], [[.8, .2], [.2, .8]], means, [[1], [1]])
        means[:] = 999
        self.assertEqual(model.means[0, 0], -1)
        with self.assertRaises(ValueError):
            model.means[0, 0] = 999
        with self.assertRaises(ValueError):
            model.means.setflags(write=True)
        for kwargs in [
            {"initial": [.3, .3]}, {"transition": [[.8, .3], [.2, .8]]},
            {"variances": [[0], [1]]}, {"means": [[float("nan")], [1]]},
        ]:
            params = dict(initial=[.5, .5], transition=[[.8, .2], [.2, .8]], means=[[-1], [1]], variances=[[1], [1]])
            params.update(kwargs)
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                GaussianHMM(**params)
        with self.assertRaises(ValueError):
            model.infer([float("inf")])


class GARCHReferences(unittest.TestCase):
    def test_recursion_likelihood_and_forecast_against_hand_reference(self):
        model = GARCH11(.04, .2, .5)
        values = np.array([.1, -.2, .3])
        h0 = .04 / .3
        h1 = .04 + .2 * .1 ** 2 + .5 * h0
        h2 = .04 + .2 * .2 ** 2 + .5 * h1
        expected_h = np.array([h0, h1, h2])
        np.testing.assert_allclose(model.conditional_variances(values), expected_h, atol=1e-15)
        expected_likelihood = sum(-.5 * (math.log(2 * math.pi * variance) + value ** 2 / variance) for value, variance in zip(values, expected_h, strict=True))
        self.assertAlmostEqual(model.log_likelihood(values), expected_likelihood, places=12)
        first = .04 + .2 * .3 ** 2 + .5 * h2
        np.testing.assert_allclose(model.forecast(values, horizon=3), [first, .04 + .7 * first, .04 + .7 * (.04 + .7 * first)])
        self.assertAlmostEqual(model.forecast(values, horizon=1000)[-1], model.unconditional_variance, places=12)

    def test_likelihood_is_scale_consistent(self):
        values = np.array([.1, -.2, .3])
        a = GARCH11(.04, .2, .5, .01)
        b = GARCH11(.04 * 100 ** 2, .2, .5, 1)
        self.assertAlmostEqual(b.log_likelihood(values * 100), a.log_likelihood(values) - len(values) * math.log(100), places=12)

    def test_mle_on_deterministic_synthetic_training_sequence(self):
        rng = np.random.default_rng(37)
        generating = GARCH11(.000004, .12, .82, .0002)
        values = np.empty(600)
        variance = generating.unconditional_variance
        for t in range(len(values)):
            values[t] = generating.mean + math.sqrt(variance) * rng.normal()
            variance = generating.omega + generating.alpha * (values[t] - generating.mean) ** 2 + generating.beta * variance
        fit = fit_garch11(values)
        self.assertTrue(fit.converged, fit.stop_reason)
        self.assertLess(fit.model.alpha + fit.model.beta, 1)
        self.assertGreater(fit.model.omega, 0)
        constant = GARCH11(float(values.var()), 0, 0, float(values.mean()))
        self.assertGreaterEqual(fit.log_likelihood, constant.log_likelihood(values) - 1e-6)
        self.assertEqual(fit.observations, len(values))
        self.assertTrue(np.isfinite(fit.model.forecast(values, 10)).all())

    def test_nonconvergence_is_explicit(self):
        values = np.random.default_rng(23).normal(size=100)
        fit = fit_garch11(values, max_iterations=1)
        self.assertFalse(fit.converged)
        self.assertEqual(fit.stop_reason, "OPTIMIZER_NOT_CONVERGED")

    def test_invalid_parameters_and_data_rejected(self):
        for values in [(0, .1, .8), (.01, -.1, .8), (.01, .2, .8), (.01, .1, float("nan"))]:
            with self.subTest(values=values), self.assertRaises(ValueError):
                GARCH11(*values)
        for values in [[0] * 40, [float("nan")] * 40, [1, 2]]:
            with self.subTest(values=values), self.assertRaises(ValueError):
                fit_garch11(values)
        for horizon in [0, -1, True, 1.5]:
            with self.subTest(horizon=horizon), self.assertRaises(ValueError):
                GARCH11(.01, .1, .8).forecast([1, 2], horizon)


if __name__ == "__main__":
    unittest.main()
