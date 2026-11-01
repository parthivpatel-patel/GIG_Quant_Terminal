"""Actual optional CPU research backends, synthetic fixtures, no economic claims."""

from __future__ import annotations

import builtins
import hashlib
import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from gig.ml.ranker import walk_forward_scores
from gig.research.stat_arb import adf_test, cointegration_test, select_pairs
from gig.research.walkforward import WalkForwardSplit
from gig.research.workflow import ModelRegistry, ResearchWorkflow


def _installed(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


def _regression_fixture():
    rng = np.random.default_rng(20261031)
    index = pd.date_range("2020-01-01", periods=95, tz="UTC")
    features = pd.DataFrame(rng.normal(size=(95, 3)), index=index, columns=["f1", "f2", "f3"])
    target = pd.Series(
        np.sin(features["f1"]) + 0.5 * features["f2"] + rng.normal(0, 0.02, 95), index=index
    )
    label_end = pd.Series(index + pd.Timedelta(days=3), index=index)
    available = pd.Series(index - pd.Timedelta(nanoseconds=1), index=index)
    return features, target, label_end, available


@unittest.skipUnless(_installed("lightgbm"), "actual LightGBM runtime is not installed")
class ActualLightGBMWorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        cls.features, cls.target, cls.label_end, cls.available = _regression_fixture()
        cls.workflow = ResearchWorkflow(Path(cls.directory.name))
        with threadpool_limits(limits=1):
            cls.result = cls.workflow.run(
                cls.features,
                cls.target,
                label_end=cls.label_end,
                feature_available_at=cls.available,
                model="lightgbm",
                min_train=60,
                test_size=15,
                embargo=1,
            )

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def test_actual_model_registration_reload_and_prediction_parity(self):
        import lightgbm

        registry = ModelRegistry(Path(self.directory.name) / "models")
        self.assertEqual(self.result["config"]["model"], "lightgbm")
        for fold in self.result["folds"]:
            model_id = fold["model_id"]
            state = registry.load(model_id)
            model_path = Path(self.directory.name) / "models" / f"{model_id}.json"
            self.assertEqual(hashlib.sha256(model_path.read_bytes()).hexdigest(), model_id)
            self.assertEqual(state["model"], "lightgbm")
            self.assertEqual(state["library_version"], lightgbm.__version__)
            self.assertIn("Tree=0", state["booster_text"])
            self.assertFalse(state["trading_approved"])
            rows = [row for row in self.result["predictions"] if row["fold"] == fold["fold"]]
            timestamps = pd.DatetimeIndex([row["timestamp"] for row in rows])
            with threadpool_limits(limits=1):
                reloaded = registry.predict(model_id, self.features.loc[timestamps])
            np.testing.assert_allclose(
                reloaded, [row["prediction"] for row in rows], rtol=0, atol=1e-12
            )

    def test_actual_workflow_purges_unavailable_labels_and_preserves_source_identity(self):
        self.assertEqual(len(self.result["folds"]), 2)
        for fold in self.result["folds"]:
            self.assertEqual(fold["purged_count"], 2)
            self.assertLess(
                pd.Timestamp(fold["max_training_label_end"]), pd.Timestamp(fold["test_start"])
            )
            train = self.features.loc[: pd.Timestamp(fold["train_end"])].to_numpy()
            np.testing.assert_allclose(
                fold["train_only_mean"], train.mean(axis=0), rtol=0, atol=1e-14
            )
        self.assertFalse(self.result["economically_validated"])
        self.assertFalse(self.result["trading_approved"])

    def test_repeated_actual_training_is_reproducible(self):
        with threadpool_limits(limits=1):
            repeated = self.workflow.run(
                self.features,
                self.target,
                label_end=self.label_end,
                feature_available_at=self.available,
                model="lightgbm",
                min_train=60,
                test_size=15,
                embargo=1,
            )
        self.assertEqual(repeated["run_id"], self.result["run_id"])
        self.assertEqual(repeated["dataset_hash"], self.result["dataset_hash"])
        self.assertEqual(repeated["predictions"], self.result["predictions"])

    def test_future_labels_do_not_change_first_fitted_model_or_predictions(self):
        changed = self.target.copy()
        changed.iloc[-1] += 1000
        with tempfile.TemporaryDirectory() as directory, threadpool_limits(limits=1):
            result = ResearchWorkflow(Path(directory)).run(
                self.features,
                changed,
                label_end=self.label_end,
                feature_available_at=self.available,
                model="lightgbm",
                min_train=60,
                test_size=15,
                embargo=1,
            )
        self.assertEqual(result["folds"][0]["model_id"], self.result["folds"][0]["model_id"])
        original = [row["prediction"] for row in self.result["predictions"] if row["fold"] == 0]
        updated = [row["prediction"] for row in result["predictions"] if row["fold"] == 0]
        np.testing.assert_array_equal(updated, original)
        self.assertNotEqual(result["dataset_hash"], self.result["dataset_hash"])

    def test_registered_lightgbm_does_not_become_ridge_if_dependency_disappears(self):
        registry = ModelRegistry(Path(self.directory.name) / "models")
        identity = self.result["folds"][0]["model_id"]
        with (
            patch("gig.research.workflow._exists", return_value=False),
            self.assertRaisesRegex(ImportError, "no model substitution"),
        ):
            registry.predict(identity, self.features.iloc[:2])


def _ranking_fixture():
    rng = np.random.default_rng(901)
    index = pd.date_range("2020-01-01", periods=29, tz="UTC")
    names = [f"SYN{i}" for i in range(8)]
    factor = pd.DataFrame(rng.normal(size=(29, 8)), index=index, columns=names)
    labels = 0.1 * factor + pd.DataFrame(rng.normal(0, 0.01, (29, 8)), index=index, columns=names)
    return factor, labels


@unittest.skipUnless(_installed("lightgbm"), "actual LightGBM runtime is not installed")
class ActualLambdaRankTests(unittest.TestCase):
    def test_actual_objective_groups_and_oos_admission(self):
        import lightgbm

        factors, labels = _ranking_fixture()
        with (
            threadpool_limits(limits=1),
            patch("lightgbm.train", wraps=lightgbm.train) as trained,
            patch(
                "gig.ml.ranker._fit_sklearn",
                side_effect=AssertionError("unexpected model substitution"),
            ),
        ):
            scores = walk_forward_scores(
                {"factor": factors},
                labels,
                min_train=12,
                test_size=6,
                embargo=2,
                backend="lightgbm",
            )
        self.assertEqual(scores.attrs["ranker_backend"], "lightgbm")
        self.assertEqual(trained.call_count, 2)
        for call in trained.call_args_list:
            params, dataset = call.args[:2]
            self.assertEqual(params["objective"], "lambdarank")
            self.assertEqual(params["seed"], 42)
            self.assertTrue(np.all(dataset.get_group() == 8))
            self.assertEqual(int(dataset.get_group().sum()), len(dataset.get_label()))
        first = next(WalkForwardSplit(12, 6, 2).split(factors.index))
        self.assertTrue(scores.loc[first.train_index].isna().all().all())
        self.assertTrue(scores.iloc[12:14].isna().all().all())
        self.assertTrue(np.isfinite(scores.loc[first.test_index]).all().all())
        self.assertGreater(float(scores.loc[first.test_index].to_numpy().std()), 0)

    def test_future_features_and_labels_do_not_change_prior_oos_predictions(self):
        factors, labels = _ranking_fixture()
        first = next(WalkForwardSplit(12, 6, 2).split(factors.index))
        changed_factors, changed_labels = factors.copy(), labels.copy()
        after = factors.index > first.test_end
        changed_factors.loc[after] += 1000
        changed_labels.loc[after] -= 1000
        with threadpool_limits(limits=1):
            original = walk_forward_scores(
                {"factor": factors},
                labels,
                min_train=12,
                test_size=6,
                embargo=2,
                backend="lightgbm",
            )
            changed = walk_forward_scores(
                {"factor": changed_factors},
                changed_labels,
                min_train=12,
                test_size=6,
                embargo=2,
                backend="lightgbm",
            )
        np.testing.assert_allclose(
            original.loc[first.test_index], changed.loc[first.test_index], rtol=0, atol=1e-12
        )


class ExplicitMissingDependencyTests(unittest.TestCase):
    def test_explicit_ranking_backend_absence_raises_without_substitution(self):
        factors, labels = _ranking_fixture()
        with (
            patch(
                "gig.ml.ranker.available_backends",
                return_value={"lightgbm": False, "sklearn": True},
            ),
            patch(
                "gig.ml.ranker._fit_sklearn", side_effect=AssertionError("unexpected substitution")
            ),
            self.assertRaises(ImportError),
        ):
            walk_forward_scores(
                {"factor": factors},
                labels,
                min_train=12,
                test_size=6,
                embargo=2,
                backend="lightgbm",
            )

    def test_explicit_workflow_dependency_absence_raises_without_ridge_artifacts(self):
        features, target, ends, available = _regression_fixture()
        with tempfile.TemporaryDirectory() as directory:
            workflow = ResearchWorkflow(Path(directory))
            with (
                patch("gig.research.workflow._exists", return_value=False),
                self.assertRaisesRegex(ImportError, "no model substitution"),
            ):
                workflow.run(
                    features,
                    target,
                    label_end=ends,
                    feature_available_at=available,
                    model="lightgbm",
                    min_train=60,
                    test_size=15,
                    embargo=1,
                )
            self.assertEqual(list((Path(directory) / "models").glob("*.json")), [])


def _pair_fixture():
    rng = np.random.default_rng(20261031)
    independent = 200 + np.cumsum(rng.normal(0, 1, 400))
    innovations = rng.normal(0, 0.25, 400)
    stationary = np.zeros(400)
    for index in range(1, len(stationary)):
        stationary[index] = 0.4 * stationary[index - 1] + innovations[index]
    dependent = 3 + 1.5 * independent + stationary
    unrelated = 300 + np.cumsum(rng.normal(0, 1, 400))
    return independent, dependent, unrelated, stationary


def _adf_ols_statistic(values, lags):
    """Independent OLS t-stat for the augmented level term, conditional on lag count."""
    changes = np.diff(values)
    response = changes[lags:]
    design = np.column_stack(
        [values[lags:-1], np.ones(len(response))]
        + [changes[lags - lag : -lag] for lag in range(1, lags + 1)]
    )
    coefficients = np.linalg.lstsq(design, response, rcond=None)[0]
    residual = response - design @ coefficients
    residual_variance = (residual @ residual) / (len(response) - design.shape[1])
    covariance = residual_variance * np.linalg.inv(design.T @ design)
    return coefficients[0] / np.sqrt(covariance[0, 0])


@unittest.skipUnless(_installed("statsmodels"), "actual statsmodels runtime is not installed")
class ActualStatisticalInferenceTests(unittest.TestCase):
    def test_engle_granger_synthetic_pair_and_independent_walk_references(self):
        independent, dependent, unrelated, _ = _pair_fixture()
        pair = cointegration_test(dependent, independent, log_prices=False)
        other = cointegration_test(unrelated, independent, log_prices=False)
        self.assertEqual(pair["state"], "AVAILABLE")
        self.assertEqual(other["state"], "AVAILABLE")
        self.assertAlmostEqual(pair["statistic"], -13.633566902761546, places=8)
        self.assertAlmostEqual(other["statistic"], -2.6380088012679375, places=8)
        self.assertLess(pair["pvalue"], 0.001)
        self.assertGreater(other["pvalue"], 0.1)
        self.assertEqual(pair["null"], "NO_COINTEGRATION")
        self.assertFalse(pair["multiple_testing_adjusted"])
        self.assertFalse(pair["trading_approved"])

    def test_actual_adf_matches_independent_ols_augmented_level_t_statistic(self):
        _, _, _, stationary = _pair_fixture()
        result = adf_test(stationary)
        self.assertEqual(result["state"], "AVAILABLE")
        reference = _adf_ols_statistic(stationary, result["used_lag"])
        self.assertAlmostEqual(result["statistic"], reference, places=9)
        self.assertLess(result["pvalue"], 0.001)
        self.assertEqual(result["null"], "UNIT_ROOT")
        self.assertFalse(result["post_selection_inference_validated"])

    def test_actual_adf_does_not_promote_random_walk_to_stationary(self):
        independent, _, _, _ = _pair_fixture()
        result = adf_test(independent)
        self.assertEqual(result["state"], "AVAILABLE")
        self.assertGreater(result["pvalue"], 0.1)

    def test_screening_adjusts_entire_pair_family_and_never_reads_future_prefix(self):
        independent, dependent, unrelated, _ = _pair_fixture()
        frame = pd.DataFrame(
            {"X": independent, "Y": dependent, "Z": unrelated},
            index=pd.date_range("2020-01-01", periods=400, tz="UTC"),
        )
        result = select_pairs(frame, training_rows=300, log_prices=False)
        changed = frame.copy()
        changed.iloc[300:] *= 100
        self.assertEqual(select_pairs(changed, training_rows=300, log_prices=False), result)
        self.assertEqual(result["family_size"], 3)
        self.assertEqual(result["correction"], "BONFERRONI_ALL_SCREENED_PAIRS")
        selected = []
        for pair in result["pairs"]:
            self.assertEqual(pair["state"], "AVAILABLE")
            self.assertEqual(pair["bonferroni_pvalue"], min(1, pair["pvalue"] * 3))
            if pair["selected"]:
                selected.append((pair["dependent"], pair["independent"]))
        self.assertEqual(selected, [("X", "Y")])

    def test_collinearity_warning_remains_a_numerical_error(self):
        independent, _, _, _ = _pair_fixture()
        result = cointegration_test(2 * independent + 3, independent, log_prices=False)
        self.assertEqual(result["state"], "ERROR")
        self.assertEqual(result["code"], "COINTEGRATION_NUMERICAL_DIAGNOSTIC")
        self.assertIsNone(result["pvalue"])

    def test_older_adf_signature_without_result_object_remains_supported(self):
        observed = []

        def legacy_adf(values, regression, autolag):
            observed.append((len(values), regression, autolag))
            return -3.0, 0.03, 1, len(values) - 2, {"1%": -3.4, "5%": -2.8, "10%": -2.5}, 10.0

        with patch("statsmodels.tsa.stattools.adfuller", legacy_adf):
            result = adf_test(np.arange(40, dtype=float))
        self.assertEqual(result["state"], "AVAILABLE")
        self.assertEqual(observed, [(40, "c", "AIC")])


class MissingStatisticalInferenceTests(unittest.TestCase):
    def test_absent_significance_backend_is_unavailable_without_replacement(self):
        original_import = builtins.__import__

        def without_statsmodels(name, *args, **kwargs):
            if name == "statsmodels.tsa.stattools":
                raise ImportError("simulated missing optional dependency")
            return original_import(name, *args, **kwargs)

        independent, dependent, _, stationary = _pair_fixture()
        with patch("builtins.__import__", side_effect=without_statsmodels):
            for result in (
                cointegration_test(dependent, independent, log_prices=False),
                adf_test(stationary),
            ):
                self.assertEqual(result["state"], "UNAVAILABLE")
                self.assertEqual(result["code"], "DEPENDENCY_STATSMODELS_MISSING")
                self.assertIsNone(result["statistic"])
                self.assertIsNone(result["pvalue"])


if __name__ == "__main__":
    unittest.main()
