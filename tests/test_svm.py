"""Actual SVR reference, strict JSON replay and purged persisted research."""
from __future__ import annotations

import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

import numpy as np

from gig.ml.svm import CACHE_MIB, EPSILON, MAX_ITER, TOLERANCE, fit_svr, predict_svr
from gig.research.model_pipeline import configuration, fit_dataset, identity, saved_predict
from gig.research.workflow import model_capabilities
from gig.service.dataset_workspace import PersistedDatasetWorkspace
from gig.service.persisted_research import PersistedResearchWorkspace
from test_persisted_data import fixture_times, seed_persisted_fixture


class SVRKernelTests(unittest.TestCase):
    def setUp(self):
        self.x = np.random.default_rng(17).normal(size=(48, 3))
        self.x[:, 2] = 7.
        self.y = .03*self.x[:, 0] - .02*self.x[:, 1]**2
        self.names = ["lag_return", "volatility", "constant"]

    def test_actual_sklearn_independent_rbf_and_training_only_scaler(self):
        from sklearn.svm import SVR
        state = fit_svr(self.x, self.y, self.names)
        mean, scale = self.x.mean(axis=0), self.x.std(axis=0)
        scale[scale == 0] = 1.
        ymean, yscale = self.y.mean(), self.y.std()
        z = (self.x-mean)/scale
        estimator = SVR(kernel="rbf", gamma=1/3, C=1., epsilon=EPSILON, tol=TOLERANCE,
                        shrinking=False, cache_size=CACHE_MIB, max_iter=MAX_ITER).fit(z, (self.y-ymean)/yscale)
        query = self.x[:9]+np.array([20., -10., 0.])
        before = deepcopy(state)
        actual = predict_svr(state, query, self.names)
        expected = estimator.predict((query-mean)/scale)*yscale+ymean
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-14)
        manual = []
        for row in (query-mean)/scale:
            total = state["intercept"]
            for vector, dual in zip(state["support_vectors"], state["dual"], strict=True):
                total += dual*np.exp(-sum((a-b)**2 for a, b in zip(row, vector, strict=True))/3)
            manual.append(total*yscale+ymean)
        np.testing.assert_allclose(actual, manual, rtol=1e-12, atol=1e-14)
        np.testing.assert_array_equal(state["mean"], mean)
        np.testing.assert_array_equal(state["scale"], scale)
        self.assertEqual(state, before)
        self.assertEqual(state["fit_backend"], "sklearn.svm.SVR")
        self.assertFalse(state["production_promoted"])

    def test_constant_targets_empty_support_and_no_fit_json_replay(self):
        for target in (self.y, np.full(48, .025)):
            state = fit_svr(self.x, target, self.names)
            expected = predict_svr(state, self.x, self.names)
            with patch("sklearn.svm.SVR.fit", side_effect=AssertionError("no inference fit")):
                actual = predict_svr(json.loads(json.dumps(state)), self.x, self.names)
            np.testing.assert_array_equal(actual, expected)
            if np.all(target == target[0]):
                self.assertEqual(state["support_vectors"], [])
                self.assertEqual(state["target_scale"], 1.)
                np.testing.assert_array_equal(actual, target)

    def test_adversarial_inputs_feature_order_and_work_bounds(self):
        for value in (True, "1", float("nan"), float("inf"), 10**1000):
            x = self.x.tolist()
            x[0][0] = value
            with self.subTest(value_type=type(value).__name__), self.assertRaises(ValueError):
                fit_svr(x, self.y, self.names)
        for count in (19, 129):
            with self.assertRaises(ValueError):
                fit_svr(np.zeros((count, 3)), np.zeros(count), self.names)
        with self.assertRaises(ValueError):
            fit_svr(np.zeros((48, 17)), self.y, [str(i) for i in range(17)])
        state = fit_svr(self.x, self.y, self.names)
        for query, names in ((self.x, self.names[::-1]), (np.zeros((2001, 3)), self.names),
                             (np.full((2, 3), 1e12), self.names)):
            with self.assertRaises(ValueError):
                predict_svr(state, query, names)
        with patch.dict("sys.modules", {"sklearn.svm": None}), self.assertRaises(ImportError):
            fit_svr(self.x, self.y, self.names)

    def test_saved_state_tampering_and_nonconvergence_rejected(self):
        original = fit_svr(self.x, self.y, self.names)
        mutations = [lambda s: s.update(threads=True), lambda s: s.update(trading_approved=True),
                     lambda s: s.update(production_promoted=True), lambda s: s.update(fit_status=True),
                     lambda s: s.update(iterations=MAX_ITER), lambda s: s["mean"].__setitem__(0, 9.),
                     lambda s: s["parameters"].update(gamma=9.), lambda s: s["parameters"].update(C=True),
                     lambda s: s["dual"].__setitem__(0, 3.),
                     lambda s: s["training_y"].__setitem__(0, .9),
                     lambda s: s["support_indices"].__setitem__(0, True),
                     lambda s: s["support_vectors"][0].__setitem__(0, 10.)]
        for number, mutate in enumerate(mutations):
            state = deepcopy(original)
            mutate(state)
            with self.subTest(case=number), self.assertRaises(ValueError):
                predict_svr(state, self.x, self.names)
        from sklearn.svm import SVR
        fit = SVR.fit
        def nonconverging(estimator, *args, **kwargs):
            fit(estimator, *args, **kwargs)
            estimator.fit_status_ = 1
            return estimator
        with patch("sklearn.svm.SVR.fit", nonconverging), self.assertRaisesRegex(ValueError, "did not converge"):
            fit_svr(self.x, self.y, self.names)


class PersistedSVRTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)
        request = seed_persisted_fixture(self.base)
        self.dataset = PersistedDatasetWorkspace(self.base).build(request).to_dict()["data"]
        self.request = {"dataset_id": self.dataset["dataset_id"], "model": "svr", "min_train": 30,
                        "test_size": 12, "embargo": 1, "seed": 17}

    def test_future_revisions_features_and_targets_cannot_change_first_fit(self):
        config = configuration(self.request)
        before, models = fit_dataset(self.dataset, config)
        changed = deepcopy(self.dataset)
        for row in changed["rows"]:
            if row["timestamp"] >= fixture_times()[55].isoformat():
                row["features"]["momentum"] += .25
                row["target"] += .1
            row.setdefault("label_vintages", [{"target": row["target"], "label_end": row["label_end"], "knowledge_at": row["label_end"]}]).append(
                {"target": .9, "label_end": row["label_end"], "knowledge_at": "2027-01-01T00:00:00Z"})
        changed["dataset_id"] = identity(changed)
        after, later = fit_dataset(changed, config)
        first, second = [[r for r in artifact["rows"] if r["fold"] == 0] for artifact in (before, after)]
        self.assertEqual(first, second)
        for row in first:
            self.assertEqual(models[row["model_id"]], later[row["model_id"]])
        self.assertFalse(before["usable_for_portfolio"])
        self.assertTrue(all(not r["usable_for_portfolio"] and r["uncertainty"]["value"] is None for r in before["rows"]))

    def test_fold_metrics_and_train_only_mean_baseline(self):
        artifact, models = fit_dataset(self.dataset, configuration(self.request))
        for fold in artifact["folds"]:
            rows = [r for r in artifact["rows"] if r["model_id"] == fold["model_id"]]
            truth, prediction = np.array([r["target"] for r in rows]), np.array([r["prediction"] for r in rows])
            train_mean = np.mean(models[fold["model_id"]]["estimator"]["training_y"])
            self.assertEqual(fold["mse"], float(np.mean((truth-prediction)**2)))
            self.assertEqual(fold["mae"], float(np.mean(np.abs(truth-prediction))))
            self.assertEqual(fold["training_mean_baseline_mse"], float(np.mean((truth-train_mean)**2)))
            self.assertLess(fold["max_training_label_end"], fold["test_start"])
            self.assertEqual(fold["economic_claim"], "NONE")

    def test_label_knowledge_at_cutoff_is_strictly_purged(self):
        baseline, _ = fit_dataset(self.dataset, configuration(self.request))
        cutoff = baseline["folds"][0]["test_start"]
        changed = deepcopy(self.dataset)
        victim = next(r for r in changed["rows"] if r["instrument_id"] == "US:AAA" and r["timestamp"] < cutoff)
        victim["label_vintages"] = [{"target": .8, "label_end": victim["label_end"], "knowledge_at": cutoff}]
        changed["dataset_id"] = identity(changed)
        artifact, models = fit_dataset(changed, configuration(self.request))
        fold = next(f for f in artifact["folds"] if f["fold"] == 0 and f["instrument_id"] == "US:AAA")
        self.assertNotIn(victim["timestamp"], models[fold["model_id"]]["training_timestamps"])

    def test_restart_saved_inference_without_fit_and_explicit_research_fence(self):
        workspace = PersistedResearchWorkspace(self.base)
        result = workspace.run(self.request).to_dict()
        reopened = PersistedResearchWorkspace(self.base)
        with patch("sklearn.svm.SVR.fit", side_effect=AssertionError("restart may not train")):
            predicted = reopened.predict({"dataset_id": self.request["dataset_id"], "run_id": result["data"]["run_id"]}).to_dict()
            repeated = PersistedResearchWorkspace(self.base).predict({"dataset_id": self.request["dataset_id"], "run_id": result["data"]["run_id"]}).to_dict()
        self.assertEqual(predicted, repeated)
        self.assertEqual(predicted["state"], "AVAILABLE")
        self.assertFalse(predicted["data"]["metrics"]["training_performed"])
        artifact, models = fit_dataset(self.dataset, configuration(self.request))
        for state in models.values():
            actual = saved_predict(state, self.dataset)
            expected = [r for r in artifact["rows"] if r["model_id"] == identity(state)]
            lookup = {r["timestamp"]: r for r in actual}
            for row in expected:
                self.assertEqual(row["prediction"], lookup[row["timestamp"]]["prediction"])
            forged = {**state, "fit_state": "CONVERGED"}
            self.assertTrue(all(not r["usable_for_portfolio"] for r in saved_predict(forged, self.dataset)))
        self.assertFalse((self.base / "execution-engine").exists())

    def test_all_plans_preflight_before_fit_and_capability_is_one_family(self):
        with patch("gig.research.svm.fit_svr", side_effect=AssertionError("preflight first")), self.assertRaises(ValueError):
            fit_dataset(self.dataset, {**configuration(self.request), "min_train": 20, "test_size": 4})
        with self.assertRaises(ValueError):
            configuration({**self.request, "min_train": 129})
        with self.assertRaises(ValueError):
            configuration({**self.request, "iterations": 5})
        capabilities = model_capabilities()
        self.assertTrue(capabilities["support_vector_machines_svr"]["implemented"])
        self.assertFalse(capabilities["support_vector_machines_svr"]["trained"])
        self.assertFalse(capabilities["support_vector_machines_svr"]["trading_approved"])
