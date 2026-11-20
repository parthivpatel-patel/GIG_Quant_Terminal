"""Genuine CPU fitting, Gaussian references, purged persistence and HTTP fences."""
from __future__ import annotations

import asyncio
import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

import httpx
import numpy as np
from scipy.stats import norm

from gig.ml.bayesian_linear import MAX_ITER, fit_bayesian_ridge, parameters, predict_bayesian_ridge
from gig.research.model_pipeline import configuration, fit_dataset, identity, saved_predict
from gig.research.workflow import model_capabilities
from gig.service.app import create_app
from gig.service.dataset_workspace import PersistedDatasetWorkspace
from gig.service.persisted_research import PersistedResearchWorkspace
from test_persisted_data import fixture_times, seed_persisted_fixture
from test_persisted_flow_http import allocation_request


class BayesianKernelTests(unittest.TestCase):
    def setUp(self):
        self.x = np.random.default_rng(31).normal(size=(48, 3))
        self.y = .02*self.x[:, 0] + np.random.default_rng(19).normal(0, .01, 48)
        self.names = ["momentum", "volatility", "return_1"]

    def test_real_fit_sklearn_and_independent_posterior_reference(self):
        from sklearn.linear_model import BayesianRidge
        original, calls = BayesianRidge.fit, []
        def fit(estimator, *args, **kwargs):
            calls.append(estimator.get_params())
            return original(estimator, *args, **kwargs)
        with patch.object(BayesianRidge, "fit", fit):
            state = fit_bayesian_ridge(self.x, self.y, self.names)
        self.assertEqual(calls, [parameters()])
        mean, scale = self.x.mean(axis=0), self.x.std(axis=0)
        z, query = (self.x-mean)/scale, self.x[:9]+.8
        actual, spread = predict_bayesian_ridge(state, query, self.names)
        oracle = BayesianRidge(**parameters()).fit(z, self.y)
        expected, expected_sd = oracle.predict((query-mean)/scale, return_std=True)
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(spread, expected_sd, rtol=1e-12, atol=1e-12)
        centered = z-z.mean(0)
        covariance = np.linalg.inv(state["weight_precision"]*np.eye(3) + state["noise_precision"]*centered.T@centered)
        coef = state["noise_precision"]*covariance@centered.T@(self.y-self.y.mean())
        q = (query-mean)/scale
        np.testing.assert_allclose(actual, q@coef+self.y.mean()-z.mean(0)@coef, rtol=1e-10, atol=1e-12)
        expected_variance = np.array([row@covariance@row + 1/state["noise_precision"] for row in q])
        np.testing.assert_allclose(spread**2, expected_variance, rtol=1e-10, atol=1e-12)
        self.assertLess(state["iterations"], MAX_ITER)
        self.assertFalse(state["interval_calibrated"])
        self.assertFalse(state["hyperparameter_uncertainty_included"])

    def test_resource_boundaries_collinearity_constants_and_json_no_fit(self):
        for count, features in [(20, 1), (256, 16)]:
            x = np.random.default_rng(9).normal(size=(count, features))
            if features > 1:
                x[:, -1] = .1
                x[:, 1] = x[:, 0]
            y = np.arange(count)/10000
            names = [str(i) for i in range(features)]
            state = fit_bayesian_ridge(x, y, names)
            query = np.tile(x[0], (2000, 1))
            before = deepcopy(state)
            expected = predict_bayesian_ridge(state, query, names)
            with patch("sklearn.linear_model.BayesianRidge.fit", side_effect=AssertionError("no fitting")):
                actual = predict_bayesian_ridge(json.loads(json.dumps(state)), query, names)
            for a, b in zip(expected, actual, strict=True):
                np.testing.assert_array_equal(a, b)
            self.assertEqual(state, before)
        state = fit_bayesian_ridge(np.ones((20, 2)), np.zeros(20), ["a", "b"])
        mean, sd = predict_bayesian_ridge(state, [[1., 1.]], ["a", "b"])
        self.assertEqual(mean[0], 0)
        self.assertGreater(sd[0], 0)

    def test_strict_numeric_shape_settings_and_missing_dependency_rejection(self):
        for bad in (True, "1", None, float("nan"), float("inf"), 10**1000):
            x, y = self.x.tolist(), self.y.tolist()
            x[0][0] = bad
            with self.subTest(value=type(bad).__name__), self.assertRaises(ValueError):
                fit_bayesian_ridge(x, y, self.names)
        for count in (0, 19, 257):
            with self.assertRaises(ValueError):
                fit_bayesian_ridge(np.zeros((count, 3)), np.zeros(count), self.names)
        for x, y, names in [(self.x, self.y[:-1], self.names), (np.zeros((48, 17)), self.y, [str(i) for i in range(17)]),
                            (self.x, self.y, ["same"]*3), (self.x.astype(bool), self.y, self.names)]:
            with self.assertRaises(ValueError):
                fit_bayesian_ridge(x, y, names)
        state = fit_bayesian_ridge(self.x, self.y, self.names)
        for q, names in [(np.zeros((2001, 3)), self.names), (self.x, self.names[::-1]), ([[True, 0, 0]], self.names)]:
            with self.assertRaises(ValueError):
                predict_bayesian_ridge(state, q, names)
        with patch.dict("sys.modules", {"sklearn.linear_model": None}), self.assertRaises(ImportError):
            fit_bayesian_ridge(self.x, self.y, self.names)

    def test_nonconvergence_rejected_and_caller_mutation_isolated(self):
        from sklearn.linear_model import BayesianRidge
        original = BayesianRidge.fit
        def capped(estimator, *args, **kwargs):
            result = original(estimator, *args, **kwargs)
            estimator.n_iter_ = MAX_ITER
            return result
        with patch.object(BayesianRidge, "fit", capped), self.assertRaisesRegex(ValueError, "converge"):
            fit_bayesian_ridge(self.x, self.y, self.names)
        x, y, names = self.x.tolist(), self.y.tolist(), self.names.copy()
        state = fit_bayesian_ridge(x, y, names)
        before = deepcopy(state)
        x[0][0], y[0], names[0] = 9., 9., "changed"
        self.assertEqual(state, before)

    def test_adversarial_saved_state_and_even_rehashed_posterior_rejected(self):
        from gig.ml.bayesian_linear import _fingerprint
        original = fit_bayesian_ridge(self.x, self.y, self.names)
        for field, value in [("noise_precision", 0), ("weight_precision", True), ("interval_calibrated", True),
                             ("hyperparameter_uncertainty_included", True), ("trading_approved", True),
                             ("iterations", True), ("threads", True), ("training_rows", True),
                             ("production_promoted", True), ("intercept", 7.)]:
            state = {**deepcopy(original), field: value}
            with self.subTest(field=field), self.assertRaises(ValueError):
                predict_bayesian_ridge(state, self.x[:2], self.names)
        for modify in [lambda s: s["mean"].__setitem__(0, 7.), lambda s: s["coef"].__setitem__(0, 7.),
                       lambda s: s["posterior_covariance"][0].__setitem__(0, -1.)]:
            state = deepcopy(original)
            modify(state)
            state["training_posterior_sha256"] = _fingerprint(state)
            with self.assertRaises(ValueError):
                predict_bayesian_ridge(state, self.x[:2], self.names)


class BayesianPersistedTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.base = Path(temporary.name)
        self.dataset = PersistedDatasetWorkspace(self.base).build(seed_persisted_fixture(self.base)).to_dict()["data"]
        self.request = {"dataset_id": self.dataset["dataset_id"], "model": "bayesian_ridge", "min_train": 30,
                        "test_size": 12, "embargo": 1, "seed": 17}

    def test_proper_oos_scores_train_only_baselines_and_unapproved_uncertainty(self):
        artifact, _ = fit_dataset(self.dataset, configuration(self.request))
        for fold in artifact["folds"]:
            rows = [r for r in artifact["rows"] if r["model_id"] == fold["model_id"]]
            y, mu, sd = [np.array([r[k] if k != "uncertainty" else r[k]["value"] for r in rows])
                         for k in ("target", "prediction", "uncertainty")]
            self.assertAlmostEqual(fold["gaussian_nll"], float(-np.mean(norm.logpdf(y, loc=mu, scale=sd))), places=12)
            self.assertEqual(fold["mse"], float(np.mean((y-mu)**2)))
            self.assertEqual(fold["nominal95_interval_coverage"], float(np.mean((y >= norm.ppf(.025, mu, sd)) & (y <= norm.ppf(.975, mu, sd)))))
            self.assertFalse(fold["uncertainty_calibrated"])
            self.assertLess(fold["max_training_label_end"], fold["test_start"])
        self.assertFalse(artifact["usable_for_portfolio"])
        self.assertTrue(all(r["uncertainty"]["value"] > 0 and not r["uncertainty"]["calibrated"] for r in artifact["rows"]))

    def test_future_features_and_later_label_revisions_cannot_rewrite_first_fold(self):
        before, models = fit_dataset(self.dataset, configuration(self.request))
        changed = deepcopy(self.dataset)
        for row in changed["rows"]:
            if row["timestamp"] >= fixture_times()[55].isoformat():
                row["features"]["momentum"] += .25
                row["target"] += .1
            row.setdefault("label_vintages", [{"target": row["target"], "label_end": row["label_end"], "knowledge_at": row["label_end"]}]).append(
                {"target": .9, "label_end": row["label_end"], "knowledge_at": "2027-01-01T00:00:00Z"})
        changed["dataset_id"] = identity(changed)
        after, later = fit_dataset(changed, configuration(self.request))
        first = [r for r in before["rows"] if r["fold"] == 0]
        self.assertEqual(first, [r for r in after["rows"] if r["fold"] == 0])
        for row in first:
            self.assertEqual(models[row["model_id"]], later[row["model_id"]])

    def test_preflight_before_any_fit_and_no_option_tuning(self):
        with patch("gig.research.bayesian_linear.fit_bayesian_ridge", side_effect=AssertionError("preflight first")), self.assertRaises(ValueError):
            fit_dataset(self.dataset, {**configuration(self.request), "min_train": 20, "test_size": 4})
        for extras in ({"min_train": 257}, {"iterations": 20}, {"alpha_1": .1}, {"min_train": True}):
            with self.assertRaises(ValueError):
                configuration({**self.request, **extras})
        caps = model_capabilities()["bayesian_linear"]
        self.assertTrue(caps["implemented"])
        self.assertFalse(caps["trained"])
        self.assertFalse(caps["trading_approved"])

    def test_exact_cutoff_label_end_and_knowledge_are_purged(self):
        original, _ = fit_dataset(self.dataset, configuration(self.request))
        cutoff = original["folds"][0]["test_start"]
        for field in ("knowledge_at", "label_end"):
            changed = deepcopy(self.dataset)
            victim = next(r for r in changed["rows"] if r["instrument_id"] == "US:AAA" and r["timestamp"] < cutoff)
            vintage = {"target": .8, "label_end": victim["label_end"], "knowledge_at": victim["label_end"]}
            vintage[field] = cutoff
            victim["label_vintages"] = [vintage]
            _, models = fit_dataset(changed, configuration(self.request))
            state = next(s for s in models.values() if s["instrument_id"] == "US:AAA" and s["training_cutoff"] == cutoff)
            self.assertNotIn(victim["timestamp"], state["training_timestamps"])

    def test_zero_training_variance_baseline_is_unavailable_not_fabricated(self):
        changed = deepcopy(self.dataset)
        for row in changed["rows"]:
            row["target"] = 0.
            for vintage in row.get("label_vintages", []):
                vintage["target"] = 0.
        changed["dataset_id"] = identity(changed)
        artifact, _ = fit_dataset(changed, configuration(self.request))
        for fold in artifact["folds"]:
            self.assertIsNone(fold["training_gaussian_baseline_nll"])
            self.assertEqual(fold["training_gaussian_baseline_state"], "UNAVAILABLE_ZERO_TRAINING_VARIANCE")
            self.assertFalse(fold["uncertainty_calibrated"])

    def test_restart_no_fit_determinism_and_relabelled_state_stays_unapproved(self):
        data = PersistedResearchWorkspace(self.base).run(self.request).to_dict()["data"]
        payload = {"run_id": data["run_id"], "dataset_id": data["dataset_id"]}
        with patch("sklearn.linear_model.BayesianRidge.fit", side_effect=AssertionError("restart may not fit")):
            a = PersistedResearchWorkspace(self.base).predict(payload).to_dict()
            b = PersistedResearchWorkspace(self.base).predict(payload).to_dict()
        self.assertEqual(a, b)
        self.assertFalse(a["data"]["metrics"]["training_performed"])
        artifact, models = fit_dataset(self.dataset, configuration(self.request))
        for state in models.values():
            expected = [r for r in artifact["rows"] if r["model_id"] == identity(state)]
            actual = {r["timestamp"]: r for r in saved_predict(state, self.dataset)}
            for row in expected:
                self.assertEqual(row["prediction"], actual[row["timestamp"]]["prediction"])
            self.assertTrue(all(not r["usable_for_portfolio"] for r in saved_predict({**state, "fit_state": "CONVERGED"}, self.dataset)))


class BayesianHTTPTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        async def heartbeat():
            while True:
                await asyncio.sleep(.01)
        self.heartbeat = asyncio.create_task(heartbeat())

    async def asyncTearDown(self):
        self.heartbeat.cancel()
        await asyncio.gather(self.heartbeat, return_exceptions=True)

    async def test_http_saved_replay_terminal_and_allocation_rejection(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            request = seed_persisted_fixture(base)
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=create_app(state_dir=base)), base_url="http://test") as client:
                dataset = (await client.post("/api/v1/workspace/research-dataset", json=request)).json()["data"]
                response = await client.post("/api/v1/workspace/research-model", json={"dataset_id": dataset["dataset_id"], "model": "bayesian_ridge", "min_train": 30, "test_size": 12})
                self.assertEqual(response.status_code, 200, response.text)
                data = response.json()["data"]
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=create_app(state_dir=base)), base_url="http://test") as client:
                with patch("sklearn.linear_model.BayesianRidge.fit", side_effect=AssertionError("no fit")):
                    replay = await client.post("/api/v1/workspace/research-predict", json={"dataset_id": dataset["dataset_id"], "run_id": data["run_id"]})
                self.assertEqual(replay.json()["state"], "AVAILABLE")
                self.assertFalse(replay.json()["data"]["usable_for_portfolio"])
                for section in ("research", "models", "signals"):
                    result = await client.get("/api/v1/terminal/"+section, params={"mode": "local", "engine": "persisted", "as_of": request["as_of"], "max_age_seconds": 60})
                    self.assertEqual(result.json()["state"], "AVAILABLE")
                    self.assertFalse(result.json()["data"]["usable_for_portfolio"])
                denied = await client.post("/api/v1/workspace/research-allocate", json=allocation_request(data["run_id"], request["as_of"]))
                self.assertEqual(denied.json()["state"], "UNAVAILABLE")
                self.assertFalse((base/"allocations").exists())
                self.assertFalse((base/"execution-engine").exists())
