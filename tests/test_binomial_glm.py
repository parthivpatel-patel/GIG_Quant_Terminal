"""Binomial likelihood references, chronological labels and no trading approval."""
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
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.metrics import brier_score_loss, log_loss

from gig.ml.binomial_glm import MAX_ITER, fit_binomial_glm, parameters, predict_binomial_glm
from gig.research.model_pipeline import configuration, fit_dataset, identity, saved_predict
from gig.research.workflow import model_capabilities
from gig.service.app import create_app
from gig.service.dataset_workspace import PersistedDatasetWorkspace
from gig.service.persisted_research import PersistedResearchWorkspace
from test_persisted_data import fixture_times, seed_persisted_fixture
from test_persisted_flow_http import allocation_request


class BinomialKernelTests(unittest.TestCase):
    def setUp(self):
        self.x = np.random.default_rng(41).normal(size=(48, 3))
        self.y = (self.x[:, 0]+.3*self.x[:, 1] > 0).astype(float)
        self.names = ["momentum", "volatility", "return_1"]

    def test_actual_cpu_fit_and_independent_regularized_likelihood_solution(self):
        from sklearn.linear_model import LogisticRegression
        original, calls = LogisticRegression.fit, []
        def fit(estimator, *args, **kwargs):
            calls.append(estimator.max_iter)
            return original(estimator, *args, **kwargs)
        with patch.object(LogisticRegression, "fit", fit):
            state = fit_binomial_glm(self.x, self.y, self.names)
        self.assertEqual(calls, [MAX_ITER])
        z = (self.x-self.x.mean(0))/self.x.std(0)
        query = self.x[:9]+.4
        values, logits = predict_binomial_glm(state, query, self.names)
        estimator = LogisticRegression(**parameters()).fit(z, self.y)
        np.testing.assert_allclose(values, estimator.predict_proba((query-self.x.mean(0))/self.x.std(0))[:, 1], rtol=1e-12, atol=1e-12)
        design = np.c_[z, np.ones(len(z))]
        penalty = np.diag([1., 1., 1., 0.])
        def objective(beta):
            eta = design@beta
            return np.sum(np.logaddexp(0, eta)-self.y*eta)+.5*beta@penalty@beta
        def gradient(beta):
            return design.T@(expit(design@beta)-self.y)+penalty@beta
        def hessian(beta):
            p = expit(design@beta)
            return design.T@((p*(1-p))[:, None]*design)+penalty
        reference = minimize(objective, np.zeros(4), jac=gradient, hess=hessian, method="trust-exact", options={"gtol": 1e-9})
        self.assertLess(np.max(np.abs(gradient(reference.x))), 1e-6)
        expected = np.c_[(query-self.x.mean(0))/self.x.std(0), np.ones(9)]@reference.x
        np.testing.assert_allclose(logits, expected, rtol=1e-6, atol=1e-6)
        self.assertFalse(state["probability_calibrated"])

    def test_resource_limits_and_strict_binary_numeric_inputs(self):
        for count, features in [(20, 1), (256, 16)]:
            x = np.random.default_rng(7).normal(size=(count, features))
            y = (np.arange(count)%2).astype(float)
            names = [str(i) for i in range(features)]
            state = fit_binomial_glm(x, y, names)
            values, _ = predict_binomial_glm(state, np.tile(x[0], (2000, 1)), names)
            self.assertEqual(len(values), 2000)
            self.assertTrue(np.all((values >= 0) & (values <= 1)))
        for y in [np.zeros(48), np.ones(48), np.full(48, .5), [True]*48, self.y[:-1]]:
            with self.assertRaises(ValueError):
                fit_binomial_glm(self.x, y, self.names)
        for bad in [True, "1", None, float("nan"), float("inf"), 10**1000]:
            x = self.x.tolist()
            x[0][0] = bad
            with self.assertRaises(ValueError):
                fit_binomial_glm(x, self.y, self.names)
        for count in [0, 19, 257]:
            with self.assertRaises(ValueError):
                fit_binomial_glm(np.zeros((count, 3)), np.arange(count)%2, self.names)
        state = fit_binomial_glm(self.x, self.y, self.names)
        for x, names in [(np.zeros((2001, 3)), self.names), (self.x, self.names[::-1]), ([[True, 0, 0]], self.names)]:
            with self.assertRaises(ValueError):
                predict_binomial_glm(state, x, names)
        with patch.dict("sys.modules", {"sklearn.linear_model": None}), self.assertRaises(ImportError):
            fit_binomial_glm(self.x, self.y, self.names)

    def test_constant_covariates_balanced_labels_and_no_refit_json(self):
        state = fit_binomial_glm(np.ones((20, 2)), np.arange(20)%2, ["a", "b"])
        self.assertEqual(state["iterations"], 0)  # Exact initial optimum; gradient validation still applies.
        np.testing.assert_allclose(predict_binomial_glm(state, [[1., 1.]], ["a", "b"])[0], [.5], atol=1e-12)
        x, y, names = self.x.tolist(), self.y.tolist(), self.names.copy()
        state = fit_binomial_glm(x, y, names)
        before = deepcopy(state)
        expected = predict_binomial_glm(state, self.x, self.names)
        x[0][0], y[0], names[0] = 9., 9., "changed"
        self.assertEqual(state, before)
        with patch("sklearn.linear_model.LogisticRegression.fit", side_effect=AssertionError("no fitting")):
            actual = predict_binomial_glm(json.loads(json.dumps(state)), self.x, self.names)
        for a, b in zip(actual, expected, strict=True):
            np.testing.assert_array_equal(a, b)

    def test_nonconvergence_and_even_rehashed_tampering_rejected(self):
        from sklearn.linear_model import LogisticRegression

        from gig.ml.binomial_glm import _fingerprint
        original = LogisticRegression.fit
        def capped(estimator, *args, **kwargs):
            result = original(estimator, *args, **kwargs)
            estimator.n_iter_[0] = MAX_ITER
            return result
        with patch.object(LogisticRegression, "fit", capped), self.assertRaisesRegex(ValueError, "converge"):
            fit_binomial_glm(self.x, self.y, self.names)
        original = fit_binomial_glm(self.x, self.y, self.names)
        for field, value in [("classes", [0, True]), ("threads", True), ("training_rows", True), ("probability_calibrated", True),
                             ("trading_approved", True), ("production_promoted", True), ("iterations", True), ("label_rule", "RETURN_GE_ZERO")]:
            state = {**deepcopy(original), field: value}
            with self.subTest(field=field), self.assertRaises(ValueError):
                predict_binomial_glm(state, self.x, self.names)
        for change in [lambda s: s["coef"].__setitem__(0, 9.), lambda s: s["mean"].__setitem__(0, 9.),
                       lambda s: s["parameters"].update(C=True)]:
            state = deepcopy(original)
            change(state)
            state["training_sha256"] = _fingerprint(state)
            with self.assertRaises(ValueError):
                predict_binomial_glm(state, self.x, self.names)


class BinomialPersistedTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.base = Path(temporary.name)
        self.dataset = PersistedDatasetWorkspace(self.base).build(seed_persisted_fixture(self.base)).to_dict()["data"]
        self.request = {"dataset_id": self.dataset["dataset_id"], "model": "binomial_glm", "min_train": 30, "test_size": 12}

    def test_model_specific_scores_labels_and_empty_calibration_states(self):
        before = deepcopy(self.dataset)
        artifact, _ = fit_dataset(self.dataset, configuration(self.request))
        self.assertEqual(self.dataset, before)
        self.assertIsNone(artifact["metrics"]["mse"])
        self.assertEqual(artifact["metrics"]["mse_state"], "UNAVAILABLE_DIFFERENT_OUTPUT_UNITS")
        self.assertEqual(artifact["prediction_kind"], "FORWARD_DIRECTION_PROBABILITY")
        self.assertFalse(artifact["usable_for_portfolio"])
        for fold in artifact["folds"]:
            rows = [r for r in artifact["rows"] if r["model_id"] == fold["model_id"]]
            y, p = [r["target"] for r in rows], [r["prediction"] for r in rows]
            self.assertAlmostEqual(fold["brier_score"], brier_score_loss(y, p), places=13)
            self.assertAlmostEqual(fold["log_loss"], log_loss(y, np.c_[1-np.array(p), p], labels=[0, 1]), places=12)
            self.assertEqual(sum(b["count"] for b in fold["calibration_bins"]), len(rows))
            for b in fold["calibration_bins"]:
                if not b["count"]:
                    self.assertIsNone(b["observed_positive_rate"])
                    self.assertEqual(b["state"], "UNAVAILABLE_NO_OBSERVATIONS")
            self.assertFalse(fold["probability_calibrated"])
        self.assertTrue(all(r["target"] == float(r["raw_return_target"] > 0) and r["uncertainty"]["value"] is None for r in artifact["rows"]))

    def test_future_revisions_and_exact_cutoff_cannot_enter_training(self):
        before, models = fit_dataset(self.dataset, configuration(self.request))
        changed = deepcopy(self.dataset)
        for row in changed["rows"]:
            if row["timestamp"] >= fixture_times()[55].isoformat():
                row["features"]["momentum"] += .25
                row["target"] = -.1
            row.setdefault("label_vintages", [{"target": row["target"], "label_end": row["label_end"], "knowledge_at": row["label_end"]}]).append(
                {"target": .9, "label_end": row["label_end"], "knowledge_at": "2027-01-01T00:00:00Z"})
        changed["dataset_id"] = identity(changed)
        after, later = fit_dataset(changed, configuration(self.request))
        first = [r for r in before["rows"] if r["fold"] == 0]
        self.assertEqual(first, [r for r in after["rows"] if r["fold"] == 0])
        for row in first:
            self.assertEqual(models[row["model_id"]], later[row["model_id"]])
        cutoff = before["folds"][0]["test_start"]
        for field in ["label_end", "knowledge_at"]:
            changed = deepcopy(self.dataset)
            victim = next(r for r in changed["rows"] if r["instrument_id"] == "US:AAA" and r["timestamp"] < cutoff)
            vintage = {"target": .8, "label_end": victim["label_end"], "knowledge_at": victim["label_end"]}
            vintage[field] = cutoff
            victim["label_vintages"] = [vintage]
            _, states = fit_dataset(changed, configuration(self.request))
            state = next(s for s in states.values() if s["instrument_id"] == "US:AAA" and s["training_cutoff"] == cutoff)
            self.assertNotIn(victim["timestamp"], state["training_timestamps"])

    def test_plan_admission_and_restart_relabelled_state_cannot_allocate(self):
        with patch("gig.research.binomial_glm.fit_binomial_glm", side_effect=AssertionError("preflight first")), self.assertRaises(ValueError):
            fit_dataset(self.dataset, {**configuration(self.request), "min_train": 20, "test_size": 4})
        for extra in [{"min_train": 257}, {"C": .2}, {"iterations": 20}]:
            with self.assertRaises(ValueError):
                configuration({**self.request, **extra})
        data = PersistedResearchWorkspace(self.base).run(self.request).to_dict()["data"]
        payload = {"dataset_id": self.dataset["dataset_id"], "run_id": data["run_id"]}
        with patch("sklearn.linear_model.LogisticRegression.fit", side_effect=AssertionError("no fit")):
            a = PersistedResearchWorkspace(self.base).predict(payload).to_dict()
            b = PersistedResearchWorkspace(self.base).predict(payload).to_dict()
        self.assertEqual(a, b)
        self.assertFalse(a["data"]["usable_for_portfolio"])
        _, states = fit_dataset(self.dataset, configuration(self.request))
        for state in states.values():
            self.assertTrue(all(not r["usable_for_portfolio"] for r in saved_predict({**state, "fit_state": "CONVERGED"}, self.dataset)))
        self.assertTrue(model_capabilities()["generalized_linear"]["implemented"])
        self.assertFalse(model_capabilities()["generalized_linear"]["trained"])


class BinomialHTTPTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        async def heartbeat():
            while True:
                await asyncio.sleep(.01)
        self.heartbeat = asyncio.create_task(heartbeat())

    async def asyncTearDown(self):
        self.heartbeat.cancel()
        await asyncio.gather(self.heartbeat, return_exceptions=True)

    async def test_http_probability_contract_saved_restart_and_allocation_rejection(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            request = seed_persisted_fixture(base)
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=create_app(state_dir=base)), base_url="http://test") as client:
                dataset = (await client.post("/api/v1/workspace/research-dataset", json=request)).json()["data"]
                result = await client.post("/api/v1/workspace/research-model", json={"dataset_id": dataset["dataset_id"], "model": "binomial_glm", "min_train": 30, "test_size": 12})
                self.assertEqual(result.status_code, 200, result.text)
                data = result.json()["data"]
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=create_app(state_dir=base)), base_url="http://test") as client:
                with patch("sklearn.linear_model.LogisticRegression.fit", side_effect=AssertionError("no fit")):
                    replay = await client.post("/api/v1/workspace/research-predict", json={"dataset_id": dataset["dataset_id"], "run_id": data["run_id"]})
                self.assertEqual(replay.json()["state"], "AVAILABLE")
                self.assertEqual(replay.json()["data"]["prediction_kind"], "FORWARD_DIRECTION_PROBABILITY")
                self.assertFalse(replay.json()["data"]["usable_for_portfolio"])
                denied = await client.post("/api/v1/workspace/research-allocate", json=allocation_request(data["run_id"], request["as_of"]))
                self.assertEqual(denied.json()["state"], "UNAVAILABLE")
                self.assertFalse((base/"allocations").exists())
                self.assertFalse((base/"execution-engine").exists())
