"""Independent Gaussian equations, saved-state safety and purged persisted replay."""
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

from gig.ml.gaussian_process import NOISE_VARIANCE, fit_gaussian_process, predict_gaussian_process
from gig.research.model_pipeline import configuration, fit_dataset, identity
from gig.service.app import create_app
from gig.service.dataset_workspace import PersistedDatasetWorkspace
from gig.service.persisted_research import PersistedResearchWorkspace
from test_persisted_data import fixture_times, seed_persisted_fixture
from test_persisted_flow_http import allocation_request


class GaussianProcessTests(unittest.TestCase):
    def setUp(self):
        self.x = np.array([[-2., 2.], [-1., 4.], [0., 1.], [1., 3.], [2., 5.]])
        self.y = np.array([.1, -.2, .3, -.1, .2])
        self.names = ["lag_return", "range"]
        self.state = fit_gaussian_process(self.x, self.y, self.names)

    def test_independent_conditional_gaussian_reference_and_predictive_noise(self):
        query = np.array([[-.5, 2.], [.7, 3.]])
        actual, spread = predict_gaussian_process(self.state, query, self.names)
        # Independent NumPy kernel/linear-solve reference, without fitted weights
        # or factors from the implementation and without calling sklearn.predict.
        z = (self.x-self.x.mean(0))/self.x.std(0)
        q = (query-self.x.mean(0))/self.x.std(0)
        kernel = np.exp(-.5*((z[:, None, :]-z[None, :, :])**2).sum(2)) + np.eye(5)*NOISE_VARIANCE
        cross = np.exp(-.5*((q[:, None, :]-z[None, :, :])**2).sum(2))
        normalized = (self.y-self.y.mean())/self.y.std()
        expected = cross@np.linalg.solve(kernel, normalized)*self.y.std()+self.y.mean()
        variance = (1.-np.einsum("ij,ji->i", cross, np.linalg.solve(kernel, cross.T))+NOISE_VARIANCE)*self.y.std()**2
        np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=1e-12)
        np.testing.assert_allclose(spread**2, variance, atol=1e-12, rtol=1e-12)
        self.assertEqual(self.state["fit_backend"], "sklearn.gaussian_process.GaussianProcessRegressor")
        self.assertEqual(self.state["optimization"], "DISABLED_FIXED_KERNEL")
        self.assertFalse(self.state["production_promoted"])

    def test_saved_json_replay_is_equal_and_does_not_fit_or_mutate(self):
        before = deepcopy(self.state)
        expected = predict_gaussian_process(self.state, self.x, self.names)
        with patch("sklearn.gaussian_process.GaussianProcessRegressor.fit", side_effect=AssertionError("no retraining")):
            actual = predict_gaussian_process(json.loads(json.dumps(self.state)), self.x, self.names)
        for a, b in zip(actual, expected, strict=True):
            np.testing.assert_array_equal(a, b)
        self.assertEqual(before, self.state)

    def test_saved_state_corruption_and_approval_rejected(self):
        for field, value in [("dual", [0.]*5), ("cholesky", np.eye(5).tolist()),
                             ("mean", [9., 9.]), ("trading_approved", True), ("target_scale", 0.),
                             ("threads", 8), ("kernel", {"kind": "RBF"})]:
            with self.subTest(field=field):
                bad = deepcopy(self.state)
                bad[field] = value
                with self.assertRaises(ValueError):
                    predict_gaussian_process(bad, self.x, self.names)
        with self.assertRaises(ValueError):
            predict_gaussian_process(self.state, self.x, list(reversed(self.names)))

    def test_adversarial_numeric_inputs_fail_without_coercion_or_truncation(self):
        for value in (True, "1", float("nan"), float("inf"), 10**1000):
            with self.subTest(value_type=type(value).__name__):
                x = self.x.tolist()
                x[0][0] = value
                with self.assertRaises(ValueError):
                    fit_gaussian_process(x, self.y, self.names)
        for x in (np.ones((129, 2)), np.ones((4, 17)), np.ones((4, 2), dtype=bool)):
            with self.assertRaises(ValueError):
                fit_gaussian_process(x, np.ones(len(x)), self.names)
        with self.assertRaises(ValueError):
            predict_gaussian_process(self.state, np.ones((2001, 2)), self.names)

    def test_missing_actual_library_never_substitutes_model(self):
        with patch.dict("sys.modules", {"sklearn.gaussian_process": None}), self.assertRaises(ImportError):
            fit_gaussian_process(self.x, self.y, self.names)

    def test_constant_features_and_targets_remain_finite(self):
        state = fit_gaussian_process(np.ones((4, 2)), np.full(4, .2), self.names)
        mean, spread = predict_gaussian_process(state, [[1., 1.]], self.names)
        self.assertAlmostEqual(mean[0], .2)
        self.assertGreater(spread[0], 0)


class PersistedGaussianTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)
        request = seed_persisted_fixture(self.base)
        self.dataset = PersistedDatasetWorkspace(self.base).build(request).to_dict()["data"]
        self.request = {"dataset_id": self.dataset["dataset_id"], "model": "gaussian_process",
                        "min_train": 30, "test_size": 12, "embargo": 1, "seed": 17}
        self.config = configuration(self.request)

    def test_revisions_and_future_features_do_not_change_first_fold(self):
        before, models = fit_dataset(self.dataset, self.config)
        changed = deepcopy(self.dataset)
        for row in changed["rows"]:
            if row["timestamp"] >= fixture_times()[55].isoformat():
                row["features"]["momentum"] += .25
                row["target"] += .10
            # A correction acquired after the final experiment cannot enter any
            # training decision, even if the historical outcome date is early.
            row["label_vintages"] = row.get("label_vintages", [{"target": row["target"], "label_end": row["label_end"], "knowledge_at": row["label_end"]}]) + [
                {"target": .9, "label_end": row["label_end"], "knowledge_at": "2027-01-01T00:00:00Z"}]
        changed["dataset_id"] = identity(changed)
        after, later_models = fit_dataset(changed, self.config)
        def first(result):
            return [r for r in result["rows"] if r["fold"] == 0]
        self.assertEqual(first(before), first(after))
        for row in first(before):
            self.assertEqual(models[row["model_id"]], later_models[row["model_id"]])
        self.assertFalse(before["usable_for_portfolio"])
        self.assertTrue(all(not r["usable_for_portfolio"] for r in before["rows"]))
        self.assertTrue(all(not f["interval_calibrated"] for f in before["folds"]))

    def test_work_limit_rejects_before_first_fit_and_persistence(self):
        # 12 folds per instrument would be 24 fits, so reject the entire plan.
        config = {**self.config, "min_train": 20, "test_size": 4}
        with patch("gig.research.gaussian_process.fit_gaussian_process", side_effect=AssertionError("must preflight")), self.assertRaises(ValueError):
            fit_dataset(self.dataset, config)
        with self.assertRaises(ValueError):
            configuration({**self.request, "min_train": 129})
        with self.assertRaises(ValueError):
            configuration({**self.request, "iterations": 1000})
        workspace = PersistedResearchWorkspace(self.base)
        with self.assertRaises(ValueError):
            workspace.run({**self.request, "min_train": 20, "test_size": 4})
        self.assertFalse((self.base / "persisted-research").exists())

    def test_reopen_exact_replay_and_training_only_statistics(self):
        workspace = PersistedResearchWorkspace(self.base)
        result = workspace.run(self.request).to_dict()
        self.assertEqual(result["state"], "AVAILABLE")
        reopened = PersistedResearchWorkspace(self.base)
        self.assertEqual(reopened.run(self.request).to_dict(), result)
        artifact = reopened.load(result["data"]["run_id"])
        prediction = reopened.predict({"dataset_id": self.request["dataset_id"], "run_id": artifact["run_id"]}).to_dict()
        self.assertEqual(prediction["state"], "AVAILABLE")
        self.assertFalse(prediction["data"]["metrics"]["training_performed"])
        for fold in artifact["folds"]:
            state = reopened._model(fold["model_id"])
            self.assertLess(state["max_training_label_end"], state["training_cutoff"])
            rows = [r for r in self.dataset["rows"] if r["instrument_id"] == state["instrument_id"] and r["timestamp"] in state["training_timestamps"]]
            expected = np.mean([[r["features"][n] for n in state["estimator"]["features"]] for r in rows], axis=0)
            np.testing.assert_array_equal(expected, state["estimator"]["mean"])


class GaussianHTTPTests(unittest.IsolatedAsyncioTestCase):
    async def test_http_saved_model_restart_and_allocation_boundary(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            request = seed_persisted_fixture(base)
            # Existing ASGI executor needs a running clock during synchronous IO.
            async def clock():
                while True:
                    await asyncio.sleep(.01)
            heartbeat = asyncio.create_task(clock())
            try:
                async with httpx.AsyncClient(transport=httpx.ASGITransport(app=create_app(state_dir=base)), base_url="http://test") as client:
                    dataset = (await client.post("/api/v1/workspace/research-dataset", json=request)).json()
                    payload = {"dataset_id": dataset["data"]["dataset_id"], "model": "gaussian_process", "min_train": 30, "test_size": 12}
                    response = await client.post("/api/v1/workspace/research-model", json=payload)
                    self.assertEqual(response.status_code, 200, response.text)
                    run = response.json()["data"]["run_id"]
                async with httpx.AsyncClient(transport=httpx.ASGITransport(app=create_app(state_dir=base)), base_url="http://test") as client:
                    replay = await client.post("/api/v1/workspace/research-model", json=payload)
                    self.assertEqual(replay.json(), response.json())
                    predicted = await client.post("/api/v1/workspace/research-predict", json={"dataset_id": payload["dataset_id"], "run_id": run})
                    self.assertEqual(predicted.json()["state"], "AVAILABLE")
                    self.assertFalse(predicted.json()["data"]["metrics"]["training_performed"])
                    section = await client.get("/api/v1/terminal/models", params={"mode": "local", "engine": "persisted", "as_of": request["as_of"], "max_age_seconds": 60})
                    self.assertEqual(section.json()["state"], "AVAILABLE")
                    rejected = await client.post("/api/v1/workspace/research-allocate", json=allocation_request(run, request["as_of"]))
                    self.assertEqual(rejected.status_code, 200)
                    self.assertEqual(rejected.json()["state"], "UNAVAILABLE")
                    self.assertIsNone(rejected.json()["data"])
                    self.assertFalse((base / "allocations").exists())
                    self.assertFalse((base / "execution-engine").exists())
            finally:
                heartbeat.cancel()
