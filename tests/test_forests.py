"""Actual forest kernels, strict JSON inference and purged persisted research."""
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

from gig.ml.forests import DEPTH, KINDS, MIN_LEAF, TREES, fit_forest, predict_forest
from gig.research.model_pipeline import configuration, fit_dataset, identity
from gig.service.app import create_app
from gig.service.dataset_workspace import PersistedDatasetWorkspace
from gig.service.persisted_research import PersistedResearchWorkspace
from test_persisted_data import fixture_times, seed_persisted_fixture
from test_persisted_flow_http import allocation_request


class ForestKernelTests(unittest.TestCase):
    def setUp(self):
        self.x = np.random.default_rng(17).normal(size=(48, 3))
        self.y = .03*self.x[:, 0] - .02*self.x[:, 1]**2
        self.names = ["lag_return", "volatility", "range"]

    def test_actual_sklearn_and_independent_tree_traversal(self):
        from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
        query = np.vstack([self.x[:8], self.x[:8]+1e-8])
        for kind, cls in zip(KINDS, (RandomForestRegressor, ExtraTreesRegressor), strict=True):
            with self.subTest(model=kind):
                state = fit_forest(self.x, self.y, self.names, model=kind)
                estimator = cls(n_estimators=TREES, max_depth=DEPTH, min_samples_leaf=MIN_LEAF,
                    max_features=1., criterion="squared_error", bootstrap=kind == "random_forest", n_jobs=1, random_state=17).fit(self.x, self.y)
                actual, spread = predict_forest(state, query, self.names)
                np.testing.assert_allclose(actual, estimator.predict(query), rtol=1e-12, atol=1e-14)
                manual = []
                for tree in state["trees"]:
                    predictions = []
                    for row in query.astype(np.float32):
                        node = 0
                        while tree["left"][node] != -1:
                            node = tree["left"][node] if row[tree["feature"][node]] <= tree["threshold"][node] else tree["right"][node]
                        predictions.append(tree["value"][node])
                    manual.append(predictions)
                np.testing.assert_array_equal(actual, np.mean(manual, axis=0))
                np.testing.assert_array_equal(spread, np.std(manual, axis=0))
                self.assertEqual(state["fit_backend"], "sklearn.ensemble."+cls.__name__)
                self.assertEqual(state["feature_dtype"], "float32")
                self.assertFalse(state["production_promoted"])

    def test_float32_threshold_boundary_and_json_replay_without_fitting(self):
        x = np.column_stack([1.+np.arange(48)*2e-8, np.arange(48), np.ones(48)])
        for kind in KINDS:
            with self.subTest(model=kind):
                state = fit_forest(x, self.y, self.names, model=kind)
                before = deepcopy(state)
                expected = predict_forest(state, x, self.names)
                with patch("sklearn.ensemble.RandomForestRegressor.fit", side_effect=AssertionError("no fit")), patch("sklearn.ensemble.ExtraTreesRegressor.fit", side_effect=AssertionError("no fit")):
                    actual = predict_forest(json.loads(json.dumps(state)), x, self.names)
                for a, b in zip(actual, expected, strict=True):
                    np.testing.assert_array_equal(a, b)
                self.assertEqual(state, before)

    def test_state_tampering_cycles_unreachable_nodes_and_approval_are_rejected(self):
        original = fit_forest(self.x, self.y, self.names, model="random_forest")
        mutations = [lambda s: s.update(threads=True), lambda s: s.update(trading_approved=True),
                     lambda s: s.update(seed=True), lambda s: s["training_y"].__setitem__(0, 9.),
                     lambda s: s["trees"][0]["left"].__setitem__(0, 0),
                     lambda s: s["trees"][0]["right"].__setitem__(0, s["trees"][0]["left"][0]),
                     lambda s: s["trees"][0]["sample_indices"].__setitem__(0, 999)]
        for i, mutate in enumerate(mutations):
            with self.subTest(case=i):
                state = deepcopy(original)
                mutate(state)
                with self.assertRaises(ValueError):
                    predict_forest(state, self.x, self.names)
        bad = deepcopy(original)
        leaf = bad["trees"][0]["left"].index(-1)
        bad["trees"][0]["value"][leaf] += 1.
        with self.assertRaises(ValueError):
            predict_forest(bad, self.x, self.names)

    def test_invalid_numeric_inputs_and_work_bounds_never_coerce(self):
        for value in (True, "1", float("nan"), float("inf"), 10**1000):
            with self.subTest(type=type(value).__name__):
                x = self.x.tolist()
                x[0][0] = value
                with self.assertRaises(ValueError):
                    fit_forest(x, self.y, self.names, model="random_forest")
        for x in (np.zeros((257, 3)), np.zeros((19, 3)), np.zeros((48, 17))):
            with self.assertRaises(ValueError):
                fit_forest(x, np.zeros(len(x)), self.names, model="extra_trees")
        with patch.dict("sys.modules", {"sklearn.ensemble": None}), self.assertRaises(ImportError):
            fit_forest(self.x, self.y, self.names, model="random_forest")


class PersistedForestTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)
        self.request = seed_persisted_fixture(self.base)
        self.dataset = PersistedDatasetWorkspace(self.base).build(self.request).to_dict()["data"]

    def request_for(self, kind):
        return {"dataset_id": self.dataset["dataset_id"], "model": kind, "min_train": 30, "test_size": 12, "embargo": 1, "seed": 17}

    def test_later_revisions_and_future_features_preserve_first_fitted_decisions(self):
        for kind in KINDS:
            with self.subTest(model=kind):
                config = configuration(self.request_for(kind))
                before, models = fit_dataset(self.dataset, config)
                changed = deepcopy(self.dataset)
                for row in changed["rows"]:
                    if row["timestamp"] >= fixture_times()[55].isoformat():
                        row["features"]["momentum"] += .25
                        row["target"] += .10
                    row.setdefault("label_vintages", [{"target": row["target"], "label_end": row["label_end"], "knowledge_at": row["label_end"]}]).append(
                        {"target": .9, "label_end": row["label_end"], "knowledge_at": "2027-01-01T00:00:00Z"})
                changed["dataset_id"] = identity(changed)
                after, later = fit_dataset(changed, config)
                a, b = [[r for r in value["rows"] if r["fold"] == 0] for value in (before, after)]
                self.assertEqual(a, b)
                for row in a:
                    self.assertEqual(models[row["model_id"]], later[row["model_id"]])
                self.assertFalse(before["usable_for_portfolio"])
                self.assertTrue(all(not row["usable_for_portfolio"] and not row["uncertainty"]["calibrated"] for row in before["rows"]))

    def test_restart_replay_and_preflight_limits(self):
        for kind in KINDS:
            with self.subTest(model=kind):
                request = self.request_for(kind)
                workspace = PersistedResearchWorkspace(self.base)
                result = workspace.run(request).to_dict()
                reopened = PersistedResearchWorkspace(self.base)
                self.assertEqual(reopened.run(request).to_dict(), result)
                predicted = reopened.predict({"dataset_id": request["dataset_id"], "run_id": result["data"]["run_id"]}).to_dict()
                self.assertEqual(predicted["state"], "AVAILABLE")
                self.assertFalse(predicted["data"]["metrics"]["training_performed"])
                artifact = reopened.load(result["data"]["run_id"])
                for fold in artifact["folds"]:
                    self.assertLess(fold["max_training_label_end"], fold["test_start"])
                with patch("gig.research.forests.fit_forest", side_effect=AssertionError("preflight first")), self.assertRaises(ValueError):
                    fit_dataset(self.dataset, {**configuration(request), "min_train": 20, "test_size": 4})
                with self.assertRaises(ValueError):
                    configuration({**request, "min_train": 257})


class ForestHTTPTests(unittest.IsolatedAsyncioTestCase):
    async def test_both_variants_persist_restart_and_reject_allocations(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            request = seed_persisted_fixture(base)
            async def clock():
                while True:
                    await asyncio.sleep(.01)
            heartbeat = asyncio.create_task(clock())
            try:
                for kind in KINDS:
                    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=create_app(state_dir=base)), base_url="http://test") as client:
                        data = (await client.post("/api/v1/workspace/research-dataset", json=request)).json()
                        payload = {"dataset_id": data["data"]["dataset_id"], "model": kind, "min_train": 30, "test_size": 12, "seed": 17}
                        response = await client.post("/api/v1/workspace/research-model", json=payload)
                        self.assertEqual(response.status_code, 200, response.text)
                        run = response.json()["data"]["run_id"]
                    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=create_app(state_dir=base)), base_url="http://test") as client:
                        self.assertEqual((await client.post("/api/v1/workspace/research-model", json=payload)).json(), response.json())
                        prediction = (await client.post("/api/v1/workspace/research-predict", json={"dataset_id": payload["dataset_id"], "run_id": run})).json()
                        self.assertEqual(prediction["state"], "AVAILABLE")
                        self.assertFalse(prediction["data"]["metrics"]["training_performed"])
                        rejected = (await client.post("/api/v1/workspace/research-allocate", json=allocation_request(run, request["as_of"]))).json()
                        self.assertEqual(rejected["state"], "UNAVAILABLE")
                        self.assertFalse((base / "execution-engine").exists())
            finally:
                heartbeat.cancel()
