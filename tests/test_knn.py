"""Actual sklearn all-neighbor queries, scalar distances and persisted KNN fences."""
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

from gig.ml.knn import K, fit_knn, predict_knn
from gig.research.model_pipeline import configuration, fit_dataset, identity, saved_predict
from gig.research.workflow import model_capabilities
from gig.service.app import create_app
from gig.service.dataset_workspace import PersistedDatasetWorkspace
from gig.service.persisted_research import PersistedResearchWorkspace
from test_persisted_data import fixture_times, seed_persisted_fixture
from test_persisted_flow_http import allocation_request


def sklearn_oracle(x, y, query, mean, scale):
    import sklearn
    from sklearn.neighbors import KNeighborsRegressor
    with sklearn.config_context(enable_cython_pairwise_dist=False, working_memory=8):
        estimator = KNeighborsRegressor(n_neighbors=5, weights="uniform", algorithm="brute",
                                       metric="sqeuclidean", n_jobs=1).fit((x-mean)/scale, y)
        distances, indices = estimator.kneighbors((query-mean)/scale, n_neighbors=len(y))
    return np.array([np.mean([y[i] for _, i in sorted(zip(ds, ix, strict=True))[:5]])
                     for ds, ix in zip(distances, indices, strict=True)])


def scalar_oracle(x, y, query, mean, scale):
    z, q = (x-mean)/scale, (query-mean)/scale
    return np.array([sum(y[i] for _, i in sorted(
        (sum((float(a)-float(b))*(float(a)-float(b)) for a, b in zip(row, train, strict=True)), i)
        for i, train in enumerate(z))[:5])/5 for row in q])


class KNNKernelTests(unittest.TestCase):
    def setUp(self):
        self.x = np.random.default_rng(17).normal(size=(48, 3))
        self.x[:, 2] = .1
        self.y = .03*self.x[:, 0] - .02*self.x[:, 1]**2
        self.names = ["lag_return", "volatility", "constant"]

    def test_actual_fit_all_neighbor_query_and_independent_scalar_oracle(self):
        import sklearn
        from sklearn.neighbors import KNeighborsRegressor
        original_fit, original_query = KNeighborsRegressor.fit, KNeighborsRegressor.kneighbors
        calls, before_config = [], sklearn.get_config()
        def fit(estimator, *args, **kwargs):
            calls.append("fit")
            return original_fit(estimator, *args, **kwargs)
        def query(estimator, *args, **kwargs):
            self.assertEqual(kwargs["n_neighbors"], len(self.y))
            self.assertFalse(sklearn.get_config()["enable_cython_pairwise_dist"])
            self.assertEqual(sklearn.get_config()["working_memory"], 8)
            calls.append("all-neighbor-query")
            return original_query(estimator, *args, **kwargs)
        with patch.object(KNeighborsRegressor, "fit", fit), patch.object(KNeighborsRegressor, "kneighbors", query):
            state = fit_knn(self.x, self.y, self.names)
        self.assertEqual(calls, ["fit", "all-neighbor-query"])
        self.assertEqual(sklearn.get_config(), before_config)
        mean, scale = self.x.mean(0), self.x.std(0)
        mean[2], scale[2] = .1, 1.
        np.testing.assert_array_equal(state["mean"], mean)
        np.testing.assert_array_equal(state["scale"], scale)
        query = self.x[:9]+np.array([20., -10., 0.])
        before = deepcopy(state)
        actual = predict_knn(state, query, self.names)
        np.testing.assert_array_equal(actual, sklearn_oracle(self.x, self.y, query, mean, scale))
        np.testing.assert_allclose(actual, scalar_oracle(self.x, self.y, query, mean, scale), rtol=0, atol=1e-16)
        self.assertEqual(state, before)
        self.assertEqual(state["fit_backend"], "sklearn.neighbors.KNeighborsRegressor")
        self.assertEqual(state["fit_semantics"], "RETAIN_OBSERVATIONS_AND_LABELS_NO_OPTIMIZED_WEIGHTS")
        self.assertNotIn("weights", state)

    def test_duplicate_equidistant_constant_and_near_tie_inputs(self):
        cases = [np.zeros((20, 2)), np.array([[1., 0.], [-1., 0.]]*10),
                 np.array([[1., 1.], [-1., -1.], [1., -1.], [-1., 1.]]*5),
                 np.column_stack([np.linspace(-1, 1, 20), np.zeros(20)])]
        for x in cases:
            y = np.arange(20)/100
            state = fit_knn(x, y, ["x", "y"])
            query = np.array([[0., 0.], [1., 0.], [np.nextafter(1., 0.), 0.]])
            actual = predict_knn(state, query, ["x", "y"])
            for oracle in (sklearn_oracle, scalar_oracle):
                np.testing.assert_allclose(actual, oracle(x, y, query, np.array(state["mean"]), np.array(state["scale"])), rtol=0, atol=1e-16)
            if np.all(np.linalg.norm(x, axis=1) == np.linalg.norm(x[0])):
                self.assertEqual(actual[0], float(y[:K].mean()))
        state = fit_knn(self.x, np.full(48, .025), self.names)
        np.testing.assert_array_equal(predict_knn(state, self.x[:2], self.names), [.025, .025])

    def test_caller_mutation_isolation_and_no_fit_json_replay(self):
        x, y, names = self.x.tolist(), self.y.tolist(), self.names.copy()
        state = fit_knn(x, y, names)
        expected = predict_knn(state, self.x, self.names)
        x[0][0], y[0], names[0] = 100., 100., "changed"
        self.assertEqual(state["features"], self.names)
        with patch("sklearn.neighbors.KNeighborsRegressor.fit", side_effect=AssertionError("no inference fit")), patch("sklearn.neighbors.KNeighborsRegressor.kneighbors", side_effect=AssertionError("JSON replay")):
            actual = predict_knn(json.loads(json.dumps(state)), self.x, self.names)
        np.testing.assert_array_equal(actual, expected)
        array_state = fit_knn(self.x, self.y, self.names)
        self.x[:] = 100.
        self.y[:] = 100.
        self.assertNotEqual(array_state["training_x"][0][0], 100.)
        self.assertNotEqual(array_state["training_y"][0], 100.)

    def test_exact_resource_limits_and_config_restoration_after_error(self):
        import sklearn
        for count in (20, 256):
            x = np.random.default_rng(3).normal(size=(count, 16))
            names = [str(i) for i in range(16)]
            y = np.arange(count)/1000
            state = fit_knn(x, y, names)
            actual = predict_knn(state, np.tile(x[0], (2000, 1)), names)
            self.assertEqual(actual.shape, (2000,))
            self.assertTrue(np.all(actual == actual[0]))
        config = sklearn.get_config()
        with patch("sklearn.neighbors.KNeighborsRegressor.kneighbors", side_effect=ValueError("query failed")), self.assertRaisesRegex(ValueError, "query failed"):
            fit_knn(self.x, self.y, self.names)
        self.assertEqual(sklearn.get_config(), config)

    def test_strict_types_finite_resource_and_missing_library_rejection(self):
        for value in (True, np.bool_(True), "1", None, float("nan"), float("inf"), 10**1000):
            for target in (False, True):
                x, y = self.x.tolist(), self.y.tolist()
                if target:
                    y[0] = value
                else:
                    x[0][0] = value
                with self.subTest(kind=type(value).__name__, target=target), self.assertRaises(ValueError):
                    fit_knn(x, y, self.names)
        for count in (0, 19, 257):
            with self.assertRaises(ValueError):
                fit_knn(np.zeros((count, 3)), np.zeros(count), self.names)
        for x, y, names in ((np.zeros((48, 17)), self.y, [str(i) for i in range(17)]),
                             (self.x, self.y[:-1], self.names), (self.x, self.y, ["x", "x", "z"]),
                             (self.x.astype(bool), self.y, self.names), (self.x.astype(str), self.y, self.names)):
            with self.assertRaises(ValueError):
                fit_knn(x, y, names)
        state = fit_knn(self.x, self.y, self.names)
        for query, names in ((self.x, self.names[::-1]), (np.zeros((2001, 3)), self.names),
                             (np.full((2, 3), 1e12), self.names), ([[True, 0., 0.]], self.names),
                             ([[float("nan"), 0., 0.]], self.names)):
            with self.assertRaises(ValueError):
                predict_knn(state, query, names)
        with patch.dict("sys.modules", {"sklearn.neighbors": None}), self.assertRaises(ImportError):
            fit_knn(self.x, self.y, self.names)

    def test_saved_state_tampering_rejected(self):
        original = fit_knn(self.x, self.y, self.names)
        mutations = [lambda s: s.update(threads=True), lambda s: s.update(training_rows=True),
                     lambda s: s.update(trading_approved=True), lambda s: s.update(production_promoted=True),
                     lambda s: s.update(tie_policy="UNSPECIFIED"), lambda s: s.update(preprocessing="ALL_DATA"),
                     lambda s: s["mean"].__setitem__(0, 9.), lambda s: s["scale"].__setitem__(0, -1.),
                     lambda s: s["parameters"].update(n_neighbors=True), lambda s: s["parameters"].update(n_jobs=True),
                     lambda s: s["parameters"].update(enable_cython_pairwise_dist=0),
                     lambda s: s["retained_indices"].__setitem__(0, True),
                     lambda s: s["retained_indices"].reverse(), lambda s: s["training_y"].__setitem__(0, .9),
                     lambda s: s["training_x"].__setitem__(0, [9., 9., 9.])]
        for number, mutate in enumerate(mutations):
            state = deepcopy(original)
            mutate(state)
            with self.subTest(case=number), self.assertRaises(ValueError):
                predict_knn(state, self.x, self.names)
        # Even a recomputed training hash cannot authorize a test-fitted scaler.
        from gig.ml.knn import _training_hash
        forged = deepcopy(original)
        forged["mean"][0] += .5
        forged["training_transform_sha256"] = _training_hash(forged)
        with self.assertRaisesRegex(ValueError, "normalization"):
            predict_knn(forged, self.x, self.names)
        forged = deepcopy(original)
        forged["scale"][0] = 0.
        forged["training_transform_sha256"] = _training_hash(forged)
        with self.assertRaisesRegex(ValueError, "normalization"):
            predict_knn(forged, self.x, self.names)


class PersistedKNNTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)
        request = seed_persisted_fixture(self.base)
        self.dataset = PersistedDatasetWorkspace(self.base).build(request).to_dict()["data"]
        self.request = {"dataset_id": self.dataset["dataset_id"], "model": "knn", "min_train": 30,
                        "test_size": 12, "embargo": 1, "seed": 17}

    def test_future_data_labels_and_scaler_invariance(self):
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

    def test_fold_metrics_train_only_baseline_and_actual_oos_queries(self):
        artifact, models = fit_dataset(self.dataset, configuration(self.request))
        for fold in artifact["folds"]:
            rows = [r for r in artifact["rows"] if r["model_id"] == fold["model_id"]]
            truth, prediction = np.array([r["target"] for r in rows]), np.array([r["prediction"] for r in rows])
            state = models[fold["model_id"]]
            estimator = state["estimator"]
            train_mean = np.mean(estimator["training_y"])
            self.assertEqual(fold["mse"], float(np.mean((truth-prediction)**2)))
            self.assertEqual(fold["mae"], float(np.mean(np.abs(truth-prediction))))
            self.assertEqual(fold["training_mean_baseline_mse"], float(np.mean((truth-train_mean)**2)))
            self.assertLess(fold["max_training_label_end"], fold["test_start"])
            self.assertEqual(state["training_timestamps"], sorted(state["training_timestamps"]))
            query = np.array([[r["features"][n] for n in estimator["features"]] for r in self.dataset["rows"]
                              if r["instrument_id"] == fold["instrument_id"] and any(r["timestamp"] == o["timestamp"] for o in rows)])
            expected = sklearn_oracle(np.array(estimator["training_x"]), np.array(estimator["training_y"]), query,
                                      np.array(estimator["mean"]), np.array(estimator["scale"]))
            np.testing.assert_array_equal(prediction, expected)
            self.assertEqual(fold["economic_claim"], "NONE")
        reversed_data = {**self.dataset, "rows": self.dataset["rows"][::-1]}
        replay, retained = fit_dataset(reversed_data, configuration(self.request))
        self.assertEqual(replay, artifact)
        self.assertEqual(retained, models)

    def test_strict_knowledge_cutoff_purge(self):
        baseline, _ = fit_dataset(self.dataset, configuration(self.request))
        cutoff = baseline["folds"][0]["test_start"]
        for field in ("knowledge_at", "label_end"):
            changed = deepcopy(self.dataset)
            victim = next(r for r in changed["rows"] if r["instrument_id"] == "US:AAA" and r["timestamp"] < cutoff)
            vintage = {"target": .8, "label_end": victim["label_end"], "knowledge_at": victim["label_end"]}
            vintage[field] = cutoff
            victim["label_vintages"] = [vintage]
            _, models = fit_dataset(changed, configuration(self.request))
            state = next(s for s in models.values() if s["instrument_id"] == "US:AAA" and s["training_cutoff"] == cutoff)
            self.assertNotIn(victim["timestamp"], state["training_timestamps"])

    def test_restart_no_refit_and_relabelled_fit_state_cannot_allocate(self):
        workspace = PersistedResearchWorkspace(self.base)
        result = workspace.run(self.request).to_dict()
        payload = {"dataset_id": self.request["dataset_id"], "run_id": result["data"]["run_id"]}
        with patch("sklearn.neighbors.KNeighborsRegressor.fit", side_effect=AssertionError("restart may not train")):
            predicted = PersistedResearchWorkspace(self.base).predict(payload).to_dict()
            repeated = PersistedResearchWorkspace(self.base).predict(payload).to_dict()
        self.assertEqual(predicted, repeated)
        self.assertEqual(predicted["state"], "AVAILABLE")
        self.assertFalse(predicted["data"]["metrics"]["training_performed"])
        self.assertFalse(predicted["data"]["usable_for_portfolio"])
        artifact, models = fit_dataset(self.dataset, configuration(self.request))
        for state in models.values():
            lookup = {r["timestamp"]: r for r in saved_predict(state, self.dataset)}
            for row in artifact["rows"]:
                if row["model_id"] == identity(state):
                    self.assertEqual(row["prediction"], lookup[row["timestamp"]]["prediction"])
            forged = {**state, "fit_state": "CONVERGED"}
            self.assertTrue(all(not r["usable_for_portfolio"] for r in saved_predict(forged, self.dataset)))
        self.assertFalse((self.base / "execution-engine").exists())

    def test_entire_plan_preflight_before_fit_and_one_family_capability(self):
        with patch("gig.research.knn.fit_knn", side_effect=AssertionError("preflight first")), self.assertRaises(ValueError):
            fit_dataset(self.dataset, {**configuration(self.request), "min_train": 20, "test_size": 4})
        for extras in ({"min_train": 257}, {"iterations": 5}, {"n_neighbors": 6}, {"min_train": True}):
            with self.assertRaises(ValueError):
                configuration({**self.request, **extras})
        capabilities = model_capabilities()["nearest_neighbors"]
        self.assertTrue(capabilities["implemented"])
        self.assertFalse(capabilities["trained"])
        self.assertFalse(capabilities["trading_approved"])


class KNNHTTPTests(unittest.IsolatedAsyncioTestCase):
    async def test_http_restart_sections_and_allocation_rejection(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            request = seed_persisted_fixture(base)
            async def clock():
                while True:
                    await asyncio.sleep(.01)
            heartbeat = asyncio.create_task(clock())
            try:
                async with httpx.AsyncClient(transport=httpx.ASGITransport(app=create_app(state_dir=base)), base_url="http://test") as client:
                    dataset = (await client.post("/api/v1/workspace/research-dataset", json=request)).json()
                    payload = {"dataset_id": dataset["data"]["dataset_id"], "model": "knn", "min_train": 30, "test_size": 12}
                    response = await client.post("/api/v1/workspace/research-model", json=payload)
                    self.assertEqual(response.status_code, 200, response.text)
                    run = response.json()["data"]["run_id"]
                async with httpx.AsyncClient(transport=httpx.ASGITransport(app=create_app(state_dir=base)), base_url="http://test") as client:
                    with patch("sklearn.neighbors.KNeighborsRegressor.fit", side_effect=AssertionError("HTTP inference cannot fit")):
                        predicted = await client.post("/api/v1/workspace/research-predict", json={"dataset_id": payload["dataset_id"], "run_id": run})
                    self.assertEqual(predicted.json()["state"], "AVAILABLE")
                    self.assertFalse(predicted.json()["data"]["metrics"]["training_performed"])
                    for section in ("models", "research", "signals"):
                        result = await client.get("/api/v1/terminal/"+section, params={"mode": "local", "engine": "persisted", "as_of": request["as_of"], "max_age_seconds": 60})
                        self.assertEqual(result.json()["state"], "AVAILABLE")
                        self.assertFalse(result.json()["data"]["usable_for_portfolio"])
                    rejected = await client.post("/api/v1/workspace/research-allocate", json=allocation_request(run, request["as_of"]))
                    self.assertEqual(rejected.json()["state"], "UNAVAILABLE")
                    self.assertIsNone(rejected.json()["data"])
                    self.assertFalse((base / "allocations").exists())
                    self.assertFalse((base / "execution-engine").exists())
            finally:
                heartbeat.cancel()
