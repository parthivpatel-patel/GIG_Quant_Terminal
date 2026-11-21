"""Genuine spline/ridge fits, independent Cox-de Boor reference and replay."""
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

from gig.ml.additive_spline import fit_additive_spline, predict_additive_spline
from gig.research.model_pipeline import configuration, fit_dataset, identity, saved_predict
from gig.research.workflow import model_capabilities
from gig.service.app import create_app
from gig.service.dataset_workspace import PersistedDatasetWorkspace
from gig.service.persisted_research import PersistedResearchWorkspace
from test_persisted_data import fixture_times, seed_persisted_fixture
from test_persisted_flow_http import allocation_request


def independent_basis(value, knots):
    def basis(i, degree, x):
        if degree == 0:
            return float(knots[i] <= x < knots[i+1])
        left = ((x-knots[i])/(knots[i+degree]-knots[i])*basis(i, degree-1, x)) if knots[i+degree] > knots[i] else 0.
        right = ((knots[i+degree+1]-x)/(knots[i+degree+1]-knots[i+1])*basis(i+1, degree-1, x)) if knots[i+degree+1] > knots[i+1] else 0.
        return left+right
    minimum, maximum = knots[3], knots[-4]
    at = max(minimum, min(maximum, value))
    result = []
    for i in range(6):
        derivative = (3/(knots[i+3]-knots[i])*basis(i, 2, at)
                      -3/(knots[i+4]-knots[i+1])*basis(i+1, 2, at))
        result.append(basis(i, 3, at)+(value-at)*derivative)
    return result


class AdditiveKernelTests(unittest.TestCase):
    def setUp(self):
        self.x = np.random.default_rng(43).normal(size=(48, 3))
        self.x[:, 2] = .1
        self.y = .02*np.sin(self.x[:, 0])+.01*self.x[:, 1]**2
        self.names = ["momentum", "volatility", "constant"]

    def test_actual_two_stage_fit_and_independent_basis_ridge_reference(self):
        from sklearn.linear_model import Ridge
        from sklearn.preprocessing import SplineTransformer
        original_spline, original_ridge, calls = SplineTransformer.fit, Ridge.fit, []
        def spline_fit(estimator, *args, **kwargs):
            calls.append("spline fit")
            return original_spline(estimator, *args, **kwargs)
        def ridge_fit(estimator, *args, **kwargs):
            calls.append("ridge fit")
            return original_ridge(estimator, *args, **kwargs)
        with patch.object(SplineTransformer, "fit", spline_fit), patch.object(Ridge, "fit", ridge_fit):
            state = fit_additive_spline(self.x, self.y, self.names)
        self.assertEqual(calls, ["spline fit", "ridge fit"])
        self.assertEqual(state["constant_features"], {"constant": .1})
        self.assertEqual(state["basis_columns"], 12)
        mean, scale = np.array(state["mean"]), np.array(state["scale"])
        z = ((self.x-mean)/scale)[:, :2]
        transformer = SplineTransformer(n_knots=5, degree=3, knots="uniform", extrapolation="linear", include_bias=False).fit(z)
        design = transformer.transform(z)
        ridge = Ridge(alpha=1., solver="svd", tol=1e-12).fit(design, self.y)
        query = np.array([[-12., 10., .1], [12., -10., .1], [0., 0., .1], *self.x[:4]])
        q = ((query-mean)/scale)[:, :2]
        actual = predict_additive_spline(state, query, self.names)
        np.testing.assert_allclose(actual, ridge.predict(transformer.transform(q)), rtol=1e-11, atol=1e-12)
        independent_train = np.array([sum((independent_basis(row[j], state["knots"][j]) for j in range(2)), []) for row in z])
        independent_query = np.array([sum((independent_basis(row[j], state["knots"][j]) for j in range(2)), []) for row in q])
        np.testing.assert_allclose(independent_train, design, rtol=1e-12, atol=1e-12)
        centered = independent_train-independent_train.mean(0)
        coef = np.linalg.solve(centered.T@centered+np.eye(12), centered.T@(self.y-self.y.mean()))
        expected = (independent_query-independent_train.mean(0))@coef+self.y.mean()
        np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-12)

    def test_additive_structure_constants_boundaries_and_no_refit(self):
        state = fit_additive_spline(self.x, self.y, self.names)
        values = predict_additive_spline(state, [[1., 2., .1], [-1., -2., .1], [1., -2., .1], [-1., 2., .1]], self.names)
        self.assertAlmostEqual(values[0]+values[1]-values[2]-values[3], 0., places=13)
        expected = predict_additive_spline(state, self.x, self.names)
        with patch("sklearn.preprocessing.SplineTransformer.fit", side_effect=AssertionError("no basis refit")), patch("sklearn.linear_model.Ridge.fit", side_effect=AssertionError("no regression refit")):
            actual = predict_additive_spline(json.loads(json.dumps(state)), self.x, self.names)
        np.testing.assert_array_equal(actual, expected)
        with self.assertRaisesRegex(ValueError, "constant training"):
            predict_additive_spline(state, [[1., 2., .2]], self.names)
        constant_y = fit_additive_spline(self.x, np.zeros(48), self.names)
        np.testing.assert_array_equal(predict_additive_spline(constant_y, self.x[:2], self.names), [0., 0.])

    def test_exact_work_bounds_types_and_caller_isolation(self):
        for count, features in [(20, 1), (256, 16)]:
            x = np.random.default_rng(9).normal(size=(count, features))
            names = [str(i) for i in range(features)]
            state = fit_additive_spline(x, np.arange(count)/10000, names)
            self.assertEqual(state["basis_columns"], features*6)
            self.assertEqual(len(predict_additive_spline(state, np.tile(x[0], (2000, 1)), names)), 2000)
        for bad in [True, "1", None, float("nan"), float("inf"), 10**1000]:
            x = self.x.tolist()
            x[0][0] = bad
            with self.assertRaises(ValueError):
                fit_additive_spline(x, self.y, self.names)
        for count in [0, 19, 257]:
            with self.assertRaises(ValueError):
                fit_additive_spline(np.zeros((count, 3)), np.zeros(count), self.names)
        with self.assertRaises(ValueError):
            fit_additive_spline(np.ones((20, 3)), np.zeros(20), self.names)
        x, y, names = self.x.tolist(), self.y.tolist(), self.names.copy()
        state = fit_additive_spline(x, y, names)
        before = deepcopy(state)
        x[0][0], y[0], names[0] = 9., 9., "changed"
        self.assertEqual(state, before)
        with self.assertRaises(ValueError):
            predict_additive_spline(state, np.zeros((2001, 3)), self.names)
        with self.assertRaises(ValueError):
            predict_additive_spline(state, self.x, self.names[::-1])
        with patch.dict("sys.modules", {"sklearn.preprocessing": None}), self.assertRaises(ImportError):
            fit_additive_spline(self.x, self.y, self.names)

    def test_tampered_even_rehashed_knots_transforms_and_coefficients_rejected(self):
        from gig.ml.additive_spline import _fingerprint
        original = fit_additive_spline(self.x, self.y, self.names)
        for field, value in [("trading_approved", True), ("production_promoted", True), ("interval_calibrated", True),
                             ("threads", True), ("basis_columns", True), ("training_rows", True)]:
            state = {**deepcopy(original), field: value}
            with self.subTest(field=field), self.assertRaises(ValueError):
                predict_additive_spline(state, self.x, self.names)
        for change in [lambda s: s["mean"].__setitem__(0, 9.), lambda s: s["coef"].__setitem__(0, 9.),
                       lambda s: s["knots"][0].__setitem__(0, -99.), lambda s: s["active_indices"].reverse(),
                       lambda s: s["parameters"].update(degree=True), lambda s: s["constant_features"].update(constant=2.)]:
            state = deepcopy(original)
            change(state)
            state["training_basis_sha256"] = _fingerprint(state)
            with self.assertRaises(ValueError):
                predict_additive_spline(state, self.x, self.names)


class AdditivePersistedTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.base = Path(temporary.name)
        self.dataset = PersistedDatasetWorkspace(self.base).build(seed_persisted_fixture(self.base)).to_dict()["data"]
        self.request = {"dataset_id": self.dataset["dataset_id"], "model": "additive_spline", "min_train": 30, "test_size": 12}

    def test_actual_oos_metrics_restart_and_relabelled_state_cannot_allocate(self):
        artifact, states = fit_dataset(self.dataset, configuration(self.request))
        for fold in artifact["folds"]:
            rows = [r for r in artifact["rows"] if r["model_id"] == fold["model_id"]]
            self.assertEqual(fold["mse"], float(np.mean([(r["target"]-r["prediction"])**2 for r in rows])))
            self.assertLessEqual(fold["basis_columns"], 96)
            self.assertLess(fold["max_training_label_end"], fold["test_start"])
            self.assertFalse(fold["uncertainty_calibrated"])
        data = PersistedResearchWorkspace(self.base).run(self.request).to_dict()["data"]
        payload = {"dataset_id": data["dataset_id"], "run_id": data["run_id"]}
        with patch("sklearn.preprocessing.SplineTransformer.fit", side_effect=AssertionError("no fitting")):
            a = PersistedResearchWorkspace(self.base).predict(payload).to_dict()
            b = PersistedResearchWorkspace(self.base).predict(payload).to_dict()
        self.assertEqual(a, b)
        self.assertFalse(a["data"]["metrics"]["training_performed"])
        self.assertFalse(a["data"]["usable_for_portfolio"])
        for state in states.values():
            rows = saved_predict(state, self.dataset)
            expected = {r["timestamp"]: r for r in artifact["rows"] if r["model_id"] == identity(state)}
            for row in rows:
                if row["timestamp"] in expected:
                    self.assertEqual(row["prediction"], expected[row["timestamp"]]["prediction"])
            self.assertTrue(all(not r["usable_for_portfolio"] for r in saved_predict({**state, "fit_state": "CONVERGED"}, self.dataset)))

    def test_future_data_revisions_and_strict_cutoff_do_not_change_first_fit(self):
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

    def test_preflight_rejection_before_any_fit_and_capability_boundary(self):
        with patch("gig.research.additive_spline.fit_additive_spline", side_effect=AssertionError("preflight first")), self.assertRaises(ValueError):
            fit_dataset(self.dataset, {**configuration(self.request), "min_train": 20, "test_size": 4})
        changed = deepcopy(self.dataset)
        for row in changed["rows"]:
            row["features"] = {n: 1. for n in row["features"]}
        with patch("gig.research.additive_spline.fit_additive_spline", side_effect=AssertionError("preflight first")), self.assertRaises(ValueError):
            fit_dataset(changed, configuration(self.request))
        for extra in [{"min_train": 257}, {"n_knots": 6}, {"iterations": 20}]:
            with self.assertRaises(ValueError):
                configuration({**self.request, **extra})
        caps = model_capabilities()["generalized_additive"]
        self.assertTrue(caps["implemented"])
        self.assertFalse(caps["trained"])
        self.assertFalse(caps["trading_approved"])


class AdditiveHTTPTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        async def heartbeat():
            while True:
                await asyncio.sleep(.01)
        self.heartbeat = asyncio.create_task(heartbeat())

    async def asyncTearDown(self):
        self.heartbeat.cancel()
        await asyncio.gather(self.heartbeat, return_exceptions=True)

    async def test_http_saved_dataset_prediction_and_allocation_rejection(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            request = seed_persisted_fixture(base)
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=create_app(state_dir=base)), base_url="http://test") as client:
                dataset = (await client.post("/api/v1/workspace/research-dataset", json=request)).json()["data"]
                result = await client.post("/api/v1/workspace/research-model", json={"dataset_id": dataset["dataset_id"], "model": "additive_spline", "min_train": 30, "test_size": 12})
                self.assertEqual(result.status_code, 200, result.text)
                data = result.json()["data"]
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=create_app(state_dir=base)), base_url="http://test") as client:
                with patch("sklearn.preprocessing.SplineTransformer.fit", side_effect=AssertionError("no fitting")):
                    replay = await client.post("/api/v1/workspace/research-predict", json={"dataset_id": dataset["dataset_id"], "run_id": data["run_id"]})
                self.assertEqual(replay.json()["state"], "AVAILABLE")
                self.assertFalse(replay.json()["data"]["usable_for_portfolio"])
                self.assertTrue(all(r["uncertainty"]["value"] is None for r in replay.json()["data"]["rows"]))
                rejected = await client.post("/api/v1/workspace/research-allocate", json=allocation_request(data["run_id"], request["as_of"]))
                self.assertEqual(rejected.json()["state"], "UNAVAILABLE")
                self.assertFalse((base/"allocations").exists())
                self.assertFalse((base/"execution-engine").exists())
