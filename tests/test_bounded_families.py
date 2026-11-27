"""Registry-wide contract tests plus independent numerical references per family."""
from __future__ import annotations

import asyncio
import json
import subprocess
import sys
import tempfile
import unittest
import warnings
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

import httpx
import numpy as np

from gig.ml.bounded import FAMILIES, capabilities, fit_family, predict_family
from gig.ml.bounded.core import _fingerprint
from gig.research.model_pipeline import (
    KINDS,
    RESEARCH_ONLY,
    configuration,
    fit_dataset,
    identity,
    saved_predict,
)
from gig.research.workflow import model_capabilities
from gig.service.app import create_app
from gig.service.dataset_workspace import PersistedDatasetWorkspace
from gig.service.persisted_research import PersistedResearchWorkspace
from test_persisted_data import fixture_times, seed_persisted_fixture
from test_persisted_flow_http import allocation_request

ROOT = Path(__file__).resolve().parents[1]
INVENTORY = {15: ["huber", "ransac"], 16: ["quantile"], 17: ["xgboost"], 18: ["catboost"], 19: ["adaboost"],
             20: ["ngboost"], 21: ["ebm"], 22: ["arima"], 23: ["ets"], 24: ["var"],
             25: ["dynamic_factor"], 26: ["gmm"], 27: ["kmeans"], 28: ["pca"], 29: ["ica"], 30: ["nmf"],
             31: ["gaussian_copula"], 32: ["evt_pot"], 33: ["bocpd"], 35: ["mlp"], 36: ["tcn"], 37: ["esn"],
             38: ["nbeats"], 39: ["attention"], 40: ["autoencoder"]}


# Diagnostics that no replay reads; every other numeric payload field must be re-derived or checked.
INFORMATIONAL = {"training_excess_kurtosis", "explained_variance_ratio", "sigma", "final_training_loss"}


# Engines from the optional `boosting` extra; absent ones are checked as UNAVAILABLE instead of fitted.
OPTIONAL = {"xgboost", "catboost", "ngboost", "ebm"}


def families():
    return {key: family for key, family in FAMILIES.items() if key not in OPTIONAL or family.available()}


def sample(key, rows=None, seed=5):
    """Authored SYNTHETIC inputs shaped for one family; returns (x, y, names)."""
    family = FAMILIES[key]
    rows = rows or max(64, family.min_train)
    rng = np.random.default_rng(seed)
    if family.data == "FEATURES":
        x = rng.normal(size=(rows, 5))
        x[:, 4] += .7*x[:, 3]
        x[:, 2] = .1
        if not family.standardize:
            x = np.abs(x)
        y = .01*x[:, 0]-.004*x[:, 1]**2+.003*rng.standard_t(4, size=rows) if family.supervised else None
        return x, y, ["a", "b", "constant", "d", "e"]
    width = 1 if family.data == "SERIES" else 3
    x = .01*rng.standard_t(5, size=(rows, width))
    x[1:, 0] += .35*x[:-1, 0]
    if width > 1:
        x[:, 1] += .5*x[:, 0]
    return x, None, ["return_1"] if width == 1 else ["US:AAA", "US:BBB", "US:CCC"]


class RegistryTests(unittest.TestCase):
    def test_inventory_numbers_kinds_capabilities_and_static_assets_match_registry(self):
        self.assertEqual({n: sorted(k for k, f in FAMILIES.items() if f.number == n) for n in INVENTORY},
                         {n: sorted(v) for n, v in INVENTORY.items()})
        self.assertEqual(sum(len(v) for v in INVENTORY.values()), len(FAMILIES))
        caps = model_capabilities()
        for key, family in FAMILIES.items():
            self.assertEqual(KINDS[key], family.kind)
            self.assertIn(key, RESEARCH_ONLY)
            self.assertIn("Allocation, promotion and trading remain unapproved", family.scope)
            entry = caps[family.capability]
            self.assertTrue(entry["implemented"])
            self.assertFalse(entry["trained"])
            self.assertFalse(entry["trading_approved"])
            self.assertIn(key, entry["variants"])
            self.assertEqual(entry["family_number"], family.number)
        self.assertEqual(set(capabilities()) & {"ridge", "lightgbm", "bilstm", "finbert", "gaussian_process"}, set())
        done = subprocess.run([sys.executable, str(ROOT/"scripts/sync_bounded_families.py"), "--check"],
                              capture_output=True, text=True, env={"PYTHONPATH": str(ROOT/"src"), "PATH": ""})
        self.assertEqual(done.returncode, 0, done.stdout+done.stderr)

    def test_missing_library_is_unavailable_without_a_fallback(self):
        family = FAMILIES["arima"]
        with patch("importlib.util.find_spec", return_value=None):
            self.assertFalse(family.available())
            self.assertEqual(capabilities()["arima"]["state"], "UNAVAILABLE")
            with self.assertRaisesRegex(ImportError, "no fallback"):
                fit_family("arima", *sample("arima"))


class ContractTests(unittest.TestCase):
    """Every registered family obeys the same replay, bound and tamper rules."""

    def test_json_replay_is_deterministic_bounded_and_isolated(self):
        for key, family in families().items():
            with self.subTest(family=key):
                x, y, names = sample(key)
                original = x.copy()
                state = fit_family(key, x, y, names, 11)
                np.testing.assert_array_equal(x, original)
                self.assertEqual(state, fit_family(key, x.tolist(), None if y is None else y.tolist(), names, 11))
                self.assertEqual((state["device"], state["threads"]), ("cpu", 1))
                self.assertFalse(state["trading_approved"] or state["production_promoted"] or state["interval_calibrated"])
                expected = predict_family(state, x, names)
                actual = predict_family(json.loads(json.dumps(state)), x, names)
                np.testing.assert_array_equal(actual, expected)
                self.assertEqual(actual.shape[0], len(x))
                self.assertEqual(actual.shape[-1], len(family.outputs))
                self.assertTrue(np.isfinite(actual[-1]).all() or key == "evt_pot")
                with self.assertRaises(ValueError):
                    predict_family(state, x, names[::-1] if len(names) > 1 else ["other"])
                with self.assertRaises(ValueError):
                    predict_family(state, np.tile(x[:1], (2001, 1)), names)

    def test_work_bounds_and_noncoercive_inputs_reject(self):
        for key, family in families().items():
            with self.subTest(family=key):
                x, y, names = sample(key, family.max_train)
                fit_family(key, x, y, names)
                for rows in (family.min_train-1, family.max_train+1):
                    big = np.vstack([x, x[:1]])[:rows] if rows > len(x) else x[:rows]
                    target = None if y is None else np.r_[y, y[:1]][:rows]
                    with self.assertRaises(ValueError):
                        fit_family(key, big, target, names)
                for bad in (True, "1", None, float("nan"), float("inf")):
                    broken = x.tolist()
                    broken[3][0] = bad
                    with self.assertRaises(ValueError):
                        fit_family(key, broken, y, names)
                for seed in (-1, 2**31, True, 1.5):
                    with self.assertRaises(ValueError):
                        fit_family(key, x, y, names, seed)
                with self.assertRaises(ValueError):
                    fit_family(key, np.ones_like(x), y, names)

    def test_boundary_flags_and_rehashed_tampering_reject(self):
        for key, family in families().items():
            x, y, names = sample(key)
            original = fit_family(key, x, y, names)
            for field, value in [("trading_approved", True), ("production_promoted", True), ("interval_calibrated", True),
                                 ("threads", True), ("threads", 2), ("device", "cuda"), ("family_number", 99),
                                 ("training_rows", True), ("model", "ridge"), ("implementation", "other")]:
                with self.subTest(family=key, field=field), self.assertRaises(ValueError):
                    predict_family({**deepcopy(original), field: value}, x, names)
            changes = [lambda s: s["mean"].__setitem__(0, 9.), lambda s: s["scale"].__setitem__(0, 9.),
                       lambda s: s["training_x"][0].__setitem__(0, s["training_x"][0][0]+1.),
                       lambda s: s["parameters"].update(extra=1), lambda s: s["outputs"].reverse() or s["outputs"].append("x"),
                       lambda s: s.__setitem__("seed", True)]
            if family.check is not None:
                for name, value in original["payload"].items():
                    if isinstance(value, (list, float)) and name not in INFORMATIONAL:
                        def corrupt(state, name=name):
                            saved = np.array(state["payload"][name], dtype=float)
                            state["payload"][name] = (saved+.37).tolist() if saved.ndim else float(saved)+.37
                        changes.append(corrupt)
            for index, change in enumerate(changes):
                state = deepcopy(original)
                change(state)
                with self.subTest(family=key, change=index, rehashed=False), self.assertRaises(ValueError):
                    predict_family(state, x, names)
                state["state_sha256"] = _fingerprint(state)
                with self.subTest(family=key, change=index, rehashed=True), self.assertRaises(ValueError):
                    predict_family(state, x, names)

    def test_history_outputs_never_depend_on_later_rows(self):
        for key, family in families().items():
            if family.data == "FEATURES":
                continue
            with self.subTest(family=key):
                x, _, names = sample(key, max(64, family.min_train)+30)
                train = max(64, family.min_train)
                state = fit_family(key, x[:train], None, names)
                before = predict_family(state, x, names)
                changed = x.copy()
                changed[train+10:] += .5
                after = predict_family(state, changed, names)
                np.testing.assert_allclose(before[:train+10], after[:train+10], rtol=1e-10, atol=1e-13)
                np.testing.assert_allclose(predict_family(state, x[:train+10], names), before[:train+10], rtol=1e-10, atol=1e-13)
                self.assertGreater(np.nanmax(np.abs(before[train+10:]-after[train+10:])), 1e-6 if key != "evt_pot" else -1)


class ReferenceTests(unittest.TestCase):
    """Independent recomputation of each family's saved replay."""

    def fit(self, key, **options):
        x, y, names = sample(key, **options)
        state = fit_family(key, x, y, names, 3)
        z = ((x-np.array(state["mean"]))/np.array(state["scale"]))[:, state["active_indices"]]
        return x, y, names, state, z, predict_family(state, x, names)

    def target(self, state, y):
        return (y-state["target_mean"][0])/state["target_scale"][0]

    def test_supervised_variants_match_fresh_library_estimators(self):
        from sklearn.ensemble import AdaBoostRegressor
        from sklearn.linear_model import HuberRegressor, LinearRegression, QuantileRegressor
        from sklearn.neural_network import MLPRegressor
        from sklearn.tree import DecisionTreeRegressor
        query = np.random.default_rng(1).normal(size=(40, 5))
        query[:, 2] = .1

        def check(key, estimator, columns=None):
            x, y, names, state, z, _ = self.fit(key)
            t = self.target(state, y)
            q = ((query-np.array(state["mean"]))/np.array(state["scale"]))[:, state["active_indices"]]
            fitted = [e.fit(z if key != "adaboost" else z.astype(np.float32), t) for e in estimator]
            expected = np.column_stack([e.predict(q if key != "adaboost" else q.astype(np.float32)) for e in fitted])
            actual = predict_family(state, query, names)
            np.testing.assert_allclose(actual, expected*state["target_scale"][0]+state["target_mean"][0], rtol=1e-9, atol=1e-12)
            self.assertEqual(state["constant_features"], ["constant"])
            return state, z, t
        state, z, t = check("huber", [HuberRegressor(epsilon=1.35, alpha=1e-4, max_iter=512, tol=1e-8)])
        self.assertGreater(state["payload"]["outliers"], 0)
        check("quantile", [QuantileRegressor(quantile=q, alpha=0., solver="highs") for q in (.1, .5, .9)])
        check("adaboost", [AdaBoostRegressor(estimator=DecisionTreeRegressor(max_depth=2, min_samples_leaf=3), n_estimators=16,
                                             learning_rate=.5, loss="linear", random_state=3)])
        check("mlp", [MLPRegressor(hidden_layer_sizes=(8,), activation="tanh", solver="lbfgs", alpha=1e-2, max_iter=256,
                                   tol=1e-8, random_state=3)])
        x, y, names, state, z, values = self.fit("ransac")
        inliers = state["payload"]["inliers"]
        ols = LinearRegression().fit(z[inliers], self.target(state, y)[inliers])
        np.testing.assert_allclose(values[:, 0], ols.predict(z)*state["target_scale"][0]+state["target_mean"][0], rtol=1e-9, atol=1e-12)
        self.assertLess(len(inliers), len(x))


    def test_boosted_engines_match_fresh_library_estimators_on_unseen_rows(self):
        query = np.random.default_rng(1).normal(size=(40, 5))
        query[:, 2] = .1

        def xgboost_reference(z, t, q):
            import xgboost
            booster = xgboost.train({"objective": "reg:squarederror", "max_depth": 2, "eta": .1, "min_child_weight": 3., "lambda": 1.,
                                     "tree_method": "exact", "base_score": 0., "nthread": 1, "verbosity": 0, "seed": 3},
                                    xgboost.DMatrix(z.astype(np.float32), label=t), num_boost_round=32)
            return booster.predict(xgboost.DMatrix(q.astype(np.float32)))[:, None]

        def catboost_reference(z, t, q):
            import catboost
            model = catboost.CatBoostRegressor(iterations=32, depth=2, learning_rate=.1, l2_leaf_reg=3., border_count=32,
                                               boosting_type="Ordered", bootstrap_type="No", random_seed=3, thread_count=1,
                                               allow_writing_files=False, verbose=False).fit(z.astype(np.float32), t)
            return model.predict(q.astype(np.float32))[:, None]

        def ngboost_reference(z, t, q):
            from ngboost import NGBRegressor
            from ngboost.distns import Normal
            from sklearn.tree import DecisionTreeRegressor
            model = NGBRegressor(Dist=Normal, Base=DecisionTreeRegressor(criterion="squared_error", max_depth=2, min_samples_leaf=3),
                                 n_estimators=32, learning_rate=.05, minibatch_frac=1., col_sample=1., natural_gradient=True,
                                 random_state=3, verbose=False).fit(z.astype(np.float32), t)
            fitted = model.pred_dist(q.astype(np.float32))
            return np.column_stack([fitted.loc, fitted.scale])

        def ebm_reference(z, t, q):
            from interpret.glassbox import ExplainableBoostingRegressor
            model = ExplainableBoostingRegressor(interactions=2, max_bins=32, max_interaction_bins=8, outer_bags=1, inner_bags=0,
                                                 max_rounds=200, learning_rate=.05, validation_size=.15, early_stopping_rounds=20,
                                                 min_samples_leaf=3, n_jobs=1, random_state=3).fit(z, t)
            return model.predict(q)[:, None]

        references = {"xgboost": (xgboost_reference, 1e-5), "catboost": (catboost_reference, 1e-9),
                      "ngboost": (ngboost_reference, 1e-9), "ebm": (ebm_reference, 1e-9)}
        self.assertEqual(set(references), OPTIONAL)
        for key, (reference, tolerance) in references.items():
            if not FAMILIES[key].available():
                self.assertEqual(capabilities()[FAMILIES[key].capability]["state"], "UNAVAILABLE")
                with self.assertRaisesRegex(ImportError, "no fallback"):
                    fit_family(key, *sample(key))
                continue
            with self.subTest(family=key), warnings.catch_warnings():
                warnings.simplefilter("ignore")
                x, y, names, state, z, _ = self.fit(key)
                q = ((query-np.array(state["mean"]))/np.array(state["scale"]))[:, state["active_indices"]]
                expected = reference(z, self.target(state, y), q)
                scale, mean = state["target_scale"][0], state["target_mean"][0]
                expected[:, 0] = expected[:, 0]*scale+mean
                if key == "ngboost":
                    expected[:, 1] *= scale
                actual = predict_family(state, query, names)
                np.testing.assert_allclose(actual, expected, rtol=tolerance, atol=tolerance*scale)
                self.assertGreater(np.ptp(actual[:, 0]), 0)
                if key == "ngboost":
                    self.assertTrue(np.all(actual[:, 1] > 0))
                    broken = deepcopy(state)
                    broken["payload"]["stages"][0]["step"] = -1.
                    broken["state_sha256"] = _fingerprint(broken)
                    with self.assertRaises(ValueError):
                        predict_family(broken, query, names)
                if key == "ebm":
                    self.assertTrue(any(len(term["features"]) == 2 for term in state["payload"]["terms"]))

    def test_history_variants_match_independent_recursions(self):
        x, _, names, state, z, values = self.fit("arima")
        mu, (a, b) = state["payload"]["mean"], state["payload"]["phi"]
        manual = [mu+a*(z[i, 0]-mu)+b*(z[i-1, 0]-mu) for i in range(1, len(z))]
        np.testing.assert_allclose(values[1:, 0], np.array(manual)*state["scale"][0]+state["mean"][0], rtol=1e-10, atol=1e-13)
        lagged = np.c_[np.ones(len(z)-2), z[1:-1, 0], z[:-2, 0]]
        conditional = np.linalg.lstsq(lagged, z[2:, 0], rcond=None)[0]
        np.testing.assert_allclose([a, b], conditional[1:], atol=.08)
        x, _, names, state, z, values = self.fit("ets")
        level, alpha, manual = state["payload"]["initial_level"], state["payload"]["smoothing_level"], []
        for value in z[:, 0]:
            level = alpha*value+(1-alpha)*level
            manual.append(level)
        np.testing.assert_allclose(values[:, 0], np.array(manual)*state["scale"][0]+state["mean"][0], rtol=1e-10, atol=1e-13)
        x, _, names, state, z, values = self.fit("var")
        design = np.c_[np.ones(len(z)-1), z[:-1]]
        solution = np.linalg.lstsq(design, z[1:], rcond=None)[0]
        expected = (np.c_[np.ones(len(z)), z]@solution)*np.array(state["scale"])+np.array(state["mean"])
        np.testing.assert_allclose(values[:, :, 0], expected, rtol=1e-8, atol=1e-11)
        x, _, names, state, z, values = self.fit("dynamic_factor")
        from statsmodels.tsa.statespace.dynamic_factor import DynamicFactor
        payload = state["payload"]
        fixed = DynamicFactor(z, k_factors=1, factor_order=1).smooth(
            [*payload["loadings"], *payload["idiosyncratic_variance"], payload["factor_autoregression"]])
        np.testing.assert_allclose(values[:-1, :, 0], fixed.predict()[1:]*np.array(state["scale"])+np.array(state["mean"]),
                                   rtol=1e-7, atol=1e-10)
        self.assertAlmostEqual(fixed.llf, payload["training_loglike"], places=6)

    def test_structure_variants_match_fresh_library_estimators(self):
        from scipy.optimize import nnls
        from sklearn.cluster import KMeans
        from sklearn.decomposition import PCA, FastICA
        from sklearn.mixture import GaussianMixture
        x, _, names, state, z, values = self.fit("gmm")
        model = GaussianMixture(n_components=2, covariance_type="diag", reg_covar=1e-6, tol=1e-8, max_iter=512, random_state=3).fit(z)
        np.testing.assert_allclose(values[:, 0], model.score_samples(z), rtol=1e-9)
        np.testing.assert_allclose(values[:, 3], model.predict_proba(z).max(1), rtol=1e-8)
        np.testing.assert_allclose(values[:, 1], -.5*(z*z).sum(1)-.5*z.shape[1]*np.log(2*np.pi), rtol=1e-12)
        x, _, names, state, z, values = self.fit("kmeans")
        model = KMeans(n_clusters=3, n_init=4, max_iter=300, tol=0., algorithm="lloyd", random_state=3).fit(z)
        np.testing.assert_allclose(values[:, 0], model.transform(z).min(1)**2, rtol=1e-9, atol=1e-12)
        self.assertEqual(len(set(zip(values[:, 2].tolist(), model.labels_.tolist(), strict=True))), 3)
        x, _, names, state, z, values = self.fit("pca")
        model = PCA(n_components=2, svd_solver="full").fit(z)
        np.testing.assert_allclose(values[:, 0], ((z-model.inverse_transform(model.transform(z)))**2).sum(1), rtol=1e-8, atol=1e-12)
        np.testing.assert_allclose(np.abs(values[:, 2:]), np.abs(model.transform(z)), rtol=1e-8, atol=1e-10)
        self.assertTrue(all(row[np.argmax(np.abs(row))] > 0 for row in np.array(state["payload"]["components"])))
        x, _, names, state, z, values = self.fit("ica")
        model = FastICA(n_components=2, whiten="unit-variance", fun="logcosh", max_iter=512, tol=1e-6, random_state=3).fit(z)
        related = np.abs(np.corrcoef(values.T, model.transform(z).T)[:2, 2:])
        np.testing.assert_allclose(np.sort(related.max(1)), [1., 1.], atol=1e-8)
        kurtosis = np.abs(state["payload"]["training_excess_kurtosis"])
        self.assertGreaterEqual(kurtosis[0], kurtosis[1])
        x, _, names, state, z, values = self.fit("nmf")
        basis = np.array(state["payload"]["basis"])
        manual = np.array([((row-nnls(basis.T, row)[0]@basis)**2).sum() for row in z])
        np.testing.assert_allclose(values[:, 0], manual, rtol=1e-10, atol=1e-14)
        self.assertEqual(state["preprocessing"], "TRAINING_ONLY_MAXIMUM_SCALING_NO_SHIFT")
        with self.assertRaisesRegex(ValueError, "nonnegative"):
            fit_family("nmf", -x, None, names)
        with self.assertRaisesRegex(ValueError, "nonnegative"):
            predict_family(state, -x, names)
        x, _, names, state, z, values = self.fit("autoencoder")
        p = state["payload"]
        code = np.tanh(z@np.array(p["encoder_weights"])+np.array(p["encoder_bias"]))
        rebuilt = code@np.array(p["decoder_weights"])+np.array(p["decoder_bias"])
        np.testing.assert_allclose(values[:, 0], ((z-rebuilt)**2).sum(1), rtol=1e-10)
        self.assertLess(values[:, 0].mean(), values[:, 1].mean())

    def test_dependence_tail_and_change_point_match_independent_formulas(self):
        from scipy.stats import genpareto, multivariate_normal, norm, rankdata
        from scipy.stats import t as student
        x, _, names, state, z, values = self.fit("gaussian_copula")
        scores = norm.ppf((np.column_stack([rankdata(z[:, j], method="max") for j in range(3)])+.5)/(len(z)+1))
        correlation = np.corrcoef(scores.T)
        expected = multivariate_normal(np.zeros(3), correlation).logpdf(scores)-norm.logpdf(scores).sum(1)
        np.testing.assert_allclose(values[:, 0], expected, rtol=1e-9, atol=1e-11)
        np.testing.assert_array_equal(values[:, 1], 0.)
        x, _, names, state, z, values = self.fit("evt_pot", rows=200)
        p = state["payload"]
        loss = -z[:, 0]
        excess = loss[loss > p["threshold"]]-p["threshold"]
        self.assertEqual(len(excess), p["exceedances"])
        tail = genpareto(p["shape"], 0., p["scale"])
        for a, b in [(p["shape"]+.02, p["scale"]), (max(p["shape"]-.02, -.5), p["scale"]), (p["shape"], p["scale"]*1.05), (p["shape"], p["scale"]/1.05)]:
            self.assertGreaterEqual(tail.logpdf(excess).sum(), genpareto(a, 0., b).logpdf(excess).sum()-1e-9)
        level = 1-.01*len(loss)/len(excess)
        quantile = p["threshold"]+tail.ppf(level)
        shortfall = p["threshold"]+tail.expect(lambda v: v, lb=tail.ppf(level))/(1-level)
        scale, mean = state["scale"][0], state["mean"][0]
        np.testing.assert_allclose(values[:, 0], -quantile*scale+mean, rtol=1e-9)
        np.testing.assert_allclose(values[:, 1], -shortfall*scale+mean, rtol=1e-6)
        np.testing.assert_allclose(values[:, 2], np.quantile(x[:, 0], .01), rtol=1e-9)
        with self.assertRaises(ValueError):
            fit_family("evt_pot", x[:99], None, names)
        x, _, names, state, z, values = self.fit("bocpd")
        runs, means = {0: 1.}, []
        stats = {0: (0., 1., 1., 1.)}
        for value in z[:, 0]:
            grown, fresh, updated = {}, 0., {0: (0., 1., 1., 1.)}
            for run, weight in runs.items():
                m, k, a, b = stats[run]
                density = student(2*a, m, np.sqrt(b*(k+1)/(a*k))).pdf(value)
                fresh += weight*density/50
                grown[run+1] = weight*density*49/50
                updated[run+1] = ((k*m+value)/(k+1), k+1, a+.5, b+k*(value-m)**2/(2*(k+1)))
            grown[0] = fresh
            total = sum(grown.values())
            runs, stats = {r: w/total for r, w in grown.items()}, updated
            means.append((sum(w*stats[r][0] for r, w in runs.items()), sum(w for r, w in runs.items() if r < 5),
                          sum(w*r for r, w in runs.items())))
        means = np.array(means)
        np.testing.assert_allclose(values[:, 0], means[:, 0]*state["scale"][0]+state["mean"][0], rtol=1e-8, atol=1e-12)
        np.testing.assert_allclose(values[:, 1:], means[:, 1:], rtol=1e-8, atol=1e-10)

    def test_sequence_networks_match_independent_forward_passes(self):
        import torch
        x, _, names, state, z, values = self.fit("esn")
        p = state["payload"]
        self.assertLessEqual(np.max(np.abs(np.linalg.eigvals(np.array(p["recurrent"])))), .9+1e-9)
        hidden, manual, rows = np.zeros(32), [], []
        for value in z[:, 0]:
            hidden = np.tanh(np.array(p["recurrent"])@hidden+np.array(p["input"])*value+np.array(p["bias"]))
            rows.append(np.r_[1., hidden])
            manual.append(rows[-1]@np.array(p["readout"]))
        np.testing.assert_allclose(values[:, 0], np.array(manual)*state["scale"][0]+state["mean"][0], rtol=1e-9, atol=1e-12)
        design, target = np.array(rows[10:-1]), z[11:, 0]
        penalty = .01*np.eye(33)
        penalty[0, 0] = 0
        np.testing.assert_allclose(design@np.array(p["readout"]), design@np.linalg.solve(design.T@design+penalty, design.T@target), rtol=1e-6, atol=1e-9)
        x, _, names, state, z, values = self.fit("tcn")
        p = {k: torch.tensor(v, dtype=torch.float64) for k, v in state["payload"].items() if k != "final_training_loss"}
        series = torch.tensor(z[:, 0])[None, None]
        first = torch.stack([p["input_previous"], p["input"]], dim=1)[:, None, :]
        hidden = torch.relu(torch.nn.functional.conv1d(series, first, p["bias_0"]))
        for index, dilation in ((1, 2), (2, 4)):
            kernel = p[f"kernel_{index}"].permute(2, 1, 0)
            hidden = torch.relu(torch.nn.functional.conv1d(hidden, kernel, p[f"bias_{index}"], dilation=dilation))
        expected = (hidden[0].T@p["readout"]+p["readout_bias"]).numpy()
        np.testing.assert_allclose(values[7:, 0], expected*state["scale"][0]+state["mean"][0], rtol=1e-9, atol=1e-12)
        self.assertTrue(np.isnan(values[:7]).all())
        for key in ("nbeats", "attention"):
            x, _, names, state, z, values = self.fit(key)
            window = z[-8:, 0]
            p = {k: np.array(v) for k, v in state["payload"].items()}
            if key == "nbeats":
                residual, forecast = window.copy(), 0.
                for block in range(2):
                    h = np.maximum(residual@p[f"hidden_{block}"]+p[f"hidden_bias_{block}"], 0)
                    h = np.maximum(h@p[f"deep_{block}"]+p[f"deep_bias_{block}"], 0)
                    residual = residual-h@p[f"backcast_{block}"]
                    forecast += h@p[f"forecast_{block}"]+p[f"forecast_bias_{block}"]
            else:
                tokens = window[:, None]*p["embed"]+p["position"]
                scores = np.array([(tokens[-1]@p["query"])@(token@p["key"]) for token in tokens])/2
                weights = np.exp(scores)/np.exp(scores).sum()
                forecast = sum(w*(token@p["value"]) for w, token in zip(weights, tokens, strict=True))@p["readout"]+p["readout_bias"]
            self.assertAlmostEqual(values[-1, 0], forecast*state["scale"][0]+state["mean"][0], places=12)
            self.assertLess(state["payload"]["final_training_loss"], 1.5)


class PersistedTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        temporary = tempfile.TemporaryDirectory()
        cls.addClassCleanup(temporary.cleanup)
        cls.base = Path(temporary.name)
        cls.datasets = {}
        for periods in (80, 160):
            base = cls.base/str(periods)
            cls.datasets[periods] = PersistedDatasetWorkspace(base).build(seed_persisted_fixture(base, periods=periods)).to_dict()["data"]

    def request(self, key):
        family = FAMILIES[key]
        periods = 160 if family.min_train > 60 else 80
        dataset = self.datasets[periods]
        return dataset, {"dataset_id": dataset["dataset_id"], "model": key,
                         "min_train": max(30, family.min_train), "test_size": 12, "seed": 42}, self.base/str(periods)

    def test_purged_folds_saved_replay_and_research_only_boundary(self):
        for key, family in families().items():
            if key == "nmf":
                continue
            with self.subTest(family=key):
                dataset, request, _ = self.request(key)
                artifact, states = fit_dataset(dataset, configuration(request))
                self.assertFalse(artifact["usable_for_portfolio"] or artifact["trading_approved"] or artifact["economically_validated"])
                self.assertEqual(artifact["family_number"], family.number)
                self.assertEqual(artifact["metrics"]["mse_state"], "AVAILABLE" if family.return_units else "UNAVAILABLE_DIFFERENT_OUTPUT_UNITS")
                self.assertEqual(artifact["metrics"]["oos_count"], sum(f["test_count"] for f in artifact["folds"]))
                for fold in artifact["folds"]:
                    rows = [r for r in artifact["rows"] if r["model_id"] == fold["model_id"]]
                    self.assertEqual(len(rows), fold["test_count"])
                    self.assertLess(fold["max_training_label_end"], fold["test_start"])
                    self.assertFalse(fold["uncertainty_calibrated"])
                    self.assertTrue(all(fold["test_start"] <= r["timestamp"] <= fold["test_end"] for r in rows))
                    if family.return_units:
                        self.assertAlmostEqual(fold["mse"], float(np.mean([(r["target"]-r["prediction"])**2 for r in rows])), places=15)
                for row in artifact["rows"]:
                    self.assertFalse(row["usable_for_portfolio"] or row["trading_approved"])
                    self.assertFalse(row["uncertainty"]["calibrated"])
                    self.assertEqual(set(row.get("outputs", {"prediction": 0})), set(family.outputs))
                for state in states.values():
                    expected = {(r["instrument_id"], r["timestamp"]): r for r in artifact["rows"] if r["model_id"] == identity(state)}
                    with patch("gig.ml.bounded.core.Family.available", side_effect=AssertionError("replay never fits")):
                        replay = {(r["instrument_id"], r["timestamp"]): r for r in saved_predict(state, dataset)}
                    self.assertTrue(expected.keys() <= replay.keys())
                    for index, row in expected.items():
                        self.assertEqual({k: row[k] for k in ("prediction", "uncertainty")}, {k: replay[index][k] for k in ("prediction", "uncertainty")})
                    self.assertTrue(all(not r["usable_for_portfolio"] for r in saved_predict({**state, "fit_state": "CONVERGED"}, dataset)))

    def test_nonnegative_family_refuses_signed_persisted_features_before_fitting(self):
        dataset, request, _ = self.request("nmf")
        with patch("gig.research.bounded.fit_family", side_effect=AssertionError("preflight first")), self.assertRaisesRegex(ValueError, "nonnegative"):
            fit_dataset(dataset, configuration(request))

    def test_future_revisions_and_strict_label_cutoff_do_not_change_first_fold(self):
        for key in ("huber", "quantile", "arima", "var", "gmm", "bocpd", "esn"):
            with self.subTest(family=key):
                dataset, request, _ = self.request(key)
                before, models = fit_dataset(dataset, configuration(request))
                changed = deepcopy(dataset)
                late = fixture_times()[60].isoformat()
                for row in changed["rows"]:
                    if row["timestamp"] >= late:
                        row["features"]["momentum"] += .25
                        row["target"] += .1
                    row.setdefault("label_vintages", [{"target": row["target"], "label_end": row["label_end"], "knowledge_at": row["label_end"]}]).append(
                        {"target": .9, "label_end": row["label_end"], "knowledge_at": "2027-01-01T00:00:00Z"})
                for row in changed["panel_rows"]:
                    if row["timestamp"] >= late and isinstance(row.get("return_1"), float):
                        row["return_1"] += .05
                changed["dataset_id"] = identity(changed)
                after, later = fit_dataset(changed, configuration(request))
                first = [r for r in before["rows"] if r["fold"] == 0]
                self.assertTrue(first)
                self.assertEqual(first, [r for r in after["rows"] if r["fold"] == 0])
                for row in first:
                    self.assertEqual(models[row["model_id"]], later[row["model_id"]])
                cutoff = before["folds"][0]["test_start"]
                changed = deepcopy(dataset)
                victims = [r for r in changed["rows"] if r["timestamp"] < cutoff]
                victim = max(victims, key=lambda r: r["timestamp"])
                for row in victims:
                    if row["timestamp"] == victim["timestamp"]:
                        row["label_vintages"] = [{"target": .8, "label_end": row["label_end"], "knowledge_at": cutoff}]
                _, states = fit_dataset(changed, configuration(request))
                for state in states.values():
                    if state["training_cutoff"] == cutoff:
                        self.assertNotIn(victim["timestamp"], state["training_timestamps"])

    def test_whole_plan_is_preflighted_and_history_families_need_their_inputs(self):
        dataset, request, _ = self.request("huber")
        with patch("gig.research.bounded.fit_family", side_effect=AssertionError("preflight first")):
            with self.assertRaises(ValueError):
                fit_dataset(dataset, {**configuration(request), "min_train": 20, "test_size": 2})
            constant = deepcopy(dataset)
            for row in constant["rows"]:
                row["features"] = {n: 1. for n in row["features"]}
            with self.assertRaises(ValueError):
                fit_dataset(constant, configuration(request))
            broken = deepcopy(dataset)
            broken["panel_rows"] = [r for r in broken["panel_rows"] if r["instrument_id"] != "US:BBB"]
            with self.assertRaises(ValueError):
                fit_dataset(broken, configuration({**request, "model": "arima"}))
            single = deepcopy(dataset)
            single["rows"] = [r for r in single["rows"] if r["instrument_id"] == "US:AAA"]
            with self.assertRaisesRegex(ValueError, "two or more instruments"):
                fit_dataset(single, configuration({**request, "model": "var"}))
            gap = deepcopy(dataset)
            gap["rows"] = [r for r in gap["rows"] if not (r["instrument_id"] == "US:BBB" and r["timestamp"] == dataset["rows"][20]["timestamp"])]
            with self.assertRaisesRegex(ValueError, "identical decision times"):
                fit_dataset(gap, configuration({**request, "model": "var"}))
        for extra in [{"min_train": 257}, {"iterations": 20}, {"states": 2}]:
            with self.assertRaises(ValueError):
                configuration({**request, **extra})
        _, states = fit_dataset(dataset, configuration({**request, "model": "arima"}))
        state = next(iter(states.values()))
        moved = deepcopy(dataset)
        for row in moved["panel_rows"]:
            if row["instrument_id"] == state["instrument_id"] and row["timestamp"] == state["training_timestamps"][3]:
                row["return_1"] += .01
        with self.assertRaisesRegex(ValueError, "unchanged admitted training history"):
            saved_predict(state, moved)

    def test_workspace_restart_replay_is_stable_and_never_refits(self):
        for key in ("ransac", "ets", "dynamic_factor", "kmeans", "gaussian_copula", "attention"):
            with self.subTest(family=key):
                dataset, request, base = self.request(key)
                data = PersistedResearchWorkspace(base).run(request).to_dict()["data"]
                payload = {"dataset_id": data["dataset_id"], "run_id": data["run_id"]}
                with patch("gig.research.bounded.fit_family", side_effect=AssertionError("no fitting")):
                    a = PersistedResearchWorkspace(base).predict(payload).to_dict()
                    b = PersistedResearchWorkspace(base).predict(payload).to_dict()
                self.assertEqual(a, b)
                self.assertEqual(a["state"], "AVAILABLE")
                self.assertFalse(a["data"]["metrics"]["training_performed"])
                self.assertFalse(a["data"]["usable_for_portfolio"])
                self.assertEqual(a["data"]["prediction_kind"], FAMILIES[key].kind)


class HTTPTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        async def heartbeat():
            while True:
                await asyncio.sleep(.01)
        self.heartbeat = asyncio.create_task(heartbeat())

    async def asyncTearDown(self):
        self.heartbeat.cancel()
        await asyncio.gather(self.heartbeat, return_exceptions=True)

    async def test_http_research_replay_and_allocation_rejection(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            request = seed_persisted_fixture(base)
            transport = httpx.ASGITransport(app=create_app(state_dir=base))
            async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
                dataset = (await client.post("/api/v1/workspace/research-dataset", json=request)).json()["data"]
                for key in ("quantile", "var", "pca"):
                    family = FAMILIES[key]
                    response = await client.post("/api/v1/workspace/research-model", json={
                        "dataset_id": dataset["dataset_id"], "model": key, "min_train": 30, "test_size": 12})
                    body = response.json()
                    self.assertEqual((response.status_code, body["state"]), (200, "AVAILABLE"), body)
                    run = body["data"]
                    self.assertEqual(run["prediction_kind"], family.kind)
                    self.assertFalse(run["usable_for_portfolio"])
                    replay = (await client.post("/api/v1/workspace/research-predict", json={
                        "dataset_id": dataset["dataset_id"], "run_id": run["run_id"]})).json()
                    self.assertEqual(replay["state"], "AVAILABLE")
                    self.assertFalse(replay["data"]["metrics"]["training_performed"])
                    rejected = await client.post("/api/v1/workspace/research-allocate",
                                                 json=allocation_request(run["run_id"], request["as_of"]))
                    self.assertEqual(rejected.json()["state"], "UNAVAILABLE", key)
                self.assertFalse((base/"allocations").exists())
                self.assertFalse((base/"execution-engine").exists())
                unknown = await client.post("/api/v1/workspace/research-model", json={
                    "dataset_id": dataset["dataset_id"], "model": "hawkes", "min_train": 30, "test_size": 12})
                self.assertNotEqual(unknown.json().get("state"), "AVAILABLE")


if __name__ == "__main__":
    unittest.main()
