"""Bounded additive cubic splines and fixed ridge; uncertainty is unavailable."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import scipy
from scipy.interpolate import BSpline
from threadpoolctl import threadpool_limits

from gig.ml.gaussian_process import _array, _names, _scaled
from gig.ml.svm import _normalizer
from gig.service.serialization import dumps

IMPLEMENTATION = "gig.additive-cubic-spline.v1"
MAX_TRAIN, MAX_FITS, MAX_PREDICT = 256, 16, 2000
DEGREE, N_KNOTS, PER_FEATURE, MAX_BASIS = 3, 5, 6, 96


def parameters():
    return {"degree": DEGREE, "n_knots": N_KNOTS, "knots": "uniform", "extrapolation": "linear",
            "include_bias": False, "ridge_alpha": 1.0, "ridge_solver": "svd", "fit_intercept": True}


def _fingerprint(state):
    return hashlib.sha256(dumps({k: state[k] for k in ("features", "training_x", "training_y", "mean", "scale",
        "active_indices", "constant_features", "knots", "coef", "intercept", "parameters")}).encode()).hexdigest()


def _basis(z, knots):
    columns = []
    for column, vector in enumerate(knots):
        spline = BSpline(vector, np.eye(PER_FEATURE+1), DEGREE, extrapolate=False)
        minimum, maximum = vector[DEGREE], vector[-DEGREE-1]
        value = z[:, column]
        basis = spline(np.clip(value, minimum, maximum))
        below, above = value < minimum, value > maximum
        if np.any(below):
            basis[below] = 0.
            basis[below, :DEGREE] = (spline(minimum)[:DEGREE] +
                (value[below]-minimum)[:, None]*spline(minimum, nu=1)[:DEGREE])
        if np.any(above):
            basis[above] = 0.
            basis[above, -DEGREE:] = (spline(maximum)[-DEGREE:] +
                (value[above]-maximum)[:, None]*spline(maximum, nu=1)[-DEGREE:])
        columns.append(basis[:, :-1])
    design = np.concatenate(columns, axis=1)
    if not np.isfinite(design).all() or np.max(np.abs(design)) > 1e8:
        raise ValueError("additive spline basis exceeds finite work bounds")
    return design


def validate(state):
    if (not isinstance(state, dict) or state.get("implementation") != IMPLEMENTATION
            or state.get("model") != "additive_spline" or state.get("device") != "cpu"
            or type(state.get("threads")) is not int or state["threads"] != 1
            or state.get("trading_approved") is not False or state.get("production_promoted") is not False
            or state.get("interval_calibrated") is not False
            or state.get("constant_feature_policy") != "REJECT_NEW_VARIATION_AT_INFERENCE"
            or state.get("preprocessing") != "TRAINING_ONLY_POPULATION_STANDARDIZATION"
            or state.get("basis_coefficients") != "FIXED_IDENTITY_7"
            or state.get("fit_state") != "FITTED_BOUNDED_ADDITIVE_SPLINES"):
        raise ValueError("unsupported additive spline state or execution boundary")
    if not {"features", "training_x", "training_y", "mean", "scale", "active_indices", "constant_features",
            "knots", "coef", "intercept", "parameters"} <= state.keys():
        raise ValueError("incomplete additive spline state")
    actual, expected = state.get("parameters"), parameters()
    if (not isinstance(actual, dict) or set(actual) != set(expected)
            or any(type(actual[k]) is not type(v) or actual[k] != v for k, v in expected.items())):
        raise ValueError("additive spline fixed parameters differ")
    names = _names(state.get("features"))
    x, y = _array(state.get("training_x"), rows=MAX_TRAIN, columns=len(names)), _array(state.get("training_y"), rows=MAX_TRAIN)
    mean, scale = _array(state.get("mean"), rows=len(names)), _array(state.get("scale"), rows=len(names))
    active = state.get("active_indices")
    expected_active = np.flatnonzero(np.any(x != x[0], axis=0)).tolist()
    if (not 20 <= len(x) == len(y) or not expected_active or not isinstance(active, list)
            or any(type(i) is not int for i in active) or active != expected_active
            or type(state.get("training_rows")) is not int or state["training_rows"] != len(y)
            or type(state.get("basis_columns")) is not int or state["basis_columns"] != PER_FEATURE*len(active)
            or state.get("training_basis_sha256") != _fingerprint(state)):
        raise ValueError("additive spline dimensions, active inputs or training identity differ")
    actual_mean, actual_scale = _normalizer(x)
    if not np.array_equal(mean, actual_mean) or not np.array_equal(scale, actual_scale):
        raise ValueError("additive spline transforms must come from training observations")
    constants = {names[i]: float(x[0, i]) for i in range(len(names)) if i not in active}
    if not isinstance(state.get("constant_features"), dict) or set(state["constant_features"]) != set(constants):
        raise ValueError("additive spline constant-feature identities differ")
    for name, level in constants.items():
        value = _array([state["constant_features"][name]], rows=1)[0]
        if value != level:
            raise ValueError("additive spline constant training levels differ")
    z = _scaled(x, mean, scale)[:, active]
    knots = _array(state.get("knots"), rows=len(active), columns=N_KNOTS+2*DEGREE)
    base = np.linspace(z.min(0), z.max(0), N_KNOTS)
    expected_knots = np.r_[base[0] - np.arange(DEGREE, 0, -1)[:, None]*(base[1]-base[0]),
                           base, base[-1]+np.arange(1, DEGREE+1)[:, None]*(base[-1]-base[-2])].T
    if knots.shape != expected_knots.shape or not np.allclose(knots, expected_knots, rtol=1e-12, atol=1e-12):
        raise ValueError("additive spline knots differ from the training-only uniform policy")
    coef = _array(state.get("coef"), rows=MAX_BASIS)
    intercept = _array([state.get("intercept")], rows=1)[0]
    if len(coef) != PER_FEATURE*len(active):
        raise ValueError("additive spline coefficient shape differs")
    design = _basis(z, knots)
    centered = design-design.mean(0)
    gradient = centered.T@(centered@coef-(y-y.mean()))+coef
    if (np.max(np.abs(gradient)) > 1e-7*(1+np.linalg.norm(y))
            or not np.isclose(intercept, y.mean()-design.mean(0)@coef, rtol=1e-10, atol=1e-10)):
        raise ValueError("additive spline coefficients/intercept do not solve the declared ridge objective")
    return names, mean, scale, active, knots, coef, intercept, constants


def predict_additive_spline(state, x, names):
    if _names(names) != state.get("features"):
        raise ValueError("additive spline feature identities/order differ")
    values = _array(x, rows=MAX_PREDICT, columns=len(names))
    with threadpool_limits(limits=1):
        _, mean, scale, active, knots, coef, intercept, constants = validate(state)
        if any(np.any(values[:, names.index(name)] != level) for name, level in constants.items()):
            raise ValueError("new variation in a constant training feature is unsupported")
        result = _basis(_scaled(values, mean, scale)[:, active], knots)@coef+intercept
    if not np.isfinite(result).all() or np.max(np.abs(result)) > 1e12:
        raise ValueError("additive spline predictions exceed numeric bounds")
    return result


def fit_additive_spline(x, y, names):
    names = _names(names)
    values, target = _array(x, rows=MAX_TRAIN, columns=len(names)), _array(y, rows=MAX_TRAIN)
    active = np.flatnonzero(np.any(values != values[0], axis=0)).tolist()
    if not 20 <= len(values) == len(target) or not active:
        raise ValueError("additive splines require20..256paired rows and at least one nonconstant input")
    try:
        import sklearn
        from sklearn.linear_model import Ridge
        from sklearn.preprocessing import SplineTransformer
    except ImportError:
        raise ImportError("actual sklearn spline and Ridge fitting required; no fallback") from None
    mean, scale = _normalizer(values)
    z = _scaled(values, mean, scale)[:, active]
    transformer = SplineTransformer(n_knots=N_KNOTS, degree=DEGREE, knots="uniform", extrapolation="linear",
                                    include_bias=False, order="C", sparse_output=False)
    estimator = Ridge(alpha=1., solver="svd", fit_intercept=True, tol=1e-12)
    with threadpool_limits(limits=1):
        design = transformer.fit_transform(z)
        estimator.fit(design, target)
        expected = estimator.predict(design)
    state = {"implementation": IMPLEMENTATION, "model": "additive_spline", "features": names,
             "training_x": values.tolist(), "training_y": target.tolist(), "mean": mean.tolist(), "scale": scale.tolist(),
             "active_indices": active, "constant_features": {names[i]: float(values[0, i]) for i in range(len(names)) if i not in active},
             "constant_feature_policy": "REJECT_NEW_VARIATION_AT_INFERENCE",
             "knots": [spline.t.tolist() for spline in transformer.bsplines_], "basis_coefficients": "FIXED_IDENTITY_7",
             "coef": estimator.coef_.tolist(), "intercept": float(estimator.intercept_),
             "basis_columns": int(design.shape[1]), "training_rows": len(target), "parameters": parameters(),
             "preprocessing": "TRAINING_ONLY_POPULATION_STANDARDIZATION", "device": "cpu", "threads": 1,
             "fit_state": "FITTED_BOUNDED_ADDITIVE_SPLINES", "fit_backend": "sklearn.SplineTransformer+Ridge",
             "inference_backend": IMPLEMENTATION+".scipy-bspline-json", "interval_calibrated": False,
             "library_versions": {"numpy": np.__version__, "scipy": scipy.__version__, "scikit-learn": sklearn.__version__},
             "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "license": "BSD-3-Clause",
             "trading_approved": False, "production_promoted": False}
    state["training_basis_sha256"] = _fingerprint(state)
    if not np.allclose(predict_additive_spline(state, values, names), expected, rtol=1e-12, atol=1e-12):
        raise ValueError("additive spline JSON replay differs from actual sklearn fitting")
    return state
