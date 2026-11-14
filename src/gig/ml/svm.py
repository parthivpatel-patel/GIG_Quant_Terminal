"""Actual bounded epsilon-SVR training and non-executable fixed-RBF replay.

The epsilon tube is a training-loss parameter, not predictive uncertainty.
Only training observations determine feature and target normalization.
"""
from __future__ import annotations

import hashlib
import warnings
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.distance import cdist
from threadpoolctl import threadpool_limits

from gig.ml.gaussian_process import _array, _names, _scaled
from gig.ml.gaussian_process import _normalizer as _base_normalizer
from gig.service.serialization import dumps

IMPLEMENTATION = "gig.fixed-rbf-svr.v1"
MAX_TRAIN, MAX_FITS, MAX_PREDICT = 128, 16, 2000
C, EPSILON, TOLERANCE = 1., .1, 1e-6
MAX_ITER, CACHE_MIB = 10000, 16


def _hash(value: Any) -> str:
    return hashlib.sha256(dumps(value).encode()).hexdigest()


def _parameters(dimensions: int) -> dict[str, Any]:
    return {"kernel": "rbf", "gamma": 1./dimensions, "C": C, "epsilon": EPSILON,
            "tol": TOLERANCE, "shrinking": False, "cache_size_mib": CACHE_MIB,
            "max_iter": MAX_ITER}


def _normalizer(values: np.ndarray):
    mean, scale = _base_normalizer(values)
    constant = np.all(values == values[0], axis=0)
    # Exact constant columns can have tiny nonzero std from mean roundoff.
    return np.where(constant, values[0], mean), np.where(constant, 1., scale)


def _training_hash(state):
    return _hash({k: state[k] for k in ("features", "training_x", "training_y", "mean", "scale",
                                       "target_mean", "target_scale")})


def validate_state(state: Any):
    if (not isinstance(state, dict) or state.get("implementation") != IMPLEMENTATION
            or state.get("model") != "svr" or state.get("family") != "support_vector_machines_svr"
            or state.get("device") != "cpu" or type(state.get("threads")) is not int or state["threads"] != 1
            or state.get("trading_approved") is not False or state.get("production_promoted") is not False
            or type(state.get("fit_status")) is not int or state["fit_status"] != 0):
        raise ValueError("unsupported SVR state or research execution boundary")
    names = _names(state.get("features"))
    x = _array(state.get("training_x"), rows=MAX_TRAIN, columns=len(names))
    y = _array(state.get("training_y"), rows=MAX_TRAIN)
    mean = _array(state.get("mean"), rows=len(names))
    scale = _array(state.get("scale"), rows=len(names))
    levels = _array([state.get("target_mean"), state.get("target_scale")], rows=2)
    expected_parameters = _parameters(len(names))
    parameters = state.get("parameters")
    if (not isinstance(parameters, dict) or set(parameters) != set(expected_parameters)
            or any(type(parameters[k]) is not type(v) or parameters[k] != v
                   for k, v in expected_parameters.items())):
        raise ValueError("SVR fixed kernel parameters differ")
    if (not 20 <= len(x) == len(y) or len(mean) != len(names) or len(scale) != len(names)
            or type(state.get("training_rows")) is not int or state["training_rows"] != len(y)
            or state.get("training_transform_sha256") != _training_hash(state)):
        raise ValueError("SVR dimensions, fixed parameters or training identity differ")
    actual_mean, actual_scale = _normalizer(x)
    target_mean, target_scale = _normalizer(y[:, None])
    if (not np.array_equal(mean, actual_mean) or not np.array_equal(scale, actual_scale)
            or not np.array_equal(levels, [target_mean[0], target_scale[0]])):
        raise ValueError("SVR normalization is not determined by training observations")
    train = _scaled(x, mean, scale)
    support = state.get("support_indices")
    if (not isinstance(support, list) or len(support) > len(y)
            or any(type(i) is not int or not 0 <= i < len(y) for i in support)
            or support != sorted(set(support))):
        raise ValueError("bounded unique ordered SVR support indices required")
    if support:
        vectors = _array(state.get("support_vectors"), rows=len(support), columns=len(names))
        dual = _array(state.get("dual"), rows=len(support))
        if (len(vectors) != len(support) or len(dual) != len(support)
                or not np.array_equal(vectors, train[support]) or np.any(np.abs(dual) > C+1e-12)
                or abs(float(dual.sum())) > 1e-8):
            raise ValueError("SVR support vectors or bounded dual coefficients differ")
    else:
        if state.get("support_vectors") != [] or state.get("dual") != []:
            raise ValueError("empty SVR support requires empty vectors and coefficients")
        vectors, dual = np.empty((0, len(names))), np.empty(0)
    intercept = _array([state.get("intercept")], rows=1)[0]
    iterations = state.get("iterations")
    if type(iterations) is not int or not 0 <= iterations < MAX_ITER:
        raise ValueError("SVR iteration/convergence work bound failed")
    return vectors, dual, float(intercept)


def predict_svr(state: dict[str, Any], x: Any, names: list[str]) -> np.ndarray:
    if _names(names) != state.get("features"):
        raise ValueError("SVR inference feature identities/order differ")
    values = _array(x, rows=MAX_PREDICT, columns=len(names))
    vectors, dual, intercept = validate_state(state)
    z = _scaled(values, np.asarray(state["mean"]), np.asarray(state["scale"]))
    with threadpool_limits(limits=1):
        kernel = np.exp(-state["parameters"]["gamma"] * cdist(z, vectors, metric="sqeuclidean"))
        prediction = (kernel @ dual + intercept) * state["target_scale"] + state["target_mean"]
    if not np.isfinite(prediction).all() or np.max(np.abs(prediction)) > 1e12:
        raise ValueError("SVR inference exceeds finite numeric range")
    return prediction


def fit_svr(x: Any, y: Any, names: list[str]) -> dict[str, Any]:
    names = _names(names)
    values = _array(x, rows=MAX_TRAIN, columns=len(names))
    target = _array(y, rows=MAX_TRAIN)
    if not 20 <= len(values) == len(target):
        raise ValueError("SVR requires20..128paired training observations")
    try:
        import sklearn
        from sklearn.exceptions import ConvergenceWarning
        from sklearn.svm import SVR
    except ImportError:
        raise ImportError("actual scikit-learn SVR library required; no substitution") from None
    mean, scale = _normalizer(values)
    target_mean, target_scale = _normalizer(target[:, None])
    z = _scaled(values, mean, scale)
    normalized = _scaled(target[:, None], target_mean, target_scale)[:, 0]
    estimator = SVR(kernel="rbf", gamma=1./len(names), C=C, epsilon=EPSILON, tol=TOLERANCE,
                    shrinking=False, cache_size=CACHE_MIB, max_iter=MAX_ITER)
    with threadpool_limits(limits=1), warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        estimator.fit(z, normalized)
    if estimator.fit_status_ != 0 or int(estimator.n_iter_) >= MAX_ITER:
        raise ValueError("bounded SVR did not converge; no fitted state published")
    state = {"implementation": IMPLEMENTATION, "model": "svr", "family": "support_vector_machines_svr",
             "features": names, "training_x": values.tolist(), "training_y": target.tolist(),
             "mean": mean.tolist(), "scale": scale.tolist(), "target_mean": float(target_mean[0]),
             "target_scale": float(target_scale[0]), "preprocessing": "TRAINING_ONLY_POPULATION_STANDARDIZATION",
             "support_indices": estimator.support_.tolist(), "support_vectors": estimator.support_vectors_.tolist(),
             "dual": estimator.dual_coef_[0].tolist(), "intercept": float(estimator.intercept_[0]),
             "parameters": _parameters(len(names)), "training_rows": len(target),
             "fit_status": int(estimator.fit_status_), "iterations": int(estimator.n_iter_),
             "device": "cpu", "threads": 1, "fit_backend": "sklearn.svm.SVR",
             "inference_backend": IMPLEMENTATION+".json-algebra",
             "library_versions": {"numpy": np.__version__, "scikit-learn": sklearn.__version__},
             "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
             "license": "BSD-3-Clause", "trading_approved": False, "production_promoted": False}
    state["training_transform_sha256"] = _training_hash(state)
    replay = predict_svr(state, values, names)
    with threadpool_limits(limits=1):
        expected = estimator.predict(z)*state["target_scale"] + state["target_mean"]
    if not np.allclose(replay, expected, rtol=1e-10, atol=1e-12):
        raise ValueError("JSON SVR replay differs from actual sklearn inference")
    return state
