"""Bounded nonparametric KNN fit and safe JSON observation replay.

Fit retains training observations and labels; it does not optimize weights.
Equal binary64 squared distances use chronological retained training index.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import scipy
from scipy.spatial.distance import cdist
from threadpoolctl import threadpool_limits

from gig.ml.gaussian_process import _array, _names, _scaled
from gig.ml.svm import _normalizer
from gig.service.serialization import dumps

IMPLEMENTATION = "gig.uniform-knn.v1"
MAX_TRAIN, MAX_FITS, MAX_PREDICT = 256, 16, 2000
K = 5
POLICY = "BINARY64_SQUARED_EUCLIDEAN_THEN_CHRONOLOGICAL_RETAINED_INDEX"
PREPROCESSING = "TRAINING_ONLY_POPULATION_STANDARDIZATION"


def _hash(value: Any) -> str:
    return hashlib.sha256(dumps(value).encode()).hexdigest()


def _parameters() -> dict[str, Any]:
    return {"n_neighbors": K, "weights": "uniform", "algorithm": "brute",
            "metric": "sqeuclidean", "n_jobs": 1, "working_memory_mib": 8,
            "enable_cython_pairwise_dist": False}


def _training_hash(state):
    return _hash({k: state[k] for k in ("features", "training_x", "training_y", "mean", "scale",
                                       "retained_indices", "parameters", "tie_policy", "preprocessing")})


def validate_state(state: Any):
    if (not isinstance(state, dict) or state.get("implementation") != IMPLEMENTATION
            or state.get("model") != "knn" or state.get("family") != "nearest_neighbors"
            or state.get("device") != "cpu" or type(state.get("threads")) is not int or state["threads"] != 1
            or state.get("trading_approved") is not False or state.get("production_promoted") is not False
            or state.get("tie_policy") != POLICY or state.get("preprocessing") != PREPROCESSING
            or state.get("fit_semantics") != "RETAIN_OBSERVATIONS_AND_LABELS_NO_OPTIMIZED_WEIGHTS"):
        raise ValueError("unsupported KNN state or research execution boundary")
    names = _names(state.get("features"))
    x = _array(state.get("training_x"), rows=MAX_TRAIN, columns=len(names))
    y = _array(state.get("training_y"), rows=MAX_TRAIN)
    mean = _array(state.get("mean"), rows=len(names))
    scale = _array(state.get("scale"), rows=len(names))
    indices, parameters = state.get("retained_indices"), state.get("parameters")
    expected = _parameters()
    if (not isinstance(parameters, dict) or set(parameters) != set(expected)
            or any(type(parameters[k]) is not type(v) or parameters[k] != v for k, v in expected.items())):
        raise ValueError("KNN fixed parameters differ")
    if (not 20 <= len(x) == len(y) or len(mean) != len(names) or len(scale) != len(names)
            or type(state.get("training_rows")) is not int or state["training_rows"] != len(y)
            or not isinstance(indices, list) or len(indices) != len(y) or any(type(i) is not int for i in indices)
            or indices != list(range(len(y)))
            or state.get("training_transform_sha256") != _training_hash(state)):
        raise ValueError("KNN dimensions, chronological indices or training identity differ")
    actual_mean, actual_scale = _normalizer(x)
    if not np.array_equal(mean, actual_mean) or not np.array_equal(scale, actual_scale):
        raise ValueError("KNN normalization is not determined by training observations")
    return _scaled(x, mean, scale), y


def _canonical(distances, indices, target):
    """Every retained neighbor must be present before canonical K selection."""
    chosen = np.array([index[np.lexsort((index, distance))[:K]]
                       for distance, index in zip(distances, indices, strict=True)])
    return target[chosen].mean(axis=1)


def predict_knn(state: dict[str, Any], x: Any, names: list[str]) -> np.ndarray:
    if _names(names) != state.get("features"):
        raise ValueError("KNN inference feature identities/order differ")
    values = _array(x, rows=MAX_PREDICT, columns=len(names))
    train, target = validate_state(state)
    z = _scaled(values, np.asarray(state["mean"]), np.asarray(state["scale"]))
    with threadpool_limits(limits=1):
        distances = cdist(z, train, metric="sqeuclidean")
        indices = np.broadcast_to(np.arange(len(target)), distances.shape)
        prediction = _canonical(distances, indices, target)
    if not np.isfinite(prediction).all() or np.max(np.abs(prediction)) > 1e12:
        raise ValueError("KNN inference exceeds finite numeric range")
    return prediction


def fit_knn(x: Any, y: Any, names: list[str]) -> dict[str, Any]:
    """Input order is chronological retained training order; no rows are dropped."""
    names = _names(names)
    values = _array(x, rows=MAX_TRAIN, columns=len(names))
    target = _array(y, rows=MAX_TRAIN)
    if not 20 <= len(values) == len(target):
        raise ValueError("KNN requires 20..256 paired chronological training observations")
    try:
        import sklearn
        from sklearn.neighbors import KNeighborsRegressor
    except ImportError:
        raise ImportError("actual scikit-learn KNeighborsRegressor required; no substitution") from None
    mean, scale = _normalizer(values)
    z = _scaled(values, mean, scale)
    estimator = KNeighborsRegressor(n_neighbors=K, weights="uniform", algorithm="brute",
                                    metric="sqeuclidean", n_jobs=1)
    with threadpool_limits(limits=1), sklearn.config_context(enable_cython_pairwise_dist=False, working_memory=8):
        estimator.fit(z, target)
        distances, indices = estimator.kneighbors(z, n_neighbors=len(target), return_distance=True)
        expected = _canonical(distances, indices, target)
    state = {"implementation": IMPLEMENTATION, "model": "knn", "family": "nearest_neighbors",
             "features": names, "training_x": values.tolist(), "training_y": target.tolist(),
             "mean": mean.tolist(), "scale": scale.tolist(), "retained_indices": list(range(len(target))),
             "preprocessing": PREPROCESSING, "tie_policy": POLICY, "parameters": _parameters(),
             "training_rows": len(target), "device": "cpu", "threads": 1,
             "fit_semantics": "RETAIN_OBSERVATIONS_AND_LABELS_NO_OPTIMIZED_WEIGHTS",
             "fit_backend": "sklearn.neighbors.KNeighborsRegressor",
             "inference_backend": IMPLEMENTATION+".scipy-cdist-json",
             "library_versions": {"numpy": np.__version__, "scipy": scipy.__version__, "scikit-learn": sklearn.__version__},
             "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
             "license": "BSD-3-Clause", "trading_approved": False, "production_promoted": False}
    state["training_transform_sha256"] = _training_hash(state)
    if not np.array_equal(predict_knn(state, values, names), expected):
        raise ValueError("JSON KNN replay differs from actual sklearn canonical neighbor inference")
    return state
