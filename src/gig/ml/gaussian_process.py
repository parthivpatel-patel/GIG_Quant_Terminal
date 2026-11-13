"""Fixed-RBF CPU Gaussian-process research, with non-executable JSON replay.

Training uses actual scikit-learn. Replay implements the saved conditional
Gaussian equations, not pickle loading or a substitute model. Posterior spread
is assumption-dependent and is never advertised as calibrated confidence.
"""
from __future__ import annotations

import hashlib
import math
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import scipy
from scipy.linalg import solve_triangular
from scipy.spatial.distance import cdist
from threadpoolctl import threadpool_limits

from gig.service.serialization import dumps

IMPLEMENTATION = "gig.fixed-rbf-gp.v1"
MAX_TRAIN_ROWS = 128
MAX_FEATURES = 16
MAX_PREDICT_ROWS = 2000
MAX_FITS = 16
NOISE_VARIANCE = 1e-4
_MAX_LEVEL = 1e12
_MAX_SCALED = 1e6


def _digest(value: Any) -> str:
    return hashlib.sha256(dumps(value).encode()).hexdigest()


def _names(names: Any) -> list[str]:
    if (not isinstance(names, list) or not 1 <= len(names) <= MAX_FEATURES
            or any(not isinstance(n, str) or not n or len(n) > 128 or any(ord(c) < 32 for c in n) for n in names)
            or len(set(names)) != len(names)):
        raise ValueError("unique bounded GP feature names required")
    return list(names)


def _array(values: Any, *, rows: int, columns: int | None = None) -> np.ndarray:
    """Bound shapes and reject coercive types before allocating binary64 copies."""
    flat: Iterable[Any]
    if isinstance(values, np.ndarray):
        if (values.dtype.kind not in "fiu" or values.dtype.itemsize > 8
                or values.ndim != (1 if columns is None else 2)
                or not 1 <= values.shape[0] <= rows
                or (columns is not None and values.shape[1] != columns)):
            raise ValueError("bounded binary64 GP array required")
        flat = values.flat
    elif isinstance(values, (list, tuple)) and 1 <= len(values) <= rows:
        if columns is None:
            flat = iter(values)
        else:
            if any(not isinstance(row, (list, tuple)) or len(row) != columns for row in values):
                raise ValueError("rectangular bounded GP features required")
            flat = (v for row in values for v in row)
    else:
        raise ValueError("bounded nonempty GP observations required")
    for v in flat:
        if (isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, float, np.integer, np.floating))
                or (isinstance(v, np.floating) and v.dtype.itemsize > 8)):
            raise ValueError("finite bounded noncoercive GP numbers required")
        try:
            numeric = float(v)
        except OverflowError:
            raise ValueError("GP numeric range exceeded") from None
        if not math.isfinite(numeric) or abs(numeric) > _MAX_LEVEL:
            raise ValueError("finite bounded GP numbers required")
    return np.array(values, dtype=np.float64, copy=True)


def _normalizer(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean, spread = values.mean(axis=0), values.std(axis=0)
    if np.any((spread == 0) & np.any(values != values[0], axis=0)):
        raise ValueError("GP training variance underflow")
    return mean, np.where(spread > 0, spread, 1.)


def _scaled(x: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        z = (x - mean) / scale
    if not np.isfinite(z).all() or np.max(np.abs(z)) > _MAX_SCALED:
        raise ValueError("GP standardized feature work/range bound exceeded")
    return z


def _kernel(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return np.exp(-.5 * cdist(left, right, metric="sqeuclidean"))


def _training_hash(state: dict[str, Any]) -> str:
    return _digest({k: state[k] for k in ("features", "mean", "scale", "target_mean", "target_scale", "x_train", "normalized_targets")})


def fit_gaussian_process(x: Any, y: Any, names: list[str]) -> dict[str, Any]:
    names = _names(names)
    values = _array(x, rows=MAX_TRAIN_ROWS, columns=len(names))
    target = _array(y, rows=MAX_TRAIN_ROWS)
    if len(values) != len(target) or len(target) < 3:
        raise ValueError("GP requires 3..128 paired training observations")
    try:
        import sklearn
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import RBF
    except ImportError:
        raise ImportError("actual scikit-learn GP training library is required; no model substitution") from None
    mean, scale = _normalizer(values)
    ymean, yscale = _normalizer(target[:, None])
    z = _scaled(values, mean, scale)
    normalized = ((target - float(ymean[0])) / float(yscale[0]))
    if not np.isfinite(normalized).all():
        raise ValueError("GP target normalization is nonfinite")
    estimator = GaussianProcessRegressor(kernel=RBF(length_scale=1., length_scale_bounds="fixed"),
                                         alpha=NOISE_VARIANCE, optimizer=None, normalize_y=False, copy_X_train=True)
    with threadpool_limits(limits=1):
        estimator.fit(z, normalized)
    state = {"implementation": IMPLEMENTATION, "model": "gaussian_process", "features": names,
             "mean": mean.tolist(), "scale": scale.tolist(), "target_mean": float(ymean[0]), "target_scale": float(yscale[0]),
             "x_train": estimator.X_train_.tolist(), "normalized_targets": normalized.tolist(),
             "dual": estimator.alpha_.tolist(), "cholesky": estimator.L_.tolist(),
             "kernel": {"kind": "RBF", "length_scale": 1., "amplitude": 1., "noise_variance": NOISE_VARIANCE},
             "optimization": "DISABLED_FIXED_KERNEL", "device": "cpu", "threads": 1,
             "fit_backend": "sklearn.gaussian_process.GaussianProcessRegressor",
             "inference_backend": IMPLEMENTATION + ".json-algebra", "training_rows": len(target),
             "library_versions": {"numpy": np.__version__, "scipy": scipy.__version__, "scikit-learn": sklearn.__version__},
             "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
             "license": "BSD-3-Clause", "trading_approved": False, "production_promoted": False}
    state["training_transform_sha256"] = _training_hash(state)
    validate_state(state)
    return state


def validate_state(state: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if (not isinstance(state, dict) or state.get("implementation") != IMPLEMENTATION
            or state.get("model") != "gaussian_process" or state.get("device") != "cpu"
            or state.get("threads") != 1 or state.get("optimization") != "DISABLED_FIXED_KERNEL"
            or state.get("trading_approved") is not False or state.get("production_promoted") is not False
            or state.get("kernel") != {"kind": "RBF", "length_scale": 1., "amplitude": 1., "noise_variance": NOISE_VARIANCE}):
        raise ValueError("unsupported GP state or approval/kernel boundary")
    names = _names(state.get("features"))
    train = _array(state.get("x_train"), rows=MAX_TRAIN_ROWS, columns=len(names))
    target = _array(state.get("normalized_targets"), rows=MAX_TRAIN_ROWS)
    dual = _array(state.get("dual"), rows=MAX_TRAIN_ROWS)
    chol = _array(state.get("cholesky"), rows=MAX_TRAIN_ROWS, columns=len(train))
    mean = _array(state.get("mean"), rows=MAX_FEATURES)
    scale = _array(state.get("scale"), rows=MAX_FEATURES)
    level = _array([state.get("target_mean"), state.get("target_scale")], rows=2)
    if (not 3 <= len(train) == len(dual) == len(target) == len(chol)
            or len(mean) != len(names) or len(scale) != len(names) or np.any(scale <= 0) or level[1] <= 0
            or state.get("training_rows") != len(train) or isinstance(state.get("training_rows"), bool)
            or np.max(np.abs(train)) > _MAX_SCALED or np.any(np.diag(chol) <= 0)
            or np.any(np.triu(chol, 1) != 0) or state.get("training_transform_sha256") != _training_hash(state)):
        raise ValueError("GP state dimension, scaling or training identity mismatch")
    with threadpool_limits(limits=1):
        gram = _kernel(train, train) + np.eye(len(train)) * NOISE_VARIANCE
        if (not np.allclose(chol @ chol.T, gram, rtol=1e-9, atol=1e-10)
                or not np.allclose(gram @ dual, target, rtol=1e-7, atol=1e-7)):
            raise ValueError("GP saved factor/weights fail independent kernel consistency")
    return train, dual, chol


def predict_gaussian_process(state: dict[str, Any], x: Any, names: list[str]) -> tuple[np.ndarray, np.ndarray]:
    if _names(names) != state.get("features"):
        raise ValueError("GP inference feature identity/order mismatch")
    values = _array(x, rows=MAX_PREDICT_ROWS, columns=len(names))
    train, dual, chol = validate_state(state)
    z = _scaled(values, np.asarray(state["mean"]), np.asarray(state["scale"]))
    with threadpool_limits(limits=1):
        covariance = _kernel(z, train)
        mean = covariance @ dual * state["target_scale"] + state["target_mean"]
        solved = solve_triangular(chol, covariance.T, lower=True, check_finite=True)
        latent_variance = 1. - np.sum(solved ** 2, axis=0)
    if np.any(latent_variance < -1e-9):
        raise ValueError("GP posterior variance violates numerical tolerance")
    # Tiny negative roundoff is clamped only for the derived kernel variance.
    # Predictive noise is a declared positive assumption, never unknown data.
    spread = np.sqrt(np.maximum(latent_variance, 0.) + NOISE_VARIANCE) * state["target_scale"]
    if not np.isfinite(mean).all() or not np.isfinite(spread).all() or np.any(spread <= 0):
        raise ValueError("GP prediction/uncertainty exceeds finite bounds")
    return mean, spread
