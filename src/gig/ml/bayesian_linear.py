"""Bounded empirical-Bayes ridge and non-executable posterior JSON replay.

Gaussian linear/noise assumptions and plug-in precisions do not establish OOS
calibration. ARD, sampling and automatic model promotion are not implemented.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import scipy
from threadpoolctl import threadpool_limits

from gig.ml.gaussian_process import _array, _names, _scaled
from gig.ml.svm import _normalizer
from gig.service.serialization import dumps

IMPLEMENTATION = "gig.bayesian-ridge.v1"
MAX_TRAIN, MAX_FITS, MAX_PREDICT, MAX_ITER = 256, 16, 2000, 512


def parameters():
    return {"max_iter": MAX_ITER, "tol": 1e-6, "alpha_1": 1e-6, "alpha_2": 1e-6,
            "lambda_1": 1e-6, "lambda_2": 1e-6, "alpha_init": None, "lambda_init": None,
            "compute_score": True, "fit_intercept": True, "copy_X": True, "verbose": False}


def _fingerprint(state):
    return hashlib.sha256(dumps({k: state[k] for k in
        ("features", "training_x", "training_y", "mean", "scale", "parameters",
         "coef", "intercept", "posterior_covariance", "noise_precision", "weight_precision")}).encode()).hexdigest()


def validate(state):
    if (not isinstance(state, dict) or state.get("implementation") != IMPLEMENTATION
            or state.get("model") != "bayesian_ridge" or state.get("device") != "cpu"
            or type(state.get("threads")) is not int or state["threads"] != 1
            or state.get("trading_approved") is not False or state.get("production_promoted") is not False
            or state.get("interval_calibrated") is not False or state.get("hyperparameter_uncertainty_included") is not False
            or state.get("posterior_assumptions") != "GAUSSIAN_LINEAR_IID_HOMOSKEDASTIC_PLUGIN_PRECISIONS"
            or state.get("preprocessing") != "TRAINING_ONLY_POPULATION_STANDARDIZATION"
            or state.get("fit_state") != "CONVERGED_BOUNDED_EMPIRICAL_BAYES"):
        raise ValueError("unsupported Bayesian ridge state or execution boundary")
    expected, actual = parameters(), state.get("parameters")
    if (not isinstance(actual, dict) or set(actual) != set(expected)
            or any(type(actual[k]) is not type(v) or actual[k] != v for k, v in expected.items())):
        raise ValueError("Bayesian ridge fixed parameters differ")
    names = _names(state.get("features"))
    x = _array(state.get("training_x"), rows=MAX_TRAIN, columns=len(names))
    y = _array(state.get("training_y"), rows=MAX_TRAIN)
    mean = _array(state.get("mean"), rows=len(names))
    scale = _array(state.get("scale"), rows=len(names))
    coef = _array(state.get("coef"), rows=len(names))
    covariance = _array(state.get("posterior_covariance"), rows=len(names), columns=len(names))
    precision = _array([state.get("noise_precision"), state.get("weight_precision"), state.get("intercept")], rows=3)
    if (not 20 <= len(x) == len(y) or len(coef) != len(names) or covariance.shape != (len(names), len(names))
            or len(mean) != len(names) or len(scale) != len(names)
            or type(state.get("training_rows")) is not int or state["training_rows"] != len(y)
            or type(state.get("iterations")) is not int or not 1 <= state["iterations"] < MAX_ITER
            or not np.all(precision[:2] >= 1e-12) or state.get("training_posterior_sha256") != _fingerprint(state)):
        raise ValueError("Bayesian ridge dimensions, precision, convergence or identity differ")
    actual_mean, actual_scale = _normalizer(x)
    if not np.array_equal(mean, actual_mean) or not np.array_equal(scale, actual_scale):
        raise ValueError("Bayesian ridge transforms must be determined by training observations")
    z = _scaled(x, mean, scale)
    centered = z-z.mean(axis=0)
    posterior_precision = precision[1]*np.eye(len(names)) + precision[0]*(centered.T @ centered)
    if (not np.allclose(covariance, covariance.T, rtol=0, atol=1e-10)
            or np.linalg.eigvalsh(covariance).min() <= 0
            or not np.allclose(posterior_precision @ covariance, np.eye(len(names)), rtol=1e-7, atol=1e-7)):
        raise ValueError("Bayesian ridge posterior covariance is inconsistent with training data")
    expected_coef = precision[0]*covariance @ centered.T @ (y-y.mean())
    if (not np.allclose(coef, expected_coef, rtol=1e-7, atol=1e-9)
            or not np.isclose(precision[2], y.mean()-z.mean(axis=0) @ coef, rtol=1e-10, atol=1e-10)):
        raise ValueError("Bayesian ridge posterior coefficients/intercept differ")
    return names, mean, scale, coef, covariance, precision


def predict_bayesian_ridge(state, x, names):
    if _names(names) != state.get("features"):
        raise ValueError("Bayesian ridge feature identities/order differ")
    values = _array(x, rows=MAX_PREDICT, columns=len(names))
    with threadpool_limits(limits=1):
        _, mean, scale, coef, covariance, precision = validate(state)
        z = _scaled(values, mean, scale)
        prediction = z @ coef + precision[2]
        variance = np.sum((z @ covariance)*z, axis=1)+1/precision[0]
        if (not np.isfinite(prediction).all() or np.max(np.abs(prediction)) > 1e12
                or not np.isfinite(variance).all() or np.any(variance <= 0) or np.max(variance) > 1e12):
            raise ValueError("Bayesian ridge prediction/variance exceeds numeric bounds")
        return prediction, np.sqrt(variance)


def fit_bayesian_ridge(x, y, names):
    names = _names(names)
    values, target = _array(x, rows=MAX_TRAIN, columns=len(names)), _array(y, rows=MAX_TRAIN)
    if not 20 <= len(values) == len(target):
        raise ValueError("Bayesian ridge requires20..256paired training rows")
    try:
        import sklearn
        from sklearn.linear_model import BayesianRidge
    except ImportError:
        raise ImportError("actual sklearn BayesianRidge is required; no fallback") from None
    mean, scale = _normalizer(values)
    z = _scaled(values, mean, scale)
    estimator = BayesianRidge(**parameters())
    with threadpool_limits(limits=1):
        estimator.fit(z, target)
        if estimator.n_iter_ >= MAX_ITER:
            raise ValueError("Bayesian ridge did not converge within its iteration budget")
        expected, spread = estimator.predict(z, return_std=True)
    state = {"implementation": IMPLEMENTATION, "model": "bayesian_ridge", "features": names,
             "training_x": values.tolist(), "training_y": target.tolist(),
             "mean": mean.tolist(), "scale": scale.tolist(), "coef": estimator.coef_.tolist(),
             "intercept": float(estimator.intercept_), "posterior_covariance": estimator.sigma_.tolist(),
             "noise_precision": float(estimator.alpha_), "weight_precision": float(estimator.lambda_),
             "iterations": int(estimator.n_iter_), "training_rows": len(target), "parameters": parameters(),
             "preprocessing": "TRAINING_ONLY_POPULATION_STANDARDIZATION",
             "fit_state": "CONVERGED_BOUNDED_EMPIRICAL_BAYES", "device": "cpu", "threads": 1,
             "fit_backend": "sklearn.linear_model.BayesianRidge", "inference_backend": IMPLEMENTATION+".json-algebra",
             "posterior_assumptions": "GAUSSIAN_LINEAR_IID_HOMOSKEDASTIC_PLUGIN_PRECISIONS",
             "interval_calibrated": False, "hyperparameter_uncertainty_included": False,
             "library_versions": {"numpy": np.__version__, "scipy": scipy.__version__, "scikit-learn": sklearn.__version__},
             "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "license": "BSD-3-Clause",
             "trading_approved": False, "production_promoted": False}
    state["training_posterior_sha256"] = _fingerprint(state)
    actual, actual_sd = predict_bayesian_ridge(state, values, names)
    if not np.allclose(actual, expected, rtol=1e-12, atol=1e-12) or not np.allclose(actual_sd, spread, rtol=1e-12, atol=1e-12):
        raise ValueError("Bayesian ridge JSON inference differs from actual fitted estimator")
    return state
