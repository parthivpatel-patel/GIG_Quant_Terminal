"""Bounded regularized binomial GLM; probabilities are not approved signals."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import scipy
from scipy.special import expit
from threadpoolctl import threadpool_limits

from gig.ml.gaussian_process import _array, _names, _scaled
from gig.ml.svm import _normalizer
from gig.service.serialization import dumps

IMPLEMENTATION = "gig.binomial-logit.v1"
MAX_TRAIN, MAX_FITS, MAX_PREDICT, MAX_ITER = 256, 16, 2000, 512
LABEL_RULE = "FORWARD_ARITHMETIC_RETURN_STRICTLY_POSITIVE.v1"


def parameters():
    return {"C": 1.0, "solver": "lbfgs", "tol": 1e-8, "max_iter": MAX_ITER,
            "fit_intercept": True, "class_weight": None, "random_state": 0, "warm_start": False}


def _fingerprint(state):
    return hashlib.sha256(dumps({k: state[k] for k in
        ("features", "training_x", "training_y", "mean", "scale", "coef", "intercept", "parameters", "label_rule")}).encode()).hexdigest()


def _paired(x, y, names):
    values = _array(x, rows=MAX_TRAIN, columns=len(names))
    target = _array(y, rows=MAX_TRAIN)
    if (not 20 <= len(values) == len(target) or not np.isin(target, [0., 1.]).all()
            or np.sum(target == 0) < 3 or np.sum(target == 1) < 3):
        raise ValueError("binomial GLM requires20..256binary training rows and at least3per class")
    return values, target


def validate(state):
    if (not isinstance(state, dict) or state.get("implementation") != IMPLEMENTATION
            or state.get("model") != "binomial_glm" or state.get("device") != "cpu"
            or type(state.get("threads")) is not int or state["threads"] != 1
            or state.get("link") != "LOGIT" or state.get("likelihood") != "BERNOULLI"
            or state.get("label_rule") != LABEL_RULE or state.get("penalty") != "L2_COEFFICIENTS_UNPENALIZED_INTERCEPT"
            or state.get("preprocessing") != "TRAINING_ONLY_POPULATION_STANDARDIZATION"
            or state.get("trading_approved") is not False or state.get("production_promoted") is not False
            or state.get("probability_calibrated") is not False
            or state.get("fit_state") != "CONVERGED_BOUNDED_BINOMIAL_GLM"
            or state.get("classes") != [0, 1] or any(type(c) is not int for c in state["classes"])):
        raise ValueError("unsupported binomial GLM state or research boundary")
    actual, expected = state.get("parameters"), parameters()
    if (not isinstance(actual, dict) or set(actual) != set(expected)
            or any(type(actual[k]) is not type(v) or actual[k] != v for k, v in expected.items())):
        raise ValueError("binomial GLM fixed settings differ")
    names = _names(state.get("features"))
    x, y = _paired(state.get("training_x"), state.get("training_y"), names)
    mean, scale = _array(state.get("mean"), rows=len(names)), _array(state.get("scale"), rows=len(names))
    coef = _array(state.get("coef"), rows=len(names))
    intercept = _array([state.get("intercept")], rows=1)[0]
    if (len(coef) != len(names) or len(mean) != len(names) or len(scale) != len(names)
            or type(state.get("training_rows")) is not int or state["training_rows"] != len(y)
            or type(state.get("iterations")) is not int or not 0 <= state["iterations"] < MAX_ITER
            or state.get("training_sha256") != _fingerprint(state)):
        raise ValueError("binomial GLM state dimensions, convergence or identity differ")
    actual_mean, actual_scale = _normalizer(x)
    if not np.array_equal(mean, actual_mean) or not np.array_equal(scale, actual_scale):
        raise ValueError("binomial GLM transforms must be fitted on training observations only")
    z = _scaled(x, mean, scale)
    residual = expit(z@coef+intercept)-y
    gradient = np.r_[z.T@residual+coef, residual.sum()]
    if not np.isfinite(gradient).all() or np.max(np.abs(gradient)) > 1e-4:
        raise ValueError("binomial GLM training objective is not stationary within the declared tolerance")
    return names, mean, scale, coef, intercept


def predict_binomial_glm(state, x, names):
    if _names(names) != state.get("features"):
        raise ValueError("binomial GLM feature identities/order differ")
    values = _array(x, rows=MAX_PREDICT, columns=len(names))
    with threadpool_limits(limits=1):
        _, mean, scale, coef, intercept = validate(state)
        logits = _scaled(values, mean, scale)@coef+intercept
        if not np.isfinite(logits).all() or np.max(np.abs(logits)) > 1e12:
            raise ValueError("binomial GLM linear predictor exceeds numeric bounds")
        return expit(logits), logits


def fit_binomial_glm(x, y, names):
    names = _names(names)
    values, target = _paired(x, y, names)
    try:
        import sklearn
        from sklearn.linear_model import LogisticRegression
    except ImportError:
        raise ImportError("actual sklearn LogisticRegression required; no fallback") from None
    mean, scale = _normalizer(values)
    z = _scaled(values, mean, scale)
    estimator = LogisticRegression(**parameters())
    with threadpool_limits(limits=1):
        estimator.fit(z, target)
        if estimator.n_iter_[0] >= MAX_ITER:
            raise ValueError("binomial GLM did not converge within its iteration budget")
        expected = estimator.predict_proba(z)[:, 1]
    state = {"implementation": IMPLEMENTATION, "model": "binomial_glm", "features": names,
             "training_x": values.tolist(), "training_y": target.tolist(), "mean": mean.tolist(), "scale": scale.tolist(),
             "coef": estimator.coef_[0].tolist(), "intercept": float(estimator.intercept_[0]),
             "classes": [int(c) for c in estimator.classes_], "parameters": parameters(),
             "label_rule": LABEL_RULE, "link": "LOGIT", "likelihood": "BERNOULLI",
             "penalty": "L2_COEFFICIENTS_UNPENALIZED_INTERCEPT", "probability_calibrated": False,
             "fit_state": "CONVERGED_BOUNDED_BINOMIAL_GLM", "training_rows": len(target),
             "iterations": int(estimator.n_iter_[0]), "preprocessing": "TRAINING_ONLY_POPULATION_STANDARDIZATION",
             "device": "cpu", "threads": 1, "fit_backend": "sklearn.linear_model.LogisticRegression",
             "inference_backend": IMPLEMENTATION+".json-algebra",
             "library_versions": {"numpy": np.__version__, "scipy": scipy.__version__, "scikit-learn": sklearn.__version__},
             "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "license": "BSD-3-Clause",
             "trading_approved": False, "production_promoted": False}
    state["training_sha256"] = _fingerprint(state)
    actual, _ = predict_binomial_glm(state, values, names)
    if not np.allclose(actual, expected, rtol=1e-13, atol=1e-13):
        raise ValueError("binomial GLM JSON inference differs from actual sklearn fit")
    return state
