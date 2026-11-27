"""Families 15, 16, 19 and 35: bounded supervised regressors on admitted features.

Every fit is the named scikit-learn estimator on training-only standardized
inputs and targets. Replay is saved algebra, never pickle or a refit.
"""
from __future__ import annotations

import math
import warnings
from typing import Any

import numpy as np

from gig.ml.bounded.core import Family, register, return_score, unavailable

HUBER_EPSILON, HUBER_ALPHA, ITERATIONS = 1.35, 1e-4, 512
RANSAC_TRIALS = 64
QUANTILES = (.1, .5, .9)
ADA_TREES, ADA_DEPTH, ADA_LEAF, ADA_RATE = 16, 2, 3, .5
MLP_HIDDEN, MLP_ALPHA, MLP_ITERATIONS = 8, 1e-2, 256


def vector(payload: dict[str, Any], key: str, shape: tuple[int, ...]) -> np.ndarray:
    """Strict finite binary64 payload array of one exact shape."""
    try:
        value = np.array(payload[key], dtype=float)
    except (KeyError, TypeError, ValueError):
        raise ValueError(f"saved {key} is missing or not numeric") from None
    if value.shape != shape or not np.isfinite(value).all() or np.max(np.abs(value), initial=0.) > 1e12:
        raise ValueError(f"saved {key} shape or numeric range differs")
    return value


def _linear(payload, z):
    return (z@vector(payload, "coef", (z.shape[1],)) + vector(payload, "intercept", ()))[:, None]


def _huber_fit(z, y, _seed):
    from sklearn.linear_model import HuberRegressor
    estimator = HuberRegressor(epsilon=HUBER_EPSILON, alpha=HUBER_ALPHA, max_iter=ITERATIONS, tol=1e-8).fit(z, y)
    if estimator.n_iter_ >= ITERATIONS:
        raise ValueError("Huber regression did not converge within the fixed iteration budget")
    return ({"coef": estimator.coef_.tolist(), "intercept": float(estimator.intercept_),
             "sigma": float(estimator.scale_), "outliers": int(estimator.outliers_.sum()),
             "iterations": int(estimator.n_iter_)}, estimator.predict(z)[:, None])


def _huber_check(payload, z, y):
    """Stationarity of the declared concomitant-scale Huber objective."""
    w, c, sigma = vector(payload, "coef", (z.shape[1],)), vector(payload, "intercept", ()), vector(payload, "sigma", ())
    if sigma <= 0:
        raise ValueError("saved Huber scale must be positive")
    r = y - z@w - c
    out = np.abs(r) > HUBER_EPSILON*sigma
    gradient = np.r_[-2*z[~out].T@r[~out]/sigma - 2*HUBER_EPSILON*z[out].T@np.sign(r[out]) + 2*HUBER_ALPHA*w,
                     -2*r[~out].sum()/sigma - 2*HUBER_EPSILON*np.sign(r[out]).sum()]
    if type(payload.get("outliers")) is not int or payload["outliers"] != int(out.sum()) or np.max(np.abs(gradient)) > 1e-3*len(y):
        raise ValueError("saved Huber coefficients do not solve the declared robust objective")


def _ransac_fit(z, y, seed):
    from sklearn.linear_model import LinearRegression, RANSACRegressor
    minimum = max(z.shape[1]+1, math.ceil(len(y)/2))
    estimator = RANSACRegressor(estimator=LinearRegression(), min_samples=minimum, max_trials=RANSAC_TRIALS,
                                random_state=seed).fit(z, y)
    return ({"coef": estimator.estimator_.coef_.tolist(), "intercept": float(estimator.estimator_.intercept_),
             "inliers": np.flatnonzero(estimator.inlier_mask_).tolist(), "minimum_samples": minimum,
             "trials": int(estimator.n_trials_)}, estimator.predict(z)[:, None])


def _ransac_check(payload, z, y):
    """The final model must be ordinary least squares on its declared consensus set."""
    inliers, minimum = payload.get("inliers"), max(z.shape[1]+1, math.ceil(len(y)/2))
    if (not isinstance(inliers, list) or any(type(i) is not int for i in inliers) or inliers != sorted(set(inliers))
            or not minimum <= len(inliers) or inliers[0] < 0 or inliers[-1] >= len(y)
            or payload.get("minimum_samples") != minimum):
        raise ValueError("saved RANSAC consensus set is invalid")
    design = np.c_[z[inliers], np.ones(len(inliers))]
    solution = np.linalg.lstsq(design, y[inliers], rcond=None)[0]
    actual = np.r_[vector(payload, "coef", (z.shape[1],)), vector(payload, "intercept", ())]
    if not np.allclose(design@actual, design@solution, rtol=1e-8, atol=1e-8):
        raise ValueError("saved RANSAC coefficients are not least squares on the consensus set")


def _quantile_fit(z, y, _seed):
    from sklearn.linear_model import QuantileRegressor
    fits = [QuantileRegressor(quantile=q, alpha=0., solver="highs").fit(z, y) for q in QUANTILES]
    return ({"quantiles": list(QUANTILES), "coef": [f.coef_.tolist() for f in fits],
             "intercept": [float(f.intercept_) for f in fits]},
            np.column_stack([f.predict(z) for f in fits]))


def _quantile_replay(payload, z):
    return z@vector(payload, "coef", (3, z.shape[1])).T + vector(payload, "intercept", (3,))


def _quantile_check(payload, z, y):
    """Necessary intercept condition of every linear-quantile optimum."""
    if payload.get("quantiles") != list(QUANTILES):
        raise ValueError("saved quantile levels differ")
    residual = y[:, None] - _quantile_replay(payload, z)
    tolerance = 1e-7*(1+float(np.max(np.abs(y))))
    for index, level in enumerate(QUANTILES):
        below, at = int(np.sum(residual[:, index] < -tolerance)), int(np.sum(np.abs(residual[:, index]) <= tolerance))
        if not below-1e-9 <= len(y)*level <= below+at+1e-9:
            raise ValueError("saved quantile coefficients violate the pinball-loss optimality condition")


def pinball(observed, estimate, level) -> float:
    error = np.asarray(observed) - estimate
    return float(np.mean(np.maximum(level*error, (level-1)*error)))


def _quantile_score(observed, outputs, train_target, _payload):
    lower, median, upper = outputs["lower"], outputs["prediction"], outputs["upper"]
    result = return_score(observed, outputs, train_target, None)
    for name, level, values in zip(("lower", "median", "upper"), QUANTILES, (lower, median, upper), strict=True):
        result[f"pinball_{name}"] = pinball(observed, values, level)
        result[f"training_quantile_baseline_pinball_{name}"] = pinball(observed, np.quantile(train_target, level), level)
    result.update(crossing_count=int(np.sum((lower > median) | (median > upper))),
                  interval_coverage=float(np.mean((observed >= lower) & (observed <= upper))),
                  nominal_interval_coverage=QUANTILES[2]-QUANTILES[0], interval_calibrated=False)
    return result


def _quantile_uncertainty(outputs):
    return {"value": None, "lower": outputs["lower"], "upper": outputs["upper"],
            "state": "UNCALIBRATED_LINEAR_QUANTILE_INTERVAL", "method": "LINEAR_PINBALL_QUANTILES_10_90",
            "nominal_coverage": QUANTILES[2]-QUANTILES[0], "crossed": bool(outputs["lower"] > outputs["upper"]),
            "unit": "arithmetic_return", "calibrated": False, "predictive_interval": False}


def tree_arrays(tree: Any, columns: int, depth: int):
    """Admit one saved binary regression tree: forward children, bounded depth, no cycles."""
    if not isinstance(tree, dict) or set(tree) != {"left", "right", "feature", "threshold", "value"}:
        raise ValueError("unsupported saved tree fields")
    count = len(tree["value"]) if isinstance(tree["value"], list) else 0
    if not 1 <= count <= 2**(depth+1)-1 or any(not isinstance(tree[k], list) or len(tree[k]) != count for k in tree):
        raise ValueError("saved tree dimensions differ")
    if any(type(v) is not int for k in ("left", "right", "feature") for v in tree[k]):
        raise ValueError("saved tree indices must be exact integers")
    left, right, feature = (np.array(tree[k], dtype=np.int64) for k in ("left", "right", "feature"))
    threshold, value = vector(tree, "threshold", (count,)), vector(tree, "value", (count,))
    seen, stack = set(), [(0, 0)]
    while stack:
        node, level = stack.pop()
        if node in seen or level > depth:
            raise ValueError("saved tree has a cycle, shared child or excessive depth")
        seen.add(node)
        if left[node] == right[node] == -1:
            continue
        if not node < left[node] < count or not node < right[node] < count or left[node] == right[node] or not 0 <= feature[node] < columns:
            raise ValueError("saved tree children or split features are invalid")
        stack.extend([(int(left[node]), level+1), (int(right[node]), level+1)])
    if len(seen) != count:
        raise ValueError("unreachable saved tree nodes")
    return left, right, feature, threshold, value


def tree_values(parts, x32: np.ndarray, depth: int) -> np.ndarray:
    left, right, feature, threshold, value = parts
    nodes = np.zeros(len(x32), dtype=np.int64)
    for _ in range(depth+1):
        active = np.flatnonzero(left[nodes] != -1)
        if not len(active):
            break
        here = nodes[active]
        nodes[active] = np.where(x32[active, feature[here]] <= threshold[here], left[here], right[here])
    return value[nodes]


def export_tree(fitted) -> dict[str, Any]:
    tree = fitted.tree_
    return {"left": tree.children_left.tolist(), "right": tree.children_right.tolist(),
            "feature": [max(int(f), -2) for f in tree.feature], "threshold": tree.threshold.tolist(),
            "value": tree.value[:, 0, 0].tolist()}


def _ada_fit(z, y, seed):
    from sklearn.ensemble import AdaBoostRegressor
    from sklearn.tree import DecisionTreeRegressor
    estimator = AdaBoostRegressor(estimator=DecisionTreeRegressor(max_depth=ADA_DEPTH, min_samples_leaf=ADA_LEAF),
                                  n_estimators=ADA_TREES, learning_rate=ADA_RATE, loss="linear", random_state=seed)
    estimator.fit(z.astype(np.float32), y)
    count = len(estimator.estimators_)
    return ({"trees": [export_tree(t) for t in estimator.estimators_],
             "weights": estimator.estimator_weights_[:count].tolist(), "feature_dtype": "float32"},
            estimator.predict(z.astype(np.float32))[:, None])


def _ada_replay(payload, z):
    trees = payload.get("trees")
    if not isinstance(trees, list) or not 1 <= len(trees) <= ADA_TREES or payload.get("feature_dtype") != "float32":
        raise ValueError("saved AdaBoost ensemble exceeds the fixed tree budget")
    weights = vector(payload, "weights", (len(trees),))
    if np.any(weights < 0) or weights.sum() <= 0:
        raise ValueError("saved AdaBoost stage weights must be nonnegative with positive mass")
    x32 = z.astype(np.float32)
    stages = np.column_stack([tree_values(tree_arrays(t, z.shape[1], ADA_DEPTH), x32, ADA_DEPTH) for t in trees])
    order = np.argsort(stages, axis=1, kind="stable")
    cumulative = np.cumsum(weights[order], axis=1)
    chosen = np.argmax(cumulative >= .5*cumulative[:, -1:], axis=1)
    rows = np.arange(len(z))
    return stages[rows, order[rows, chosen]][:, None]


def _mlp_fit(z, y, seed):
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.neural_network import MLPRegressor
    estimator = MLPRegressor(hidden_layer_sizes=(MLP_HIDDEN,), activation="tanh", solver="lbfgs", alpha=MLP_ALPHA,
                             max_iter=MLP_ITERATIONS, tol=1e-8, random_state=seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        estimator.fit(z, y)
    return ({"hidden_weights": estimator.coefs_[0].tolist(), "hidden_bias": estimator.intercepts_[0].tolist(),
             "output_weights": estimator.coefs_[1][:, 0].tolist(), "output_bias": float(estimator.intercepts_[1][0]),
             "iterations": int(estimator.n_iter_), "iteration_budget_exhausted": bool(estimator.n_iter_ >= MLP_ITERATIONS)},
            estimator.predict(z)[:, None])


def _mlp_replay(payload, z):
    hidden = np.tanh(z@vector(payload, "hidden_weights", (z.shape[1], MLP_HIDDEN)) + vector(payload, "hidden_bias", (MLP_HIDDEN,)))
    return (hidden@vector(payload, "output_weights", (MLP_HIDDEN,)) + vector(payload, "output_bias", ()))[:, None]


_RESEARCH = " Uncertainty is UNAVAILABLE. Allocation, promotion and trading remain unapproved."
register(Family(
    key="huber", number=15, capability="robust_regression", label="Huber robust regression · saved dataset",
    scope="Bounded CPU Huber regression (epsilon 1.35, fixed L2 1e-4) on admitted saved data: 20–256 training rows, 16 features, 16 fits. Outlier down-weighting uses training residuals only; no hindsight filtering." + _RESEARCH,
    kind="FORWARD_RETURN_ESTIMATE", fit=_huber_fit, replay=_linear, check=_huber_check,
    backend="sklearn.linear_model.HuberRegressor",
    parameters={"epsilon": HUBER_EPSILON, "alpha": HUBER_ALPHA, "max_iter": ITERATIONS, "tol": 1e-8},
    uncertainty=unavailable("HUBER_SCALE_IS_A_TRAINING_RESIDUAL_PARAMETER_NOT_AN_INTERVAL"), tolerance=1e-9))
register(Family(
    key="ransac", number=15, capability="robust_regression", label="RANSAC consensus regression · saved dataset",
    scope="Bounded CPU RANSAC least squares on admitted saved data: 64 seeded trials, majority consensus set, training-MAD threshold, 20–256 training rows and 16 fits. Excluded training rows are recorded, not deleted from evidence." + _RESEARCH,
    kind="FORWARD_RETURN_ESTIMATE", fit=_ransac_fit, replay=_linear, check=_ransac_check,
    backend="sklearn.linear_model.RANSACRegressor",
    parameters={"max_trials": RANSAC_TRIALS, "min_samples": "MAX_FEATURES_PLUS_ONE_OR_HALF", "residual_threshold": "TRAINING_TARGET_MAD"},
    uncertainty=unavailable("CONSENSUS_SET_IS_NOT_A_PREDICTIVE_DISTRIBUTION"), tolerance=1e-9))
register(Family(
    key="quantile", number=16, capability="linear_quantile_regression", label="Linear quantile regression · 10/50/90",
    scope="Bounded CPU linear quantile regression at fixed 10%, 50% and 90% levels (HiGHS linear programs, no penalty): 20–256 training rows, 16 features, 16 folds. The estimate is a conditional median; the 10–90 band is NOT calibrated and quantiles may cross. Pinball loss, crossing and coverage are reported. Allocation, promotion and trading remain unapproved.",
    kind="FORWARD_RETURN_MEDIAN_ESTIMATE", fit=_quantile_fit, replay=_quantile_replay, check=_quantile_check,
    backend="sklearn.linear_model.QuantileRegressor", outputs=("lower", "prediction", "upper"),
    affine=("lower", "prediction", "upper"),
    parameters={"quantiles": list(QUANTILES), "alpha": 0., "solver": "highs"},
    uncertainty=_quantile_uncertainty, score=_quantile_score, tolerance=1e-9))
register(Family(
    key="adaboost", number=19, capability="adaboost", label="AdaBoost.R2 regression · saved dataset",
    scope="Bounded CPU AdaBoost.R2 on admitted saved data: 16 depth-2 trees, learning rate 0.5, linear loss, seeded training-only reweighting, 20–256 training rows and 16 fits. The weighted-median estimate has no predictive distribution." + _RESEARCH,
    kind="FORWARD_RETURN_ESTIMATE", fit=_ada_fit, replay=_ada_replay, backend="sklearn.ensemble.AdaBoostRegressor",
    parameters={"n_estimators": ADA_TREES, "max_depth": ADA_DEPTH, "min_samples_leaf": ADA_LEAF,
                "learning_rate": ADA_RATE, "loss": "linear"},
    uncertainty=unavailable("WEIGHTED_MEDIAN_OF_WEAK_LEARNERS_HAS_NO_PREDICTIVE_DISTRIBUTION"), tolerance=1e-10))
register(Family(
    key="mlp", number=35, capability="multilayer_perceptron", label="Multilayer perceptron · saved dataset",
    scope="Bounded CPU multilayer perceptron on admitted saved data: one 8-unit tanh layer, fixed L2 0.01, at most 256 L-BFGS iterations, seeded initialization, 20–256 training rows and 16 fits. An exhausted iteration budget is recorded, not hidden." + _RESEARCH,
    kind="FORWARD_RETURN_ESTIMATE", fit=_mlp_fit, replay=_mlp_replay, backend="sklearn.neural_network.MLPRegressor",
    parameters={"hidden_units": MLP_HIDDEN, "activation": "tanh", "solver": "lbfgs", "alpha": MLP_ALPHA,
                "max_iter": MLP_ITERATIONS},
    uncertainty=unavailable("POINT_NETWORK_HAS_NO_PREDICTIVE_DISTRIBUTION"), tolerance=1e-9))
