"""Bounded actual sklearn forests with audited non-executable tree replay.

Both estimators belong to one research family. Tree disagreement is descriptive,
not a calibrated predictive interval or permission to allocate a portfolio.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
from threadpoolctl import threadpool_limits

from gig.ml.gaussian_process import _array, _names
from gig.service.serialization import dumps

IMPLEMENTATION = "gig.bounded-forest.v1"
KINDS = ("random_forest", "extra_trees")
TREES, DEPTH, MIN_LEAF = 16, 4, 3
MAX_TRAIN, MAX_FITS, MAX_PREDICT = 256, 16, 2000


def _hash(value: Any) -> str:
    return hashlib.sha256(dumps(value).encode()).hexdigest()


def _indices(value: Any, count: int, low: int, high: int) -> np.ndarray:
    if (not isinstance(value, list) or len(value) != count
            or any(type(v) is not int or not low <= v <= high for v in value)):
        raise ValueError("bounded exact tree indices required")
    return np.array(value, dtype=np.int64)


def _tree(tree: Any, dimensions: int):
    if not isinstance(tree, dict) or set(tree) != {"left", "right", "feature", "threshold", "value", "sample_indices"}:
        raise ValueError("unsupported saved tree fields")
    value = _array(tree["value"], rows=2**(DEPTH+1)-1)
    count = len(value)
    left = _indices(tree["left"], count, -1, count-1)
    right = _indices(tree["right"], count, -1, count-1)
    feature = _indices(tree["feature"], count, -2, dimensions-1)
    threshold = _array(tree["threshold"], rows=count)
    if len(threshold) != count:
        raise ValueError("saved tree dimensions differ")
    seen, stack = set(), [(0, 0)]
    while stack:
        node, depth = stack.pop()
        if node in seen or depth > DEPTH:
            raise ValueError("tree has a cycle, shared child or excessive depth")
        seen.add(node)
        if left[node] == right[node] == -1:
            if feature[node] != -2:
                raise ValueError("leaf feature sentinel is invalid")
        else:
            if min(left[node], right[node]) <= node or feature[node] < 0:
                raise ValueError("tree children must be distinct forward nodes")
            stack.extend([(int(left[node]), depth+1), (int(right[node]), depth+1)])
    if len(seen) != count:
        raise ValueError("unreachable saved tree nodes")
    return left, right, feature, threshold, value


def _leaves(parts, x):
    left, right, feature, threshold, _ = parts
    nodes = np.zeros(len(x), dtype=np.int64)
    for _ in range(DEPTH+1):
        active = np.flatnonzero(left[nodes] != -1)
        if not len(active):
            return nodes
        here = nodes[active]
        go_left = x[active, feature[here]] <= threshold[here]
        nodes[active] = np.where(go_left, left[here], right[here])
    raise ValueError("saved tree traversal exceeded the bound")


def _training_hash(state):
    return _hash({k: state[k] for k in ("features", "training_x", "training_y")})


def validate_state(state: Any):
    if (not isinstance(state, dict) or state.get("implementation") != IMPLEMENTATION
            or state.get("model") not in KINDS or state.get("device") != "cpu"
            or type(state.get("threads")) is not int or state["threads"] != 1
            or state.get("trading_approved") is not False or state.get("production_promoted") is not False
            or state.get("parameters") != {"trees": TREES, "depth": DEPTH, "min_leaf": MIN_LEAF,
                                           "criterion": "squared_error", "max_features": 1.0}
            or state.get("feature_dtype") != "float32"):
        raise ValueError("unsupported forest state or execution boundary")
    names = _names(state.get("features"))
    x = _array(state.get("training_x"), rows=MAX_TRAIN, columns=len(names)).astype(np.float32)
    y = _array(state.get("training_y"), rows=MAX_TRAIN)
    if (not 20 <= len(x) == len(y) or type(state.get("seed")) is not int or not 0 <= state["seed"] < 2**31
            or not isinstance(state.get("trees"), list) or len(state["trees"]) != TREES
            or state.get("training_data_sha256") != _training_hash(state)):
        raise ValueError("forest training identity or work bound failed")
    parsed = []
    for tree in state["trees"]:
        parts = _tree(tree, len(names))
        samples = _indices(tree["sample_indices"], len(y), 0, len(y)-1)
        if state["model"] == "extra_trees" and samples.tolist() != list(range(len(y))):
            raise ValueError("ExtraTrees must use the complete training sample")
        leaves = _leaves(parts, x)[samples]
        for node in np.unique(leaves):
            actual = float(np.mean(y[samples[leaves == node]]))
            if not np.isclose(parts[-1][node], actual, rtol=1e-10, atol=max(1e-12, 1e-12*float(np.max(np.abs(y))))):
                raise ValueError("saved leaf mean disagrees with its actual training sample")
        parsed.append(parts)
    return names, parsed


def predict_forest(state: dict[str, Any], x: Any, names: list[str]):
    if _names(names) != state.get("features"):
        raise ValueError("forest feature identities/order differ")
    values = _array(x, rows=MAX_PREDICT, columns=len(names)).astype(np.float32)
    _, trees = validate_state(state)
    per_tree = np.array([tree[-1][_leaves(tree, values)] for tree in trees])
    mean, spread = per_tree.mean(axis=0), per_tree.std(axis=0)
    if not np.isfinite(mean).all() or not np.isfinite(spread).all():
        raise ValueError("forest estimates exceed numeric range")
    return mean, spread


def fit_forest(x: Any, y: Any, names: list[str], *, model: str, seed: int = 17):
    names = _names(names)
    values = _array(x, rows=MAX_TRAIN, columns=len(names)).astype(np.float32)
    target = _array(y, rows=MAX_TRAIN)
    if (model not in KINDS or not 20 <= len(values) == len(target)
            or type(seed) is not int or not 0 <= seed < 2**31):
        raise ValueError("explicit bounded forest training required")
    try:
        import sklearn
        from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
    except ImportError:
        raise ImportError("actual scikit-learn forest library is required; no substitution") from None
    cls = RandomForestRegressor if model == "random_forest" else ExtraTreesRegressor
    estimator = cls(n_estimators=TREES, max_depth=DEPTH, min_samples_leaf=MIN_LEAF,
                    max_features=1.0, criterion="squared_error", bootstrap=model == "random_forest",
                    n_jobs=1, random_state=seed)
    with threadpool_limits(limits=1):
        estimator.fit(values, target)
        trees = [{"left": fitted.tree_.children_left.tolist(), "right": fitted.tree_.children_right.tolist(),
                  "feature": fitted.tree_.feature.tolist(), "threshold": fitted.tree_.threshold.tolist(),
                  "value": fitted.tree_.value[:, 0, 0].tolist(), "sample_indices": samples.tolist()}
                 for fitted, samples in zip(estimator.estimators_, estimator.estimators_samples_, strict=True)]
    state = {"implementation": IMPLEMENTATION, "model": model, "family": "random_forest_extratrees",
             "features": names, "feature_dtype": "float32", "preprocessing": "SKLEARN_FLOAT32_INPUT_NO_FITTED_SCALER",
             "training_x": values.tolist(), "training_y": target.tolist(), "trees": trees,
             "feature_min": values.min(axis=0).tolist(), "feature_max": values.max(axis=0).tolist(),
             "seed": seed, "device": "cpu", "threads": 1,
             "parameters": {"trees": TREES, "depth": DEPTH, "min_leaf": MIN_LEAF,
                            "criterion": "squared_error", "max_features": 1.0},
             "fit_backend": "sklearn.ensemble."+cls.__name__, "library_versions": {"numpy": np.__version__, "scikit-learn": sklearn.__version__},
             "license": "BSD-3-Clause", "trading_approved": False, "production_promoted": False,
             "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    state["training_data_sha256"] = _training_hash(state)
    replay, _ = predict_forest(state, values, names)
    with threadpool_limits(limits=1):
        if not np.allclose(replay, estimator.predict(values), rtol=1e-10, atol=1e-12):
            raise ValueError("JSON forest replay differs from actual sklearn inference")
    return state
