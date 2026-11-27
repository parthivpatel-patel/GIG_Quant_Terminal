"""Families 17, 18, 20 and 21: optional boosted engines with saved JSON replay.

Each fit is the named library on training-only standardized inputs and targets.
The library is never needed to predict: trees, bins and scores are exported to
JSON and replayed in NumPy, and the fit is refused unless that replay reproduces
the library's own training-row outputs. A missing library is UNAVAILABLE; no
other engine is substituted.
"""
from __future__ import annotations

import json
import math
import tempfile
import warnings
from pathlib import Path

import numpy as np

from gig.ml.bounded.core import Family, register, return_score, unavailable
from gig.ml.bounded.supervised import export_tree, tree_arrays, tree_values, vector

ROUNDS, DEPTH, RATE = 32, 2, .1
XGB = {"objective": "reg:squarederror", "max_depth": DEPTH, "eta": RATE, "min_child_weight": 3., "lambda": 1.,
       "tree_method": "exact", "base_score": 0., "nthread": 1, "verbosity": 0}
CAT = {"iterations": ROUNDS, "depth": DEPTH, "learning_rate": RATE, "l2_leaf_reg": 3., "border_count": 32,
       "boosting_type": "Ordered", "bootstrap_type": "No"}
NGB_RATE, NGB_LEAF, LOG_SCALE_LIMIT, INTERVAL_Z = .05, 3, 20., 1.6448536269514722
EBM = {"interactions": 2, "max_bins": 32, "max_interaction_bins": 8, "outer_bags": 1, "inner_bags": 0, "max_rounds": 200,
       "learning_rate": .05, "validation_size": .15, "early_stopping_rounds": 20, "min_samples_leaf": 3}


def _trees(payload, limit):
    trees = payload.get("trees")
    if not isinstance(trees, list) or not 1 <= len(trees) <= limit or payload.get("feature_dtype") != "float32":
        raise ValueError("saved boosted ensemble exceeds the fixed tree budget")
    return trees


def _xgb_fit(z, y, seed):
    import xgboost
    x32 = z.astype(np.float32)
    booster = xgboost.train({**XGB, "seed": seed}, xgboost.DMatrix(x32, label=y), num_boost_round=ROUNDS)
    model = json.loads(booster.save_raw("json"))["learner"]["gradient_booster"]["model"]
    trees = []
    for tree in model["trees"]:
        leaf = [child == -1 for child in tree["left_children"]]
        trees.append({"left": tree["left_children"], "right": tree["right_children"],
                      "feature": [-2 if end else int(f) for f, end in zip(tree["split_indices"], leaf, strict=True)],
                      "threshold": [0. if end else float(c) for c, end in zip(tree["split_conditions"], leaf, strict=True)],
                      "value": [float(c) if end else 0. for c, end in zip(tree["split_conditions"], leaf, strict=True)]})
    return {"trees": trees, "feature_dtype": "float32"}, booster.predict(xgboost.DMatrix(x32))[:, None]


def _xgb_replay(payload, z):
    x32, total = z.astype(np.float32), np.zeros(len(z), dtype=np.float32)
    for tree in _trees(payload, ROUNDS):
        # XGBoost sends a row left when its float32 feature is strictly below the split.
        total += tree_values(tree_arrays(tree, z.shape[1], DEPTH), x32, DEPTH, strict=True).astype(np.float32)
    return total.astype(float)[:, None]


def _cat_fit(z, y, seed):
    import catboost
    x32 = z.astype(np.float32)
    estimator = catboost.CatBoostRegressor(**CAT, random_seed=seed, thread_count=1, allow_writing_files=False, verbose=False)
    estimator.fit(x32, y)
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory)/"model.json"
        estimator.save_model(str(path), format="json")
        model = json.loads(path.read_text())
    scale, bias = model["scale_and_bias"]
    trees = []
    for tree in model["oblivious_trees"]:
        splits = tree.get("splits", [])
        if any(split["split_type"] != "FloatFeature" for split in splits):
            raise ValueError("CatBoost produced a non-numeric split; categorical statistics are not admitted")
        trees.append({"features": [int(split["float_feature_index"]) for split in splits],
                      "borders": [float(split["border"]) for split in splits],
                      "leaves": [float(v) for v in tree["leaf_values"]]})
    return ({"trees": trees, "scale": float(scale), "bias": float(bias[0]), "feature_dtype": "float32"},
            estimator.predict(x32)[:, None])


def _cat_replay(payload, z):
    x32, total = z.astype(np.float32), np.zeros(len(z))
    for tree in _trees(payload, ROUNDS):
        if not isinstance(tree, dict) or set(tree) != {"features", "borders", "leaves"}:
            raise ValueError("unsupported saved oblivious tree fields")
        features = tree["features"]
        if (not isinstance(features, list) or len(features) > DEPTH
                or any(type(f) is not int or not 0 <= f < z.shape[1] for f in features)):
            raise ValueError("saved oblivious tree splits are invalid")
        borders, leaves = vector(tree, "borders", (len(features),)), vector(tree, "leaves", (2**len(features),))
        index = np.zeros(len(z), dtype=np.int64)
        for bit, (feature, border) in enumerate(zip(features, borders, strict=True)):
            index |= (x32[:, feature] > np.float32(border)).astype(np.int64) << bit
        total += leaves[index]
    return (total*vector(payload, "scale", ()) + vector(payload, "bias", ()))[:, None]


def _ngb_fit(z, y, seed):
    x32 = z.astype(np.float32)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from ngboost import NGBRegressor
        from ngboost.distns import Normal
        from sklearn.tree import DecisionTreeRegressor
        base = DecisionTreeRegressor(criterion="squared_error", max_depth=DEPTH, min_samples_leaf=NGB_LEAF)
        estimator = NGBRegressor(Dist=Normal, Base=base, n_estimators=ROUNDS, learning_rate=NGB_RATE, minibatch_frac=1.,
                                 col_sample=1., natural_gradient=True, random_state=seed, verbose=False).fit(x32, y)
        fitted = estimator.pred_dist(x32)
    stages = [{"location": export_tree(models[0]), "log_scale": export_tree(models[1]), "step": float(step)}
              for models, step in zip(estimator.base_models, estimator.scalings, strict=True)]
    return ({"initial": [float(v) for v in estimator.init_params], "stages": stages, "feature_dtype": "float32"},
            np.column_stack([fitted.loc, fitted.scale]))


def _ngb_replay(payload, z):
    stages = payload.get("stages")
    if not isinstance(stages, list) or not 1 <= len(stages) <= ROUNDS or payload.get("feature_dtype") != "float32":
        raise ValueError("saved natural-gradient ensemble exceeds the fixed stage budget")
    x32 = z.astype(np.float32)
    parameters = np.tile(vector(payload, "initial", (2,)), (len(z), 1))
    for stage in stages:
        if not isinstance(stage, dict) or set(stage) != {"location", "log_scale", "step"}:
            raise ValueError("unsupported saved natural-gradient stage fields")
        step = vector(stage, "step", ())
        if step < 0:
            raise ValueError("saved natural-gradient step must be nonnegative")
        update = np.column_stack([tree_values(tree_arrays(stage[name], z.shape[1], DEPTH), x32, DEPTH)
                                  for name in ("location", "log_scale")])
        parameters -= NGB_RATE*step*update
    if np.max(np.abs(parameters[:, 1]), initial=0.) > LOG_SCALE_LIMIT:
        raise ValueError("saved natural-gradient scale is outside the supported numeric range")
    return np.column_stack([parameters[:, 0], np.exp(parameters[:, 1])])


def _normal_scores(observed, mean, sd):
    from scipy.special import ndtr
    standard = (observed-mean)/sd
    density = np.exp(-.5*standard**2)/math.sqrt(2*math.pi)
    return (float(np.mean(.5*math.log(2*math.pi) + np.log(sd) + .5*standard**2)),
            float(np.mean(sd*(standard*(2*ndtr(standard)-1) + 2*density - 1/math.sqrt(math.pi)))))


def _ngb_score(observed, outputs, train_target, _payload):
    """Proper scores against the unconditional training Normal; lower is better."""
    mean, sd = outputs["prediction"], outputs["sd"]
    result = return_score(observed, outputs, train_target, None)
    result["gaussian_nll"], result["crps"] = _normal_scores(observed, mean, sd)
    baseline = np.std(train_target)
    if baseline > 0:
        result["training_normal_baseline_nll"], result["training_normal_baseline_crps"] = _normal_scores(
            observed, np.mean(train_target), baseline)
    result.update(interval_coverage=float(np.mean(np.abs(observed-mean) <= INTERVAL_Z*sd)),
                  nominal_interval_coverage=.9, interval_calibrated=False)
    return result


def _ngb_uncertainty(outputs):
    return {"value": outputs["sd"], "state": "UNCALIBRATED_MODEL_NORMAL_STANDARD_DEVIATION",
            "method": "NGBOOST_NORMAL_NATURAL_GRADIENT", "unit": "arithmetic_return", "calibrated": False,
            "predictive_interval": False}


def _ebm_fit(z, y, seed):
    from interpret.glassbox import ExplainableBoostingRegressor
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        estimator = ExplainableBoostingRegressor(**EBM, n_jobs=1, random_state=seed).fit(z, y)
        expected = estimator.predict(z)
    terms = []
    for features, scores in zip(estimator.term_features_, estimator.term_scores_, strict=True):
        cuts = []
        for feature in features:
            levels = estimator.bins_[feature]
            # Pair terms use the coarser interaction binning when one exists.
            level = levels[min(len(levels), len(features))-1]
            if isinstance(level, dict):
                raise ValueError("EBM produced a categorical bin; only continuous features are admitted")
            cuts.append(np.asarray(level, dtype=float).tolist())
        terms.append({"features": [int(f) for f in features], "cuts": cuts, "scores": np.asarray(scores, dtype=float).tolist()})
    return {"intercept": float(estimator.intercept_), "terms": terms}, expected[:, None]


def _ebm_replay(payload, z):
    terms = payload.get("terms")
    if not isinstance(terms, list) or not 1 <= len(terms) <= z.shape[1]+EBM["interactions"]:
        raise ValueError("saved additive terms exceed the fixed main-effect and pair budget")
    total = np.full(len(z), vector(payload, "intercept", ()))
    seen = set()
    for term in terms:
        if not isinstance(term, dict) or set(term) != {"features", "cuts", "scores"}:
            raise ValueError("unsupported saved additive term fields")
        features, cuts = term["features"], term["cuts"]
        if (not isinstance(features, list) or len(features) not in (1, 2) or tuple(features) in seen
                or any(type(f) is not int or not 0 <= f < z.shape[1] for f in features) or len(set(features)) != len(features)
                or not isinstance(cuts, list) or len(cuts) != len(features)
                or any(not isinstance(c, list) or len(c) > EBM["max_bins"] for c in cuts)):
            raise ValueError("saved additive term features or bins are invalid")
        seen.add(tuple(features))
        edges = [vector({"cuts": c}, "cuts", (len(c),)) for c in cuts]
        if any(np.any(np.diff(edge) <= 0) for edge in edges):
            raise ValueError("saved additive bin cuts must increase strictly")
        # Bin 0 is reserved for missing values and the last bin for unseen categories; neither is reachable here.
        scores = vector(term, "scores", tuple(len(edge)+3 for edge in edges))
        total += scores[tuple(np.digitize(z[:, f], edge)+1 for f, edge in zip(features, edges, strict=True))]
    return total[:, None]


_RESEARCH = " Uncertainty is UNAVAILABLE. Allocation, promotion and trading remain unapproved."
register(Family(
    key="xgboost", number=17, capability="xgboost", label="XGBoost exact trees · saved dataset",
    scope="Bounded CPU XGBoost on admitted saved data: 32 depth-2 trees by the exact greedy method, learning rate 0.1, L2 1, minimum child weight 3, squared error, one thread, 20–256 training rows and 16 fits. Trees are exported to JSON and replayed without the library; the CPU-only package is used, never a GPU build." + _RESEARCH,
    kind="FORWARD_RETURN_ESTIMATE", fit=_xgb_fit, replay=_xgb_replay, backend="xgboost.train", requires=("xgboost",),
    parameters={"num_boost_round": ROUNDS, **{k: v for k, v in XGB.items() if k not in {"nthread", "verbosity"}}},
    uncertainty=unavailable("BOOSTED_POINT_ESTIMATE_HAS_NO_PREDICTIVE_DISTRIBUTION"), tolerance=1e-6, license="Apache-2.0"))
register(Family(
    key="catboost", number=18, capability="catboost", label="CatBoost ordered boosting · saved dataset",
    scope="Bounded CPU CatBoost on admitted saved data: 32 depth-2 oblivious trees with ordered boosting, learning rate 0.1, L2 3, 32 borders, no bootstrap, one thread, 20–256 training rows and 16 fits. Only numeric features are admitted, so no categorical target statistics are computed. Trees are exported to JSON and replayed without the library." + _RESEARCH,
    kind="FORWARD_RETURN_ESTIMATE", fit=_cat_fit, replay=_cat_replay, backend="catboost.CatBoostRegressor",
    requires=("catboost",), parameters=dict(CAT),
    uncertainty=unavailable("BOOSTED_POINT_ESTIMATE_HAS_NO_PREDICTIVE_DISTRIBUTION"), tolerance=1e-9, license="Apache-2.0"))
register(Family(
    key="ngboost", number=20, capability="natural_gradient_boosting", label="NGBoost Normal distribution · saved dataset",
    scope="Bounded CPU natural-gradient boosting of a Normal distribution on admitted saved data: 32 stages of depth-2 trees for location and log scale, learning rate 0.05, full batches, 20–256 training rows and 16 fits. The standard deviation is a model output, NOT a calibrated interval; Gaussian log score, CRPS and 90% coverage are reported against the unconditional training Normal. Allocation, promotion and trading remain unapproved.",
    kind="FORWARD_RETURN_DISTRIBUTION_ESTIMATE", fit=_ngb_fit, replay=_ngb_replay, backend="ngboost.NGBRegressor",
    requires=("ngboost", "sklearn"), outputs=("prediction", "sd"), scaled=("sd",),
    parameters={"distribution": "Normal", "n_estimators": ROUNDS, "learning_rate": NGB_RATE, "max_depth": DEPTH,
                "min_samples_leaf": NGB_LEAF, "minibatch_frac": 1., "col_sample": 1., "natural_gradient": True},
    uncertainty=_ngb_uncertainty, score=_ngb_score, tolerance=1e-9, license="Apache-2.0"))
register(Family(
    key="ebm", number=21, capability="explainable_boosting_machine", label="Explainable boosting machine · saved dataset",
    scope="Bounded CPU explainable boosting machine on admitted saved data: additive per-feature shape functions on at most 32 bins plus at most two pair terms chosen on the training fold only, one bag, at most 200 rounds with a seeded 15% training-fold stopping split, 20–256 training rows and 16 fits. Bins and scores are exported to JSON and replayed without the library; shape functions describe the fit, not causes." + _RESEARCH,
    kind="FORWARD_RETURN_ESTIMATE", fit=_ebm_fit, replay=_ebm_replay,
    backend="interpret.glassbox.ExplainableBoostingRegressor", requires=("interpret",), parameters=dict(EBM),
    uncertainty=unavailable("ADDITIVE_POINT_ESTIMATE_HAS_NO_PREDICTIVE_DISTRIBUTION"), tolerance=1e-9, license="MIT"))
