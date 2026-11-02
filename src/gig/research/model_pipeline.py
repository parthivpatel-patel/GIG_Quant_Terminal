"""Purged model adapters for admitted point-in-time dataset documents.

Existing numerical kernels remain canonical. This module binds their inputs,
training cutoffs, output meaning and saved-state inference to persisted evidence.
"""

from __future__ import annotations

import hashlib
import math
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from gig.ml.ranker import _group_sizes, _relevance_labels
from gig.ml.regimes import GaussianHMM, fit_gaussian_hmm
from gig.ml.sequence import predict_sequence, train_sequence
from gig.ml.state_space import kalman_filter
from gig.ml.volatility import GARCH11, fit_garch11
from gig.research.walkforward import WalkForwardSplit
from gig.research.workflow import ResearchWorkflow
from gig.service.serialization import dumps, loads
from gig.timestamps import aware_timestamp

IMPLEMENTATION = "gig.persisted-models.v1"
KINDS = {
    "ridge": "FORWARD_RETURN_ESTIMATE",
    "lightgbm": "CROSS_SECTIONAL_RANK_SCORE",
    "bilstm": "FORWARD_RETURN_ESTIMATE",
    "hmm": "NEXT_OBSERVATION_RETURN_ESTIMATE",
    "garch": "NEXT_OBSERVATION_VARIANCE",
    "kalman": "FILTERED_RETURN_STATE",
}


def identity(value: Any) -> str:
    return hashlib.sha256(dumps(value).encode()).hexdigest()


def sha256(value: Any) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError("identity requires lowercase SHA-256")
    return value


def integer(value: Any, low: int, high: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
        raise ValueError(f"invalid bounded {name}")
    return value


def configuration(payload: dict[str, Any]) -> dict[str, Any]:
    allowed = {"dataset_id", "model", "min_train", "test_size", "embargo", "seed",
               "sequence_length", "hidden_size", "epochs", "states", "iterations"}
    if not isinstance(payload, dict) or set(payload) - allowed or not {"dataset_id", "model"} <= payload.keys():
        raise ValueError("unknown or missing persisted research fields")
    sha256(payload["dataset_id"])
    model = payload["model"]
    if not isinstance(model, str) or model not in KINDS:
        raise ValueError("explicit supported CPU model required")
    options = {"sequence_length", "hidden_size", "epochs"} if model == "bilstm" else {"states", "iterations"} if model == "hmm" else {"iterations"} if model == "garch" else set()
    if set(payload) & ({"sequence_length", "hidden_size", "epochs", "states", "iterations"} - options):
        raise ValueError("settings do not belong to selected model")
    config = {"model": model,
              "min_train": integer(payload.get("min_train", 40), 20, 800, "min_train"),
              "test_size": integer(payload.get("test_size", 10), 1, 200, "test_size"),
              "embargo": integer(payload.get("embargo", 1), 0, 100, "embargo"),
              "seed": integer(payload.get("seed", 42), 0, 2**31 - 1, "seed")}
    if model == "bilstm":
        config.update(sequence_length=integer(payload.get("sequence_length", 4), 2, 16, "sequence_length"),
                      hidden_size=integer(payload.get("hidden_size", 4), 1, 16, "hidden_size"),
                      epochs=integer(payload.get("epochs", 4), 1, 20, "epochs"))
    if model == "hmm":
        config.update(states=integer(payload.get("states", 2), 1, 4, "states"),
                      iterations=integer(payload.get("iterations", 50), 1, 200, "iterations"))
    if model == "garch":
        config["iterations"] = integer(payload.get("iterations", 200), 1, 1000, "iterations")
    return config


def dataset_rows(dataset: dict[str, Any]) -> tuple[list[dict[str, Any]], list[str]]:
    rows = dataset.get("rows")
    if not isinstance(rows, list) or not 24 <= len(rows) <= 2000:
        raise ValueError("model dataset requires 24..2000 admitted feature rows")
    rows = loads(dumps(rows))
    names = sorted(rows[0]["features"])
    if not 1 <= len(names) <= 16:
        raise ValueError("unsupported feature dimensions")
    seen = set()
    for row in rows:
        key = (row["instrument_id"], aware_timestamp(row["timestamp"]))
        if key in seen:
            raise ValueError("duplicate model decision identity")
        seen.add(key)
        if sorted(row["features"]) != names:
            raise ValueError("inconsistent model feature identity")
        numbers = [*row["features"].values(), row["target"]]
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in numbers):
            raise ValueError("finite admitted model features and labels required")
        if aware_timestamp(row["available_at"]) > key[1]:
            raise ValueError("model feature is unavailable at decision")
        if not key[1] <= aware_timestamp(row["label_end"]) <= aware_timestamp(dataset["as_of"]):
            raise ValueError("model label lies outside research knowledge cutoff")
    rows.sort(key=lambda r: (aware_timestamp(r["timestamp"]), r["instrument_id"]))
    if len({r["instrument_id"] for r in rows}) > 8:
        raise ValueError("model dataset exceeds eight instruments")
    return rows, names


def _wrapper(dataset, model, instrument, estimator, cutoff, max_label, **extra):
    return {"implementation": IMPLEMENTATION, "model": model, "instrument_id": instrument,
            "source": dataset["source"], "input_classification": dataset["classification"],
            "training_cutoff": cutoff, "max_training_label_end": max_label,
            "training_scope": "RETROSPECTIVE_PURGED_RESEARCH_NOT_HISTORICALLY_PUBLISHED_MODEL",
            "estimator": estimator, "trading_approved": False, **extra}


def _prediction(row, value, model_id, fold, model, uncertainty, **extra):
    if not math.isfinite(float(value)):
        raise ValueError("nonfinite research model output")
    return {"instrument_id": row["instrument_id"], "timestamp": row["timestamp"],
            "prediction": float(value), "target": row["target"], "label_end": row["label_end"],
            "fold": fold, "model_id": model_id, "prediction_kind": KINDS[model],
            "uncertainty": uncertainty, "usable_for_portfolio": model not in {"garch", "hmm"},
            "trading_approved": False, **extra}


def _folds(rows, config):
    index = pd.DatetimeIndex([aware_timestamp(r["timestamp"]) for r in rows])
    output = list(WalkForwardSplit(config["min_train"], config["test_size"], config["embargo"]).split(index))
    if not output or len(output) > 16:
        raise ValueError("dataset must admit one to 16 complete walk-forward folds per instrument")
    return output


def _training_vintage(row, cutoff):
    """Use the outcome revision actually knowable before this fitted model."""
    candidates = row.get("label_vintages", [{"target": row["target"],
        "label_end": row["label_end"], "knowledge_at": row["label_end"]}])
    visible = [v for v in candidates if aware_timestamp(v["knowledge_at"]) < cutoff
               and aware_timestamp(v["label_end"]) < cutoff]
    if not visible:
        return None
    latest = max(visible, key=lambda v: (aware_timestamp(v["knowledge_at"]), aware_timestamp(v["label_end"])))
    # Full future revision histories must not enter a training-data identity.
    result = {k: v for k, v in row.items() if k != "label_vintages"}
    result.update({k: v for k, v in latest.items() if k != "knowledge_at"})
    return result


def _ridge(dataset, rows, names, config):
    models, predictions, folds = {}, [], []
    groups = defaultdict(list)
    for row in rows:
        groups[row["instrument_id"]].append(row)
    for instrument, own in sorted(groups.items()):
        for number, schedule in enumerate(_folds(own, config)):
            candidates = [r for r in own if aware_timestamp(r["timestamp"]) in schedule.train_index]
            train = [v for r in candidates if (v := _training_vintage(r, schedule.test_start)) is not None]
            test = [r for r in own if aware_timestamp(r["timestamp"]) in schedule.test_index]
            if len(train) < 20:
                raise ValueError("ridge requires 20 purged known outcome revisions")
            combined = train + test
            index = pd.DatetimeIndex([aware_timestamp(r["timestamp"]) for r in combined])
            frame = pd.DataFrame([[r["features"][n] for n in names] for r in combined], index=index, columns=names)
            with tempfile.TemporaryDirectory(prefix="gig-persisted-fit-") as temporary:
                workflow = ResearchWorkflow(Path(temporary))
                result = workflow.run(frame, pd.Series([r["target"] for r in combined], index=index),
                                      label_end=pd.Series([r["label_end"] for r in combined], index=index),
                                      feature_available_at=pd.Series([r["available_at"] for r in combined], index=index),
                                      model="ridge", min_train=len(train), test_size=len(test),
                                      embargo=0, seed=config["seed"])
                by_time = {r["timestamp"]: r for r in test}
                fold = result["folds"][0]
                old_id = fold["model_id"]
                wrapped = _wrapper(dataset, "ridge", instrument, workflow.registry.load(old_id),
                                   fold["test_start"], fold["max_training_label_end"], fit_state="CONVERGED")
                model_id = identity(wrapped)
                models[model_id] = wrapped
                folds.append({**fold, "fold": number, "instrument_id": instrument, "model_id": model_id,
                              "purged_count": len(candidates)-len(train)})
                uncertainty = {"value": fold["training_residual_abs_q90"], "state": "NOT_OOS_CALIBRATED",
                               "method": "TRAINING_ABSOLUTE_RESIDUAL_Q90", "unit": "arithmetic_return"}
                for observed in result["predictions"]:
                    if observed["model_id"] == old_id:
                        predictions.append(_prediction(by_time[observed["timestamp"]], observed["prediction"],
                                                       model_id, number, "ridge", uncertainty))
    return models, predictions, folds


def _ranking(dataset, rows, names, config):
    try:
        import lightgbm as lgb
    except ImportError:
        raise ImportError("Actual LightGBM ranking library is required") from None
    dates = pd.DatetimeIndex(sorted({aware_timestamp(r["timestamp"]) for r in rows}))
    split = list(WalkForwardSplit(config["min_train"], config["test_size"], config["embargo"]).split(dates))
    if not split or len(split) > 16:
        raise ValueError("ranking requires one to16 complete date-grouped folds")
    models, predictions, folds = {}, [], []
    for number, fold in enumerate(split):
        candidates = [r for r in rows if aware_timestamp(r["timestamp"]) in fold.train_index]
        train = [v for r in candidates if (v := _training_vintage(r, fold.test_start)) is not None]
        test = [r for r in rows if aware_timestamp(r["timestamp"]) in fold.test_index]
        if len(train) < 20:
            raise ValueError("ranking has too few purged training observations")
        index = pd.MultiIndex.from_tuples([(r["timestamp"], r["instrument_id"]) for r in train])
        groups = _group_sizes(index)
        if min(groups) < 2:
            raise ValueError("ranking requires at least two known instruments per training date")
        x = np.array([[r["features"][n] for n in names] for r in train])
        mean, scale = x.mean(axis=0), x.std(axis=0)
        scale = np.where(scale > 0, scale, 1.)
        if not np.isfinite(mean).all() or not np.isfinite(scale).all():
            raise ValueError("ranking normalization overflow")
        scaled = (x-mean)/scale
        labels = _relevance_labels(pd.Series([r["target"] for r in train], index=index))
        estimator = lgb.train({"objective": "lambdarank", "metric": "ndcg", "learning_rate": .05,
                               "num_leaves": 7, "min_data_in_leaf": 3, "verbosity": -1,
                               "seed": config["seed"], "num_threads": 1, "deterministic": True,
                               "force_col_wise": True, "label_gain": list(range(5))},
                              lgb.Dataset(scaled, label=labels, group=groups, feature_name=names),
                              num_boost_round=32)
        state = {"model": "lightgbm", "objective": "lambdarank", "features": names,
                 "mean": mean.tolist(), "scale": scale.tolist(),
                 "booster_text": estimator.model_to_string(), "library_version": lgb.__version__,
                 "training_data_hash": identity(train), "group_sizes": groups, "seed": config["seed"]}
        maximum = max(aware_timestamp(r["label_end"]) for r in train).isoformat()
        wrapped = _wrapper(dataset, "lightgbm", "*", state, fold.test_start.isoformat(), maximum,
                           fit_state="CONVERGED")
        model_id = identity(wrapped)
        models[model_id] = wrapped
        values = estimator.predict((np.array([[r["features"][n] for n in names] for r in test])-mean)/scale)
        uncertainty = {"value": None, "state": "UNAVAILABLE", "method": "RANK_SCORE_HAS_NO_CALIBRATED_RETURN_INTERVAL"}
        predictions.extend(_prediction(row, value, model_id, number, "lightgbm", uncertainty)
                           for row, value in zip(test, values, strict=True))
        folds.append({"fold": number, "instrument_id": "*", "model_id": model_id,
                      "train_count": len(train), "test_count": len(test),
                      "purged_count": len(candidates)-len(train),
                      "test_start": fold.test_start.isoformat(), "test_end": fold.test_end.isoformat(),
                      "max_training_label_end": maximum, "query_groups": len(groups),
                      "objective": "lambdarank", "uncertainty_state": "UNAVAILABLE"})
    return models, predictions, folds


def _sequence(dataset, rows, config):
    models, predictions, folds = {}, [], []
    groups = defaultdict(list)
    for row in rows:
        groups[row["instrument_id"]].append(row)
    for instrument, own in sorted(groups.items()):
        for number, fold in enumerate(_folds(own, config)):
            train = [_training_vintage(r, fold.test_start) or r for r in own
                     if aware_timestamp(r["timestamp"]) in fold.train_index]
            test = [r for r in own if aware_timestamp(r["timestamp"]) in fold.test_index]
            combined = train+test
            fields = ("timestamp", "available_at", "features", "target", "label_end")
            result = train_sequence({"rows": [{k: r[k] for k in fields} for r in combined],
                                     "source": dataset["source"], "classification": dataset["classification"],
                                     "as_of": dataset["as_of"], "train_size": len(train),
                                     **{k: config[k] for k in ("sequence_length", "hidden_size", "epochs", "seed")}})
            excluded = [r["timestamp"] for r in own
                        if fold.train_end < aware_timestamp(r["timestamp"]) < fold.test_start]
            wrapped = _wrapper(dataset, "bilstm", instrument, result["model_state"],
                               fold.test_start.isoformat(), result["model_state"]["max_training_label_end"],
                               fit_state="CONVERGED", excluded_window_timestamps=excluded)
            model_id = identity(wrapped)
            models[model_id] = wrapped
            by_time = {r["timestamp"]: r for r in test}
            uncertainty = {"value": None, "state": "UNAVAILABLE", "method": "NO_CALIBRATED_PREDICTIVE_INTERVAL"}
            predictions.extend(_prediction(by_time[r["timestamp"]], r["prediction"], model_id,
                                           number, "bilstm", uncertainty) for r in result["predictions"])
            folds.append({"fold": number, "instrument_id": instrument, "model_id": model_id,
                          "test_start": fold.test_start.isoformat(), "test_end": fold.test_end.isoformat(),
                          "max_training_label_end": wrapped["max_training_label_end"],
                          **result["metrics"], "window_policy": "EMBARGO_OBSERVATIONS_EXCLUDED",
                          "training_loss": result["training_loss"]})
    return models, predictions, folds


def _state_values(state, values):
    kind = state["model"]
    estimator = state["estimator"]
    x = np.array(values, dtype=float)
    if kind == "hmm":
        model = GaussianHMM(*(estimator[k] for k in ("initial", "transition", "means", "variances")))
        filtered = model.infer(x).filtered_probabilities
        predictions = filtered @ model.transition @ model.means[:, 0]
        uncertainty = [{"state": "MODEL_PROBABILITY_NOT_CALIBRATED_CONFIDENCE",
                        "filtered_probabilities": probabilities.tolist(),
                        "entropy": float(-sum(p*math.log(p) for p in probabilities if p > 0))}
                       for probabilities in filtered]
    elif kind == "garch":
        model = GARCH11(**{k: estimator[k] for k in ("omega", "alpha", "beta", "mean")})
        predictions = np.array([model.forecast(x[:i+1], 1)[0] for i in range(len(x))])
        uncertainty = [{"value": float(value), "state": "MODEL_VARIANCE_NOT_CALIBRATED_INTERVAL",
                        "unit": "squared_return_per_admitted_observation"} for value in predictions]
    else:
        variance = estimator["training_variance"]
        result = kalman_filter(x[:, None], [[1.]], [[1.]], [[variance*.01]], [[variance]],
                               [estimator["training_mean"]], [[variance]])
        predictions = result.filtered_mean[:, 0]
        uncertainty = [{"value": float(math.sqrt(v+variance*.01)),
                        "state": "LATENT_STATE_STD_NOT_CALIBRATED_RETURN_INTERVAL",
                        "unit": "arithmetic_return"} for v in result.filtered_covariance[:, 0, 0]]
    return predictions, uncertainty


def _state_models(dataset, rows, config):
    if dataset.get("request", dataset.get("canonical_request", {})).get("horizon", 1) != 1:
        raise ValueError("one-observation state models require a one-observation target horizon")
    panel = {(r["instrument_id"], r["timestamp"]): r for r in dataset["panel_rows"]}
    groups = defaultdict(list)
    for row in rows:
        value = panel.get((row["instrument_id"], row["timestamp"]), {}).get("return_1")
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError("state model requires an admitted causal one-observation return")
        groups[row["instrument_id"]].append({**row, "return_1": float(value)})
    models, predictions, folds = {}, [], []
    kind = config["model"]
    for instrument, own in sorted(groups.items()):
        for number, fold in enumerate(_folds(own, config)):
            candidates = [r for r in own if aware_timestamp(r["timestamp"]) in fold.train_index]
            train = [v for r in candidates if (v := _training_vintage(r, fold.test_start)) is not None]
            if len(train) < 20:
                raise ValueError("state model requires 20 purged training observations")
            values = np.array([r["return_1"] for r in train])
            fitted: dict[str, Any]
            fit_state = "CONVERGED"
            if kind == "hmm":
                fit = fit_gaussian_hmm(values, states=config["states"], max_iterations=config["iterations"])
                fitted = {k: getattr(fit.model, k).tolist() for k in ("initial", "transition", "means", "variances")}
                fitted.update(converged=fit.converged, stop_reason=fit.stop_reason, iterations=fit.iterations)
                fit_state = "CONVERGED" if fit.converged else "NOT_CONVERGED"
            elif kind == "garch":
                fit = fit_garch11(values, max_iterations=config["iterations"])
                fitted = {k: getattr(fit.model, k) for k in ("omega", "alpha", "beta", "mean")}
                fitted.update(converged=fit.converged, stop_reason=fit.stop_reason, iterations=fit.iterations)
                fit_state = "CONVERGED" if fit.converged else "NOT_CONVERGED"
            else:
                variance = float(values.var())
                if not math.isfinite(variance) or variance <= 0:
                    raise ValueError("Kalman training requires positive finite return variance")
                fitted = {"training_variance": variance, "training_mean": float(values.mean()),
                          "parameter_method": "TRAINING_VARIANCE_HEURISTIC_NOT_MLE"}
            wrapped = _wrapper(dataset, kind, instrument, fitted, fold.test_start.isoformat(),
                               max(aware_timestamp(r["label_end"]) for r in train).isoformat(),
                               fit_state=fit_state, training_returns=[r["return_1"] for r in train],
                               training_timestamps=[r["timestamp"] for r in train])
            model_id = identity(wrapped)
            models[model_id] = wrapped
            # Filtering through holdout uses prefix-causal observations; no smoothing is emitted.
            values_all = [r["return_1"] for r in own]
            values_pred, uncertainty = _state_values(wrapped, values_all)
            for i, row in enumerate(own):
                if aware_timestamp(row["timestamp"]) in fold.test_index:
                    predictions.append(_prediction(row, values_pred[i], model_id, number, kind,
                                                   uncertainty[i], fit_state=fit_state,
                                                   usable_for_portfolio=kind not in {"garch", "hmm"} and fit_state == "CONVERGED"))
            folds.append({"fold": number, "instrument_id": instrument, "model_id": model_id,
                          "train_count": len(train), "purged_count": len(candidates)-len(train),
                          "test_count": len(fold.test_index), "test_start": fold.test_start.isoformat(),
                          "test_end": fold.test_end.isoformat(), "max_training_label_end": wrapped["max_training_label_end"],
                          "fit_state": fit_state})
    return models, predictions, folds


def fit_dataset(dataset: dict[str, Any], config: dict[str, Any]):
    rows, names = dataset_rows(dataset)
    if config["model"] == "ridge":
        models, predictions, folds = _ridge(dataset, rows, names, config)
    elif config["model"] == "lightgbm":
        models, predictions, folds = _ranking(dataset, rows, names, config)
    elif config["model"] == "bilstm":
        models, predictions, folds = _sequence(dataset, rows, config)
    else:
        models, predictions, folds = _state_models(dataset, rows, config)
    if len(models) > 64:
        raise ValueError("research exceeds 64 fitted model states")
    predictions.sort(key=lambda r: (aware_timestamp(r["timestamp"]), r["instrument_id"]))
    errors = [(r["prediction"]-r["target"])**2 for r in predictions]
    metric = float(np.mean(errors)) if config["model"] not in {"lightgbm", "garch"} else None
    if metric is not None and not math.isfinite(metric):
        raise ValueError("research metric exceeds finite range")
    latest = {}
    for row in predictions:
        latest[row["instrument_id"]] = row
    artifact = {"schema_version": IMPLEMENTATION, "dataset_id": dataset["dataset_id"],
                "model": config["model"],
                "source": dataset["source"], "input_classification": dataset["classification"],
                "as_of": dataset["as_of"], "configuration": config,
                "prediction_kind": KINDS[config["model"]], "rows": predictions,
                "model_ids": sorted(models), "folds": folds,
                "latest_scores": [latest[k] for k in sorted(latest)],
                "metrics": {"oos_count": len(predictions), "fold_count": len(folds),
                            "model_count": len(models), "mse": metric,
                            "mse_state": "AVAILABLE" if metric is not None else "UNAVAILABLE_DIFFERENT_OUTPUT_UNITS"},
                "dataset_diagnostics": dataset.get("diagnostics", {}),
                "provenance": dataset.get("provenance", {}),
                "usable_for_portfolio": config["model"] not in {"garch", "hmm"}
                    and all(r.get("fit_state", "CONVERGED") == "CONVERGED" for r in predictions),
                "trading_approved": False, "economically_validated": False}
    # Admit the whole durable artifact before any immutable state is published.
    return {**artifact, "run_id": identity(artifact)}, models


def saved_predict(state: dict[str, Any], dataset: dict[str, Any]) -> list[dict[str, Any]]:
    if state.get("implementation") != IMPLEMENTATION:
        raise ValueError("unsupported saved model implementation")
    rows, names = dataset_rows(dataset)
    own = [r for r in rows if state["instrument_id"] in {"*", r["instrument_id"]}]
    if not own:
        return []
    estimator = state["estimator"]
    kind = state["model"]
    if kind in {"ridge", "lightgbm"}:
        if names != estimator["features"]:
            raise ValueError("saved model feature names do not match admitted dataset")
        x = np.array([[r["features"][n] for n in names] for r in own])
        x = (x-np.array(estimator["mean"]))/np.array(estimator["scale"])
        if kind == "ridge":
            prediction = x @ np.array(estimator["coefficients"]) + estimator["intercept"]
        else:
            import lightgbm as lgb
            prediction = lgb.Booster(model_str=estimator["booster_text"]).predict(x, num_threads=1)
        output = [dict(row, prediction=float(value)) for row, value in zip(own, prediction, strict=True)]
    elif kind == "bilstm":
        own = [r for r in own if r["timestamp"] not in state["excluded_window_timestamps"]]
        values = predict_sequence(estimator, [{k: r[k] for k in ("timestamp", "available_at", "features")} for r in own])
        lookup = {r["timestamp"]: r for r in own}
        output = [dict(lookup[p["timestamp"]], prediction=p["prediction"]) for p in values]
    else:
        panel = {(r["instrument_id"], r["timestamp"]): r for r in dataset["panel_rows"]}
        returns = [panel[(r["instrument_id"], r["timestamp"])]["return_1"] for r in own]
        expected = dict(zip(state["training_timestamps"], state["training_returns"], strict=True))
        observed = {r["timestamp"]: value for r, value in zip(own, returns, strict=True)}
        if any(observed.get(k) != value for k, value in expected.items()):
            raise ValueError("saved state inference requires its unchanged admitted training history")
        state_values, uncertainty = _state_values(state, returns)
        output = [dict(r, prediction=float(v), uncertainty=u) for r, v, u in zip(own, state_values, uncertainty, strict=True)]
    cutoff = aware_timestamp(state["training_cutoff"])
    model_id = identity(state)
    final = []
    for row in output:
        if aware_timestamp(row["timestamp"]) >= cutoff:
            final.append(_prediction(row, row["prediction"], model_id, None, kind,
                                     row.get("uncertainty", {"value": None, "state": "UNAVAILABLE"}),
                                     fit_state=state["fit_state"],
                                     usable_for_portfolio=kind not in {"garch", "hmm"} and state["fit_state"] == "CONVERGED"))
    return final
