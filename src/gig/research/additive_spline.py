"""Purged additive spline return research with no calibrated uncertainty."""
from __future__ import annotations

import hashlib
from collections import defaultdict
from pathlib import Path

import numpy as np

from gig.ml.additive_spline import MAX_FITS, MAX_TRAIN, fit_additive_spline, predict_additive_spline
from gig.research.model_pipeline import _folds, _prediction, _training_vintage, _wrapper, identity
from gig.timestamps import aware_timestamp


def uncertainty():
    return {"value": None, "state": "UNAVAILABLE_NO_CALIBRATED_PREDICTIVE_INTERVAL",
            "method": "ADDITIVE_SPLINE_RIDGE_HAS_NO_PREDICTIVE_DISTRIBUTION",
            "unit": "arithmetic_return", "calibrated": False, "predictive_interval": False}


def fit_persisted_additive_spline(dataset, rows, names, config):
    groups = defaultdict(list)
    for row in rows:
        groups[row["instrument_id"]].append(row)
    plans = []
    for instrument, own in sorted(groups.items()):
        own.sort(key=lambda r: aware_timestamp(r["timestamp"]))
        for number, fold in enumerate(_folds(own, config)):
            candidates = [r for r in own if aware_timestamp(r["timestamp"]) in fold.train_index]
            train = [v for r in candidates if (v := _training_vintage(r, fold.test_start)) is not None]
            test = [r for r in own if aware_timestamp(r["timestamp"]) in fold.test_index]
            if not 20 <= len(train) <= MAX_TRAIN:
                raise ValueError("additive splines require20..256purged training observations")
            x, query = [np.array([[r["features"][n] for n in names] for r in records]) for records in (train, test)]
            varying = np.any(x != x[0], axis=0)
            if not varying.any() or np.any(query[:, ~varying] != x[0, ~varying]):
                raise ValueError("additive spline plan has no varying inputs or unsupported variation in constant training inputs")
            plans.append((instrument, number, fold, candidates, train, test))
    if not 1 <= len(plans) <= MAX_FITS:
        raise ValueError("additive spline plan exceeds16total fits")
    models, predictions, folds = {}, [], []
    for instrument, number, fold, candidates, train, test in plans:
        x, y = [[r["features"][n] for n in names] for r in train], [r["target"] for r in train]
        estimator = fit_additive_spline(x, y, names)
        maximum = max(aware_timestamp(r["label_end"]) for r in train).isoformat()
        state = _wrapper(dataset, "additive_spline", instrument, estimator, fold.test_start.isoformat(), maximum,
                         fit_state="FITTED_BOUNDED_ADDITIVE_SPLINES", training_timestamps=[r["timestamp"] for r in train],
                         adapter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        model_id = identity(state)
        models[model_id] = state
        values = predict_additive_spline(estimator, [[r["features"][n] for n in names] for r in test], names)
        observed = np.array([r["target"] for r in test])
        folds.append({"fold": number, "instrument_id": instrument, "model_id": model_id,
                      "train_count": len(train), "test_count": len(test), "purged_count": len(candidates)-len(train),
                      "test_start": fold.test_start.isoformat(), "test_end": fold.test_end.isoformat(),
                      "max_training_label_end": maximum, "fit_state": state["fit_state"],
                      "active_features": [names[i] for i in estimator["active_indices"]],
                      "constant_features": estimator["constant_features"], "basis_columns": estimator["basis_columns"],
                      "mse": float(np.mean((observed-values)**2)), "mae": float(np.mean(np.abs(observed-values))),
                      "training_mean_baseline_mse": float(np.mean((observed-np.mean(y))**2)),
                      "uncertainty_calibrated": False, "economic_claim": "NONE"})
        predictions.extend(_prediction(row, value, model_id, number, "additive_spline", uncertainty(),
                                       fit_state=state["fit_state"], usable_for_portfolio=False)
                           for row, value in zip(test, values, strict=True))
    return models, predictions, folds
