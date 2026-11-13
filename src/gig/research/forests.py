"""Purged chronological research for family09; no allocator admission."""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from gig.ml.forests import MAX_FITS, MAX_TRAIN, fit_forest, predict_forest
from gig.research.model_pipeline import _folds, _prediction, _training_vintage, _wrapper, identity
from gig.timestamps import aware_timestamp


def uncertainty(value):
    return {"value": float(value), "state": "TREE_DISAGREEMENT_NOT_CALIBRATED",
            "method": "POPULATION_STD_ACROSS_CORRELATED_TREES", "unit": "arithmetic_return",
            "calibrated": False, "predictive_interval": False}


def fit_persisted_forest(dataset, rows, names, config):
    groups = defaultdict(list)
    for row in rows:
        groups[row["instrument_id"]].append(row)
    plans = []
    for instrument, own in sorted(groups.items()):
        for number, fold in enumerate(_folds(own, config)):
            candidates = [r for r in own if aware_timestamp(r["timestamp"]) in fold.train_index]
            train = [v for r in candidates if (v := _training_vintage(r, fold.test_start)) is not None]
            test = [r for r in own if aware_timestamp(r["timestamp"]) in fold.test_index]
            if not 20 <= len(train) <= MAX_TRAIN:
                raise ValueError("forests require 20..256 purged training observations")
            plans.append((instrument, number, fold, candidates, train, test))
    if not 1 <= len(plans) <= MAX_FITS:
        raise ValueError("forest work plan exceeds16total fits")
    models, predictions, folds = {}, [], []
    kind = config["model"]
    for instrument, number, fold, candidates, train, test in plans:
        x = [[r["features"][n] for n in names] for r in train]
        y = [r["target"] for r in train]
        estimator = fit_forest(x, y, names, model=kind, seed=config["seed"])
        maximum = max(aware_timestamp(r["label_end"]) for r in train).isoformat()
        state = _wrapper(dataset, kind, instrument, estimator, fold.test_start.isoformat(), maximum,
                         fit_state="FITTED_BOUNDED_FOREST", training_timestamps=[r["timestamp"] for r in train])
        model_id = identity(state)
        models[model_id] = state
        values, spread = predict_forest(estimator, [[r["features"][n] for n in names] for r in test], names)
        observed = np.array([r["target"] for r in test])
        folds.append({"fold": number, "instrument_id": instrument, "model_id": model_id,
                      "train_count": len(train), "test_count": len(test), "purged_count": len(candidates)-len(train),
                      "test_start": fold.test_start.isoformat(), "test_end": fold.test_end.isoformat(),
                      "max_training_label_end": maximum, "fit_state": "FITTED_BOUNDED_FOREST",
                      "mse": float(np.mean((observed-values)**2)), "mae": float(np.mean(np.abs(observed-values))),
                      "training_mean_baseline_mse": float(np.mean((observed-np.mean(y))**2)),
                      "uncertainty_calibrated": False, "economic_claim": "NONE"})
        predictions.extend(_prediction(row, value, model_id, number, kind, uncertainty(sd),
                                       fit_state="FITTED_BOUNDED_FOREST", usable_for_portfolio=False)
                           for row, value, sd in zip(test, values, spread, strict=True))
    return models, predictions, folds
