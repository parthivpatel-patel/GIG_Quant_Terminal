"""Train-only, purged adapter for bounded fixed-kernel Gaussian-process research."""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from gig.ml.gaussian_process import (
    MAX_FITS,
    MAX_TRAIN_ROWS,
    fit_gaussian_process,
    predict_gaussian_process,
)
from gig.research.model_pipeline import _folds, _prediction, _training_vintage, _wrapper, identity
from gig.timestamps import aware_timestamp


def uncertainty(value):
    return {"value": float(value), "state": "MODEL_POSTERIOR_NOT_CALIBRATED",
            "method": "FIXED_RBF_PREDICTIVE_STD_INCLUDING_ASSUMED_NOISE",
            "unit": "arithmetic_return", "calibrated": False}


def fit_persisted_gp(dataset, rows, names, config):
    groups = defaultdict(list)
    for row in rows:
        groups[row["instrument_id"]].append(row)
    plans = []
    # Validate every work plan before the first fit. Never silently subsample,
    # discard folds or replace the requested model when a resource bound fails.
    for instrument, own in sorted(groups.items()):
        for number, fold in enumerate(_folds(own, config)):
            candidates = [r for r in own if aware_timestamp(r["timestamp"]) in fold.train_index]
            train = [v for r in candidates if (v := _training_vintage(r, fold.test_start)) is not None]
            test = [r for r in own if aware_timestamp(r["timestamp"]) in fold.test_index]
            if not 20 <= len(train) <= MAX_TRAIN_ROWS:
                raise ValueError("GP requires 20..128 purged training rows per fit")
            plans.append((instrument, number, fold, candidates, train, test))
    if not 1 <= len(plans) <= MAX_FITS:
        raise ValueError("GP exceeds 16 total CPU fits across instruments and folds")
    models, predictions, folds = {}, [], []
    for instrument, number, fold, candidates, train, test in plans:
        x = [[r["features"][n] for n in names] for r in train]
        y = [r["target"] for r in train]
        estimator = fit_gaussian_process(x, y, names)
        maximum = max(aware_timestamp(r["label_end"]) for r in train).isoformat()
        state = _wrapper(dataset, "gaussian_process", instrument, estimator, fold.test_start.isoformat(), maximum,
                         fit_state="FITTED_FIXED_KERNEL", training_timestamps=[r["timestamp"] for r in train])
        model_id = identity(state)
        models[model_id] = state
        values, spread = predict_gaussian_process(estimator, [[r["features"][n] for n in names] for r in test], names)
        observed = np.asarray([r["target"] for r in test])
        error = observed - values
        baseline = np.mean(y)
        folds.append({"fold": number, "instrument_id": instrument, "model_id": model_id,
                      "train_count": len(train), "test_count": len(test), "purged_count": len(candidates)-len(train),
                      "test_start": fold.test_start.isoformat(), "test_end": fold.test_end.isoformat(),
                      "max_training_label_end": maximum, "fit_state": "FITTED_FIXED_KERNEL",
                      "mse": float(np.mean(error ** 2)), "training_mean_baseline_mse": float(np.mean((observed-baseline)**2)),
                      "descriptive_95pct_coverage": float(np.mean(np.abs(error) <= 1.959963984540054 * spread)),
                      "interval_calibrated": False, "economic_claim": "NONE"})
        predictions.extend(_prediction(row, value, model_id, number, "gaussian_process", uncertainty(sd),
                                       fit_state="FITTED_FIXED_KERNEL", usable_for_portfolio=False)
                           for row, value, sd in zip(test, values, spread, strict=True))
    return models, predictions, folds
