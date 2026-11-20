"""Purged Bayesian linear research with proper Gaussian scores, not alpha."""
from __future__ import annotations

import hashlib
from collections import defaultdict
from pathlib import Path

import numpy as np

from gig.ml.bayesian_linear import MAX_FITS, MAX_TRAIN, fit_bayesian_ridge, predict_bayesian_ridge
from gig.research.model_pipeline import _folds, _prediction, _training_vintage, _wrapper, identity
from gig.timestamps import aware_timestamp


def uncertainty(value):
    return {"value": float(value), "state": "MODEL_STD_NOT_OOS_CALIBRATED",
            "method": "GAUSSIAN_EMPIRICAL_BAYES_POSTERIOR_PREDICTIVE_STD",
            "unit": "arithmetic_return", "calibrated": False,
            "predictive_interval": True, "hyperparameter_uncertainty_included": False}


def fit_persisted_bayesian_ridge(dataset, rows, names, config):
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
                raise ValueError("Bayesian ridge requires20..256purged training rows")
            plans.append((instrument, number, fold, candidates, train, test))
    if not 1 <= len(plans) <= MAX_FITS:
        raise ValueError("Bayesian ridge exceeds16total fits")
    models, predictions, folds = {}, [], []
    for instrument, number, fold, candidates, train, test in plans:
        x, y = [[r["features"][n] for n in names] for r in train], [r["target"] for r in train]
        estimator = fit_bayesian_ridge(x, y, names)
        maximum = max(aware_timestamp(r["label_end"]) for r in train).isoformat()
        state = _wrapper(dataset, "bayesian_ridge", instrument, estimator, fold.test_start.isoformat(), maximum,
                         fit_state="CONVERGED_BOUNDED_EMPIRICAL_BAYES", training_timestamps=[r["timestamp"] for r in train],
                         adapter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        model_id = identity(state)
        models[model_id] = state
        values, sd = predict_bayesian_ridge(estimator, [[r["features"][n] for n in names] for r in test], names)
        observed = np.array([r["target"] for r in test])
        baseline_variance = float(np.var(y))
        baseline_nll = (float(np.mean(.5*np.log(2*np.pi*baseline_variance)+(observed-np.mean(y))**2/(2*baseline_variance)))
                        if baseline_variance > 0 else None)
        folds.append({"fold": number, "instrument_id": instrument, "model_id": model_id,
                      "train_count": len(train), "test_count": len(test), "purged_count": len(candidates)-len(train),
                      "test_start": fold.test_start.isoformat(), "test_end": fold.test_end.isoformat(),
                      "max_training_label_end": maximum, "fit_state": state["fit_state"],
                      "iterations": estimator["iterations"], "mse": float(np.mean((observed-values)**2)),
                      "mae": float(np.mean(np.abs(observed-values))),
                      "training_mean_baseline_mse": float(np.mean((observed-np.mean(y))**2)),
                      "gaussian_nll": float(np.mean(np.log(sd*np.sqrt(2*np.pi))+(observed-values)**2/(2*sd**2))),
                      "training_gaussian_baseline_nll": baseline_nll,
                      "training_gaussian_baseline_state": "AVAILABLE" if baseline_nll is not None else "UNAVAILABLE_ZERO_TRAINING_VARIANCE",
                      "nominal95_interval_coverage": float(np.mean(np.abs(observed-values) <= 1.959963984540054*sd)),
                      "nominal95_mean_interval_width": float(np.mean(2*1.959963984540054*sd)),
                      "uncertainty_calibrated": False, "economic_claim": "NONE"})
        predictions.extend(_prediction(row, value, model_id, number, "bayesian_ridge", uncertainty(spread),
                                       fit_state=state["fit_state"], usable_for_portfolio=False)
                           for row, value, spread in zip(test, values, sd, strict=True))
    return models, predictions, folds
