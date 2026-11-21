"""Chronological directional research scored as probabilities, never returns."""
from __future__ import annotations

import hashlib
from collections import defaultdict
from pathlib import Path

import numpy as np

from gig.ml.binomial_glm import LABEL_RULE, MAX_FITS, MAX_TRAIN, fit_binomial_glm, predict_binomial_glm
from gig.research.model_pipeline import _folds, _prediction, _training_vintage, _wrapper, identity
from gig.timestamps import aware_timestamp


def uncertainty(probability):
    return {"value": None, "state": "UNAVAILABLE_PARAMETER_UNCERTAINTY",
            "method": "REGULARIZED_BINOMIAL_GLM_HAS_NO_PARAMETER_POSTERIOR",
            "bernoulli_outcome_variance": float(probability*(1-probability)),
            "calibrated": False, "predictive_interval": False}


def prediction(row, probability, logit, model_id, fold):
    return _prediction({**row, "target": float(row["target"] > 0)}, probability, model_id, fold,
                       "binomial_glm", uncertainty(probability), fit_state="CONVERGED_BOUNDED_BINOMIAL_GLM",
                       raw_return_target=row["target"], label_rule=LABEL_RULE, linear_predictor=float(logit),
                       usable_for_portfolio=False)


def fit_persisted_binomial_glm(dataset, rows, names, config):
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
            labels = [float(r["target"] > 0) for r in train]
            if not 20 <= len(train) <= MAX_TRAIN or labels.count(0.) < 3 or labels.count(1.) < 3:
                raise ValueError("binomial GLM requires20..256purged rows and at least3observations per class")
            plans.append((instrument, number, fold, candidates, train, test, labels))
    if not 1 <= len(plans) <= MAX_FITS:
        raise ValueError("binomial GLM exceeds16total fits")
    models, predictions, folds = {}, [], []
    for instrument, number, fold, candidates, train, test, labels in plans:
        estimator = fit_binomial_glm([[r["features"][n] for n in names] for r in train], labels, names)
        maximum = max(aware_timestamp(r["label_end"]) for r in train).isoformat()
        state = _wrapper(dataset, "binomial_glm", instrument, estimator, fold.test_start.isoformat(), maximum,
                         fit_state="CONVERGED_BOUNDED_BINOMIAL_GLM", training_timestamps=[r["timestamp"] for r in train],
                         adapter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), label_rule=LABEL_RULE)
        model_id = identity(state)
        models[model_id] = state
        values, logits = predict_binomial_glm(estimator, [[r["features"][n] for n in names] for r in test], names)
        observed = np.array([float(r["target"] > 0) for r in test])
        baseline = float(np.mean(labels))
        bins = []
        for k in range(5):
            mask = (values >= k/5) & ((values < (k+1)/5) if k < 4 else (values <= 1))
            count = int(mask.sum())
            bins.append({"lower": k/5, "upper": (k+1)/5, "count": count,
                         "mean_probability": float(values[mask].mean()) if count else None,
                         "observed_positive_rate": float(observed[mask].mean()) if count else None,
                         "state": "AVAILABLE" if count else "UNAVAILABLE_NO_OBSERVATIONS"})
        folds.append({"fold": number, "instrument_id": instrument, "model_id": model_id,
                      "train_count": len(train), "test_count": len(test), "purged_count": len(candidates)-len(train),
                      "test_start": fold.test_start.isoformat(), "test_end": fold.test_end.isoformat(),
                      "max_training_label_end": maximum, "fit_state": state["fit_state"], "label_rule": LABEL_RULE,
                      "positive_training_labels": int(sum(labels)), "training_positive_rate": baseline,
                      "brier_score": float(np.mean((observed-values)**2)),
                      "log_loss": float(np.mean(np.logaddexp(0, logits)-observed*logits)),
                      "training_rate_baseline_brier": float(np.mean((observed-baseline)**2)),
                      "training_rate_baseline_log_loss": float(np.mean(-observed*np.log(baseline)-(1-observed)*np.log1p(-baseline))),
                      "calibration_bins": bins, "probability_calibrated": False, "economic_claim": "NONE"})
        predictions.extend(prediction(row, p, logit, model_id, number) for row, p, logit in zip(test, values, logits, strict=True))
    return models, predictions, folds
