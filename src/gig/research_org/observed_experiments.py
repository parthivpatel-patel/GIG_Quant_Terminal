"""Offline, bounded observed-reference risk studies on two central-bank archives.

Observed reference levels are not trade prices. Publication/revision vintages are
unknown; chronological feature ordering does not establish tradable PIT data.
Both studies are exploratory, with no model promotion or economic claim.
"""
from __future__ import annotations

import base64
import csv
import hashlib
import io
import json
import math
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

import numpy as np

from gig.research.experiment import config_hash

FIXTURES = Path(__file__).parent / "fixtures"
ECB_DATASET = "ecb-reference-fx-2023-usd-gbp-jpy"
BOC_DATASET = "boc-valet-fxusdcad-2023"
ECB_HYPOTHESIS = "ecb-fx-correlation-stability-2023-v1"
BOC_HYPOTHESIS = "boc-usdcad-variance-calibration-2023-v1"
CATALOG = {
    ECB_DATASET: ("observed_ecb_2023.json", "c6c37a1c04da4ab43c2d0c371f43987c949fa9405d010fab40139b090184272b", ("USD", "GBP", "JPY")),
    BOC_DATASET: ("observed_boc_2023.json", "f5acbea85e4ebe9561e323bcab64115f2f4593a73dc90d72b1fd94369a53c607", ("USDCAD",)),
}
SEED = 113
BOOTSTRAP_DRAWS = 256
BLOCK_LENGTH = 5
VARIANCE_FLOOR = 1e-12
EWMA_ALPHAS = (0.94, 0.97)


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _validate(observations: list[dict[str, Any]], columns: tuple[str, ...]) -> np.ndarray:
    if not isinstance(observations, list) or not 230 <= len(observations) <= 300:
        raise ValueError("bounded complete 2023 reference sample required")
    values = []
    previous = ""
    for row in observations:
        if not isinstance(row, dict) or set(row) != {"date", *columns}:
            raise ValueError("reference identity/column mismatch")
        day = row["date"]
        if not isinstance(day, str) or date.fromisoformat(day).isoformat() != day or not day.startswith("2023-") or day <= previous:
            raise ValueError("unique increasing 2023 observation dates required")
        numeric = []
        for column in columns:
            value = row[column]
            if not isinstance(value, str):
                raise ValueError("original exact decimal strings required")
            try:
                number = Decimal(value)
            except InvalidOperation as exc:
                raise ValueError("invalid reference value") from exc
            if not number.is_finite() or not 0 < number < Decimal("1000000"):
                raise ValueError("positive finite reference levels required")
            numeric.append(float(number))
        values.append(numeric)
        previous = day
    if observations[0]["date"] > "2023-01-10" or observations[-1]["date"] < "2023-12-20":
        raise ValueError("whole-calendar observation coverage required")
    levels = np.asarray(values, dtype=float)
    if np.any(np.var(np.diff(np.log(levels), axis=0), axis=0) <= VARIANCE_FLOOR):
        raise ValueError("insufficient reference variation")
    return levels


def _load(dataset_id: str) -> dict[str, Any]:
    if dataset_id not in CATALOG:
        raise ValueError("unknown approved reference dataset")
    filename, expected, columns = CATALOG[dataset_id]
    raw = (FIXTURES / filename).read_bytes()
    if len(raw) > 300000 or hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError("reference fixture hash mismatch")
    document = json.loads(raw)
    if (document["dataset_id"] != dataset_id or document["classification"] != "HISTORICAL"
            or document["source_authenticity"] != "OFFICIAL_HTTPS_FETCH"
            or document["publication_time"] != "UNKNOWN" or document["revision_knowledge_time"] != "UNKNOWN"
            or document["point_in_time_backtest_eligible"] or document["funded_allocation_eligible"]):
        raise ValueError("reference provenance scope mismatch")
    acquired = datetime.fromisoformat(document["acquired_at"])
    if acquired.tzinfo is None or acquired.date() <= date(2023, 12, 31):
        raise ValueError("explicit later acquisition clock required")
    source_name = document["source_file"]
    expected_source = "observed_ecb_2023_source.json" if dataset_id == ECB_DATASET else "observed_boc_2023_source.json"
    if source_name != expected_source:
        raise ValueError("reference source path mismatch")
    source = (FIXTURES / source_name).read_bytes()
    if len(source) > 5000000 or hashlib.sha256(source).hexdigest() != document["source_sha256"]:
        raise ValueError("reference original source hash mismatch")
    original = json.loads(source)
    if dataset_id == ECB_DATASET:
        decoded = base64.b64decode(original["data"], validate=True)
        if hashlib.sha256(decoded).hexdigest() != document["csv_sha256"]:
            raise ValueError("original ECB CSV hash mismatch")
        reference = [{"date": row["Date"], **{column: row[column].strip() for column in columns}}
                     for row in csv.DictReader(io.StringIO(decoded.decode("utf-8-sig")))
                     if "2023-01-01" <= row["Date"] <= "2023-12-31"]
        reference.sort(key=lambda row: row["date"])
    else:
        if original["seriesDetail"]["FXUSDCAD"]["metadata"]["unit"] != "Canadian dollars per US dollar":
            raise ValueError("BoC quote identity/units mismatch")
        reference = [{"date": row["d"], "USDCAD": row["FXUSDCAD"]["v"]} for row in original["observations"]]
    if reference != document["observations"]:
        raise ValueError("normalized observations differ from official source")
    _validate(document["observations"], columns)
    return document


def dataset_readiness(dataset_id: str) -> dict[str, Any]:
    """Verify pinned source and scope, without running forecasts or acquiring data."""
    try:
        document = _load(dataset_id)
    except FileNotFoundError:
        return {"state": "UNAVAILABLE", "dataset_id": dataset_id, "sha256": None,
                "reason": "Approved source artifact not installed; no fabricated replacement."}
    except (ValueError, KeyError, TypeError, UnicodeError, OverflowError, OSError):
        return {"state": "ERROR", "dataset_id": dataset_id, "sha256": None,
                "reason": "Pinned source integrity, identity or provenance verification failed."}
    return {"state": "AVAILABLE", "dataset_id": dataset_id, "provider": document["provider"],
            "reason": "Pinned official historical reference observations; not tradable publication vintages.",
            "sha256": CATALOG[dataset_id][1], "source_sha256": document["source_sha256"],
            "source_url": document["source_url"], "license_url": document["license_url"],
            "attribution": document["attribution"], "acquired_at": document["acquired_at"],
            "publication_time": "UNKNOWN", "revision_knowledge_time": "UNKNOWN",
            "observation_count": len(document["observations"]), "classification": "HISTORICAL",
            "point_in_time_backtest_eligible": False, "funded_allocation_eligible": False}


def _windows(total: int) -> tuple[tuple[int, int], ...]:
    return ((120, 165), (165, 210), (210, total))


def _bootstrap(records: list[dict[str, Any]]) -> dict[str, Any]:
    differences = np.array([row["model_score"] - row["baseline_score"] for row in records])
    rng = np.random.default_rng(SEED)
    samples = []
    for _ in range(BOOTSTRAP_DRAWS):
        sampled = []
        for fold in range(3):
            values = differences[[row["fold"] == fold for row in records]]
            starts = rng.integers(0, len(values), math.ceil(len(values) / BLOCK_LENGTH))
            indices = np.concatenate([(start + np.arange(BLOCK_LENGTH)) % len(values) for start in starts])
            sampled.extend(values[indices[:len(values)]])
        samples.append(float(np.mean(sampled)))
    observed = float(np.mean(differences))
    centered = np.asarray(samples) - observed
    return {"mean_score_difference": observed,
            "bootstrap_interval_95": [float(v) for v in np.quantile(samples, [.025, .975])],
            "diagnostic_p_value": (1 + int(np.sum(np.abs(centered) >= abs(observed)))) / (BOOTSTRAP_DRAWS + 1),
            "bootstrap": {"method": "within-fold circular moving-block bootstrap", "draws": BOOTSTRAP_DRAWS,
                          "block_length": BLOCK_LENGTH, "seed": SEED,
                          "limits": "Exploratory dependence-sensitive approximation on one year; not an exact test."}}


def _fold_summary(records: list[dict[str, Any]], observations: list[dict[str, Any]], start: int, end: int,
                  fold: int, training_pairs: int) -> dict[str, Any]:
    return {"fold": fold, "training_pairs": training_pairs, "validation_pairs": end - start,
            "training_last_target_date": observations[start - 2]["date"],
            "validation_first_feature_date": observations[start - 1]["date"],
            "validation_start": observations[start]["date"], "validation_end": observations[end - 1]["date"],
            "purge_observations": 1,
            "model_mean_score": float(np.mean([row["model_score"] for row in records])),
            "baseline_mean_score": float(np.mean([row["baseline_score"] for row in records]))}


def _moments(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = np.mean(values, axis=0)
    covariance = np.cov(values, rowvar=False, ddof=1)
    eigenvalues, vectors = np.linalg.eigh(covariance)
    covariance = (vectors * np.maximum(eigenvalues, VARIANCE_FLOOR)) @ vectors.T
    return mean, covariance


def evaluate_ecb(observations: list[dict[str, Any]]) -> dict[str, Any]:
    levels = _validate(observations, ("USD", "GBP", "JPY"))
    changes = np.diff(np.log(levels), axis=0)
    records, folds = [], []
    for fold, (start, end) in enumerate(_windows(len(levels))):
        training = changes[:start - 2]
        mean, covariance = _moments(training)
        diagonal = np.diag(np.diag(covariance))
        inverse, inverse_base = np.linalg.inv(covariance), np.linalg.inv(diagonal)
        logdet, base_logdet = float(np.linalg.slogdet(covariance)[1]), float(np.linalg.slogdet(diagonal)[1])
        fold_records = []
        for index in range(start, end):
            residual = changes[index - 1] - mean
            row = {"fold": fold, "observation_index": index, "observation_date": observations[index]["date"],
                   "feature_date": observations[index - 1]["date"],
                   "last_training_target_date": observations[start - 2]["date"],
                   "model_score": logdet + float(residual @ inverse @ residual),
                   "baseline_score": base_logdet + float(residual @ inverse_base @ residual),
                   "predicted_covariance": covariance.tolist(), "baseline_covariance": diagonal.tolist()}
            fold_records.append(row)
        summary = _fold_summary(fold_records, observations, start, end, fold, len(training))
        train_corr = np.corrcoef(training, rowvar=False)
        heldout_corr = np.corrcoef(changes[start - 1:end - 1], rowvar=False)
        summary.update(training_correlation=train_corr.tolist(), heldout_correlation=heldout_corr.tolist(),
                       correlation_frobenius_drift=float(np.linalg.norm(heldout_corr - train_corr)),
                       minimum_covariance_eigenvalue=float(np.min(np.linalg.eigvalsh(covariance))))
        records.extend(fold_records)
        folds.append(summary)
    return {"score_name": "Multivariate Gaussian covariance log score", "score_units": "daily reference-log-change density score",
            "lower_score_is_better": True, "model_mean_score": float(np.mean([r["model_score"] for r in records])),
            "baseline_mean_score": float(np.mean([r["baseline_score"] for r in records])),
            "folds": folds, "prediction_rows": records, **_bootstrap(records),
            "assumptions": "Training-only full sample covariance versus diagonal covariance, same training mean; fixed 1e-12 eigenvalue floor."}


def _variance_records(observations: list[dict[str, Any]], alpha: float) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    levels = _validate(observations, ("USDCAD",))[:, 0]
    changes = np.diff(np.log(levels))
    records, folds = [], []
    for fold, (start, end) in enumerate(_windows(len(levels))):
        training = changes[:start - 2]
        baseline = max(float(np.mean(training ** 2)), VARIANCE_FLOOR)
        variance = baseline
        for change in training:
            variance = alpha * variance + (1 - alpha) * float(change ** 2)
        fold_records = []
        for index in range(start, end):
            variance = max(alpha * variance + (1 - alpha) * float(changes[index - 2] ** 2), VARIANCE_FLOOR)
            target = float(changes[index - 1] ** 2)
            fold_records.append({"fold": fold, "observation_index": index,
                "observation_date": observations[index]["date"], "feature_date": observations[index - 1]["date"],
                "last_training_target_date": observations[start - 2]["date"],
                "predicted_variance": variance, "baseline_variance": baseline, "observed_squared_change": target,
                "model_score": math.log(variance) + target / variance,
                "baseline_score": math.log(baseline) + target / baseline})
        records.extend(fold_records)
        folds.append(_fold_summary(fold_records, observations, start, end, fold, len(training)))
    return records, folds


def _holm(values: list[float]) -> list[float]:
    result, running = [0.0] * len(values), 0.0
    for rank, index in enumerate(sorted(range(len(values)), key=values.__getitem__)):
        running = max(running, min(1.0, (len(values) - rank) * values[index]))
        result[index] = running
    return result


def evaluate_boc(observations: list[dict[str, Any]]) -> dict[str, Any]:
    candidates = []
    for alpha in EWMA_ALPHAS:
        records, folds = _variance_records(observations, alpha)
        candidates.append({"alpha": alpha, "model_mean_score": float(np.mean([r["model_score"] for r in records])),
                           "baseline_mean_score": float(np.mean([r["baseline_score"] for r in records])),
                           "folds": folds, "prediction_rows": records, **_bootstrap(records)})
    adjusted = _holm([row["diagnostic_p_value"] for row in candidates])
    for row, pvalue in zip(candidates, adjusted, strict=True):
        row["holm_adjusted_diagnostic_p_value"] = pvalue
    selected = candidates[0]
    return {**selected, "score_name": "Variance QLIKE (target-only constant omitted)",
            "score_units": "daily squared reference-log-change quasi-likelihood score", "lower_score_is_better": True,
            "candidates": candidates, "multiple_testing": {"method": "Holm step-down", "family_size": 2,
            "scope": "Two fixed EWMA variance forecasts versus the same training-only baseline; exploratory diagnostic p-values."},
            "selection": "alpha 0.94 fixed before evaluation; no outcome-based selection or production promotion.",
            "assumptions": "Conditional mean zero; squared daily indicative-reference change is a noisy variance proxy; fixed 1e-12 floor."}


def challenge(observations: list[dict[str, Any]], metrics: dict[str, Any], *, dataset_id: str) -> dict[str, Any]:
    """Separate Decimal/loop reference; not independent data or human review."""
    columns = CATALOG[dataset_id][2]
    _validate(observations, columns)
    changes = [[float((Decimal(observations[i][c]) / Decimal(observations[i - 1][c])).ln())
                for c in columns] for i in range(1, len(observations))]
    expected = []
    psd = True
    for fold, (start, end) in enumerate(_windows(len(observations))):
        train = changes[:start - 2]
        if dataset_id == ECB_DATASET:
            mean = [math.fsum(row[j] for row in train) / len(train) for j in range(3)]
            covariance = np.array([[math.fsum((row[j] - mean[j]) * (row[k] - mean[k]) for row in train) /
                                    (len(train) - 1) for k in range(3)] for j in range(3)])
            eigenvalues, vectors = np.linalg.eigh(covariance)
            covariance = (vectors * np.maximum(eigenvalues, VARIANCE_FLOOR)) @ vectors.T
            psd &= bool(np.min(np.linalg.eigvalsh(covariance)) > 0)
            base = np.diag(np.diag(covariance))
            for index in range(start, end):
                residual = np.array([changes[index - 1][j] - mean[j] for j in range(3)])
                model_score = float(np.linalg.slogdet(covariance)[1] + residual @ np.linalg.solve(covariance, residual))
                baseline_score = float(np.linalg.slogdet(base)[1] + residual @ np.linalg.solve(base, residual))
                expected.append({"fold": fold, "observation_index": index,
                                 "last_training_target_date": observations[start - 2]["date"],
                                 "model_score": model_score, "baseline_score": baseline_score,
                                 "prediction": covariance.tolist(), "baseline": base.tolist()})
        else:
            baseline = max(math.fsum(row[0] ** 2 for row in train) / len(train), VARIANCE_FLOOR)
            alpha = Decimal(str(metrics["alpha"]))
            variance = Decimal(str(baseline))
            for training_row in train:
                variance = alpha * variance + (1 - alpha) * Decimal(str(training_row[0])) ** 2
            for index in range(start, end):
                variance = max(alpha * variance + (1 - alpha) * Decimal(str(changes[index - 2][0])) ** 2,
                               Decimal(str(VARIANCE_FLOOR)))
                forecast = float(variance)
                psd &= forecast > 0
                target = changes[index - 1][0] ** 2
                expected.append({"fold": fold, "observation_index": index,
                    "last_training_target_date": observations[start - 2]["date"],
                    "model_score": math.log(forecast) + target / forecast,
                    "baseline_score": math.log(baseline) + target / baseline,
                    "prediction": forecast, "baseline": baseline})
    records = metrics.get("prediction_rows", [])
    identity_ok = len(records) == len(expected)
    prediction_ok = score_ok = identity_ok
    for row, reference in zip(records, expected, strict=False):
        identity_ok &= row.get("fold") == reference["fold"] and row.get("observation_index") == reference["observation_index"]
        index = reference["observation_index"]
        identity_ok &= (row.get("observation_date") == observations[index]["date"]
                        and row.get("feature_date") == observations[index - 1]["date"]
                        and row.get("last_training_target_date") == reference["last_training_target_date"])
        prediction_key = "predicted_covariance" if dataset_id == ECB_DATASET else "predicted_variance"
        baseline_key = "baseline_covariance" if dataset_id == ECB_DATASET else "baseline_variance"
        prediction_ok &= bool(np.allclose(row.get(prediction_key), reference["prediction"], rtol=1e-9, atol=1e-16))
        prediction_ok &= bool(np.allclose(row.get(baseline_key), reference["baseline"], rtol=1e-9, atol=1e-16))
        score_ok &= math.isclose(row.get("model_score", math.inf), reference["model_score"], rel_tol=1e-9, abs_tol=1e-7)
        score_ok &= math.isclose(row.get("baseline_score", math.inf), reference["baseline_score"], rel_tol=1e-9, abs_tol=1e-7)
    expected_mean = math.fsum(row["model_score"] for row in expected) / len(expected)
    expected_base = math.fsum(row["baseline_score"] for row in expected) / len(expected)
    score_ok &= math.isclose(metrics.get("model_mean_score", math.inf), expected_mean, rel_tol=1e-9, abs_tol=1e-7)
    score_ok &= math.isclose(metrics.get("baseline_mean_score", math.inf), expected_base, rel_tol=1e-9, abs_tol=1e-7)
    temporal_ok = all(row.get("last_training_target_date", "") < row.get("feature_date", "") < row.get("observation_date", "")
                      for row in records)
    fold_ok = len(metrics.get("folds", [])) == 3
    for fold, (start, end) in enumerate(_windows(len(observations))):
        if fold >= len(metrics.get("folds", [])):
            continue
        summary = metrics["folds"][fold]
        fold_ok &= (summary.get("training_pairs") == start - 2 and summary.get("validation_pairs") == end - start
                    and summary.get("training_last_target_date") == observations[start - 2]["date"]
                    and summary.get("validation_first_feature_date") == observations[start - 1]["date"]
                    and summary.get("validation_start") == observations[start]["date"]
                    and summary.get("validation_end") == observations[end - 1]["date"])
    checks = {"independent_prediction_identity": bool(identity_ok), "independent_numeric_forecast_reference": bool(prediction_ok),
              "independent_score_reference": bool(score_ok), "purged_temporal_boundary": bool(temporal_ok),
              "independent_fold_metadata": bool(fold_ok),
              "positive_semidefinite_forecast": bool(psd)}
    return {"verdict": "INCONCLUSIVE" if all(checks.values()) else "REJECTED_NUMERICAL_OR_TEMPORAL_INTEGRITY",
            "checks": checks,
            "review_independence": "Separate Decimal-log/loop forecast and linear-solve score reference; same source data, not independent confirmation or human review.",
            "economic_claim": "NONE", "allocation_eligibility": "UNAVAILABLE",
            "reason": "Exploratory reference-level risk diagnostics; absent historical publication/revision clocks and transaction prices."}


def _report(dataset_id: str, hypothesis_id: str, metrics: dict[str, Any], document: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    skeptic = challenge(document["observations"], metrics, dataset_id=dataset_id)
    if dataset_id == BOC_DATASET:
        reviews = [{"alpha": item["alpha"], **challenge(document["observations"], item, dataset_id=dataset_id)}
                   for item in metrics["candidates"]]
        skeptic["candidate_reviews"] = reviews
        skeptic["checks"]["all_declared_candidates_reviewed"] = all(all(item["checks"].values()) for item in reviews)
        if not skeptic["checks"]["all_declared_candidates_reviewed"]:
            skeptic["verdict"] = "REJECTED_NUMERICAL_OR_TEMPORAL_INTEGRITY"
    config: dict[str, Any] = {"hypothesis_id": hypothesis_id, "dataset": dataset_id, "fixture_sha256": CATALOG[dataset_id][1],
              "source_sha256": document["source_sha256"], "version": 1,
              "folds": _windows(len(document["observations"])), "purge_observations": 1,
              "seed": SEED, "bootstrap_draws": BOOTSTRAP_DRAWS, "block_length": BLOCK_LENGTH,
              "variance_floor": VARIANCE_FLOOR, "method": metrics["score_name"],
              "parameters": {"alphas": EWMA_ALPHAS} if dataset_id == BOC_DATASET else {"covariance": "sample-full-vs-diagonal"}}
    questions = {ECB_DATASET: "Does a training-only cross-currency covariance improve held-out reference-change risk scoring over a diagonal baseline?",
                 BOC_DATASET: "Do fixed EWMA variance forecasts improve held-out USD/CAD reference-change QLIKE over a training-only constant variance?"}
    dataset = {key: document[key] for key in ("dataset_id", "provider", "source_url", "license_url", "information_url", "attribution", "acquired_at", "source_sha256", "observation_semantics")}
    dataset.update(id=dataset_id, sha256=CATALOG[dataset_id][1], classification="HISTORICAL",
                   publication_time="UNKNOWN", revision_knowledge_time="UNKNOWN", point_in_time="Observation dates only; no admitted historical publication vintages.")
    observations = document["observations"]
    report = {"hypothesis_id": hypothesis_id, "question": questions[dataset_id],
              "strategy": "gig.research_org.observed_experiments", "experiment_version": 1,
              "classification": "HISTORICAL", "verdict": skeptic["verdict"], "evidence_class": skeptic["verdict"],
              "economic_claim": "NONE", "publication_time": "UNKNOWN", "revision_knowledge_time": "UNKNOWN",
              "n_obs": len(observations), "period": {"start": observations[0]["date"], "end": observations[-1]["date"]},
              "dataset": dataset, "configuration": config, "metrics": metrics, "skeptic": skeptic,
              "lookahead_detected": not skeptic["checks"]["purged_temporal_boundary"],
              "post_selection_reuse": None, "independent_confirmation": False,
              "design_status": "EXPLORATORY_NOT_PREREGISTERED; fixed methods and comparison family are retained, not selected by results.",
              "missingness": {"imputed_values": 0, "missing_quotes": 0, "calendar": "Source observation dates; bank holidays are not filled."},
              "holdout": {"status": "PURGED_CHRONOLOGICAL", "independent": False,
                          "note": "No temporal training leakage detected by reference checks; source vintage availability remains unknown."},
              "total_return": None, "sharpe_after_cost": None, "max_drawdown": None,
              "allocation_eligibility": "UNAVAILABLE", "orders_submitted": 0, "models_promoted": 0, "code_rewritten": False,
              "cost_bps_model": "UNAVAILABLE: indicative reference observations are not executed trades.",
              "method": metrics["score_name"], "training_boundary": "Training targets end before the first validation feature; forecasts use completed observations only.",
              "interpretation": f"Exploratory and inconclusive. Mean model score {metrics['model_mean_score']:.6f} versus training-only baseline {metrics['baseline_mean_score']:.6f}; lower is better. No economically useful signal or executable return is established.",
              "failure_conditions": ["Historical publication/revision vintages are unknown.", "One year and limited instruments.",
                                     "Reference changes are not transaction returns.", "Block bootstrap is an exploratory approximation, not exact confirmation."],
              "role_outputs": {
                  "macro": {"state": "COMPLETED", "output": "Verified admitted official source hash, quote identity, provider reuse terms and reference-only scope."},
                  "quant": {"state": "COMPLETED", "output": "Computed purged chronological risk forecast scores on observed reference changes."},
                  "ml": {"state": "COMPLETED", "output": "Estimated training-only benchmark; fixed candidate parameters, no production promotion."},
                  "skeptic": {"state": "COMPLETED", "output": skeptic},
                  "risk": {"state": "COMPLETED", "output": "Denied allocation: absent tradable prices, cost assumptions and historical knowledge clocks."},
                  "developer": {"state": "COMPLETED", "output": "Pinned source/fixture configuration and content hashes; deterministic offline replay."}},
              "reproduce": f"python -m gig.research_org --compute {hypothesis_id}",
              "lineage": {"dataset": dataset_id, "sha256": CATALOG[dataset_id][1],
                          "source_sha256": document["source_sha256"], "strategy": "gig.research_org.observed_experiments",
                          "strategy_version": 1, "model_version": "observed-reference-risk-v1",
                          "point_in_time": "DATE_ONLY; publication and revision clocks UNKNOWN", "costs": "UNAVAILABLE",
                          "assumptions": metrics.get("assumptions"), "configuration": config,
                          "skeptic_verdict": skeptic["verdict"], "evidence_class": skeptic["verdict"],
                          "artifacts": "Pinned JSON source, observation fixture, permissions evidence and ledger report."}}
    if dataset_id == ECB_DATASET:
        report["multiple_testing"] = {"method": "single declared model-versus-baseline comparison", "family_size": 1,
                                      "scope": "Not a global correction for unrelated archived experiments."}
    else:
        report["multiple_testing"] = metrics["multiple_testing"]
    report["result_hash"] = hashlib.sha256(_canonical(report)).hexdigest()
    return config_hash(config), report


def run_ecb_risk() -> tuple[str, dict[str, Any]]:
    document = _load(ECB_DATASET)
    return _report(ECB_DATASET, ECB_HYPOTHESIS, evaluate_ecb(document["observations"]), document)


def run_boc_volatility() -> tuple[str, dict[str, Any]]:
    document = _load(BOC_DATASET)
    return _report(BOC_DATASET, BOC_HYPOTHESIS, evaluate_boc(document["observations"]), document)
