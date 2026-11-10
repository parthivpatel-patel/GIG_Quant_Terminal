"""Observed source provenance and independent quantitative reference checks."""
from __future__ import annotations

import copy
import hashlib
import json
import math
import shutil
from decimal import Decimal
from unittest.mock import patch

import numpy as np
import pytest

from gig.research_org import observed_experiments as science
from gig.research_org.treasury_experiment import run_treasury


@pytest.fixture(scope="module")
def studies():
    result = {}
    for dataset, runner in ((science.ECB_DATASET, science.run_ecb_risk), (science.BOC_DATASET, science.run_boc_volatility)):
        document = science._load(dataset)
        digest, report = runner()
        result[dataset] = document, digest, report
    return result


def test_distinct_official_datasets_are_historical_and_hash_verified(studies):
    identities = set()
    for dataset, (document, _, report) in studies.items():
        ready = science.dataset_readiness(dataset)
        assert ready["state"] == "AVAILABLE"
        assert ready["dataset_id"] == report["dataset"]["id"] == dataset
        assert ready["sha256"] == report["dataset"]["sha256"]
        assert document["source_authenticity"] == "OFFICIAL_HTTPS_FETCH"
        assert ready["publication_time"] == ready["revision_knowledge_time"] == "UNKNOWN"
        assert not ready["point_in_time_backtest_eligible"] and not ready["funded_allocation_eligible"]
        identities.add((ready["provider"], ready["source_url"], ready["sha256"]))
    assert len(identities) == 2
    assert len(studies[science.ECB_DATASET][0]["observations"]) == 255
    assert len(studies[science.BOC_DATASET][0]["observations"]) == 248


def test_exact_source_ecb_and_boc_identity_and_refresh_failure(studies):
    ecb = studies[science.ECB_DATASET][0]
    boc = studies[science.BOC_DATASET][0]
    assert ecb["attribution"].startswith("Source: European Central Bank.")
    assert boc["attribution"].startswith("Source: Bank of Canada.")
    assert boc["response"]["status"] == 200 and boc["subsequent_refresh"]["status"] == 503
    assert not boc["subsequent_refresh"]["data_replaced"]
    assert "Local persistence" in boc["acquisition_clock_semantics"]
    assert boc["series"]["metadata"]["unit"] == "Canadian dollars per US dollar"
    assert boc["observation_semantics"] == "INDICATIVE_DAILY_REFERENCE_NOT_TRANSACTION_PRICES"
    permissions = json.loads((science.FIXTURES / "observed_source_permissions.json").read_text())
    assert len(permissions["evidence"]) == 4
    assert all(item["status"] == 200 for item in permissions["evidence"])


@pytest.mark.parametrize("dataset", [science.ECB_DATASET, science.BOC_DATASET])
def test_deterministic_results_and_content_hash(studies, dataset):
    document, digest, report = studies[dataset]
    runner = science.run_ecb_risk if dataset == science.ECB_DATASET else science.run_boc_volatility
    second_digest, replay = runner()
    assert (second_digest, replay) == (digest, report)
    body = copy.deepcopy(report)
    expected = body.pop("result_hash")
    assert hashlib.sha256(science._canonical(body)).hexdigest() == expected
    assert report["verdict"] == "INCONCLUSIVE"
    assert all(report["skeptic"]["checks"].values())
    assert report["n_obs"] == len(document["observations"])
    assert report["economic_claim"] == "NONE" and report["allocation_eligibility"] == "UNAVAILABLE"
    assert report["total_return"] is None and report["sharpe_after_cost"] is None
    assert report["orders_submitted"] == report["models_promoted"] == 0
    assert not report["independent_confirmation"]
    json.dumps(report, allow_nan=False)


def test_boc_variance_qlike_matches_independent_decimal_forecast(studies):
    document, _, report = studies[science.BOC_DATASET]
    observations = document["observations"]
    changes = [(Decimal(observations[index]["USDCAD"]) / Decimal(observations[index - 1]["USDCAD"])).ln()
               for index in range(1, len(observations))]
    first = report["metrics"]["prediction_rows"][0]
    start = first["observation_index"]
    training = changes[:start - 2]
    baseline = sum(value ** 2 for value in training) / Decimal(len(training))
    variance = baseline
    for change in training:
        variance = Decimal(".94") * variance + Decimal(".06") * change ** 2
    variance = Decimal(".94") * variance + Decimal(".06") * changes[start - 2] ** 2
    target = float(changes[start - 1] ** 2)
    assert first["predicted_variance"] == pytest.approx(float(variance), rel=1e-10)
    assert first["model_score"] == pytest.approx(math.log(float(variance)) + target / float(variance), abs=1e-8)
    assert report["metrics"]["model_mean_score"] == pytest.approx(-10.244119476102405)
    assert report["metrics"]["baseline_mean_score"] == pytest.approx(-10.356258370725135)
    assert report["metrics"]["model_mean_score"] > report["metrics"]["baseline_mean_score"]
    assert len(report["skeptic"]["candidate_reviews"]) == 2
    assert report["metrics"]["multiple_testing"]["family_size"] == 2
    assert all(item["holm_adjusted_diagnostic_p_value"] >= item["diagnostic_p_value"] for item in report["metrics"]["candidates"])


def test_ecb_covariance_is_psd_and_reference_score_is_retained(studies):
    _, _, report = studies[science.ECB_DATASET]
    assert report["metrics"]["model_mean_score"] == pytest.approx(-29.853298978752516)
    assert report["metrics"]["baseline_mean_score"] == pytest.approx(-29.741033433991053)
    for row in report["metrics"]["prediction_rows"]:
        covariance = np.asarray(row["predicted_covariance"])
        assert np.linalg.eigvalsh(covariance).min() > 0
        np.testing.assert_allclose(covariance, covariance.T, rtol=1e-12, atol=1e-16)
        assert row["last_training_target_date"] < row["feature_date"] < row["observation_date"]
    assert all(math.isfinite(fold["correlation_frobenius_drift"]) for fold in report["metrics"]["folds"])


@pytest.mark.parametrize("dataset", [science.ECB_DATASET, science.BOC_DATASET])
def test_future_changes_do_not_rewrite_earlier_forecasts(studies, dataset):
    document, _, report = studies[dataset]
    changed = copy.deepcopy(document["observations"])
    columns = science.CATALOG[dataset][2]
    for index in range(180, len(changed)):
        for column in columns:
            changed[index][column] = str(Decimal(changed[index][column]) * (1 + Decimal(index % 5) / 100))
    evaluate = science.evaluate_ecb if dataset == science.ECB_DATASET else science.evaluate_boc
    altered = evaluate(changed)
    def earlier(rows):
        return [row for row in rows if row["observation_index"] < 180]
    assert earlier(altered["prediction_rows"]) == earlier(report["metrics"]["prediction_rows"])
    assert altered["prediction_rows"] != report["metrics"]["prediction_rows"]


@pytest.mark.parametrize("dataset", [science.ECB_DATASET, science.BOC_DATASET])
def test_independent_skeptic_rejects_tampered_scores_forecasts_and_boundaries(studies, dataset):
    document, _, report = studies[dataset]
    for field in ("model_score", "baseline_score", "last_training_target_date"):
        metrics = copy.deepcopy(report["metrics"])
        metrics["prediction_rows"][0][field] = "2099-01-01" if field.endswith("date") else 123.0
        result = science.challenge(document["observations"], metrics, dataset_id=dataset)
        assert result["verdict"].startswith("REJECTED")
    metrics = copy.deepcopy(report["metrics"])
    metrics["model_mean_score"] += 1
    assert not science.challenge(document["observations"], metrics, dataset_id=dataset)["checks"]["independent_score_reference"]
    metrics = copy.deepcopy(report["metrics"])
    metrics["folds"][0]["training_pairs"] += 1
    assert not science.challenge(document["observations"], metrics, dataset_id=dataset)["checks"]["independent_fold_metadata"]


def test_invalid_missing_duplicate_nonfinite_and_constant_data_rejected(studies):
    for dataset, (document, _, _) in studies.items():
        column = science.CATALOG[dataset][2][0]
        evaluate = science.evaluate_ecb if dataset == science.ECB_DATASET else science.evaluate_boc
        for value in ("NaN", "Infinity", "-1", "0", "bad", 1.0):
            changed = copy.deepcopy(document["observations"])
            changed[5][column] = value
            with pytest.raises(ValueError):
                evaluate(changed)
        changed = copy.deepcopy(document["observations"])
        changed[5]["date"] = changed[4]["date"]
        with pytest.raises(ValueError):
            evaluate(changed)
        with pytest.raises(ValueError):
            evaluate(changed[:100])
        changed = copy.deepcopy(document["observations"])
        for row in changed:
            row[column] = "1.0"
        with pytest.raises(ValueError):
            evaluate(changed)


def test_corrupt_original_or_normalized_data_fails_closed(tmp_path):
    for name in science.FIXTURES.glob("observed_*.json"):
        shutil.copyfile(name, tmp_path / name.name)
    with patch.object(science, "FIXTURES", tmp_path):
        assert science.dataset_readiness(science.BOC_DATASET)["state"] == "AVAILABLE"
        (tmp_path / "observed_boc_2023_source.json").write_text('{"forged":true}')
        assert science.dataset_readiness(science.BOC_DATASET)["state"] == "ERROR"
        with pytest.raises(ValueError, match="source hash"):
            science.run_boc_volatility()
        (tmp_path / "observed_ecb_2023.json").unlink()
        assert science.dataset_readiness(science.ECB_DATASET)["state"] == "UNAVAILABLE"
    assert science.dataset_readiness("unknown")["state"] == "ERROR"


def test_previous_treasury_negative_finding_unchanged():
    _, report = run_treasury()
    assert (report["hits"], report["pairs"], report["verdict"]) == (105, 228, "INCONCLUSIVE")
