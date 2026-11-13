"""Retrospective equity archive diagnostics, never PIT trading research.

Two fixed questions on five source labels. Unknown adjustment, universe and
publication vintages prohibit an executable-return or allocation interpretation.
"""
from __future__ import annotations

import hashlib
import json
import math
import statistics
from decimal import Decimal
from pathlib import Path
from typing import Any

import numpy as np

from gig.data.public_equity import (
    ATTRIBUTION,
    INFORMATION_URL,
    LICENSE_URL,
    PROVIDER,
    SOURCE_URL,
    parse_equity_csv,
)
from gig.research.experiment import config_hash

DATASET = "uci-dow-jones-weekly-2011"
FIXTURE = Path(__file__).parent / "fixtures/observed_uci_2011.json"
FIXTURE_SHA = "cf77b64c2afbb3decfeb629d42cb2096f32632e9ca8e5904c74822a0fa8f2d17"
SOURCE_SHA = "08ad8e30915f071b46f6f5bef2b1ccdfe70aacfa31aee116117048ad2adec0bf"
CSV_SHA = "24b8f83168153d9cd0bc6121d3979bd8c70fb7b58a94b920194135cbe003090b"
LABELS = ("AA", "IBM", "KO", "MSFT", "XOM")
WINDOWS = ((10, 15), (15, 20), (20, 25))
HYPOTHESES = {"range": "uci-equity-weekly-range-persistence-2011-v1",
              "volume": "uci-equity-weekly-volume-persistence-2011-v1"}


def load_equity() -> tuple[dict[str, Any], dict[str, Any]]:
    raw = FIXTURE.read_bytes()
    if len(raw) > 150_000 or hashlib.sha256(raw).hexdigest() != FIXTURE_SHA:
        raise ValueError("pinned equity fixture integrity failed")
    source = json.loads(raw)
    if (source["dataset_id"] != DATASET or source["license"] != "CC-BY-4.0"
            or source["source_sha256"] != SOURCE_SHA
            or source["csv_sha256"] != CSV_SHA
            or hashlib.sha256(source["csv_text"].encode()).hexdigest() != CSV_SHA):
        raise ValueError("equity source identity failed")
    document = parse_equity_csv(source["csv_text"], {"provider": PROVIDER, "symbols": list(LABELS),
                               "start": "2011-01-07", "end": "2011-06-24"},
                                source["source_acquired_at"], observed=True)
    if document["diagnostics"]["selected_rows"] != 125:
        raise ValueError("complete fixed source selection required")
    return source, document


def dataset_readiness() -> dict[str, Any]:
    try:
        source, document = load_equity()
    except FileNotFoundError:
        return {"dataset_id": DATASET, "state": "UNAVAILABLE", "sha256": None, "reason": "Pinned equity archive not installed."}
    except (OSError, ValueError, KeyError, TypeError):
        return {"dataset_id": DATASET, "state": "ERROR", "sha256": None, "reason": "Equity source admission failed; no replacement data."}
    return {"dataset_id": DATASET, "state": "AVAILABLE", "sha256": FIXTURE_SHA,
            "source_sha256": SOURCE_SHA, "provider": PROVIDER, "source_url": SOURCE_URL,
            "license_url": LICENSE_URL, "attribution": ATTRIBUTION, "acquired_at": source["source_acquired_at"],
            "publication_time": "UNKNOWN", "revision_knowledge_time": "UNKNOWN", "classification": "HISTORICAL",
            "corporate_actions_state": "UNAVAILABLE", "price_adjustment_basis_state": "UNKNOWN",
            "point_in_time_backtest_eligible": False, "funded_allocation_eligible": False,
            "observation_count": len(document["rows"]),
            "reason": "CC BY 4.0 source weekly equity aggregates; retrospective diagnostics only, not adjusted or PIT-admitted returns."}


def _series(document: dict[str, Any], kind: str) -> tuple[list[str], dict[str, np.ndarray]]:
    if kind not in HYPOTHESES:
        raise ValueError("registered range or volume diagnostic required")
    by_label = {label: [r for r in document["rows"] if r["source_symbol"] == label] for label in LABELS}
    dates = [r["week_ending_date"] for r in by_label[LABELS[0]]]
    if len(dates) != 25 or dates != sorted(set(dates)):
        raise ValueError("25 unique source weeks required")
    values = {}
    for label, rows in by_label.items():
        if [r["week_ending_date"] for r in rows] != dates:
            raise ValueError("unaligned equity panel; no missing security imputation")
        series = []
        for row in rows:
            if kind == "range":
                high, low, opening = (Decimal(row[k]) for k in ("high", "low", "open"))
                if not all(v.is_finite() for v in (high, low, opening)) or not 0 < low <= opening <= high:
                    raise ValueError("positive consistent range observations required")
                value = float((high - low) / opening)
            else:
                volume = row["volume"]
                if isinstance(volume, bool) or not isinstance(volume, int) or not 0 < volume <= 10**15:
                    raise ValueError("positive observed volume required for log-volume diagnostic")
                value = math.log(volume)
            if not math.isfinite(value):
                raise ValueError("finite archive diagnostic required")
            series.append(value)
        values[label] = np.asarray(series)
    return dates, values


def evaluate(document: dict[str, Any], kind: str) -> dict[str, Any]:
    dates, values = _series(document, kind)
    rows: list[dict[str, Any]] = []
    for fold, (start, end) in enumerate(WINDOWS):
        for label in LABELS:
            series = values[label]
            # Exclude the week immediately before validation from estimation.
            baseline = float(np.median(series[:start - 1]))
            for i in range(start, end):
                prediction = float(series[i - 1])
                target = float(series[i])
                rows.append({"fold": fold, "source_label": label, "observation_date": dates[i],
                             "feature_date": dates[i - 1], "last_training_target_date": dates[start - 2],
                             "prediction": prediction, "baseline_prediction": baseline,
                             "model_score": abs(target - prediction), "baseline_score": abs(target - baseline)})
    model = math.fsum(r["model_score"] for r in rows) / len(rows)
    baseline = math.fsum(r["baseline_score"] for r in rows) / len(rows)
    # Cross-sectional observations are never treated as independent weeks.
    differences = [statistics.mean(r["model_score"] - r["baseline_score"] for r in rows if r["observation_date"] == day)
                   for day in dates[10:]]
    rng = np.random.default_rng(113)
    boot = []
    for _ in range(512):
        starts = rng.integers(0, 15, 5)
        idx = [(int(s) + offset) % 15 for s in starts for offset in range(3)]
        boot.append(statistics.mean(differences[i] for i in idx))
    observed = statistics.mean(differences)
    p = (1 + sum(abs(v - observed) >= abs(observed) for v in boot)) / 513
    return {"score_name": "Mean absolute weekly range error" if kind == "range" else "Mean absolute log-volume error",
            "score_units": "fraction of source weekly opening level" if kind == "range" else "natural-log source volume",
            "model_mean_score": model, "baseline_mean_score": baseline, "lower_score_is_better": True,
            "prediction_rows": rows, "validation_weeks": 15, "panel_observations": len(rows),
            "mean_score_difference": observed, "bootstrap_interval_95": np.quantile(boot, [.025, .975]).tolist(),
            "diagnostic_p_value": p, "bonferroni_two_question_p_value": min(1.0, 2 * p),
            "bootstrap": {"method": "circular moving blocks over weekly cross-sectional mean loss differences",
                          "draws": 512, "block_weeks": 3, "seed": 113,
                          "limits": "Only 15 held-out weeks; descriptive uncertainty, not independent confirmation."},
            "folds": [{"fold": f, "training_weeks": start - 1, "purge_weeks": 1,
                       "training_last_target_date": dates[start - 2], "validation_start": dates[start],
                       "validation_end": dates[end - 1], "training_pairs": len(LABELS) * (start - 1),
                       "validation_pairs": len(LABELS) * (end - start), "sample_unit": "source-label weeks",
                       "model_mean_score": statistics.mean(r["model_score"] for r in rows if r["fold"] == f),
                       "baseline_mean_score": statistics.mean(r["baseline_score"] for r in rows if r["fold"] == f)}
                      for f, (start, end) in enumerate(WINDOWS)]}


def challenge(document: dict[str, Any], metrics: dict[str, Any], kind: str) -> dict[str, Any]:
    """Separate Decimal transforms and Python median/score reference, no fitted model reuse."""
    checks = {"independent_numeric_reference": True, "purged_temporal_boundary": True, "complete_fixed_panel": True}
    expected = []
    for fold, (start, end) in enumerate(WINDOWS):
        for label in LABELS:
            source = [r for r in document["rows"] if r["source_symbol"] == label]
            vals = [(Decimal(r["high"]) - Decimal(r["low"])) / Decimal(r["open"]) if kind == "range"
                    else Decimal(r["volume"]).ln() for r in source]
            baseline = statistics.median(vals[:start - 1])
            for i in range(start, end):
                expected.append({"fold": fold, "source_label": label, "observation_date": source[i]["week_ending_date"],
                                 "feature_date": source[i - 1]["week_ending_date"],
                                 "last_training_target_date": source[start - 2]["week_ending_date"],
                                 "prediction": float(vals[i - 1]), "baseline_prediction": float(baseline),
                                 "model_score": float(abs(vals[i] - vals[i - 1])), "baseline_score": float(abs(vals[i] - baseline))})
    records = metrics.get("prediction_rows", [])
    checks["complete_fixed_panel"] = len(records) == len(expected)
    for actual, reference in zip(records, expected, strict=False):
        for key in ("fold", "source_label", "observation_date", "feature_date", "last_training_target_date"):
            checks["complete_fixed_panel"] &= actual.get(key) == reference[key]
        for key in ("prediction", "baseline_prediction", "model_score", "baseline_score"):
            val = actual.get(key)
            checks["independent_numeric_reference"] &= (isinstance(val, (int, float)) and not isinstance(val, bool)
                                                       and math.isclose(val, reference[key], rel_tol=1e-10, abs_tol=1e-12))
        checks["purged_temporal_boundary"] &= actual.get("last_training_target_date", "") < actual.get("feature_date", "") < actual.get("observation_date", "")
    checks["independent_fold_metadata"] = len(metrics.get("folds", [])) == len(WINDOWS)
    for fold, ((start, end), summary) in enumerate(zip(WINDOWS, metrics.get("folds", []), strict=False)):
        checks["independent_fold_metadata"] &= (summary.get("fold") == fold
            and summary.get("training_pairs") == len(LABELS) * (start - 1)
            and summary.get("validation_pairs") == len(LABELS) * (end - start)
            and summary.get("purge_weeks") == 1
            and summary.get("training_last_target_date") == expected[fold * 25]["last_training_target_date"])
        for key, column in (("model_mean_score", "model_score"), ("baseline_mean_score", "baseline_score")):
            val = summary.get(key)
            ref = statistics.mean(r[column] for r in expected if r["fold"] == fold)
            checks["independent_fold_metadata"] &= isinstance(val, (int, float)) and math.isclose(val, ref, rel_tol=1e-10, abs_tol=1e-12)
    for key, column in (("model_mean_score", "model_score"), ("baseline_mean_score", "baseline_score")):
        expected_mean = math.fsum(r[column] for r in expected) / len(expected)
        val = metrics.get(key)
        checks["independent_numeric_reference"] &= isinstance(val, (int, float)) and math.isclose(val, expected_mean, rel_tol=1e-10, abs_tol=1e-12)
    return {"verdict": "INCONCLUSIVE" if all(checks.values()) else "REJECTED_NUMERICAL_OR_TEMPORAL_INTEGRITY",
            "checks": checks, "review_independence": "Separate Decimal transforms and Python median/loss reference; same source, not independent economic confirmation.",
            "reason": "Small retrospective selected panel; unknown adjustment, historical universe and publication clocks. No return or alpha claim.",
            "economic_claim": "NONE", "allocation_eligibility": "UNAVAILABLE"}


def run_equity(kind: str) -> tuple[str, dict[str, Any]]:
    source, document = load_equity()
    metrics = evaluate(document, kind)
    skeptic = challenge(document, metrics, kind)
    hypothesis = HYPOTHESES[kind]
    config = {"hypothesis_id": hypothesis, "dataset": DATASET, "fixture_sha256": FIXTURE_SHA,
              "method_version": 1, "labels": LABELS, "windows": WINDOWS, "kind": kind,
              "bootstrap_seed": 113, "bootstrap_draws": 512, "bootstrap_block_weeks": 3,
              "multiple_testing_family": list(HYPOTHESES.values())}
    dataset = {**dataset_readiness(), "id": DATASET, "information_url": INFORMATION_URL,
               "source_acquired_at": source["source_acquired_at"], "observation_semantics": "SOURCE_REPORTED_WEEKLY_AGGREGATES"}
    report = {"hypothesis_id": hypothesis, "question": f"Does lagged weekly {kind} beat a purged training-median baseline on the selected UCI archive panel?",
              "strategy": "gig.research_org.equity_experiments", "experiment_version": 1,
              "classification": "HISTORICAL", "verdict": skeptic["verdict"], "evidence_class": skeptic["verdict"],
              "economic_claim": "NONE", "n_obs": 125, "dataset": dataset, "configuration": config,
              "metrics": metrics, "skeptic": skeptic, "independent_confirmation": False,
              "design_status": "EXPLORATORY_NOT_PREREGISTERED; fixed five-label software research sample, not an investment universe.",
              "period": {"start": document["start"], "end": document["end"]}, "method": metrics["score_name"],
              "training_boundary": "One purged week; expanding training-only median; lag feature from completed source week.",
              "interpretation": f"Retrospective {kind} error {metrics['model_mean_score']:.6f} versus {metrics['baseline_mean_score']:.6f}; lower is better. Small selected archive panel, no economic claim.",
              "multiple_testing": {"method": "Bonferroni", "family_size": 2, "scope": "Two fixed UCI questions, not all historical research."},
              "lookahead_detected": not skeptic["checks"]["purged_temporal_boundary"],
              "missingness": document["diagnostics"], "quarantined_future_columns": document["quarantined_future_columns"],
              "publication_time": "UNKNOWN", "revision_knowledge_time": "UNKNOWN", "allocation_eligibility": "UNAVAILABLE",
              "total_return": None, "sharpe_after_cost": None, "max_drawdown": None,
              "orders_submitted": 0, "models_promoted": 0, "code_rewritten": False,
              "failure_conditions": ["Corporate actions and adjustment basis unavailable.", "Historical publication and membership evidence unavailable.", "Only five source labels and 25 weeks; no survivorship-free universe."],
              "role_outputs": {"quant": {"state": "COMPLETED", "output": "Two fixed purged chronological persistence diagnostics with training-only median and weekly block uncertainty."},
                               "skeptic": {"state": "COMPLETED", "output": skeptic},
                               "risk": {"state": "COMPLETED", "output": "Rejected allocation eligibility: unknown actions, identities and knowledge clocks."},
                               "developer": {"state": "COMPLETED", "output": "Verified pinned licensed archive, future-label quarantine and deterministic source/config/result hashes."}},
              "reproduce": f"python -m gig.research_org --compute {hypothesis}",
              "lineage": {"dataset": DATASET, "sha256": FIXTURE_SHA, "source_sha256": SOURCE_SHA,
                          "model_version": f"weekly-{kind}-persistence-v1", "point_in_time": "UNKNOWN publication/revision clocks; chronological archive diagnostic only",
                          "skeptic_verdict": skeptic["verdict"], "evidence_class": skeptic["verdict"],
                          "artifacts": "Packaged UCI source fixture and saved immutable ledger report."}}
    report["result_hash"] = hashlib.sha256(json.dumps(report, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    return config_hash(config), report


def run_equity_range() -> tuple[str, dict[str, Any]]:
    return run_equity("range")


def run_equity_volume() -> tuple[str, dict[str, Any]]:
    return run_equity("volume")
