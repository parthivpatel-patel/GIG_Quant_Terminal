"""Bounded, exploratory reversal follow-up on the admitted 2023 Treasury archive.

The negative whole-year persistence study motivated this question. Chronological
folds therefore do NOT turn this reused archive into independent confirmation.
Observation ordering is tested; historical publication/revision clocks are not
known. This module makes no tradability, return, or profitability claim.
"""

from __future__ import annotations

import hashlib
import json
import math
from datetime import date
from decimal import Decimal, InvalidOperation
from typing import Any

import numpy as np
from scipy.stats import binomtest

from gig.research.experiment import config_hash
from gig.research_org.treasury_experiment import HYPOTHESIS_ID as PARENT_HYPOTHESIS_ID
from gig.research_org.treasury_experiment import _load

HYPOTHESIS_ID = "treasury-10y-reversal-robustness-2023-v1"
EXPERIMENT_VERSION = 1
WINDOWS = ((125, 166), (166, 207), (207, 250))
PURGE_OBSERVATIONS = 1
BOOTSTRAP_DRAWS = 256
BLOCK_LENGTH = 5
PLACEBO_DRAWS = 64
SEED = 67


def _levels(series: list[dict[str, Any]]) -> list[Decimal]:
    if len(series) != 250:
        raise ValueError("this approved study requires all 250 Treasury observations")
    levels = []
    previous: str | None = None
    for row in series:
        day = row.get("event_date")
        if not isinstance(day, str) or date.fromisoformat(day).isoformat() != day:
            raise ValueError("Treasury observation date is invalid")
        if previous is not None and day <= previous:
            raise ValueError("Treasury dates must be unique and increasing")
        if row.get("tenor") != "BC_10YEAR":
            raise ValueError("Treasury instrument identity does not match the approved study")
        try:
            value = Decimal(str(row["yield_percent"]))
        except (KeyError, InvalidOperation) as exc:
            raise ValueError("Treasury yield is invalid") from exc
        if not value.is_finite() or abs(value) > Decimal("1000"):
            raise ValueError("Treasury yield is nonfinite or outside the study bound")
        levels.append(value)
        previous = day
    return levels


def _holm(values: list[float]) -> list[float]:
    result = [0.0] * len(values)
    running = 0.0
    for rank, index in enumerate(sorted(range(len(values)), key=values.__getitem__)):
        running = max(running, min(1.0, (len(values) - rank) * values[index]))
        result[index] = running
    return result


def _uncertainty(records: list[dict[str, Any]]) -> dict[str, Any]:
    hits = sum(row["prediction"] == row["target"] for row in records)
    baseline_hits = sum(row["baseline"] == row["target"] for row in records)
    total = len(records)
    test = binomtest(hits, total, p=0.5)
    interval = test.proportion_ci(confidence_level=0.95, method="exact")
    better = sum(row["prediction"] == row["target"] and row["baseline"] != row["target"]
                 for row in records)
    worse = sum(row["prediction"] != row["target"] and row["baseline"] == row["target"]
                for row in records)
    paired_p = float(binomtest(better, better + worse, p=0.5).pvalue) if better + worse else 1.0
    rng = np.random.default_rng(SEED)
    differences = []
    for _ in range(BOOTSTRAP_DRAWS):
        sample: list[float] = []
        for fold in range(len(WINDOWS)):
            values = np.array([
                int(row["prediction"] == row["target"]) - int(row["baseline"] == row["target"])
                for row in records if row["fold"] == fold
            ], dtype=float)
            # Circular moving blocks stay inside their own validation fold.
            starts = rng.integers(0, len(values), size=math.ceil(len(values) / BLOCK_LENGTH))
            indices = np.concatenate([(start + np.arange(BLOCK_LENGTH)) % len(values) for start in starts])
            sample.extend(values[indices[:len(values)]].tolist())
        differences.append(float(np.mean(sample)))
    targets = np.array([row["target"] for row in records])
    predictions = np.array([row["prediction"] for row in records])
    observed = hits / total
    # Cyclic shifts preserve target ordering, unlike an IID target permutation.
    # This bounded diagnostic is not an exact randomization test under arbitrary
    # serial dependence, nor an independent sample.
    shifts = rng.choice(np.arange(1, total), size=min(PLACEBO_DRAWS, total - 1), replace=False)
    placebo = [float(np.mean(predictions == np.roll(targets, int(shift)))) for shift in shifts]
    return {
        "hits": hits, "pairs": total, "hit_rate": observed,
        "baseline_hits": baseline_hits, "baseline_hit_rate": baseline_hits / total,
        "accuracy_difference": (hits - baseline_hits) / total,
        "binomial_p_value": float(test.pvalue),
        "hit_rate_interval_95": [float(interval.low), float(interval.high)],
        "binomial_assumption": "IID Bernoulli reference only; serial dependence can invalidate nominal coverage.",
        "paired_baseline_p_value": paired_p,
        "paired_discordant_counts": {"better": better, "worse": worse},
        "bootstrap": {
            "method": "within-fold circular moving-block bootstrap of paired accuracy difference",
            "draws": BOOTSTRAP_DRAWS, "block_length": BLOCK_LENGTH, "seed": SEED,
            "interval_95": [float(value) for value in np.quantile(differences, [0.025, 0.975])],
            "limits": "One block length on one short reused sample; uncertainty is exploratory.",
        },
        "placebo": {
            "method": "distinct nonzero circular shifts of validation targets",
            "draws": len(placebo), "seed": SEED,
            "hit_rate_median": float(np.median(placebo)),
            "upper_tail_fraction": (1 + sum(value >= observed for value in placebo)) / (1 + len(placebo)),
            "limits": "Diagnostic only; cyclic exchangeability is not established.",
        },
    }


def evaluate_series(series: list[dict[str, Any]]) -> dict[str, Any]:
    """Pure evaluation; fixed observation windows, no fitting on validation targets.

    A record's feature is the previous completed yield change. Training targets
    end strictly before the first validation feature date, including a purged
    boundary observation. Zero changes are excluded, never imputed.
    """
    levels = _levels(series)
    changes = [levels[index] - levels[index - 1] for index in range(1, len(levels))]
    eligible = [(index, 1 if changes[index - 2] > 0 else -1,
                 1 if changes[index - 1] > 0 else -1)
                for index in range(2, len(levels)) if changes[index - 2] and changes[index - 1]]
    records: list[dict[str, Any]] = []
    folds = []
    for fold, (start, end) in enumerate(WINDOWS):
        train = [row for row in eligible if row[0] < start - PURGE_OBSERVATIONS]
        test = [row for row in eligible if start <= row[0] < end]
        if len(train) < 30 or len(test) < 20:
            raise ValueError("insufficient nonzero Treasury pairs for the approved folds")
        baseline = 1 if sum(row[2] for row in train) >= 0 else -1
        for index, feature, target in test:
            records.append({
                "fold": fold, "observation_index": index,
                "observation_date": series[index]["event_date"],
                "feature_date": series[index - 1]["event_date"],
                "last_training_target_date": series[train[-1][0]]["event_date"],
                "prediction": -feature, "target": target, "baseline": baseline,
            })
        folds.append({
            "fold": fold, "training_pairs": len(train), "validation_pairs": len(test),
            "training_last_target_date": series[train[-1][0]]["event_date"],
            "validation_first_feature_date": series[test[0][0] - 1]["event_date"],
            "validation_start": series[start]["event_date"],
            "validation_end": series[end - 1]["event_date"],
            "purge_observations": PURGE_OBSERVATIONS,
            "excluded_zero_pairs": end - start - len(test),
            "baseline_sign": baseline,
            "hits": sum(-row[1] == row[2] for row in test),
            "baseline_hits": sum(baseline == row[2] for row in test),
        })
    return {**_uncertainty(records), "folds": folds, "prediction_rows": records}


def challenge(series: list[dict[str, Any]], metrics: dict[str, Any]) -> dict[str, Any]:
    """Independent loop reference checks predictions, baselines and boundaries.

    It does not call evaluate_series, consume its feature arrays, or accept its
    claimed counts. Independence here is computational, not an independent
    dataset, human review, or separately trained AI model.
    """
    _levels(series)
    expected = []
    for fold, (start, end) in enumerate(WINDOWS):
        up = down = 0
        last_training_date = None
        for index in range(2, start - 1):
            earlier = Decimal(str(series[index - 2]["yield_percent"]))
            previous = Decimal(str(series[index - 1]["yield_percent"]))
            current = Decimal(str(series[index]["yield_percent"]))
            if previous == earlier or current == previous:
                continue
            up += current > previous
            down += current < previous
            last_training_date = series[index]["event_date"]
        baseline = 1 if up >= down else -1
        for index in range(start, end):
            previous_change = Decimal(str(series[index - 1]["yield_percent"])) - Decimal(str(series[index - 2]["yield_percent"]))
            next_change = Decimal(str(series[index]["yield_percent"])) - Decimal(str(series[index - 1]["yield_percent"]))
            if previous_change == 0 or next_change == 0:
                continue
            expected.append({
                "fold": fold, "observation_index": index,
                "observation_date": series[index]["event_date"],
                "feature_date": series[index - 1]["event_date"],
                "last_training_target_date": last_training_date,
                "prediction": -1 if previous_change > 0 else 1,
                "target": 1 if next_change > 0 else -1, "baseline": baseline,
            })
    expected_hits = sum(row["prediction"] == row["target"] for row in expected)
    expected_baseline = sum(row["baseline"] == row["target"] for row in expected)
    checks = {
        "independent_prediction_reference": metrics.get("prediction_rows") == expected,
        "independent_count_reference": (metrics.get("hits") == expected_hits
                                        and metrics.get("pairs") == len(expected)
                                        and metrics.get("baseline_hits") == expected_baseline),
        "independent_accuracy_reference": (metrics.get("hit_rate") == expected_hits / len(expected)
                                           and metrics.get("baseline_hit_rate") == expected_baseline / len(expected)),
        "purged_training_boundary": all(row["last_training_target_date"] < row["feature_date"] for row in expected),
        "lagged_feature_boundary": all(row["feature_date"] < row["observation_date"] for row in expected),
    }
    return {
        "verdict": "INCONCLUSIVE" if all(checks.values()) else "REJECTED_NUMERICAL_OR_TEMPORAL_INTEGRITY",
        "checks": checks,
        "independence": "Separate Decimal loop reference; same archive, not independent confirmation.",
        "reasons": [
            "Exploratory follow-up selected after reviewing the same full-year archive.",
            "Publication and revision knowledge clocks are UNKNOWN; tradable point-in-time admission is blocked.",
            "Short, single-year, single-tenor sample; no transaction prices or executable return series.",
            "Nominal IID significance does not establish significance under serial dependence.",
        ],
    }


def run_followup() -> tuple[str, dict[str, Any]]:
    meta, series = _load()
    metrics = evaluate_series(series)
    skeptic = challenge(series, metrics)
    config = {
        "hypothesis_id": HYPOTHESIS_ID, "version": EXPERIMENT_VERSION,
        "parent": PARENT_HYPOTHESIS_ID, "dataset": meta["dataset_id"], "sha256": meta["sha256"],
        "tenor": "BC_10YEAR", "rule": "opposite sign of previous completed yield change",
        "windows": WINDOWS, "purge_observations": PURGE_OBSERVATIONS,
        "baseline": "majority nonzero target sign in expanding purged training; ties predict up",
        "bootstrap_draws": BOOTSTRAP_DRAWS, "block_length": BLOCK_LENGTH,
        "placebo_draws": PLACEBO_DRAWS, "seed": SEED,
        "post_selection_reuse": True,
    }
    parent_p = float(binomtest(105, 228, p=0.5).pvalue)
    family = [parent_p, metrics["binomial_p_value"], metrics["paired_baseline_p_value"]]
    reproduce = f"python -m gig.research_org --compute {HYPOTHESIS_ID}"
    report: dict[str, Any] = {
        "hypothesis_id": HYPOTHESIS_ID,
        "question": "Does the opposite of the previous Treasury 10-year yield-change sign survive purged chronological checks and a training-only baseline?",
        "strategy": "gig.research_org.treasury_followup", "experiment_version": EXPERIMENT_VERSION,
        "configuration": config, "classification": "HISTORICAL",
        "evidence_class": "INCONCLUSIVE" if skeptic["verdict"] == "INCONCLUSIVE" else "REJECTED",
        "verdict": skeptic["verdict"], "economic_claim": "NONE",
        "publication_time": "UNKNOWN", "revision_knowledge_time": "UNKNOWN",
        "lookahead_detected": not all(skeptic["checks"].values()),
        "post_selection_reuse": True, "independent_confirmation": False,
        "parent": {"hypothesis_id": PARENT_HYPOTHESIS_ID, "hits": 105, "pairs": 228,
                   "verdict": "INCONCLUSIVE", "finding": "Negative persistence finding preserved; not overwritten."},
        "dataset": {
            "id": meta["dataset_id"], "provider": "U.S. Treasury", "classification": "HISTORICAL",
            "source_url": meta["source_url"], "license": meta["license"], "sha256": meta["sha256"],
            "retrieved_at": meta["retrieved_at"], "instrument": "BC_10YEAR constant-maturity par yield",
            "units": "yield percent", "publication_time": "UNKNOWN", "revision_knowledge_time": "UNKNOWN",
            "point_in_time": "Observation-date ordering only; historical information-availability admission is blocked.",
        },
        "period": {"start": series[0]["event_date"], "end": series[-1]["event_date"]},
        "n_obs": len(series), "metrics": metrics, "skeptic": skeptic,
        "hits": metrics["hits"], "pairs": metrics["pairs"], "hit_rate": metrics["hit_rate"],
        "multiple_testing": {
            "method": "Holm step-down", "family_size": 3,
            "family": ["parent persistence versus chance", "follow-up reversal versus chance",
                       "follow-up reversal versus training-only majority baseline"],
            "raw_p_values": family, "adjusted_p_values": _holm(family),
            "limits": "Declared local family only. Does not correct for all prior research or remove post-selection bias.",
        },
        "role_outputs": {
            "macro": {"state": "COMPLETED", "output": "Verified official admitted Treasury source hash and single-tenor observation scope."},
            "quant": {"state": "COMPLETED", "output": "Fixed reversal predictions and purged expanding-window comparisons.", "pairs": metrics["pairs"]},
            "ml": {"state": "COMPLETED", "output": "Estimated the majority-sign baseline using training targets only; no model promotion."},
            "risk": {"state": "COMPLETED", "output": "Denied allocation eligibility: no transaction prices, cost model or publication clock."},
            "skeptic": {"state": "COMPLETED", "output": skeptic},
            "developer": {"state": "COMPLETED", "output": "Deterministic replay configuration and result hash; no self-modifying code."},
        },
        "allocation_eligibility": "UNAVAILABLE", "allocation_blocker": "No tradable instruments, returns, costs or historical publication clock.",
        "orders_submitted": 0, "models_promoted": 0, "code_rewritten": False,
        "total_return": None, "sharpe_after_cost": None, "max_drawdown": None,
        "cost_bps_model": "UNAVAILABLE: yields are not transaction prices; no executable strategy is scored.",
        "method": "Fixed one-lag reversal, three expanding purged chronological folds, training-only majority baseline; zero pairs excluded.",
        "missingness": {
            "imputation": "NONE", "calendar": "Source observation dates; no exchange-calendar or publication-time assumption.",
            "zero_target_policy": "Flat target changes are excluded only from directional accuracy scoring, not asserted absent before observation.",
            "validation_zero_pairs_excluded": sum(fold["excluded_zero_pairs"] for fold in metrics["folds"]),
        },
        "training_boundary": "Every training target date is strictly earlier than the first validation feature date.",
        "holdout": {"status": "CHRONOLOGICAL_REUSED_DATA", "independent": False,
                    "note": "The parent already inspected the full archive. These folds cannot undo that selection."},
        "interpretation": (
            f"Exploratory and inconclusive: reversal matched {metrics['hits']}/{metrics['pairs']} targets "
            f"versus {metrics['baseline_hits']}/{metrics['pairs']} for the training-only baseline. "
            "No significant or economically useful signal is established; this is not a traded return backtest."
        ),
        "reproduce": reproduce,
        "lineage": {
            "dataset": meta["dataset_id"], "sha256": meta["sha256"], "parent_hypothesis": PARENT_HYPOTHESIS_ID,
            "strategy": "gig.research_org.treasury_followup", "strategy_version": EXPERIMENT_VERSION,
            "model_version": "training-majority-sign-v1", "point_in_time": "DATE_ONLY; publication/revision knowledge UNKNOWN",
            "configuration": config, "costs": "UNAVAILABLE", "assumptions": "Reused yield observations; not traded prices.",
            "skeptic_verdict": skeptic["verdict"], "evidence_class": "INCONCLUSIVE",
            "reproduce": reproduce, "artifacts": "Vendored Treasury archive, provenance JSON, configuration and ledger report.",
        },
    }
    report["result_hash"] = hashlib.sha256(json.dumps(report, sort_keys=True, allow_nan=False).encode()).hexdigest()
    return config_hash(config), report
