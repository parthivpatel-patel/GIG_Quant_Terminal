"""Saved reference-return diagnostics against training-only baselines.

Loss comparisons are retrospective predictive diagnostics, never an account
backtest, calibrated confidence, multiplicity-adjusted alpha or trade approval.
"""
from __future__ import annotations

import numpy as np

from gig.research.reference_research import digest
from gig.service.workspace import _fields
from gig.timestamps import aware_timestamp

VERSION = "reference-predictive-baselines-v1"


def configuration(payload):
    _fields(payload, {"report_id", "bootstrap_replications", "block_length", "seed"}, {"report_id"})
    result = dict(payload)
    identity = result["report_id"]
    if not isinstance(identity, str) or len(identity) != 64 or any(c not in "0123456789abcdef" for c in identity):
        raise ValueError("canonical saved reference report required")
    for name, default, low, high in (("bootstrap_replications", 500, 100, 2000), ("block_length", 5, 1, 20), ("seed", 17, 0, 2**31-1)):
        result.setdefault(name, default)
        if isinstance(result[name], bool) or not isinstance(result[name], int) or not low <= result[name] <= high:
            raise ValueError("bounded integer predictive evaluation settings required")
    return result


def paired_interval(differences, *, replications, block_length, seed):
    """Circular moving-block bootstrap of date-level paired loss differences."""
    values = np.asarray(differences, dtype=float)
    if values.ndim != 1 or len(values) < 5 or block_length > len(values) or not np.isfinite(values).all():
        raise ValueError("sufficient finite paired dates and bounded block length required")
    generator = np.random.default_rng(seed)
    count = (len(values)+block_length-1)//block_length
    starts = generator.integers(0, len(values), size=(replications, count))
    indices = ((starts[:, :, None]+np.arange(block_length)) % len(values)).reshape(replications, -1)[:, :len(values)]
    means = np.mean(values[indices], axis=1)
    return {"mean_model_minus_baseline_loss": float(values.mean()),
            "percentile_95_interval": [float(x) for x in np.quantile(means, [.025, .975])],
            "method": "CIRCULAR_MOVING_BLOCK_BOOTSTRAP_PAIRED_OBSERVATION_DATES",
            "block_length": block_length, "replications": replications,
            "assumptions": "Date blocks approximate temporal dependence; currencies are kept together. No stationarity or family-completeness guarantee.",
            "multiplicity_adjusted": False, "calibrated_confidence_state": "UNKNOWN"}


def evaluate_reference(report, models, config, available_at):
    if aware_timestamp(available_at) < aware_timestamp(report["actual_acquired_at"]):
        raise ValueError("evaluation cannot precede actual source acquisition")
    common = {"schema_version": VERSION, "title": "Saved reference model versus training-only baselines",
              "report_id": report["report_id"], "snapshot_id": report["snapshot_id"], "as_of": available_at,
              "parent_as_of": report["as_of"], "actual_acquired_at": report["actual_acquired_at"],
              "input_classification": report["input_classification"], "request": config,
              "model": report["model"], "model_ids": report["model_ids"], "prediction_kind": report["prediction_kind"],
              "research_mode": report["research_mode"], "historical_point_in_time_validated": False,
              "trading_approved": False, "funded_allocation_eligible": False, "economically_validated": False,
              "portfolio_pnl_state": "UNAVAILABLE", "capacity_state": "UNKNOWN",
              "historical_publication_time_state": "UNKNOWN", "attribution": report["attribution"],
              "parent_report_hash": digest({k: v for k, v in report.items() if k != "report_id"}),
              "baseline_training_only": True, "model_selection_performed": False,
              "economic_interpretation": "Predictive loss is not after-cost portfolio performance, alpha or a validated trading signal."}
    if common["parent_report_hash"] != report["report_id"] or config["report_id"] != report["report_id"]:
        raise ValueError("reference evaluation parent identity mismatch")
    if report["prediction_kind"] != "DERIVED_REFERENCE_RETURN_ESTIMATE":
        return {**common, "evaluation_state": "UNAVAILABLE", "metrics": None, "comparisons": None,
                "reason": "Return MSE baselines do not apply to variance forecasts or cross-reference rank scores."}
    if set(models) != set(report["model_ids"]):
        raise ValueError("complete saved models required")
    for identity, state in models.items():
        if digest(state) != identity:
            raise ValueError("reference evaluation model integrity failure")
    dates, seen, excluded = {}, set(), []
    for row in report["rows"]:
        key = row["instrument_id"], row["observation_date"]
        if key in seen:
            raise ValueError("duplicate reference prediction identity")
        seen.add(key)
        state = models[row["model_id"]]
        if state["fit_state"] != "CONVERGED" or row["prediction"] is None:
            excluded.append({"instrument_id": key[0], "observation_date": key[1], "state": "UNAVAILABLE", "fit_state": state["fit_state"]})
            continue
        training = np.asarray(state["training_targets"], dtype=float)
        actual, predicted = row["target_derived_return"], row["prediction"]
        if training.ndim != 1 or not len(training) or not np.isfinite(training).all() or not np.isfinite([actual, predicted]).all():
            raise ValueError("finite actual, predicted and training-only targets required")
        with np.errstate(over="ignore", invalid="ignore"):
            losses = np.square(np.asarray([predicted-actual, actual, actual-float(training.mean())])).tolist()
        if not np.isfinite(losses).all():
            raise ValueError("finite predictive losses required")
        dates.setdefault(key[1], []).append(losses)
    if len(dates) < 5:
        return {**common, "evaluation_state": "UNAVAILABLE", "metrics": None, "comparisons": None,
                "reason": "Insufficient converged held-out observation dates.", "excluded_rows": excluded}
    ordered = sorted(dates)
    losses = np.asarray([np.mean(dates[day], axis=0) for day in ordered])
    means = losses.mean(axis=0)
    comparisons = {}
    for index, name in ((1, "zero_return"), (2, "training_mean_return")):
        comparisons[name] = paired_interval(losses[:, 0]-losses[:, index], replications=config["bootstrap_replications"],
                                            block_length=config["block_length"], seed=config["seed"])
        comparisons[name]["matched_date_equal_weight_baseline_mse"] = float(means[index])
    return {**common, "evaluation_state": "AVAILABLE", "coverage_state": "COMPLETE" if not excluded else "PARTIAL",
            "metrics": {"matched_date_equal_weight_model_mse": float(means[0]), "paired_dates": len(ordered),
                        "paired_rows": sum(len(v) for v in dates.values()), "excluded_rows": len(excluded)},
            "comparisons": comparisons, "excluded_rows": excluded, "observation_dates": ordered,
            "coverage_note": "All losses use identical available model/target rows within each date. Dates have equal weight; missing fits are disclosed, never zero-filled.",
            "interpretation": "MODEL_WORSE_THAN_BOTH_BASELINES_ON_MATCHED_DATES" if means[0] > max(means[1:]) else
                              "MODEL_BETTER_THAN_BOTH_BASELINES_ON_MATCHED_DATES" if means[0] < min(means[1:]) else "MIXED_OR_EQUAL_BASELINE_COMPARISON"}
