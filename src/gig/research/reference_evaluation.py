"""Saved reference-return diagnostics against training-only baselines.

Loss comparisons are retrospective predictive diagnostics, never an account
backtest, calibrated confidence, multiplicity-adjusted alpha or trade approval.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import rankdata

from gig.research.reference_research import digest
from gig.service.workspace import _fields
from gig.timestamps import aware_timestamp

VERSION_V1 = "reference-predictive-baselines-v1"
VERSION = "reference-model-appropriate-evaluation-v2"
SUPPORTED_VERSIONS = {VERSION_V1, VERSION}


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


def evaluate_reference_v1(report, models, config, available_at):
    if aware_timestamp(available_at) < aware_timestamp(report["actual_acquired_at"]):
        raise ValueError("evaluation cannot precede actual source acquisition")
    common = {"schema_version": VERSION_V1, "title": "Saved reference model versus training-only baselines",
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


def _finite(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
        raise ValueError("finite scalar model evidence required")
    return float(value)


def _diagnostics(report, models):
    entries = []
    for identity in report["model_ids"]:
        state = models[identity]
        rows = [r for r in report["rows"] if r["model_id"] == identity]
        count = sum(r["prediction"] is not None and state["fit_state"] == "CONVERGED" for r in rows)
        entries.append({"model_id": identity, "fit_state": state["fit_state"],
                        "stop_reason": state.get("stop_reason", "NOT_RECORDED"),
                        "expected_rows": len(rows), "available_rows": count,
                        "excluded_rows": len(rows)-count,
                        "training_end_observation_date": state.get("training_end_observation_date"),
                        "first_test_observation_date": state.get("first_test_observation_date")})
    excluded = sum(e["excluded_rows"] for e in entries)
    return {"models": len(entries), "converged_models": sum(e["fit_state"] == "CONVERGED" for e in entries),
            "nonconverged_models": sum(e["fit_state"] != "CONVERGED" for e in entries),
            "expected_rows": len(report["rows"]), "excluded_rows": excluded,
            "coverage_state": "PARTIAL" if excluded else "COMPLETE", "folds": entries,
            "selection_or_refitting_performed": False}


def _validate_parent(report, models, config, available_at):
    if aware_timestamp(available_at) < aware_timestamp(report["actual_acquired_at"]):
        raise ValueError("evaluation cannot precede actual source acquisition")
    if config["report_id"] != report["report_id"] or digest({k: v for k, v in report.items() if k != "report_id"}) != report["report_id"]:
        raise ValueError("reference evaluation parent identity mismatch")
    if set(models) != set(report["model_ids"]) or any(digest(state) != key for key, state in models.items()):
        raise ValueError("reference evaluation model integrity failure")
    seen = set()
    for row in report["rows"]:
        key = row["instrument_id"], row["observation_date"]
        if key in seen or row["model_id"] not in models:
            raise ValueError("duplicate or unknown reference prediction identity")
        seen.add(key)
        _finite(row["target_derived_return"])
        if row["prediction"] is not None:
            _finite(row["prediction"])
        state = models[row["model_id"]]
        if row.get("fit_state", state["fit_state"]) != state["fit_state"]:
            raise ValueError("prediction and model convergence evidence differs")
        end, start = state.get("training_end_observation_date"), state.get("first_test_observation_date")
        if end is not None and start is not None and not end < start <= row["observation_date"]:
            raise ValueError("model training/held-out observation boundary differs")


def rank_scores(actual, scores):
    """Linear-gain NDCG with expected tie ordering; average-rank Spearman IC."""
    # Retain integer ordering exactly rather than creating ties above 2**53.
    actual, scores = np.asarray(actual), np.asarray(scores)
    if actual.dtype.kind not in "iuf" or scores.dtype.kind not in "iuf":
        raise ValueError("numeric nonboolean ranking evidence required")
    if actual.ndim != 1 or scores.shape != actual.shape or len(actual) < 2 or not np.isfinite(actual).all() or not np.isfinite(scores).all():
        raise ValueError("aligned finite ranking cross-section required")
    _, gains = np.unique(actual, return_inverse=True)
    discounts = 1/np.log2(np.arange(len(actual))+2)
    ideal = float(np.sort(gains)[::-1]@discounts)
    dcg, offset = 0., 0
    for score in sorted(set(scores.tolist()), reverse=True):
        selected = gains[scores == score]
        dcg += float(selected.mean()*discounts[offset:offset+len(selected)].sum())
        offset += len(selected)
    a, b = rankdata(actual), rankdata(scores)
    ic = float(np.corrcoef(a, b)[0, 1]) if np.ptp(a) and np.ptp(b) else None
    return {"ndcg": dcg/ideal if ideal > 0 else None, "spearman_ic": ic,
            "ic_state": "AVAILABLE" if ic is not None else "UNKNOWN",
            "ndcg_state": "AVAILABLE" if ideal > 0 else "UNKNOWN"}


def _training_rank_baseline(state, snapshot):
    frozen = state.get("ranking_baseline")
    if frozen is not None:
        return {key: _finite(value) for key, value in frozen["mean_return_by_instrument"].items()}, "SAVED_TRAINING_ONLY_STATISTICS"
    if snapshot is None:
        return None, "UNKNOWN: original training statistics and snapshot unavailable"
    if snapshot["snapshot_id"] != state["snapshot_id"] or digest({k: v for k, v in snapshot.items() if k != "snapshot_id"}) != snapshot["snapshot_id"]:
        raise ValueError("ranking source snapshot integrity failure")
    from gig.research.reference_research import _features
    groups = _features(snapshot)
    train = sorted([r for rows in groups.values() for r in rows if r["target_observation_date"] <= state["training_end_observation_date"]],
                   key=lambda r: (r["observation_date"], r["instrument_id"]))
    if not train or digest(train) != state["training_evidence_hash"]:
        raise ValueError("ranking baseline training evidence differs")
    values = {key: float(np.mean([r["target"] for r in train if r["instrument_id"] == key])) for key in groups}
    return values, "RECONSTRUCTED_FROM_HASH_VERIFIED_ORIGINAL_TRAINING_ROWS"


def _comparison(values, config):
    if len(values) < 5 or config["block_length"] > len(values):
        return {"state": "UNKNOWN", "reason": "insufficient paired dates for requested bootstrap block"}
    return {"state": "AVAILABLE", **paired_interval(values, replications=config["bootstrap_replications"],
                block_length=config["block_length"], seed=config["seed"])}


def _ranking(report, models, config, snapshot):
    grouped, scored, excluded = {}, [], []
    for row in report["rows"]:
        grouped.setdefault(row["observation_date"], []).append(row)
    baselines = {key: _training_rank_baseline(state, snapshot) for key, state in models.items()}
    for day, rows in sorted(grouped.items()):
        identities = {r["model_id"] for r in rows}
        if len(identities) != 1:
            raise ValueError("ranking date must use one saved cross-sectional model")
        state = models[rows[0]["model_id"]]
        baseline, evidence = baselines[rows[0]["model_id"]]
        if state["fit_state"] != "CONVERGED" or any(r["prediction"] is None for r in rows) or len(rows) < 2:
            excluded.append({"observation_date": day, "state": "UNAVAILABLE", "reason": "incomplete converged cross-section"})
            continue
        if baseline is not None and {r["instrument_id"] for r in rows} != set(baseline):
            excluded.append({"observation_date": day, "state": "UNAVAILABLE", "reason": "held-out cross-section differs from training universe"})
            continue
        actual, predicted = [r["target_derived_return"] for r in rows], [r["prediction"] for r in rows]
        score, uniform = rank_scores(actual, predicted), rank_scores(actual, [0.]*len(rows))
        training = rank_scores(actual, [baseline[r["instrument_id"]] for r in rows]) if baseline is not None else None
        scored.append({"observation_date": day, "model": score, "uniform_tied_scores": uniform,
                       "training_mean_return_rank": training, "training_baseline_evidence": evidence,
                       "training_universe_state": "AVAILABLE" if baseline is not None else "UNKNOWN"})
    valid = [row for row in scored if row["model"]["ndcg"] is not None]
    if not valid:
        return {"evaluation_state": "UNAVAILABLE", "metrics": None, "comparisons": None, "excluded_dates": excluded,
                "reason": "no admissible nonconstant held-out ranking cross-sections", "date_scores": scored}
    ics = [r["model"]["spearman_ic"] for r in valid if r["model"]["spearman_ic"] is not None]
    comparisons = {}
    for name in ("uniform_tied_scores", "training_mean_return_rank"):
        paired = [r for r in valid if r[name] is not None and r[name]["ndcg"] is not None]
        comparisons[name] = {**_comparison([r[name]["ndcg"]-r["model"]["ndcg"] for r in paired], config),
                             "paired_dates": len(paired), "loss": "ONE_MINUS_LINEAR_GAIN_TIE_AVERAGED_NDCG"}
    undefined_dates = sum(row["model"]["ndcg"] is None for row in scored)
    coverage = "PARTIAL" if excluded or undefined_dates else "UNKNOWN" if any(r["training_universe_state"] == "UNKNOWN" for r in scored) else "COMPLETE"
    return {"evaluation_state": "AVAILABLE", "scoring_family": "CROSS_SECTIONAL_RANKING", "coverage_state": coverage,
            "metrics": {"mean_ndcg": float(np.mean([r["model"]["ndcg"] for r in valid])),
                        "mean_spearman_ic": float(np.mean(ics)) if ics else None,
                        "spearman_ic_state": "AVAILABLE" if ics else "UNKNOWN", "defined_ic_dates": len(ics),
                        "scored_dates": len(valid), "undefined_target_dates": undefined_dates},
            "comparisons": comparisons, "date_scores": scored, "excluded_dates": excluded,
            "definition": "Linear target ordinal gains match the saved LambdaRank label-gain convention. Scores are dimensionless, never return forecasts.",
            "calibrated_confidence_state": "UNKNOWN"}


def variance_scores(actual, mean, forecast):
    actual, mean, forecast = _finite(actual), _finite(mean), _finite(forecast)
    if forecast <= 0:
        raise ValueError("strictly positive variance forecast required")
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        residual = actual-mean
        proxy = float(np.square(residual))
        qlike = float(np.log(forecast)+proxy/forecast)
        nll = .5*(float(np.log(2*np.pi))+qlike)
        mse = float(np.square(forecast-proxy))
        z = residual/float(np.sqrt(forecast))
    if not np.isfinite([proxy, qlike, nll, mse, z]).all():
        raise ValueError("variance scoring numerical overflow")
    return {"qlike": qlike, "gaussian_nll": nll, "variance_proxy_squared_error": mse,
            "innovation_squared_proxy": proxy, "forecast_variance": forecast, "standardized_innovation": z}


def _variance(report, models, config):
    dates, scores, excluded = {}, [], []
    for row in report["rows"]:
        state = models[row["model_id"]]
        if state["fit_state"] != "CONVERGED" or row["prediction"] is None:
            excluded.append({"instrument_id": row["instrument_id"], "observation_date": row["observation_date"], "state": "UNAVAILABLE"})
            continue
        train = np.asarray(state["training_targets"], dtype=float)
        mean = _finite(state["parameters"]["mean"])
        if train.ndim != 1 or not len(train) or not np.isfinite(train).all():
            raise ValueError("finite frozen variance training targets required")
        with np.errstate(over="ignore", invalid="ignore"):
            baseline = float(np.mean((train-mean)**2))
        actual = variance_scores(row["target_derived_return"], mean, row["prediction"])
        if not np.isfinite(baseline):
            raise ValueError("finite training-only variance baseline required")
        # A constant training series has no admissible strictly positive QLIKE
        # baseline. Preserve the valid model score, but never add an epsilon.
        reference = variance_scores(row["target_derived_return"], mean, baseline) if baseline > 0 else None
        if reference is not None:
            dates.setdefault(row["observation_date"], []).append(actual["qlike"]-reference["qlike"])
        scores.append({**actual, "observation_date": row["observation_date"], "instrument_id": row["instrument_id"],
                       "training_second_moment_baseline": baseline if reference is not None else None,
                       "baseline_state": "AVAILABLE" if reference is not None else "UNKNOWN",
                       "baseline_qlike": reference["qlike"] if reference is not None else None})
    if not scores:
        return {"evaluation_state": "UNAVAILABLE", "metrics": None, "comparisons": None, "excluded_rows": excluded,
                "reason": "no converged held-out conditional variance forecasts"}
    z = np.asarray([r["standardized_innovation"] for r in scores])
    return {"evaluation_state": "AVAILABLE", "scoring_family": "CONDITIONAL_VARIANCE", "coverage_state": "PARTIAL" if excluded else "COMPLETE",
            "metrics": {"mean_qlike": float(np.mean([r["qlike"] for r in scores])),
                        "mean_gaussian_nll": float(np.mean([r["gaussian_nll"] for r in scores])),
                        "variance_proxy_mse": float(np.mean([r["variance_proxy_squared_error"] for r in scores])),
                        "scored_rows": len(scores), "scored_dates": len({r["observation_date"] for r in scores}),
                        "paired_dates": len(dates)},
            "comparisons": {"training_second_moment_variance": {**_comparison([float(np.mean(v)) for _, v in sorted(dates.items())], config),
                            "paired_dates": len(dates), "paired_rows": sum(map(len, dates.values())),
                            "loss": "QLIKE_LOG_VARIANCE_PLUS_SQUARED_INNOVATION_OVER_VARIANCE"}},
            "calibration": {"state": "DIAGNOSTIC_ONLY", "standardized_innovation_mean": float(z.mean()),
                            "standardized_innovation_second_moment": float(np.mean(z*z)),
                            "observed_innovation_squared_over_forecast_variance": float(sum(r["innovation_squared_proxy"] for r in scores)/sum(r["forecast_variance"] for r in scores)),
                            "gaussian_95_interval_coverage": float(np.mean(np.abs(z) <= 1.959963984540054)),
                            "calibrated_confidence_state": "UNKNOWN", "distributional_assumption": "CONDITIONAL_GAUSSIAN_NOT_VERIFIED"},
            "score_rows": scores, "excluded_rows": excluded,
            "aggregation": "Model scoring metrics weight available rows equally; paired baseline loss differences weight dates equally using matching rows only.",
            "definition": "Next-observation fitted-mean innovations squared are noisy variance proxies, not observed latent volatility. QLIKE/NLL use return-squared variance; proxy MSE has return-fourth units. Degenerate training-only baseline comparison remains UNKNOWN without epsilon replacement."}


def evaluate_reference(report, models, config, available_at, *, schema_version=VERSION, snapshot=None):
    """Version-dispatched immutable evaluation; v1 calculations remain frozen."""
    if schema_version == VERSION_V1:
        return evaluate_reference_v1(report, models, config, available_at)
    if schema_version != VERSION:
        raise ValueError("unsupported reference evaluation version")
    _validate_parent(report, models, config, available_at)
    if report["prediction_kind"] == "DERIVED_REFERENCE_RETURN_ESTIMATE":
        result = evaluate_reference_v1(report, models, config, available_at)
        result["schema_version"] = VERSION
        result["scoring_family"] = "RETURN_ESTIMATE"
    else:
        # Freeze common research/provenance flags without imposing return-unit loss.
        result = evaluate_reference_v1(report, models, config, available_at)
        family = report["prediction_kind"]
        if family == "CROSS_REFERENCE_RANK_SCORE" and report["model"] == "lightgbm" and all(s["model"] == "lightgbm" for s in models.values()):
            result.update(_ranking(report, models, config, snapshot))
        elif family == "CONDITIONAL_VARIANCE_PER_REFERENCE_OBSERVATION" and report["model"] == "garch" and all(s["model"] == "garch" for s in models.values()):
            result.update(_variance(report, models, config))
        result["schema_version"] = VERSION
        if result["evaluation_state"] == "AVAILABLE":
            result.pop("reason", None)
    result["fit_diagnostics"] = _diagnostics(report, models)
    return result
