"""Bounded OOS economic diagnostics on immutable admitted research evidence.

This is an evaluation framework, not profitability certification. Unknown exits,
missing members and unknown capacity are never replaced by zero returns.
"""
from __future__ import annotations

import math
from collections import defaultdict
from importlib.metadata import version
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from gig.research.model_pipeline import integer, sha256
from gig.research.persisted_data import digest
from gig.research.persisted_pairs import _panel
from gig.service.serialization import dumps, loads
from gig.service.workspace import _fields, _number, _text
from gig.timestamps import aware_timestamp

VERSION = "persisted-economic-evaluation-v1"


def configuration(payload: dict[str, Any]) -> dict[str, Any]:
    required = {"kind", "run_ids", "now", "annualization", "cost_bps_grid", "slippage_bps_grid",
                "notional", "participation_limit", "declared_trials"}
    optional = {"bootstrap_repetitions", "block_size", "seed", "benchmark_instrument_id", "factor_instrument_ids"}
    _fields(payload, required | optional, required)
    request = loads(dumps(payload))
    if request["kind"] not in {"CPU", "PAIRS"}:
        raise ValueError("saved CPU or PAIRS experiments required")
    ids = request["run_ids"]
    if not isinstance(ids, list) or not 1 <= len(ids) <= 4:
        raise ValueError("one to four saved experiment identities required")
    request["run_ids"] = sorted(sha256(value) for value in ids)
    if len(set(request["run_ids"])) != len(ids):
        raise ValueError("experiment family must contain distinct identities")
    request["now"] = aware_timestamp(request["now"]).isoformat()
    request["annualization"] = integer(request["annualization"], 1, 525600, "annualization")
    for field in ("cost_bps_grid", "slippage_bps_grid"):
        values = request[field]
        if not isinstance(values, list) or not 1 <= len(values) <= 3:
            raise ValueError("one to three assumptions per cost axis required")
        checked = [_number(value) for value in values]
        if any(not 0 <= value <= 1000 for value in checked) or len(set(checked)) != len(checked):
            raise ValueError("distinct finite cost assumptions within zero to 1000 bps required")
        request[field] = sorted(checked)
    request["notional"] = _number(request["notional"], positive=True)
    request["participation_limit"] = _number(request["participation_limit"], positive=True)
    if request["notional"] > 1e12 or request["participation_limit"] > 1:
        raise ValueError("bounded notional and participation fraction required")
    request["declared_trials"] = integer(request["declared_trials"], 1, 1_000_000, "declared_trials")
    scenarios = len(ids)*len(request["cost_bps_grid"])*len(request["slippage_bps_grid"])
    if request["declared_trials"] < scenarios:
        raise ValueError("declared family cannot omit evaluated model/cost settings")
    request["bootstrap_repetitions"] = integer(request.get("bootstrap_repetitions", 100), 100, 500, "bootstrap_repetitions")
    request["block_size"] = integer(request.get("block_size", 5), 1, 64, "block_size")
    request["seed"] = integer(request.get("seed", 17), 0, 2**31-1, "seed")
    if "benchmark_instrument_id" in request:
        _text(request["benchmark_instrument_id"], bound=128)
    factors = request.setdefault("factor_instrument_ids", {})
    if not isinstance(factors, dict) or len(factors) > 4:
        raise ValueError("at most four explicitly admitted factor proxy identities required")
    for name, instrument in factors.items():
        _text(name, bound=64)
        _text(instrument, bound=128)
    if len(set(factors.values())) != len(factors):
        raise ValueError("factor proxies must have distinct instrument identities")
    return request


def uncertainty(returns: list[float], *, block_size: int, repetitions: int, seed: int) -> dict[str, Any]:
    """HAC positive-mean diagnostic and circular moving-block mean interval."""
    x = np.asarray(returns, dtype=float)
    if len(x) < max(20, 2*block_size) or not np.isfinite(x).all():
        return {"state": "UNAVAILABLE", "mean_standard_error": None, "positive_mean_pvalue": None,
                "mean_return_ci95": None, "reason": "INSUFFICIENT_COMPLETE_DEPENDENT_OBSERVATIONS"}
    n, mean = len(x), float(x.mean())
    centered = x-mean
    lag = min(block_size-1, n-1)
    variance = float(centered@centered/n)
    for offset in range(1, lag+1):
        variance += 2*(1-offset/(lag+1))*float(centered[offset:]@centered[:-offset]/n)
    se = math.sqrt(max(0., variance)/n)
    generator = np.random.default_rng(seed)
    starts = generator.integers(0, n, size=(repetitions, math.ceil(n/block_size)))
    indices = (starts[:, :, None]+np.arange(block_size)[None, None, :]) % n
    means = x[indices.reshape(repetitions, -1)[:, :n]].mean(axis=1)
    return {"state": "AVAILABLE" if se > 0 else "UNKNOWN", "mean_standard_error": se if se > 0 else None,
            "positive_mean_pvalue": float(stats.t.sf(mean/se, df=n-1)) if se > 0 else None,
            "mean_return_ci95": np.quantile(means, [.025, .975]).tolist(),
            "method": "APPROXIMATE_NEWEY_WEST_T_AND_CIRCULAR_MOVING_BLOCK_BOOTSTRAP",
            "hac_lags": lag, "bootstrap_repetitions": repetitions, "block_size": block_size,
            "seed": seed, "independence_or_stationarity_validated": False}


def _performance(values: list[float], annualization: int) -> dict[str, Any]:
    x = np.asarray(values, dtype=float)
    if not len(x) or not np.isfinite(x).all() or np.any(x <= -1):
        return {"state": "UNAVAILABLE", "compounded_return": None, "annualized_mean": None,
                "annualized_volatility": None, "sharpe": None, "max_drawdown": None}
    nav = np.r_[1., np.cumprod(1+x)]
    volatility = float(x.std(ddof=1))*math.sqrt(annualization) if len(x) > 1 else None
    mean = float(x.mean())*annualization
    return {"state": "AVAILABLE", "compounded_return": float(nav[-1]-1), "annualized_mean": mean,
            "annualized_volatility": volatility, "sharpe": mean/volatility if volatility else None,
            "max_drawdown": float(np.min(nav/np.maximum.accumulate(nav)-1)),
            "return_definition": "FRACTIONAL_CONSTANT_NOTIONAL_RESEARCH_RETURNS_COMPOUNDED_FOR_DIAGNOSTIC_ONLY",
            "sharpe_definition": "ANNUALIZED_MEAN_OVER_VOLATILITY_ZERO_RATE_ASSUMPTION_NOT_RISK_ADJUSTED_ALPHA"}


def _membership(universes, stamp):
    known = [u for u in universes if aware_timestamp(u["available_at"]) <= aware_timestamp(stamp)
             and aware_timestamp(u["effective_at"]) <= aware_timestamp(stamp)]
    if not known:
        return None
    selected = max(known, key=lambda u: (aware_timestamp(u["effective_at"]), aware_timestamp(u["available_at"])))
    return {member["instrument_id"] for member in selected["members"]}


def _fold_integrity(run, kind):
    for fold in run["folds"]:
        if kind == "CPU":
            if aware_timestamp(fold["max_training_label_end"]) >= aware_timestamp(fold["test_start"]):
                raise ValueError("training outcomes overlap the saved evaluation fold")
        elif aware_timestamp(fold["training_end"]) >= aware_timestamp(fold["training_cutoff"]):
            raise ValueError("pair training clock overlaps evaluation")
    if kind == "CPU":
        by_id = {fold["model_id"]: fold for fold in run["folds"]}
        for row in run["rows"]:
            fold = by_id.get(row["model_id"])
            if fold is None or not (aware_timestamp(fold["test_start"]) <= aware_timestamp(row["timestamp"])
                                    <= aware_timestamp(fold["test_end"])):
                raise ValueError("prediction lies outside its preserved OOS model fold")


def _periods(run, dataset, universes, request):
    stamps, panel = _panel(dataset)
    index = {stamp: i for i, stamp in enumerate(stamps)}
    by_time: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in run["rows"]:
        by_time[row["timestamp"]].append(row)
    if request["kind"] == "CPU":
        # The saved fold intervals declare the evaluation clocks. An absent
        # entire cross-section must not silently disappear from the report.
        for stamp in stamps:
            if any(aware_timestamp(fold["test_start"]) <= aware_timestamp(stamp)
                   <= aware_timestamp(fold["test_end"]) for fold in run["folds"]):
                by_time.setdefault(stamp, [])
    previous: dict[str, float] = {}
    periods = []
    for stamp in sorted(by_time, key=aware_timestamp):
        if stamp not in index or aware_timestamp(stamp) > aware_timestamp(request["now"]):
            raise ValueError("prediction decision differs from admitted dataset clocks")
        rows, reasons = by_time[stamp], []
        eligible = _membership(universes, stamp)
        current = {symbol: values[stamp] for symbol, values in panel.items() if stamp in values}
        if eligible is None or eligible - current.keys():
            reasons.append("MISSING_POINT_IN_TIME_UNIVERSE_OR_CURRENT_MARK")
        if request["kind"] == "CPU":
            if len({r["instrument_id"] for r in rows}) != len(rows):
                raise ValueError("duplicate model decision identity")
            scores = {r["instrument_id"]: float(r["prediction"]) for r in rows}
            if not np.isfinite(list(scores.values())).all():
                raise ValueError("nonfinite saved prediction")
            if eligible is None or set(scores) != eligible or len(scores) < 2:
                reasons.append("MISSING_ELIGIBLE_SECURITY_PREDICTION")
            if any(r.get("fit_state", "CONVERGED") != "CONVERGED" for r in rows):
                reasons.append("MODEL_NOT_CONVERGED")
            ranks = pd.Series(scores, dtype=float).rank(method="average")
            centered = ranks-ranks.mean()
            denominator = float(centered.abs().sum())
            weights = {symbol: float(value/denominator) if denominator else 0.
                       for symbol, value in centered.items()}
            regime = "UNCLASSIFIED_NO_SAVED_REGIME_MODEL"
        else:
            weights: dict[str, float] = {}
            for row in rows:
                for symbol, weight in row["weights"].items():
                    weights[symbol] = weights.get(symbol, 0.)+float(weight)
                if row["state"] != "AVAILABLE" and any(row["weights"].values()):
                    reasons.append("PAIR_MODEL_UNAVAILABLE_FOR_HELD_RISK")
            regime = ";".join(sorted({row["regime"] for row in rows}))
        if any(symbol not in current for symbol, value in previous.items() if value):
            reasons.append("HELD_SECURITY_DISAPPEARED_NO_LIQUIDATION_MARK")
        if eligible is not None and set(weights)-eligible:
            reasons.append("POSITION_OUTSIDE_POINT_IN_TIME_UNIVERSE")
        next_stamp = stamps[index[stamp]+1] if index[stamp]+1 < len(stamps) else None
        attribution: dict[str, float] = {}
        known_at = []
        for symbol, weight in weights.items():
            if not weight:
                continue
            before, after = current.get(symbol), panel.get(symbol, {}).get(next_stamp)
            if (before is None or after is None
                    or before["pair_segment_start"] != after["pair_segment_start"]
                    or aware_timestamp(after["available_at"]) > aware_timestamp(request["now"])):
                reasons.append("MISSING_HELD_OUTCOME_DELISTING_OR_KNOWLEDGE")
                continue
            attribution[symbol] = weight*(after["pair_price"]/before["pair_price"]-1)
            known_at.append(after["available_at"])
        if next_stamp is None:
            reasons.append("NO_FUTURE_MARK_FOR_FINAL_DECISION")
        turnover = sum(abs(weights.get(s, 0.)-previous.get(s, 0.)) for s in set(weights) | set(previous))
        is_last = stamp == max(by_time, key=aware_timestamp)
        liquidation = sum(abs(value) for value in weights.values()) if is_last else 0.
        limits, participation = [], {}
        for symbol in set(weights) | set(previous):
            change = abs(weights.get(symbol, 0.)-previous.get(symbol, 0.))
            if not change:
                continue
            row = current.get(symbol)
            if row is None:
                reasons.append("MISSING_TRADE_VOLUME_OR_PRICE")
                continue
            volume, price = float(row["volume"]), float(row["raw_close"])
            participation[symbol] = request["notional"]*change/(volume*price) if volume > 0 else None
            limits.append(request["participation_limit"]*volume*price/change)
        if is_last and next_stamp is not None:
            for symbol, weight in weights.items():
                if not weight:
                    continue
                row = panel.get(symbol, {}).get(next_stamp)
                if row is None:
                    reasons.append("TERMINAL_LIQUIDATION_VOLUME_UNKNOWN")
                else:
                    limits.append(request["participation_limit"]*float(row["volume"])*row["raw_close"]/abs(weight))
        capacity = min(limits) if limits else None
        capacity_state = ("UNKNOWN" if reasons else "NOT_REQUIRED_NO_TURNOVER" if not limits else
                          "BREACH" if request["notional"] > capacity else "WITHIN_ASSUMPTION")
        gross = None if reasons else float(sum(attribution.values()))
        periods.append({"timestamp": stamp, "outcome_at": next_stamp, "outcome_knowledge_at": max(known_at, key=aware_timestamp) if known_at else None,
            "weights": weights, "gross_return": gross, "turnover": turnover+liquidation,
            "terminal_liquidation_turnover": liquidation, "instrument_attribution": attribution,
            "capacity_notional": capacity, "capacity_state": capacity_state, "participation": participation,
            "regime": regime, "folds": sorted({int(row["fold"]) for row in rows}),
            "state": "UNKNOWN" if reasons else "AVAILABLE", "unknown_reasons": sorted(set(reasons)),
            "source_evidence_hash": digest([current[s] for s in sorted(current)])})
        previous = weights
    return periods, panel


def _proxy_return(panel, instrument, period):
    before = panel.get(instrument, {}).get(period["timestamp"])
    after = panel.get(instrument, {}).get(period["outcome_at"])
    if before is None or after is None or before["pair_segment_start"] != after["pair_segment_start"]:
        return None
    return after["pair_price"]/before["pair_price"]-1


def evaluate_runs(runs, dataset, universes, request):
    if request["kind"] == "CPU" and dataset["request"]["horizon"] != 1:
        raise ValueError("economic CPU evaluation currently supports nonoverlapping one-step horizons only")
    if aware_timestamp(dataset["as_of"]) > aware_timestamp(request["now"]):
        raise ValueError("dataset was not available by requested evaluation cutoff")
    experiments, scenarios = [], []
    total_periods = 0
    evaluated_trials = len(runs)*len(request["cost_bps_grid"])*len(request["slippage_bps_grid"])
    for run in runs:
        if aware_timestamp(run["as_of"]) > aware_timestamp(request["now"]):
            raise ValueError("saved experiment unavailable by evaluation cutoff")
        if request["kind"] == "CPU" and not run["usable_for_portfolio"]:
            raise ValueError("regime probabilities and variance estimates are not economic return signals")
        _fold_integrity(run, request["kind"])
        periods, panel = _periods(run, dataset, universes, request)
        total_periods += len(periods)
        if total_periods > 1000:
            raise ValueError("economic evaluation exceeds 1000 retained experiment periods")
        complete = bool(periods) and all(p["gross_return"] is not None for p in periods)
        selected = [float(p["gross_return"]) for p in periods if p["gross_return"] is not None]
        expected_complete = len(selected) == len(periods)
        if not expected_complete:
            selected = []
        benchmark_id = request.get("benchmark_instrument_id")
        benchmark = [_proxy_return(panel, benchmark_id, period) for period in periods] if benchmark_id else []
        benchmark_complete = bool(benchmark_id and periods and all(v is not None for v in benchmark))
        capacity_complete = bool(periods) and all(p["capacity_state"] != "UNKNOWN" for p in periods)
        for cost in request["cost_bps_grid"]:
            for slippage in request["slippage_bps_grid"]:
                net = [float(p["gross_return"])-p["turnover"]*(cost+slippage)/10000 for p in periods] if complete else []
                inference = uncertainty(net, block_size=request["block_size"], repetitions=request["bootstrap_repetitions"], seed=request["seed"])
                raw_p = inference["positive_mean_pvalue"]
                adjusted = min(1., raw_p*request["declared_trials"]) if raw_p is not None else None
                sensitivity_id = digest({"run_id": run["run_id"], "cost_bps": cost, "slippage_bps": slippage})
                scenario = {"scenario_id": sensitivity_id, "run_id": run["run_id"], "cost_bps": cost,
                    "slippage_bps": slippage, "state": "AVAILABLE" if complete else "UNKNOWN",
                    "performance": _performance(net, request["annualization"]), "uncertainty": inference,
                    "bonferroni_positive_mean_pvalue": adjusted, "declared_trials": request["declared_trials"],
                    "statistical_interpretation": "APPROXIMATE_DEPENDENT_SAMPLE_DIAGNOSTIC_NOT_CAUSAL_ALPHA_OR_VALIDATED_PROFIT",
                    "cost_return_sum": sum(p["turnover"]*(cost+slippage)/10000 for p in periods),
                    "gross_return_sum": sum(selected) if complete else None,
                    "net_return_sum": sum(net) if complete else None,
                    "capacity_state": "UNKNOWN" if not capacity_complete else "BREACH" if any(p["capacity_state"] == "BREACH" for p in periods) else "WITHIN_ASSUMPTION",
                    "minimum_capacity_notional": min((p["capacity_notional"] for p in periods if p["capacity_notional"] is not None), default=None),
                    "economic_usefulness": "NOT_ESTABLISHED", "profitability_validated": False,
                    "benchmark": {"state": "AVAILABLE" if benchmark_complete and complete else "UNAVAILABLE",
                        "instrument_id": benchmark_id, "active_mean_return": float(np.mean(np.asarray(net)-np.asarray(benchmark, dtype=float))) if benchmark_complete and complete else None,
                        "transaction_costs": "EXCLUDED_FROM_BENCHMARK_COMPARISON_ASSUMPTION"},
                    "factor_comparison": {"state": "UNAVAILABLE", "reason": "NO_COMPLETE_ADMITTED_FACTOR_PROXY_COMPARISON"},
                    "regimes": [], "fold_performance": [], "trading_approved": False}
                if complete:
                    for regime in sorted({p["regime"] for p in periods}):
                        values = [value for value, period in zip(net, periods, strict=True) if period["regime"] == regime]
                        scenario["regimes"].append({"regime": regime, "observations": len(values), "mean_return": float(np.mean(values)),
                            "method": "SAVED_PAIR_VOLATILITY_HEURISTIC_OR_UNCLASSIFIED_CPU_NOT_HMM"})
                    for fold in sorted({f for p in periods for f in p["folds"]}):
                        values = [value for value, period in zip(net, periods, strict=True) if fold in period["folds"]]
                        scenario["fold_performance"].append({"fold": fold, "observations": len(values), "mean_return": float(np.mean(values))})
                    factors = request["factor_instrument_ids"]
                    columns = [[_proxy_return(panel, instrument, period) for period in periods] for instrument in factors.values()]
                    if columns and len(periods) >= max(20, len(columns)+5) and all(all(v is not None for v in column) for column in columns):
                        x = np.column_stack([np.ones(len(periods)), *columns]).astype(float)
                        if np.linalg.matrix_rank(x) == x.shape[1]:
                            beta = np.linalg.lstsq(x, net, rcond=None)[0]
                            scenario["factor_comparison"] = {"state": "AVAILABLE", "method": "DESCRIPTIVE_OOS_RETURN_OLS_ON_ADMITTED_PRICE_PROXY_RETURNS",
                                "intercept_per_period": float(beta[0]), "coefficients": dict(zip(factors, beta[1:].tolist(), strict=True)),
                                "residual_rms": float(np.sqrt(np.mean((np.asarray(net)-x@beta)**2))),
                                "risk_free_series": "UNAVAILABLE", "alpha_validated": False}
                scenario["degradation"] = {"training_to_oos": {"state": "UNKNOWN", "value": None,
                    "reason": "COMPARABLE_TRAINING_ECONOMIC_PATH_NOT_PRESERVED"},
                    "early_to_late_oos_mean_change": float(np.mean(net[len(net)//2:])-np.mean(net[:len(net)//2])) if len(net) >= 20 else None,
                    "interpretation": "DESCRIPTIVE_ONLY_NO_INDEPENDENT_MODEL_SELECTION_HOLDOUT"}
                scenarios.append(scenario)
        experiments.append({"run_id": run["run_id"], "model": run.get("model", "PAIRS"),
            "model_ids": run["model_ids"], "fold_provenance": run["folds"], "periods": periods,
            "complete_outcomes": complete, "unknown_periods": sum(p["state"] != "AVAILABLE" for p in periods),
            "total_turnover": sum(p["turnover"] for p in periods), "gross_performance": _performance(selected, request["annualization"]),
            "survivorship": {"state": "UNVERIFIED_COMPLETENESS", "policy": "PIT_UNIVERSE_AND_MISSING_HELD_EXITS_RETAINED_AS_UNKNOWN",
                "authoritative_delisting_master": "UNAVAILABLE", "dropped_unknown_periods": 0}})
    return {"schema_version": VERSION, "request": request, "as_of": request["now"],
        "dataset_id": dataset["dataset_id"], "source": dataset["source"], "input_classification": dataset["classification"],
        "run_ids": request["run_ids"], "rows": scenarios, "experiments": experiments,
        "metrics": {"experiments": len(experiments), "scenarios": len(scenarios),
            "unknown_periods": sum(e["unknown_periods"] for e in experiments), "trading_approved": False,
            "evaluated_trials": evaluated_trials, "declared_trials": request["declared_trials"]},
        "family": {"evaluated_trials": evaluated_trials, "declared_trials": request["declared_trials"],
            "unobserved_trials": request["declared_trials"]-evaluated_trials,
            "method": "BONFERRONI_OVER_CALLER_DECLARED_MODEL_AND_COST_FAMILY",
            "family_completeness_verified": False, "strategy_selection_performed": False},
        "library_versions": {"numpy": np.__version__, "pandas": pd.__version__, "scipy": version("scipy")},
        "assumptions": {"positions": "FRACTIONAL_CONSTANT_NOTIONAL_UNAPPROVED_RESEARCH_WEIGHTS",
            "cpu_weights": "CURRENT_OOS_CENTERED_CROSS_SECTIONAL_RANKS_NORMALIZED_TO_UNIT_GROSS",
            "outcomes": "NEXT_FROZEN_DECISION_MARK_RATIO_OR_CAUSAL_ACTION_AWARE_INDEX; NEVER_REVISED_ENTRY_PRICE",
            "costs": "FLAT_BPS_ON_ABSOLUTE_WEIGHT_TURNOVER_PLUS_EXPLICIT_FINAL_LIQUIDATION",
            "capacity": "DECLARED_PARTICIPATION_OF_ADMITTED_BAR_VOLUME_TIMES_RAW_CLOSE; ASSUMED_UNDERLYING_VOLUME_UNITS_NOT_EXECUTABLE_LIQUIDITY",
            "terminal_capacity_clock": "RETROSPECTIVE_EXIT_VOLUME_CHECK_NOT_KNOWN_AT_ORIGINAL_ENTRY",
            "excluded": ["funded_cash_accounting", "borrow", "margin", "financing", "nonlinear_market_impact", "vendor_authentication"]},
        "evidence_hashes": {"universe_snapshots": sorted(u["source_sha256"] for u in universes),
            "dataset_provenance": digest(dataset["provenance"])},
        "evaluation_state": "UNKNOWN" if any(not e["complete_outcomes"] for e in experiments) else "AVAILABLE",
        "economic_validation_status": "NOT_ESTABLISHED_SYNTHETIC_FIXTURE" if dataset["classification"] == "SYNTHETIC" else "NOT_ESTABLISHED_RESEARCH_DIAGNOSTICS_ONLY",
        "economically_validated": False, "production_suitable": False, "trading_approved": False,
        "notes": ["A statistically small p-value is neither economic usefulness nor validated profitability.",
            "Candidate family, stationarity, survivorship coverage and market capacity are not independently certified.",
            "Missing security outcomes remain UNKNOWN for the entire scenario; no losing or delisted observation is silently discarded.",
            "No model selection is performed on these same OOS reports; selecting afterwards requires a new independent holdout."]}
