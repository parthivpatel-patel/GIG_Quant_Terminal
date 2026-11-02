"""Causal, hash-addressed pair research on admitted per-decision observations.

The portfolio path is a hypothetical constant-dollar, fractional-weight research
backtest. It is neither a cash/share ledger nor an order execution simulator.
"""
from __future__ import annotations

import math
from dataclasses import asdict
from itertools import combinations
from typing import Any

import numpy as np

from gig.research.model_pipeline import integer, sha256
from gig.research.persisted_data import digest
from gig.research.stat_arb import (
    adf_test,
    cointegration_test,
    fit_pair,
    kalman_hedge,
    mean_reversion_diagnostics,
)
from gig.service.serialization import dumps, loads
from gig.timestamps import aware_timestamp

IMPLEMENTATION = "gig.persisted-pairs.v1"


def _number(value: Any, low: float, high: float, name: str, *, strict=False) -> float:
    if (isinstance(value, bool) or not isinstance(value, (float, int))
            or not math.isfinite(value) or not low <= value <= high
            or strict and value == low):
        raise ValueError(f"bounded finite {name} required")
    return float(value)


def configuration(payload: Any) -> dict[str, Any]:
    fields = {"dataset_id", "training_window", "test_size", "embargo", "alpha", "entry_z",
              "cost_bps", "borrow_bps", "max_gross", "process_variance", "observation_variance"}
    if not isinstance(payload, dict) or set(payload)-fields or "dataset_id" not in payload:
        raise ValueError("pair research requires an admitted dataset and explicit supported settings")
    sha256(payload["dataset_id"])
    return {"training_window": integer(payload.get("training_window", 60), 30, 300, "training_window"),
            "test_size": integer(payload.get("test_size", 20), 1, 100, "test_size"),
            "embargo": integer(payload.get("embargo", 1), 1, 30, "embargo"),
            "alpha": _number(payload.get("alpha", .05), 0, .5, "alpha", strict=True),
            "entry_z": _number(payload.get("entry_z", 1.5), 0, 10, "entry_z"),
            "cost_bps": _number(payload.get("cost_bps", 10), 0, 1000, "cost_bps"),
            "borrow_bps": _number(payload.get("borrow_bps", 0), 0, 10000, "borrow_bps"),
            "max_gross": _number(payload.get("max_gross", 1), 0, 1, "max_gross", strict=True),
            "process_variance": _number(payload.get("process_variance", 1e-5), 0, 1, "process_variance"),
            "observation_variance": _number(payload.get("observation_variance", .01), 0, 10,
                                             "observation_variance", strict=True)}


def _panel(dataset):
    panel = loads(dumps(dataset["panel_rows"]))
    if not 2 <= len(panel) <= 1000:
        raise ValueError("bounded admitted market panel required")
    stamps = sorted({row["timestamp"] for row in panel}, key=aware_timestamp)
    if len(stamps) > 500:
        raise ValueError("pair research is bounded to 500 decision clocks")
    instruments = sorted({r["instrument_id"] for r in panel})
    if not 2 <= len(instruments) <= 8:
        raise ValueError("pair research requires two to eight point-in-time instrument identities")
    clock_index = {stamp: i for i, stamp in enumerate(stamps)}
    by_instrument: dict[str, dict[str, dict[str, Any]]] = {key: {} for key in instruments}
    adjusted = dataset["request"]["adjustment"] != "RAW"
    for identity in instruments:
        previous_index, causal_price, segment_start = -2, 100., None
        for row in sorted((r for r in panel if r["instrument_id"] == identity),
                          key=lambda r: aware_timestamp(r["timestamp"])):
            stamp = row["timestamp"]
            if stamp in by_instrument[identity] or aware_timestamp(row["available_at"]) > aware_timestamp(stamp):
                raise ValueError("duplicate or future pair observation")
            value = row["raw_close"]
            index = clock_index[stamp]
            new_segment = index != previous_index + 1
            if new_segment:
                segment_start = stamp
            if adjusted:
                if new_segment:
                    causal_price = 100.
                elif row["return_1"] is None:
                    raise ValueError("causal adjusted observations require an admitted return")
                else:
                    causal_price *= 1 + row["return_1"]
                value = causal_price
            previous_index = index
            if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or value <= 0:
                raise ValueError("positive finite pair price required")
            by_instrument[identity][stamp] = {**row, "pair_price": float(value), "pair_segment_start": segment_start}
    return stamps, by_instrument


def fit_pairs(dataset: dict[str, Any], config: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Fit only preceding eligible observations; never use outcomes for selection.

    Every price is frozen at its own historical decision, rather than revised to
    the experiment's final cutoff. Adjusted inputs use a causal return index;
    they do not misrepresent raw closes as retrospectively adjusted levels.
    """
    stamps, panel = _panel(dataset)
    window, embargo = config["training_window"], config["embargo"]
    if len(stamps) <= window + embargo:
        raise ValueError("pair research requires a training window, embargo and evaluation observations")
    models, folds, rows = {}, [], []
    prior_weights: dict[tuple[str, str], dict[str, float]] = {}
    latest_proposals: list[dict[str, Any]] = []
    missing_outcomes, missing_windows = 0, 0
    for fold_id, start in enumerate(range(window+embargo, len(stamps), config["test_size"])):
        end = min(len(stamps), start+config["test_size"])
        train_stamps = stamps[start-embargo-window:start-embargo]
        candidates = [identity for identity in sorted(panel)
                      if all(stamp in panel[identity] for stamp in train_stamps)]
        family = len(candidates)*(len(candidates)-1)//2
        screening, accepted = [], []
        for dependent, independent in combinations(candidates, 2):
            y = np.array([panel[dependent][stamp]["pair_price"] for stamp in train_stamps])
            x = np.array([panel[independent][stamp]["pair_price"] for stamp in train_stamps])
            evidence = [[panel[instrument][stamp] for instrument in (dependent, independent)]
                        for stamp in train_stamps]
            try:
                fitted = fit_pair(y, x, dependent=dependent, independent=independent, log_prices=True)
                coint = cointegration_test(y, x, log_prices=True)
                spread = np.log(y)-fitted.alpha-fitted.beta*np.log(x)
                adf, reversion = adf_test(spread), mean_reversion_diagnostics(spread)
                adjusted_p = None if coint["pvalue"] is None else min(1., coint["pvalue"]*family)
                selected = (coint["state"] == "AVAILABLE" and adf["state"] == "AVAILABLE"
                            and adjusted_p is not None and adjusted_p < config["alpha"]
                            and fitted.spread_std > 1e-10 and abs(fitted.beta) <= 20)
                state = {"implementation": IMPLEMENTATION, "dependent": dependent, "independent": independent,
                         "training_start": train_stamps[0], "training_end": train_stamps[-1],
                         "training_cutoff": stamps[start], "training_evidence_sha256": digest(evidence),
                         "fit": asdict(fitted), "cointegration": coint, "adf": adf,
                         "mean_reversion": reversion, "bonferroni_pvalue": adjusted_p,
                         "family_size": family, "selected": selected,
                         "input_classification": dataset["classification"],
                         "price_basis": "RAW_OBSERVED_LEVEL" if dataset["request"]["adjustment"] == "RAW"
                         else "CAUSAL_ACTION_AWARE_RETURN_INDEX",
                         "kalman_settings": {"process_variance": config["process_variance"],
                                             "observation_variance": config["observation_variance"]},
                         "trading_approved": False}
                model_id = digest(state)
                models[model_id] = state
                rejection = None if selected else "COINTEGRATION_NOT_SIGNIFICANT_AFTER_CORRECTION"
                if coint["state"] != "AVAILABLE":
                    rejection = coint.get("code", "COINTEGRATION_UNAVAILABLE")
                elif adf["state"] != "AVAILABLE":
                    rejection = adf.get("code", "ADF_UNAVAILABLE")
                elif fitted.spread_std <= 1e-10 or abs(fitted.beta) > 20:
                    rejection = "SPREAD_OR_HEDGE_UNIDENTIFIABLE_WITHIN_BOUNDS"
                screening.append({"dependent": dependent, "independent": independent, "state": coint["state"],
                                  "rejection_reason": rejection,
                                  "model_id": model_id, "selected": selected, "pvalue": coint["pvalue"],
                                  "bonferroni_pvalue": adjusted_p, "adf_pvalue": adf["pvalue"],
                                  "half_life_steps": reversion["half_life_steps"]})
                if selected:
                    accepted.append((dependent, independent, fitted, model_id))
            except ValueError:
                screening.append({"dependent": dependent, "independent": independent, "state": "ERROR",
                                  "selected": False, "code": "PAIR_UNIDENTIFIABLE_OR_NUMERICALLY_INVALID"})
        missing_windows += len(panel)-len(candidates)
        folds.append({"fold": fold_id, "training_start": train_stamps[0], "training_end": train_stamps[-1],
                      "training_cutoff": stamps[start], "test_end": stamps[end-1], "embargo_observations": embargo,
                      "family_size": family, "screening": screening,
                      "selection_method": "TRAINING_ONLY_ENGLE_GRANGER_BONFERRONI",
                      "adf_interpretation": "DESCRIPTIVE_RESIDUAL_TEST_POST_SELECTION_NOT_VALIDATED"})
        # Causal filters restart from the preceding fold's OLS fit. Only observed
        # current/prior prices enter a given innovation; no RTS smoothing.
        active: dict[tuple[str, str], list[Any]] = {}
        for dependent, independent, fitted, model_id in accepted:
            active[(dependent, independent)] = [fitted, model_id, [], []]
        for index in range(start, end):
            stamp = stamps[index]
            latest_proposals = []
            all_pairs = sorted(set(active) | set(prior_weights))
            for dependent, independent in all_pairs:
                pair = dependent, independent
                previous = prior_weights.get(pair, {dependent: 0., independent: 0.})
                weights = {dependent: 0., independent: 0.}
                zscore = innovation = beta = uncertainty = None
                status, regime, risk_multiplier = "UNAVAILABLE", "UNKNOWN", 0.
                model_id = active[pair][1] if pair in active else None
                current = [panel[identity].get(stamp) for identity in pair]
                continuous = all(current) and all(aware_timestamp(row["pair_segment_start"]) <= aware_timestamp(train_stamps[0])
                                                  for row in current)
                if pair in active and not continuous:
                    # Eligibility/observation gaps invalidate this fitted path.
                    # A later segment must earn a new complete training window;
                    # never reconnect a reset adjusted index to an older fit.
                    active.pop(pair)
                if pair in active and all(current):
                    fitted, model_id, test_y, test_x = active[pair]
                    test_y.append(current[0]["pair_price"])
                    test_x.append(current[1]["pair_price"])
                    # The existing kernel requires >=3 observations; prepend
                    # the final three training values and process them causally.
                    warm = train_stamps[-3:]
                    yy = [panel[dependent][t]["pair_price"] for t in warm] + test_y
                    xx = [panel[independent][t]["pair_price"] for t in warm] + test_x
                    filtered = kalman_hedge(yy, xx, process_variance=config["process_variance"],
                        observation_variance=config["observation_variance"], initial_alpha=fitted.alpha,
                        initial_beta=fitted.beta)
                    innovation = float(np.asarray(filtered["innovation"], dtype=float)[-1])
                    uncertainty = float(np.asarray(filtered["innovation_variance"], dtype=float)[-1])
                    beta = float(np.asarray(filtered["beta"], dtype=float)[-1])
                    zscore = innovation/math.sqrt(uncertainty)
                    # Regime is descriptive trailing return volatility, not an
                    # HMM state or an independently calibrated probability.
                    training_returns = np.diff(np.log([panel[dependent][t]["pair_price"] for t in train_stamps]))
                    trail = np.diff(np.log(yy[-9:]))
                    baseline = float(np.std(training_returns, ddof=1))
                    high = len(trail) >= 3 and float(np.std(trail, ddof=1)) > 1.5*baseline
                    regime, risk_multiplier = ("HIGH_TRAILING_VOLATILITY", .5) if high else ("BASE_TRAILING_VOLATILITY", 1.)
                    if all(math.isfinite(v) for v in (zscore, beta, uncertainty)) and abs(beta) <= 20:
                        status = "AVAILABLE"
                        if abs(zscore) >= config["entry_z"]:
                            direction = -1. if zscore > 0 else 1.
                            scale = config["max_gross"]*risk_multiplier/max(1, len(accepted))/(1+abs(beta))
                            weights = {dependent: direction*scale, independent: -direction*beta*scale}
                turnover = sum(abs(weights[identity]-previous[identity]) for identity in pair)
                cost = turnover*config["cost_bps"]/10000
                gross_return = borrow = None
                if index+1 < len(stamps):
                    future = [panel[identity].get(stamps[index+1]) for identity in pair]
                    if all(future) and all(current):
                        # Consecutive frozen decision marks form the return.
                        # RAW return_1 can revise the prior mark after a late
                        # correction, so must not replace the price actually
                        # used for the preceding hypothetical decision.
                        step_returns = [after["pair_price"]/before["pair_price"]-1
                                        for before, after in zip(current, future, strict=True)]
                        gross_return = sum(weights[identity]*value for identity, value in zip(pair, step_returns, strict=True))
                        years = (aware_timestamp(stamps[index+1])-aware_timestamp(stamp)).total_seconds()/(365.25*86400)
                        borrow = sum(max(0., -value) for value in weights.values())*config["borrow_bps"]/10000*years
                    if gross_return is None and any(weights.values()):
                        missing_outcomes += 1
                else:
                    # Explicit mark-to-final-clock liquidation, no final future
                    # return or hidden zero imputation.
                    gross_return, borrow = 0., 0.
                    cost += sum(abs(v) for v in weights.values())*config["cost_bps"]/10000
                if not any(weights.values()):
                    gross_return, borrow = 0., 0.
                net_return = None if gross_return is None else gross_return-cost-borrow
                row = {"timestamp": stamp, "dependent": dependent, "independent": independent,
                       "fold": fold_id, "model_id": model_id, "state": status,
                       "innovation": innovation, "innovation_zscore": zscore, "kalman_beta": beta,
                       "innovation_variance": uncertainty, "regime": regime,
                       "regime_method": "DESCRIPTIVE_TRAILING_VOLATILITY_NOT_HMM",
                       "weights": weights, "turnover": turnover, "transaction_cost_return": cost,
                       "borrow_cost_return": borrow, "gross_return": gross_return, "net_return": net_return,
                       "outcome_at": stamps[index+1] if index+1 < len(stamps) else None,
                       "outcome_state": "AVAILABLE" if gross_return is not None else "UNAVAILABLE",
                       "source_sha256": [value["source_sha256"] for value in current if value],
                       "universe_sha256": [value["universe_sha256"] for value in current if value],
                       "trading_approved": False}
                rows.append(row)
                prior_weights[pair] = weights
                if status == "AVAILABLE" and any(weights.values()):
                    latest_proposals.append({"timestamp": stamp, "model_id": model_id, "dependent": dependent,
                        "independent": independent, "weights": weights, "state": "AVAILABLE",
                        "prediction_kind": "UNAPPROVED_PAIR_TARGET_WEIGHTS",
                        "uncertainty": {"innovation_variance": uncertainty,
                                        "calibrated_confidence": False},
                        "input_classification": dataset["classification"], "trading_approved": False})
        if len(rows) > 500:
            raise ValueError("pair evaluation exceeds bounded experiment size")
    period_returns = []
    for stamp in sorted({row["timestamp"] for row in rows}, key=aware_timestamp):
        values = [r["net_return"] for r in rows if r["timestamp"] == stamp]
        period_returns.append(None if any(v is None for v in values) else sum(float(v) for v in values if v is not None))
    known_returns = [float(value) for value in period_returns if value is not None]
    simulated_return = (None if not period_returns or len(known_returns) != len(period_returns)
                        or any(value <= -1 for value in known_returns)
                        else float(np.prod([1+value for value in known_returns])-1))
    body = {"schema_version": IMPLEMENTATION, "dataset_id": dataset["dataset_id"], "config": config,
            "source": dataset["source"], "as_of": dataset["as_of"],
            "input_classification": dataset["classification"], "folds": folds,
            "model_ids": sorted(models), "rows": rows, "proposals": latest_proposals,
            "usable_for_portfolio": bool(latest_proposals),
            "metrics": {"folds": len(folds), "fitted_models": len(models), "evaluations": len(rows),
                        "selected_pair_folds": sum(r["selected"] for f in folds for r in f["screening"]),
                        "missing_training_memberships": missing_windows, "missing_held_outcomes": missing_outcomes,
                        "hypothetical_compounded_net_return": simulated_return,
                        "transaction_cost_return_sum": sum(r["transaction_cost_return"] for r in rows),
                        "trading_approved": False},
            "provenance": {"dataset_policy": dataset["provenance"],
                           "revision_policy": "FROZEN_AT_EACH_ORIGINAL_DECISION",
                           "model_scope": "RECONSTRUCTED_RESEARCH_NOT_HISTORICALLY_PUBLISHED",
                           "return_basis": "FROZEN_DECISION_MARK_RATIO_RAW_OR_CAUSAL_ACTION_AWARE_INDEX",
                           "backtest": "HYPOTHETICAL_FRACTIONAL_CONSTANT_DOLLAR_WEIGHTS_NOT_SHARE_LEDGER",
                           "costs": "TURNOVER_BPS_PLUS_SHORT_BORROW_ACT_365_25_AND_FINAL_LIQUIDATION",
                           "regime": "CAUSAL_TRAILING_VOLATILITY_HEURISTIC_NOT_HMM"},
            "trading_approved": False, "economically_validated": False}
    if simulated_return is not None and not math.isfinite(simulated_return):
        raise ValueError("pair backtest numerical overflow")
    return {"run_id": digest(body), **body}, models
