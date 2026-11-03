"""Observation-order research on acquired reference quotes, never historical PIT.

Source dates are preserved as dates. Acquisition times are not moved into the
past to satisfy a trading-clock contract. Existing numerical kernels are reused
on explicit chronological train/test arrays instead of fabricating OHLCV bars.
"""
from __future__ import annotations

import hashlib
from dataclasses import asdict
from importlib.metadata import version

import numpy as np

from gig.ml.regimes import GaussianHMM, fit_gaussian_hmm
from gig.ml.state_space import kalman_filter
from gig.ml.volatility import GARCH11, fit_garch11
from gig.service.serialization import dumps
from gig.service.workspace import _fields

VERSION = "reference-observation-order-research-v1"
MODELS = {"ridge", "lightgbm", "bilstm", "hmm", "garch", "kalman"}
FEATURES = ["return_1", "momentum_5", "volatility_5", "log_reference_level"]


def digest(value):
    return hashlib.sha256(dumps(value).encode()).hexdigest()


def configuration(payload):
    required = {"snapshot_id", "model"}
    _fields(payload, required | {"training_observations", "test_observations", "embargo", "seed"}, required)
    result = dict(payload)
    if not isinstance(result["snapshot_id"], str) or len(result["snapshot_id"]) != 64 or any(c not in "0123456789abcdef" for c in result["snapshot_id"]):
        raise ValueError("canonical admitted reference identity required")
    if not isinstance(result["model"], str) or result["model"] not in MODELS:
        raise ValueError("explicit supported CPU reference model required")
    for key, default, low, high in (("training_observations", 60, 20, 200), ("test_observations", 20, 5, 60), ("embargo", 1, 0, 10), ("seed", 17, 0, 2**31-1)):
        result.setdefault(key, default)
        if isinstance(result[key], bool) or not isinstance(result[key], int) or not low <= result[key] <= high:
            raise ValueError("bounded integer reference-model settings required")
    return result


def _features(snapshot):
    groups = {}
    for row in snapshot["rows"]:
        groups.setdefault(row["instrument_id"], []).append(row)
    output = {}
    for instrument, rows in sorted(groups.items()):
        rows.sort(key=lambda row: row["observation_date"])
        if any(row["value"] is None for row in rows):
            raise ValueError("missing reference quotes require a different explicit window; no imputation or security dropping")
        prices = np.asarray([row["value"] for row in rows], dtype=float)
        returns = prices[1:]/prices[:-1]-1
        features = []
        for index in range(5, len(rows)-1):
            window = returns[index-5:index]
            features.append({"instrument_id": instrument, "observation_date": rows[index]["observation_date"],
                "target_observation_date": rows[index+1]["observation_date"],
                "features": [float(returns[index-1]), float(prices[index]/prices[index-5]-1), float(window.std(ddof=1)), float(np.log(prices[index]))],
                "target": float(returns[index]), "quote_currency": rows[index]["quote_currency"]})
        output[instrument] = features
    if len({tuple(row["observation_date"] for row in rows) for rows in output.values()}) != 1:
        raise ValueError("reference currency dates must align explicitly; no inferred calendar rows")
    return output


def _fit(model, train, config):
    x = np.asarray([row["features"] for row in train])
    y = np.asarray([row["target"] for row in train])
    mean, scale = x.mean(axis=0), x.std(axis=0)
    scale = np.where(scale > 0, scale, 1.)
    state = {"model": model, "feature_names": FEATURES, "mean": mean.tolist(), "scale": scale.tolist(), "training_targets": y.tolist(), "fit_state": "CONVERGED"}
    if model == "ridge":
        design = np.column_stack((np.ones(len(x)), (x-mean)/scale))
        penalty = np.eye(design.shape[1])*.001
        penalty[0, 0] = 0
        state["coefficients"] = np.linalg.solve(design.T@design+penalty, design.T@y).tolist()
    elif model == "garch":
        fitted = fit_garch11(y, max_iterations=200)
        state.update(parameters=asdict(fitted.model), fit_state="CONVERGED" if fitted.converged else "NONCONVERGED", stop_reason=fitted.stop_reason)
    elif model == "hmm":
        fitted = fit_gaussian_hmm(y, 2, max_iterations=100)
        state.update(parameters={name: getattr(fitted.model, name).tolist() for name in ("initial", "transition", "means", "variances")},
                     fit_state="CONVERGED" if fitted.converged else "NONCONVERGED", stop_reason=fitted.stop_reason)
    elif model == "kalman":
        state["variance"] = max(float(np.var(y)), 1e-12)
    elif model == "bilstm":
        from gig.ml.sequence import _cpu, _encode_weights, _network, _torch
        torch = _torch()
        z = (x-mean)/scale
        length = 4
        with _cpu(torch, config["seed"]):
            network = _network(torch, len(FEATURES), 4)
            optimizer = torch.optim.Adam(network.parameters(), lr=.01)
            tensors = torch.tensor(np.stack([z[i-length+1:i+1] for i in range(length-1, len(z))]), dtype=torch.float32)
            targets = torch.tensor(y[length-1:], dtype=torch.float32)
            network.train()
            for _ in range(4):
                optimizer.zero_grad()
                loss = torch.mean((network(tensors)-targets)**2)
                if not torch.isfinite(loss):
                    raise ValueError("nonfinite CPU sequence training")
                loss.backward()
                optimizer.step()
            state["parameters"] = _encode_weights(network)
            state["training_context"] = z[-3:].tolist()
        state["library_version"] = version("torch")
    return state


def _predict(state, test):
    model = state["model"]
    if state["fit_state"] != "CONVERGED":
        return [None]*len(test)
    x = np.asarray([row["features"] for row in test])
    mean, scale = np.asarray(state["mean"]), np.asarray(state["scale"])
    if model == "ridge":
        return (np.column_stack((np.ones(len(x)), (x-mean)/scale))@np.asarray(state["coefficients"])).tolist()
    if model == "bilstm":
        from gig.ml.sequence import IMPLEMENTATION, _cpu, _restore, _torch
        torch = _torch()
        values = np.concatenate((np.asarray(state["training_context"]), (x-mean)/scale))
        encoded = {"implementation": IMPLEMENTATION, "features": FEATURES, "configuration": {"hidden_size": 4, "sequence_length": 4}, "parameters": state["parameters"]}
        with _cpu(torch, 0), torch.no_grad():
            network = _restore(torch, encoded)
            return network(torch.tensor(np.stack([values[i:i+4] for i in range(len(x))]), dtype=torch.float32)).numpy().astype(float).tolist()
    predictions = []
    history = list(state["training_targets"])+list(state.get("conditioning_returns", []))
    for row in test:
        history.append(row["features"][0])
        if model == "garch":
            value = GARCH11(**state["parameters"]).forecast(history)[0]
        elif model == "hmm":
            fitted = GaussianHMM(**state["parameters"])
            filtered = fitted.infer(history).filtered_probabilities[-1]
            value = float(filtered@fitted.transition@fitted.means[:, 0])
        else:
            variance = state["variance"]
            filtered = kalman_filter(np.asarray(history)[:, None], [[1.]], [[1.]], [[variance*.01]], [[variance]], [history[0]], [[variance]])
            value = filtered.filtered_mean[-1, 0]
        predictions.append(float(value))
    return predictions


def fit_reference(snapshot, config):
    groups = _features(snapshot)
    count = min(len(rows) for rows in groups.values())
    minimum, test_size, gap = config["training_observations"], config["test_observations"], max(1, config["embargo"])
    if count < minimum+gap+test_size:
        raise ValueError("insufficient complete reference observations for chronological training, purging and holdout")
    models, output, folds = {}, [], []
    for fold, start in enumerate(range(minimum+gap, min(count, minimum+gap+4*test_size), test_size)):
        end = min(count, start+test_size)
        if config["model"] == "lightgbm":
            import lightgbm as lgb
            if len(groups) < 2:
                raise ValueError("reference ranking needs at least two currencies")
            train = sorted([r for rows in groups.values() for r in rows[:start-gap]], key=lambda r: (r["observation_date"], r["instrument_id"]))
            test = sorted([r for rows in groups.values() for r in rows[start:end]], key=lambda r: (r["observation_date"], r["instrument_id"]))
            x = np.asarray([r["features"] for r in train])
            y = np.asarray([r["target"] for r in train])
            labels = []
            for i in range(0, len(train), len(groups)):
                _, ranks = np.unique(y[i:i+len(groups)], return_inverse=True)
                labels.extend(ranks.tolist())
            mean, scale = x.mean(axis=0), np.maximum(x.std(axis=0), 1e-12)
            booster = lgb.train({"objective": "lambdarank", "verbosity": -1, "num_threads": 1, "deterministic": True, "force_col_wise": True,
                "seed": config["seed"], "min_data_in_leaf": 3, "num_leaves": 7, "label_gain": list(range(5))},
                lgb.Dataset((x-mean)/scale, label=labels, group=[len(groups)]*(len(train)//len(groups))), num_boost_round=32)
            state = {"model": "lightgbm", "fit_state": "CONVERGED", "booster_text": booster.model_to_string(), "mean": mean.tolist(), "scale": scale.tolist(), "library_version": lgb.__version__}
            predicted = booster.predict((np.asarray([r["features"] for r in test])-mean)/scale).tolist()
            batches = [("ALL_REQUESTED_REFERENCE_CURRENCIES", train, test, state, predicted)]
        else:
            batches = []
            for instrument, rows in sorted(groups.items()):
                train, test = rows[:start-gap], rows[start:end]
                state = _fit(config["model"], train, config)
                state["conditioning_returns"] = [row["features"][0] for row in rows[start-gap+1:start]]
                state["conditioning_observation_dates"] = [row["observation_date"] for row in rows[start-gap+1:start]]
                if config["model"] == "bilstm":
                    context = np.asarray([r["features"] for r in rows[start-3:start]])
                    state["training_context"] = ((context-np.asarray(state["mean"]))/np.asarray(state["scale"])).tolist()
                batches.append((instrument, train, test, state, _predict(state, test)))
        for instrument, train, test, state, predictions in batches:
            state.update({"implementation": VERSION, "snapshot_id": snapshot["snapshot_id"], "instrument_id": instrument,
                "training_end_observation_date": train[-1]["target_observation_date"], "first_test_observation_date": test[0]["observation_date"],
                "training_evidence_hash": digest(train), "research_scope": "RETROSPECTIVE_OBSERVATION_ORDER_NOT_HISTORICAL_AVAILABILITY"})
            model_id = digest(state)
            models[model_id] = state
            folds.append({"fold": fold, "model_id": model_id, "instrument_id": instrument, "training_rows": len(train),
                "training_end_observation_date": train[-1]["target_observation_date"], "first_test_observation_date": test[0]["observation_date"],
                "fit_state": state["fit_state"], "embargo_observations": gap})
            for row, prediction in zip(test, predictions, strict=True):
                output.append({"instrument_id": row["instrument_id"], "observation_date": row["observation_date"], "target_observation_date": row["target_observation_date"],
                    "quote_currency": row["quote_currency"], "prediction": prediction, "target_derived_return": row["target"], "model_id": model_id,
                    "fold": fold, "state": "AVAILABLE" if prediction is not None else "UNAVAILABLE", "fit_state": state["fit_state"]})
    output.sort(key=lambda r: (r["observation_date"], r["instrument_id"]))
    known = [r for r in output if r["prediction"] is not None]
    kind = "CONDITIONAL_VARIANCE_PER_REFERENCE_OBSERVATION" if config["model"] == "garch" else "CROSS_REFERENCE_RANK_SCORE" if config["model"] == "lightgbm" else "DERIVED_REFERENCE_RETURN_ESTIMATE"
    mse = float(np.mean([(r["prediction"]-r["target_derived_return"])**2 for r in known])) if known and config["model"] not in {"garch", "lightgbm"} else None
    return {"schema_version": VERSION, "snapshot_id": snapshot["snapshot_id"], "request": config, "model": config["model"],
        "as_of": snapshot["acquired_at"], "input_classification": snapshot["classification"], "rows": output, "folds": folds,
        "model_ids": sorted(models), "prediction_kind": kind, "feature_names": FEATURES,
        "metrics": {"predictions": len(known), "unavailable_predictions": len(output)-len(known), "heldout_mse": mse},
        "research_mode": "RETROSPECTIVE_OBSERVATION_ORDER_RESEARCH", "historical_point_in_time_validated": False,
        "historical_publication_time_state": "UNKNOWN", "volume_state": "UNAVAILABLE", "capacity_state": "UNKNOWN",
        "portfolio_pnl_state": "UNAVAILABLE", "funded_allocation_eligible": False, "trading_approved": False, "economically_validated": False,
        "actual_acquired_at": snapshot["acquired_at"], "observation_clock": "PROVIDER_DATE_ONLY_NO_INTRADAY_TIME_INVENTED",
        "units": "FOREIGN_QUOTE_CURRENCY_PER_EUR; derived fractional changes are not USD account P&L",
        "attribution": snapshot["attribution"], "library_versions": {"numpy": np.__version__, "scipy": version("scipy")}}, models
