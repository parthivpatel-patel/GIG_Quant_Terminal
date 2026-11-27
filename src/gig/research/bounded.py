"""Purged chronological research for every registered bounded family.

FEATURES families use admitted feature rows per instrument. SERIES families
use the admitted causal one-observation return history per instrument. PANEL
families use the balanced cross-instrument return matrix. Every fold is
preflighted before the first fit; no output is portfolio-usable.
"""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Any

import numpy as np
import pandas as pd

from gig.ml.bounded import FAMILIES, MAX_FITS, MAX_PREDICT, admit, fit_family, predict_family
from gig.research.model_pipeline import _folds, _prediction, _training_vintage, _wrapper, identity
from gig.research.walkforward import WalkForwardSplit
from gig.timestamps import aware_timestamp

SERIES_NAME = "return_1"


def _groups(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row["instrument_id"]].append(row)
    return dict(sorted(groups.items()))


def _returns(dataset, rows):
    """Attach the admitted trailing return whose next value a one-step family forecasts."""
    if dataset.get("request", dataset.get("canonical_request", {})).get("horizon", 1) != 1:
        raise ValueError("return-history families require a one-observation target horizon")
    panel = {(r["instrument_id"], r["timestamp"]): r for r in dataset["panel_rows"]}
    output = []
    for row in rows:
        value = panel.get((row["instrument_id"], row["timestamp"]), {}).get(SERIES_NAME)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError("return-history family requires an admitted causal one-observation return")
        output.append({**row, SERIES_NAME: float(value)})
    return output


def _balanced(rows):
    """Dates x instruments matrix; an unbalanced panel is refused, never filled."""
    groups = _groups(rows)
    ids = list(groups)
    dates = [r["timestamp"] for r in groups[ids[0]]]
    if len(ids) < 2 or any([r["timestamp"] for r in groups[i]] != dates for i in ids):
        raise ValueError("panel family requires two or more instruments observed on identical decision times")
    return ids, dates, [[groups[i][t] for i in ids] for t in range(len(dates))]


def _prefix(train, own, family):
    """History families replay recursions, so purged training must be one unbroken prefix."""
    if [r["timestamp"] for r in train] != [r["timestamp"] for r in own[:len(train)]]:
        raise ValueError(f"{family.key} requires a contiguous purged training history")


def _plans(family, dataset, rows, names, config):
    plans: list[dict[str, Any]] = []
    if family.data == "PANEL":
        ids, dates, grid = _balanced(_returns(dataset, rows))
        index = pd.DatetimeIndex([aware_timestamp(t) for t in dates])
        split = list(WalkForwardSplit(config["min_train"], config["test_size"], config["embargo"]).split(index))
        if not split:
            raise ValueError("panel family requires one complete date-grouped fold")
        for number, fold in enumerate(split):
            members = [g for g, t in zip(grid, index, strict=True) if t in fold.train_index]
            train = [known for g in members if None not in (known := [_training_vintage(r, fold.test_start) for r in g])]
            _prefix([g[0] for g in train], [g[0] for g in grid], family)
            query = [g for g, t in zip(grid, index, strict=True) if t <= fold.test_end]
            plans.append({"instrument": "*", "fold": number, "schedule": fold, "names": ids,
                          "candidates": len(members)*len(ids), "train_rows": [r for g in train for r in g],
                          "x": [[r[SERIES_NAME] for r in g] for g in train], "y": None,
                          "train_target": np.array([[r["target"] for r in g] for g in train]),
                          "query": [[r[SERIES_NAME] for r in g] for g in query],
                          "test": [(i, g) for i, g in enumerate(query) if aware_timestamp(g[0]["timestamp"]) in fold.test_index],
                          "training_timestamps": [g[0]["timestamp"] for g in train]})
    else:
        source = rows if family.data == "FEATURES" else _returns(dataset, rows)
        for instrument, own in _groups(source).items():
            for number, fold in enumerate(_folds(own, config)):
                candidates = [r for r in own if aware_timestamp(r["timestamp"]) in fold.train_index]
                train = [v for r in candidates if (v := _training_vintage(r, fold.test_start)) is not None]
                plan = {"instrument": instrument, "fold": number, "schedule": fold, "candidates": len(candidates),
                        "train_rows": train, "train_target": np.array([r["target"] for r in train]),
                        "training_timestamps": [r["timestamp"] for r in train]}
                if family.data == "FEATURES":
                    test = [r for r in own if aware_timestamp(r["timestamp"]) in fold.test_index]
                    plan.update(names=names, x=[[r["features"][n] for n in names] for r in train],
                                y=[r["target"] for r in train] if family.supervised else None,
                                query=[[r["features"][n] for n in names] for r in test],
                                test=[(i, [r]) for i, r in enumerate(test)])
                else:
                    _prefix(train, own, family)
                    query = [r for r in own if aware_timestamp(r["timestamp"]) <= fold.test_end]
                    plan.update(names=[SERIES_NAME], x=[[r[SERIES_NAME]] for r in train], y=None,
                                query=[[r[SERIES_NAME]] for r in query],
                                test=[(i, [r]) for i, r in enumerate(query)
                                      if aware_timestamp(r["timestamp"]) in fold.test_index])
                plans.append(plan)
    if not 1 <= len(plans) <= MAX_FITS:
        raise ValueError(f"{family.key} work plan exceeds {MAX_FITS} total fits")
    for plan in plans:
        if not family.min_train <= len(plan["x"]) <= family.max_train:
            raise ValueError(f"{family.key} requires {family.min_train}..{family.max_train} purged training observations")
        if not plan["test"] or len(plan["query"]) > MAX_PREDICT or len(plan["names"]) > family.max_columns:
            raise ValueError(f"{family.key} holdout, replay or column budget failed")
        admit(family.key, plan["x"], plan["y"], plan["names"], config["seed"])
    return plans


def _cells(family, values, position, width):
    """Named outputs for each instrument at one replay row; missing values stay null."""
    block = values[position] if family.data == "PANEL" and not family.joint else np.repeat(values[position][None], width, axis=0)
    return [{name: (float(v) if math.isfinite(v) else None) for name, v in zip(family.outputs, cell, strict=True)}
            for cell in block]


def _extra(family, cell):
    extra = {"outputs": cell} if len(family.outputs) > 1 else {}
    if family.data == "PANEL" and family.joint:
        extra["output_scope"] = "JOINT_ACROSS_INSTRUMENTS_REPEATED_PER_ROW"
    return extra


def fit_persisted_family(dataset, rows, names, config):
    family = FAMILIES[config["model"]]
    plans = _plans(family, dataset, rows, names, config)
    models, predictions, folds = {}, [], []
    for plan in plans:
        fold = plan["schedule"]
        estimator = fit_family(family.key, plan["x"], plan["y"], plan["names"], config["seed"])
        maximum = max(aware_timestamp(r["label_end"]) for r in plan["train_rows"]).isoformat()
        state = _wrapper(dataset, family.key, plan["instrument"], estimator, fold.test_start.isoformat(), maximum,
                         fit_state=family.fit_state, training_timestamps=plan["training_timestamps"],
                         evaluation_end=fold.test_end.isoformat())
        model_id = identity(state)
        models[model_id] = state
        values = predict_family(estimator, plan["query"], plan["names"])
        positions = [i for i, _ in plan["test"]]
        observed = np.array([[r["target"] for r in group] for _, group in plan["test"]])
        chosen = values[positions]
        per_series = family.data == "PANEL" and not family.joint
        outputs = {name: chosen[..., k] if per_series else chosen[:, k][:, None] for k, name in enumerate(family.outputs)}
        if family.data != "PANEL":
            observed, outputs = observed[:, 0], {k: v[:, 0] for k, v in outputs.items()}
        folds.append({"fold": plan["fold"], "instrument_id": plan["instrument"], "model_id": model_id,
                      "train_count": len(plan["train_rows"]), "test_count": sum(len(g) for _, g in plan["test"]),
                      "purged_count": plan["candidates"]-len(plan["train_rows"]),
                      "test_start": fold.test_start.isoformat(), "test_end": fold.test_end.isoformat(),
                      "max_training_label_end": maximum, "fit_state": family.fit_state,
                      **family.score(observed, outputs, plan["train_target"], estimator["payload"]),
                      "uncertainty_calibrated": False, "economic_claim": "NONE"})
        for position, group in plan["test"]:
            for row, cell in zip(group, _cells(family, values, position, len(group)), strict=True):
                if cell["prediction"] is None:
                    raise ValueError(f"{family.key} produced no finite holdout output")
                predictions.append(_prediction(row, cell["prediction"], model_id, plan["fold"], family.key,
                                               family.uncertainty(cell), fit_state=family.fit_state,
                                               usable_for_portfolio=False, **_extra(family, cell)))
    return models, predictions, folds


def summarize(folds: list[dict[str, Any]]) -> dict[str, Any]:
    """Holdout-weighted fold scores; counts are summed. Negative findings stay visible."""
    total = sum(f["test_count"] for f in folds)
    reserved = {"fold", "train_count", "test_count", "purged_count"}
    summary: dict[str, Any] = {}
    for key in folds[0]:
        values = [f.get(key) for f in folds]
        if key in reserved or any(isinstance(v, bool) or not isinstance(v, (int, float)) for v in values):
            continue
        if key.endswith("_count"):
            summary[key] = int(sum(v for v in values if v is not None))
        elif len(set(values)) == 1:
            summary[key] = values[0]
        else:
            summary[key] = float(sum(v*f["test_count"] for v, f in zip(values, folds, strict=True))/total)
    return summary


def saved_family_rows(state, dataset, own, names):
    """No-refit inference rows for one saved bounded state on its admitted dataset.

    The original holdout is replayed on exactly the inputs used when it was
    first evaluated, so saved estimates are identical to the published ones;
    later rows are evaluated separately.
    """
    start, end = aware_timestamp(state["training_cutoff"]), aware_timestamp(state["evaluation_end"])
    estimator = state["estimator"]
    family = FAMILIES[state["model"]]

    def holdout(row):
        return start <= aware_timestamp(row["timestamp"]) <= end
    if family.data == "FEATURES":
        output = []
        for block in ([r for r in own if holdout(r)], [r for r in own if aware_timestamp(r["timestamp"]) > end]):
            if block:
                values = predict_family(estimator, [[r["features"][n] for n in names] for r in block], names)
                output += _rows(family, state, values, [(i, [r]) for i, r in enumerate(block)])
        return output
    else:
        own = _returns(dataset, own)
        if family.data == "SERIES":
            columns, grid = [SERIES_NAME], [[r] for r in own]
        else:
            columns, _, grid = _balanced(own)
        history = [[r[SERIES_NAME] for r in g] for g in grid]
        observed = {g[0]["timestamp"]: values for g, values in zip(grid, history, strict=True)}
        expected = dict(zip(state["training_timestamps"], estimator["training_x"], strict=True))
        if any(observed.get(k) != v for k, v in expected.items()):
            raise ValueError("saved history inference requires its unchanged admitted training history")
        inside = sum(aware_timestamp(g[0]["timestamp"]) <= end for g in grid)
        output = _rows(family, state, predict_family(estimator, history[:inside], columns),
                       [(i, g) for i, g in enumerate(grid[:inside]) if holdout(g[0])]) if inside else []
        if inside < len(grid):
            output += _rows(family, state, predict_family(estimator, history, columns), list(enumerate(grid))[inside:])
        return output


def _rows(family, state, values, groups):
    output = []
    for position, group in groups:
        for row, cell in zip(group, _cells(family, values, position, len(group)), strict=True):
            if cell["prediction"] is None:
                if aware_timestamp(row["timestamp"]) >= aware_timestamp(state["training_cutoff"]):
                    raise ValueError(f"{family.key} produced no finite saved output")
                continue
            output.append(dict({k: v for k, v in row.items() if k != SERIES_NAME}, prediction=cell["prediction"],
                               uncertainty=family.uncertainty(cell), extra=_extra(family, cell)))
    return output
