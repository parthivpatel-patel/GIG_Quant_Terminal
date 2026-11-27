"""Shared bounded CPU contract for expansion-inventory research families.

A registry entry declares one actual fit and one non-executable JSON replay.
Generic code owns admission, training-only transforms, state identity, replay
verification and the research-only boundary. Nothing here calibrates
uncertainty, approves allocation or promotes a model.
"""
from __future__ import annotations

import hashlib
import importlib
import importlib.util
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import scipy
from threadpoolctl import threadpool_limits

from gig.ml.gaussian_process import _array, _names, _scaled
from gig.ml.svm import _normalizer
from gig.service.serialization import dumps, loads

IMPLEMENTATION = "gig.bounded-family.v1"
MAX_FITS, MAX_PREDICT = 16, 2000
FAMILIES: dict[str, Family] = {}
_FIELDS = ("model", "features", "data", "training_x", "training_y", "mean", "scale", "active_indices",
           "target_mean", "target_scale", "seed", "parameters", "outputs", "payload")


def unavailable(method: str, unit: str = "arithmetic_return") -> Callable[[dict[str, Any]], dict[str, Any]]:
    """Null is the honest uncertainty when a method has no predictive distribution."""
    def build(_outputs: dict[str, Any]) -> dict[str, Any]:
        return {"value": None, "state": "UNAVAILABLE_NO_CALIBRATED_PREDICTIVE_INTERVAL", "method": method,
                "unit": unit, "calibrated": False, "predictive_interval": False}
    return build


def return_score(observed, outputs, train_target, _payload) -> dict[str, Any]:
    """Out-of-sample error against the training-mean forecast, per series when 2-D."""
    values = outputs["prediction"]
    return {"mse": float(np.mean((observed-values)**2)), "mae": float(np.mean(np.abs(observed-values))),
            "training_mean_baseline_mse": float(np.mean((observed-np.mean(train_target, axis=0))**2))}


@dataclass(frozen=True)
class Family:
    """Fixed declaration of one bounded variant; variants may share a family number."""
    key: str
    number: int
    capability: str
    label: str
    scope: str
    kind: str
    fit: Callable[..., tuple[dict[str, Any], np.ndarray | None]]
    replay: Callable[[dict[str, Any], np.ndarray], np.ndarray]
    backend: str
    parameters: dict[str, Any]
    requires: tuple[str, ...] = ("sklearn",)
    outputs: tuple[str, ...] = ("prediction",)
    affine: tuple[str, ...] = ("prediction",)
    scaled: tuple[str, ...] = ()
    data: str = "FEATURES"
    supervised: bool = True
    standardize: bool = True
    joint: bool = False
    min_train: int = 20
    max_train: int = 256
    max_columns: int = 16
    tolerance: float = 1e-9
    check: Callable[[dict[str, Any], np.ndarray, np.ndarray | None], None] | None = None
    uncertainty: Callable[[dict[str, Any]], dict[str, Any]] = unavailable("NO_PREDICTIVE_DISTRIBUTION")
    score: Callable[..., dict[str, Any]] = return_score
    return_units: bool = True
    license: str = "BSD-3-Clause"

    @property
    def fit_state(self) -> str:
        return "FITTED_BOUNDED_" + self.key.upper()

    def available(self) -> bool:
        return all(importlib.util.find_spec(name) is not None for name in self.requires)


def register(family: Family) -> Family:
    if family.key in FAMILIES or family.data not in {"FEATURES", "SERIES", "PANEL"}:
        raise ValueError("duplicate or unsupported bounded family declaration")
    FAMILIES[family.key] = family
    return family


def _fingerprint(state: dict[str, Any]) -> str:
    return hashlib.sha256(dumps({k: state[k] for k in _FIELDS}).encode()).hexdigest()


def _code(family: Family) -> str:
    digest = hashlib.sha256(Path(__file__).read_bytes())
    digest.update(Path(str(sys.modules[family.fit.__module__].__file__)).read_bytes())
    return digest.hexdigest()


def _versions(family: Family) -> dict[str, str]:
    versions = {"numpy": np.__version__, "scipy": scipy.__version__}
    for name in family.requires:
        versions[name] = str(getattr(importlib.import_module(name), "__version__", "UNKNOWN"))
    return versions


def _transform(family: Family, values: np.ndarray):
    """Training-only transforms; constant feature columns are excluded, never imputed."""
    if family.standardize:
        mean, scale = _normalizer(values)
    else:
        if np.any(values < 0):
            raise ValueError("this family admits only nonnegative features; signed inputs are never shifted")
        mean, scale = np.zeros(values.shape[1]), np.where(values.max(0) > 0, values.max(0), 1.)
    active = np.flatnonzero(np.any(values != values[0], axis=0)).tolist()
    if not active or (family.data != "FEATURES" and len(active) != values.shape[1]):
        raise ValueError("bounded family requires varying training inputs; constant series are unsupported")
    return mean, scale, active


def _targets(family: Family, values: np.ndarray, y, mean, scale):
    if family.data != "FEATURES":
        return mean.copy(), scale.copy(), _scaled(values, mean, scale)
    if not family.supervised:
        return np.zeros(1), np.ones(1), None
    target = _array(y, rows=family.max_train)
    if len(target) != len(values):
        raise ValueError("paired bounded training observations required")
    tmean, tscale = _normalizer(target[:, None])
    return tmean, tscale, (target-tmean[0])/tscale[0]


def _original(family: Family, columns: np.ndarray, tmean: np.ndarray, tscale: np.ndarray) -> np.ndarray:
    """Return standardized replay columns in admitted units.

    Shapes: (rows, outputs), or (rows, series, outputs) for a per-series PANEL
    family, whose columns each use their own training location and scale.
    """
    result = np.array(columns, dtype=float, copy=True)
    panel = family.data == "PANEL" and not family.joint
    if result.shape[1:] != ((len(tmean), len(family.outputs)) if panel else (len(family.outputs),)):
        raise ValueError("bounded family replay shape differs from declared outputs")
    shift, spread = (tmean, tscale) if panel else (tmean[0], tscale[0])
    for index, name in enumerate(family.outputs):
        if name in family.affine:
            result[..., index] = result[..., index]*spread + shift
        elif name in family.scaled:
            result[..., index] = result[..., index]*spread
    return result


def validate(state: Any):
    if (not isinstance(state, dict) or state.get("implementation") != IMPLEMENTATION
            or state.get("model") not in FAMILIES or state.get("device") != "cpu"
            or type(state.get("threads")) is not int or state["threads"] != 1
            or state.get("trading_approved") is not False or state.get("production_promoted") is not False
            or state.get("interval_calibrated") is not False or not set(_FIELDS) <= state.keys()):
        raise ValueError("unsupported bounded family state or research execution boundary")
    family = FAMILIES[state["model"]]
    actual = state["parameters"]
    if (state.get("family_number") != family.number or state.get("fit_state") != family.fit_state
            or state["data"] != family.data or state["outputs"] != list(family.outputs)
            or not isinstance(actual, dict) or set(actual) != set(family.parameters)
            or any(type(actual[k]) is not type(v) or actual[k] != v for k, v in family.parameters.items())
            or type(state["seed"]) is not int or not 0 <= state["seed"] < 2**31
            or not isinstance(state["payload"], dict)):
        raise ValueError("bounded family fixed declaration, parameters or payload differ")
    names = _names(state["features"])
    if len(names) > family.max_columns:
        raise ValueError("bounded family column budget exceeded")
    x = _array(state["training_x"], rows=family.max_train, columns=len(names))
    mean, scale = _array(state["mean"], rows=len(names)), _array(state["scale"], rows=len(names))
    expected_mean, expected_scale, active = _transform(family, x)
    y = state["training_y"]
    if (not family.min_train <= len(x) or state["active_indices"] != active
            or type(state.get("training_rows")) is not int or state["training_rows"] != len(x)
            or not np.array_equal(mean, expected_mean) or not np.array_equal(scale, expected_scale)
            or (y is None) != (family.data != "FEATURES" or not family.supervised)
            or state.get("state_sha256") != _fingerprint(state)):
        raise ValueError("bounded family dimensions, training transforms or identity differ")
    tmean, tscale, target = _targets(family, x, y, mean, scale)
    if (not np.array_equal(_array(state["target_mean"], rows=len(tmean)), tmean)
            or not np.array_equal(_array(state["target_scale"], rows=len(tscale)), tscale)):
        raise ValueError("bounded family target transforms must come from training observations")
    if family.check is not None:
        family.check(state["payload"], _scaled(x, mean, scale)[:, active], target)
    return family, names, mean, scale, active, tmean, tscale


def predict_family(state: dict[str, Any], x: Any, names: list[str]) -> np.ndarray:
    """Replay saved JSON. SERIES/PANEL inputs are chronological histories and each
    output row may depend only on rows at or before it."""
    if _names(names) != state.get("features"):
        raise ValueError("bounded family feature identities/order differ")
    values = _array(x, rows=MAX_PREDICT, columns=len(names))
    with threadpool_limits(limits=1):
        family, _, mean, scale, active, tmean, tscale = validate(state)
        if not family.standardize and np.any(values < 0):
            raise ValueError("this family admits only nonnegative features at inference")
        columns = np.asarray(family.replay(state["payload"], _scaled(values, mean, scale)[:, active]), dtype=float)
    result = _original(family, columns, tmean, tscale)
    if len(result) != len(values) or np.isinf(result).any() or np.nanmax(np.abs(result), initial=0.) > 1e12:
        raise ValueError("bounded family outputs exceed numeric bounds")
    return result


def admit(key: str, x: Any, y: Any, names: list[str], seed: int = 42):
    """Every admission check of a fit, without fitting; used to preflight whole work plans."""
    family = FAMILIES.get(key)
    if family is None or isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**31:
        raise ValueError("registered bounded family and bounded seed required")
    names = _names(names)
    if len(names) > family.max_columns:
        raise ValueError("bounded family column budget exceeded")
    values = _array(x, rows=family.max_train, columns=len(names))
    if len(values) < family.min_train:
        raise ValueError(f"{key} requires {family.min_train}..{family.max_train} training observations")
    if not family.available():
        raise ImportError(f"actual {family.backend} library required; no fallback estimator")
    mean, scale, active = _transform(family, values)
    tmean, tscale, target = _targets(family, values, y, mean, scale)
    return family, names, values, mean, scale, active, tmean, tscale, target


def fit_family(key: str, x: Any, y: Any, names: list[str], seed: int = 42) -> dict[str, Any]:
    family, names, values, mean, scale, active, tmean, tscale, target = admit(key, x, y, names, seed)
    z = _scaled(values, mean, scale)[:, active]
    with threadpool_limits(limits=1):
        payload, expected = family.fit(z, target, seed)
    payload = loads(dumps(payload))
    state = {"implementation": IMPLEMENTATION, "model": key, "family_number": family.number,
             "features": names, "data": family.data, "training_x": values.tolist(),
             "training_y": None if family.data != "FEATURES" or not family.supervised else _array(y, rows=family.max_train).tolist(),
             "mean": mean.tolist(), "scale": scale.tolist(), "active_indices": active,
             "constant_features": [names[i] for i in range(len(names)) if i not in active],
             "constant_feature_policy": "EXCLUDED_FROM_MODEL_NOT_IMPUTED",
             "target_mean": tmean.tolist(), "target_scale": tscale.tolist(), "seed": seed,
             "parameters": dict(family.parameters), "outputs": list(family.outputs), "payload": payload,
             "training_rows": len(values), "device": "cpu", "threads": 1, "fit_state": family.fit_state,
             "fit_backend": family.backend, "inference_backend": IMPLEMENTATION+".json-replay",
             "preprocessing": "TRAINING_ONLY_POPULATION_STANDARDIZATION" if family.standardize
                              else "TRAINING_ONLY_MAXIMUM_SCALING_NO_SHIFT",
             "library_versions": _versions(family), "code_sha256": _code(family), "license": family.license,
             "interval_calibrated": False, "trading_approved": False, "production_promoted": False}
    state["state_sha256"] = _fingerprint(state)
    actual = predict_family(state, values, names)
    if expected is not None:
        wanted = _original(family, np.asarray(expected, dtype=float), tmean, tscale)
        known = np.isfinite(wanted)
        if wanted.shape != actual.shape or not np.allclose(actual[known], wanted[known], rtol=family.tolerance,
                                                         atol=family.tolerance*(1+float(np.max(np.abs(tscale))))):
            raise ValueError(f"{key} JSON replay differs from the actual library fit")
    return state
