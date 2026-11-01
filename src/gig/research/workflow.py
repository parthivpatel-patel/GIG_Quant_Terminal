"""Bounded, reproducible research with purged folds and content-addressed models.

These are statistical research outputs requiring approval before any trading use.
Models are explicitly selected; an unavailable model never silently becomes a
baseline. This module performs no network access and downloads no model weights.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from gig.research.walkforward import WalkForwardSplit
from gig.artifacts import write_immutable
from gig.timestamps import aware_timestamp

_MAX_ROWS = 20_000
_MAX_FEATURES = 128
_MAX_FOLDS = 100


def _bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, allow_nan=False, ensure_ascii=False,
                      separators=(",", ":")).encode("utf-8")


def _hash(value: Any) -> str:
    return hashlib.sha256(_bytes(value)).hexdigest()


def _exists(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


def model_capabilities() -> dict[str, dict[str, Any]]:
    """Dependency presence is separate from implemented/trained/approved state."""
    dependencies = {"lightgbm": _exists("lightgbm"), "torch": _exists("torch"),
                    "transformers": _exists("transformers")}
    return {
        "ridge": {"state": "AVAILABLE", "implemented": True, "trained": False,
                  "trading_approved": False, "dependency_available": True},
        "lightgbm": {"state": "AVAILABLE" if dependencies["lightgbm"] else "UNAVAILABLE",
                     "implemented": True, "trained": False, "trading_approved": False,
                     "dependency_available": dependencies["lightgbm"]},
        "bilstm": {"state": "UNAVAILABLE", "implemented": False, "trained": False,
                   "trading_approved": False, "dependency_available": dependencies["torch"],
                   "reason": "No sequence training adapter/weights supplied in this workflow"},
        "finbert": {"state": "UNAVAILABLE", "implemented": True, "trained": False,
                    "trading_approved": False,
                    "dependency_available": dependencies["torch"] and dependencies["transformers"],
                    "load_verified": False, "downloads_enabled": False,
                    "reason": "Local-only adapter exists; no hash-verified FinBERT checkpoint loaded"},
    }


def _index(index: pd.Index) -> pd.DatetimeIndex:
    if not isinstance(index, pd.DatetimeIndex) or index.tz is None:
        raise ValueError("features require timezone-aware DatetimeIndex")
    result = index.tz_convert("UTC")
    if result.hasnans or not result.is_unique or not result.is_monotonic_increasing:
        raise ValueError("feature timestamps must be nonmissing, unique and increasing")
    return result


def _times(series: pd.Series, index: pd.DatetimeIndex, name: str) -> pd.DatetimeIndex:
    if not isinstance(series, pd.Series) or not series.index.equals(index):
        raise ValueError(f"{name} must align exactly to feature timestamps")
    values = []
    for value in series:
        values.append(aware_timestamp(value))
    return pd.DatetimeIndex(values)


class ModelRegistry:
    """Hash-verified JSON model states; no executable pickle deserialization."""

    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    def register(self, model_state: dict[str, Any]) -> str:
        payload = _bytes(model_state)
        identity = hashlib.sha256(payload).hexdigest()
        target = self.directory / f"{identity}.json"
        write_immutable(target, payload)
        return identity

    def load(self, identity: str) -> dict[str, Any]:
        if not isinstance(identity, str) or len(identity) != 64 or any(c not in "0123456789abcdef" for c in identity):
            raise ValueError("model identity must be lowercase SHA-256")
        payload = (self.directory / f"{identity}.json").read_bytes()
        if hashlib.sha256(payload).hexdigest() != identity:
            raise ValueError("registered model state failed hash verification")
        return json.loads(payload)

    def predict(self, identity: str, features: pd.DataFrame) -> list[float]:
        model = self.load(identity)
        if list(features.columns) != model["features"]:
            raise ValueError("inference feature names/order do not match registered model")
        if not 0 < len(features) <= _MAX_ROWS:
            raise ValueError("inference row bound exceeded or no rows supplied")
        values = features.to_numpy(dtype=float, copy=True)
        if not np.isfinite(values).all():
            raise ValueError("inference features must be finite")
        scaled = (values - np.array(model["mean"])) / np.array(model["scale"])
        if model["model"] == "ridge":
            prediction = scaled @ np.array(model["coefficients"]) + model["intercept"]
        elif model["model"] == "lightgbm":
            if not _exists("lightgbm"):
                raise ImportError("LightGBM is unavailable; no model substitution allowed")
            import lightgbm as lgb
            prediction = lgb.Booster(model_str=model["booster_text"]).predict(scaled)
        else:
            raise ValueError("unsupported registered model kind")
        if not np.isfinite(prediction).all():
            raise ValueError("model emitted nonfinite predictions")
        return np.asarray(prediction).tolist()


def drift_report(reference: Any, current: Any, *, bins: int = 10) -> dict[str, Any]:
    """Quantile-bin PSI and standardized mean shift; diagnostic, not a decision."""
    if not isinstance(bins, int) or isinstance(bins, bool) or not 2 <= bins <= 100:
        raise ValueError("bins must be an integer from 2 to 100")
    a, b = np.asarray(reference, dtype=float), np.asarray(current, dtype=float)
    if a.ndim != 1 or b.ndim != 1 or min(a.size, b.size) < 10 or max(a.size, b.size) > _MAX_ROWS:
        raise ValueError("drift needs two bounded one-dimensional samples of at least 10")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("drift samples must be finite")
    # Common positive scaling leaves quantile bins/PSI unchanged and prevents
    # overflow in interpolation, mean, variance, or a difference of huge means.
    magnitude = max(float(np.max(np.abs(a))), float(np.max(np.abs(b)))) or 1.
    scaled_a, scaled_b = a / magnitude, b / magnitude
    cuts = np.unique(np.quantile(scaled_a, np.linspace(0, 1, bins + 1)[1:-1]))
    edges = np.concatenate(([-np.inf], cuts, [np.inf]))
    # Jeffreys pseudocount avoids infinite PSI for unobserved bins.
    count_a, _ = np.histogram(scaled_a, edges)
    count_b, _ = np.histogram(scaled_b, edges)
    p = (count_a + .5) / (a.size + .5 * count_a.size)
    q = (count_b + .5) / (b.size + .5 * count_b.size)
    sigma = float(scaled_a.std())
    if sigma == 0 and np.any(a != a[0]):
        raise ValueError("reference variance underflows at the sample magnitude")
    shift = float((scaled_b.mean() - scaled_a.mean()) / sigma) if sigma > 0 else None
    if shift is not None and not math.isfinite(shift):
        raise ValueError("standardized drift shift exceeds finite numeric range")
    return {"psi": float(np.sum((q-p) * np.log(q/p))), "standardized_mean_shift": shift,
            "reference_count": int(a.size), "current_count": int(b.size), "bins": int(count_a.size),
            "binning": "reference_quantiles", "pseudocount": .5,
            "state": "AVAILABLE" if sigma > 0 else "UNKNOWN"}


class ResearchWorkflow:
    """Train-only normalization, label-end purge, OOS predictions and registry.

    All features at index t must be knowable at t. ``feature_available_at`` is
    explicit evidence when available; omission records a caller assumption.
    Labels may look forward, but a fold only trains on labels ending strictly
    before that fold's first test timestamp. Fold metrics are research evidence
    and do not establish implementable/economically validated returns.
    """

    def __init__(self, output_dir: str | Path) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.registry = ModelRegistry(self.output_dir / "models")

    def run(
        self, features: pd.DataFrame, target: pd.Series, *, label_end: pd.Series,
        feature_available_at: pd.Series | None = None, model: str = "ridge",
        min_train: int = 60, test_size: int = 20, embargo: int = 1,
        ridge_alpha: float = 1., seed: int = 42,
    ) -> dict[str, Any]:
        if not isinstance(features, pd.DataFrame) or not 1 <= features.shape[1] <= _MAX_FEATURES or not 1 <= len(features) <= _MAX_ROWS:
            raise ValueError("features exceed declared row/column bounds or are empty")
        if not all(isinstance(name, str) and 0 < len(name) <= 256 for name in features.columns) or not features.columns.is_unique:
            raise ValueError("feature names must be unique bounded strings")
        index = _index(features.index)
        if not isinstance(target, pd.Series) or not target.index.equals(features.index):
            raise ValueError("target index must align exactly to features")
        for name, value, low in (("min_train", min_train, 3), ("test_size", test_size, 1), ("embargo", embargo, 0)):
            if not isinstance(value, int) or isinstance(value, bool) or value < low:
                raise ValueError(f"invalid {name}")
        if not isinstance(seed, int) or isinstance(seed, bool) or not 0 <= seed < 2**31:
            raise ValueError("seed must be a bounded nonnegative integer")
        if not math.isfinite(ridge_alpha) or ridge_alpha <= 0:
            raise ValueError("ridge_alpha must be finite and positive")
        if model not in {"ridge", "lightgbm"}:
            raise ValueError("supported explicit models: ridge or lightgbm")
        if model == "lightgbm" and not _exists("lightgbm"):
            raise ImportError("LightGBM is unavailable; no model substitution allowed")
        # Source indices may use an equivalent timezone; preserve alignment check
        # before converting the time values to UTC for comparisons and artifacts.
        ends = _times(label_end, features.index, "label_end")
        if (ends < index).any():
            raise ValueError("label_end cannot precede the feature observation")
        available = index if feature_available_at is None else _times(feature_available_at, features.index, "feature_available_at")
        if (available > index).any():
            raise ValueError("look-ahead: feature was unavailable at its observation time")
        x, y = features.to_numpy(dtype=float, copy=True), target.to_numpy(dtype=float, copy=True)
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError("research features and labels must be finite")
        dataset = {"features": list(features.columns), "timestamps": [t.isoformat() for t in index],
                   "available_at": [t.isoformat() for t in available], "label_end": [t.isoformat() for t in ends],
                   "x": x.tolist(), "y": y.tolist()}
        config = {"model": model, "min_train": min_train, "test_size": test_size,
                  "embargo": embargo, "ridge_alpha": ridge_alpha, "seed": seed,
                  "feature_availability": "caller_assumed_at_observation" if feature_available_at is None else "explicit"}
        rows: list[dict[str, Any]] = []
        folds: list[dict[str, Any]] = []
        for fold in WalkForwardSplit(min_train, test_size, embargo).split(index):
            if len(folds) >= _MAX_FOLDS:
                raise ValueError("research fold bound exceeded")
            candidates = index.isin(fold.train_index)
            train = candidates & (ends < fold.test_start)
            test = index.isin(fold.test_index)
            if int(train.sum()) < 3:
                raise ValueError("too few training labels remain after label-end purge")
            mean, scale = x[train].mean(axis=0), x[train].std(axis=0)
            scale = np.where(scale > 0, scale, 1.)
            train_x, test_x = (x[train] - mean) / scale, (x[test] - mean) / scale
            intercept = float(y[train].mean())
            model_state: dict[str, Any] = {"schema_version": "1", "model": model,
                "features": list(features.columns), "mean": mean.tolist(), "scale": scale.tolist(),
                "training_data_hash": _hash({"x": x[train].tolist(), "y": y[train].tolist(),
                                              "timestamp": [t.isoformat() for t in index[train]]}),
                "training_end": index[train][-1].isoformat(), "max_training_label_end": max(ends[train]).isoformat(),
                "seed": seed, "trading_approved": False}
            if model == "ridge":
                coefficients = np.linalg.solve(train_x.T @ train_x + ridge_alpha * np.eye(x.shape[1]), train_x.T @ (y[train] - intercept))
                predictions = test_x @ coefficients + intercept
                train_predictions = train_x @ coefficients + intercept
                model_state.update({"coefficients": coefficients.tolist(), "intercept": intercept, "alpha": ridge_alpha})
            else:
                import lightgbm as lgb
                estimator = lgb.LGBMRegressor(n_estimators=80, max_depth=3, num_leaves=7,
                    learning_rate=.05, random_state=seed, n_jobs=1, verbosity=-1,
                    deterministic=True, force_col_wise=True)
                estimator.fit(train_x, y[train])
                predictions, train_predictions = estimator.predict(test_x), estimator.predict(train_x)
                model_state.update({"booster_text": estimator.booster_.model_to_string(), "library_version": lgb.__version__})
            if not np.isfinite(predictions).all() or not np.isfinite(train_predictions).all():
                raise ValueError("model emitted nonfinite predictions")
            model_id = self.registry.register(model_state)
            residual = y[train] - train_predictions
            # Training residual coverage is descriptive, explicitly not conformal
            # or OOS-calibrated predictive uncertainty.
            residual_q90 = float(np.quantile(np.abs(residual), .9))
            fold_id = len(folds)
            folds.append({"fold": fold_id, "train_count": int(train.sum()), "test_count": int(test.sum()),
                "purged_count": int(candidates.sum()-train.sum()), "train_end": index[train][-1].isoformat(),
                "max_training_label_end": max(ends[train]).isoformat(), "test_start": fold.test_start.isoformat(),
                "test_end": fold.test_end.isoformat(), "model_id": model_id,
                "training_residual_abs_q90": residual_q90, "uncertainty_state": "NOT_OOS_CALIBRATED",
                "mse": float(np.mean((y[test]-predictions)**2)),
                "train_only_mean": mean.tolist(), "train_only_scale": scale.tolist()})
            for timestamp, actual, prediction in zip(index[test], y[test], predictions, strict=True):
                rows.append({"timestamp": timestamp.isoformat(), "target": float(actual),
                             "prediction": float(prediction), "fold": fold_id, "model_id": model_id})
        if not folds:
            raise ValueError("no complete walk-forward fold fits the dataset")
        actual = np.array([row["target"] for row in rows])
        predicted = np.array([row["prediction"] for row in rows])
        variance = float(np.sum((actual-actual.mean())**2))
        error = float(np.sum((actual-predicted)**2))
        artifact = {"schema_version": "1", "dataset_hash": _hash(dataset), "config": config,
                    "config_hash": _hash(config), "state": "AVAILABLE", "maturity": "RESEARCH_ONLY",
                    "economically_validated": False, "trading_approved": False,
                    "folds": folds, "predictions": rows,
                    "metrics": {"mse": error/len(rows), "r2": 1-error/variance if variance > 0 else None,
                                "oos_count": len(rows), "fold_count": len(folds)}}
        run_id = _hash(artifact)
        path = self.output_dir / f"{run_id}.json"
        payload = _bytes(artifact)
        write_immutable(path, payload)
        return {**artifact, "run_id": run_id}
