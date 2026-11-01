"""Real CPU BiLSTM research on bounded historical windows, with JSON states.

Bidirectionality is confined to a window ending at its decision timestamp.
No subsequent observation enters that prediction. This is statistical research,
not an approved signal, brokerage adapter, or economic-performance assertion.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import math
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np

from gig.artifacts import write_immutable
from gig.ml.runtime import TORCH_CPU_LOCK
from gig.service.serialization import dumps
from gig.timestamps import aware_timestamp

IMPLEMENTATION = "gig.cpu-bilstm.v1"
_LOCK = TORCH_CPU_LOCK
_CLASSIFICATIONS = {"SYNTHETIC", "HISTORICAL", "DELAYED", "SNAPSHOT", "REAL_TIME"}


def _bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":")
    ).encode()


def _hash(value: Any) -> str:
    return hashlib.sha256(_bytes(value)).hexdigest()


def _torch():
    try:
        return importlib.import_module("torch")
    except ImportError:
        raise ImportError(
            "Actual PyTorch is required; no sequence-model substitution is available"
        ) from None


def _integer(value: Any, name: str, low: int, high: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
        raise ValueError(f"{name} is outside its integer bounds")
    return value


def _number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    try:
        result = float(value)
    except (ValueError, OverflowError):
        raise ValueError(f"{name} must be finite") from None
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > 256:
        raise ValueError(f"{name} must be bounded nonempty text")
    return value


@contextmanager
def _cpu(torch, seed: int):
    """Isolate CPU RNG and restore process execution settings after use."""
    with _LOCK:
        threads = torch.get_num_threads()
        deterministic = torch.are_deterministic_algorithms_enabled()
        warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
        mkldnn = torch.backends.mkldnn.enabled
        try:
            torch.set_num_threads(1)
            torch.use_deterministic_algorithms(True)
            torch.backends.mkldnn.enabled = False
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(seed)
                yield
        finally:
            torch.backends.mkldnn.enabled = mkldnn
            torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)
            torch.set_num_threads(threads)


def _network(torch, features: int, hidden: int):
    class HistoryWindowBiLSTM(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = torch.nn.LSTM(
                features, hidden, batch_first=True, bidirectional=True, num_layers=1
            )
            self.output = torch.nn.Linear(2 * hidden, 1)

        def forward(self, values):
            _, (final, _) = self.lstm(values)
            # Both directional states summarize this same past-only window.
            return self.output(torch.cat((final[-2], final[-1]), dim=1)).squeeze(-1)

    return HistoryWindowBiLSTM().to(device="cpu", dtype=torch.float32)


def _rows(rows: Any, *, targets: bool):
    if not isinstance(rows, list) or not 1 <= len(rows) <= 1000:
        raise ValueError("sequence rows must contain 1..1000 observations")
    names: list[str] | None = None
    values: list[list[float]] = []
    times: list[Any] = []
    available: list[Any] = []
    ends: list[Any] = []
    labels: list[float] = []
    required = {"timestamp", "features", "available_at"} | (
        {"target", "label_end"} if targets else set()
    )
    for row in rows:
        if not isinstance(row, dict) or set(row) != required:
            raise ValueError("sequence row fields do not match the contract")
        features = row["features"]
        if not isinstance(features, dict) or not 1 <= len(features) <= 16:
            raise ValueError("sequence requires 1..16 features")
        columns = sorted(_text(name, "feature") for name in features)
        if names is not None and names != columns:
            raise ValueError("sequence feature identities must remain aligned")
        names = columns
        stamp, knowledge = aware_timestamp(row["timestamp"]), aware_timestamp(row["available_at"])
        if times and stamp <= times[-1]:
            raise ValueError("sequence timestamps must increase strictly")
        if knowledge > stamp:
            raise ValueError("look-ahead: a feature is unavailable at its decision timestamp")
        times.append(stamp)
        available.append(knowledge)
        values.append([_number(features[name], "feature") for name in names])
        if targets:
            end = aware_timestamp(row["label_end"])
            if end < stamp:
                raise ValueError("label_end cannot precede the observation")
            ends.append(end)
            labels.append(_number(row["target"], "target"))
    return (
        names,
        np.array(values, dtype=float),
        times,
        available,
        ends,
        np.array(labels, dtype=float),
    )


def _encode_weights(model) -> dict[str, Any]:
    return {
        name: {
            "shape": list(value.shape),
            "dtype": "float32",
            "values": value.detach().cpu().contiguous().reshape(-1).tolist(),
        }
        for name, value in sorted(model.state_dict().items())
    }


def _restore(torch, state: dict[str, Any]):
    if not isinstance(state, dict) or state.get("implementation") != IMPLEMENTATION:
        raise ValueError("unsupported sequence-model implementation")
    names = state.get("features")
    if not isinstance(names, list) or not 1 <= len(names) <= 16 or len(set(names)) != len(names):
        raise ValueError("invalid registered feature identities")
    if any(not isinstance(name, str) or not name for name in names):
        raise ValueError("invalid registered feature identities")
    config = state.get("configuration", {})
    hidden = _integer(config.get("hidden_size"), "hidden_size", 1, 32)
    _integer(config.get("sequence_length"), "sequence_length", 2, 32)
    model = _network(torch, len(names), hidden)
    serialized = state.get("parameters")
    if not isinstance(serialized, dict) or serialized.keys() != model.state_dict().keys():
        raise ValueError("registered tensor names do not match the architecture")
    tensors = {}
    for key, expected in model.state_dict().items():
        item = serialized[key]
        if not isinstance(item, dict) or set(item) != {"shape", "dtype", "values"}:
            raise ValueError("invalid registered tensor fields")
        if item["shape"] != list(expected.shape) or item["dtype"] != "float32":
            raise ValueError("registered tensor shape or dtype mismatch")
        if not isinstance(item["values"], list) or len(item["values"]) != expected.numel():
            raise ValueError("registered tensor allocation bound mismatch")
        array = np.array([_number(v, "parameter") for v in item["values"]], dtype=np.float32)
        if not np.isfinite(array).all():
            raise ValueError("registered float32 tensor overflow")
        tensors[key] = torch.tensor(
            array.reshape(expected.shape), device="cpu", dtype=torch.float32
        )
    model.load_state_dict(tensors, strict=True)
    model.eval()
    return model


def train_sequence(payload: dict[str, Any]) -> dict[str, Any]:
    """Train a real bidirectional LSTM on a purged chronological prefix.

    Returned model/run identities exclude volatile duration measurements. Training
    uses full batches and a single CPU thread. Outputs do not approve trading.
    """
    allowed = {
        "rows",
        "source",
        "classification",
        "sequence_length",
        "train_size",
        "hidden_size",
        "epochs",
        "learning_rate",
        "seed",
        "as_of",
    }
    if (
        not isinstance(payload, dict)
        or set(payload) - allowed
        or not {"rows", "source", "classification"} <= set(payload)
    ):
        raise ValueError("unknown or missing sequence request fields")
    dumps(payload)  # Apply the same bounded wire contract before allocation.
    payload = json.loads(_bytes(payload))
    source = _text(payload["source"], "source")
    classification = payload["classification"]
    if classification not in _CLASSIFICATIONS:
        raise ValueError("explicit source classification required")
    names, x, times, available, ends, y = _rows(payload["rows"], targets=True)
    if len(x) < 24:
        raise ValueError("sequence training requires at least 24 observations")
    window = _integer(payload.get("sequence_length", 8), "sequence_length", 2, 32)
    train_size = _integer(
        payload.get("train_size", len(x) * 3 // 4), "train_size", window + 7, len(x) - 1
    )
    hidden = _integer(payload.get("hidden_size", 8), "hidden_size", 1, 32)
    epochs = _integer(payload.get("epochs", 12), "epochs", 1, 100)
    seed = _integer(payload.get("seed", 42), "seed", 0, 2**31 - 1)
    rate = _number(payload.get("learning_rate", 0.01), "learning_rate")
    if not 1e-5 <= rate <= 0.1:
        raise ValueError("learning_rate must be between 1e-5 and .1")
    cutoff = aware_timestamp(payload["as_of"]) if "as_of" in payload else max(ends)
    if any(end > cutoff for end in ends):
        raise ValueError("labels are unknown at the research cutoff")
    train = [i for i in range(window - 1, train_size) if ends[i] < times[train_size]]
    test = list(range(train_size, len(x)))
    if len(train) < 8:
        raise ValueError("too few training windows after purging unavailable labels")
    # Each direction computes four recurrent gates per time step. Count both
    # directions rather than admitting eight times the stated recurrent budget.
    work = 2 * 4 * epochs * len(train) * window * (hidden + x.shape[1]) * hidden
    if work > 100_000_000:
        raise ValueError("sequence training work budget exceeded")
    used = sorted({j for i in train for j in range(i - window + 1, i + 1)})
    with np.errstate(over="ignore", invalid="ignore"):
        mean, scale = x[used].mean(axis=0), x[used].std(axis=0)
        target_mean = float(y[train].mean())
        target_scale = float(y[train].std())
    if (
        not np.isfinite(mean).all()
        or not np.isfinite(scale).all()
        or not math.isfinite(target_mean)
        or not math.isfinite(target_scale)
    ):
        raise ValueError("sequence normalization overflow")
    scale = np.where(scale > 0, scale, 1.0)
    target_scale = target_scale or 1.0
    with np.errstate(over="ignore", invalid="ignore"):
        scaled = (x - mean) / scale
        training_y = (y[train] - target_mean) / target_scale
    if not np.isfinite(scaled).all() or not np.isfinite(training_y).all():
        raise ValueError("sequence normalization overflow")
    windows = np.stack([scaled[i - window + 1 : i + 1] for i in train + test]).astype(np.float32)
    if not np.isfinite(windows).all() or not np.isfinite(training_y.astype(np.float32)).all():
        raise ValueError("sequence input exceeds float32 execution range")
    config = {
        "sequence_length": window,
        "hidden_size": hidden,
        "epochs": epochs,
        "learning_rate": rate,
        "seed": seed,
        "train_size": train_size,
        "optimizer": "Adam",
        "gradient_clip_norm": 1.0,
        "device": "cpu",
        "threads": 1,
    }
    training_identity = {
        "source": source,
        "classification": classification,
        "samples": [
            {
                "decision_timestamp": times[i].isoformat(),
                "label_end": ends[i].isoformat(),
                "window": x[i - window + 1 : i + 1].tolist(),
                "target": float(y[i]),
            }
            for i in train
        ],
        "feature_available_at": [available[i].isoformat() for i in used],
    }
    torch = _torch()
    started = time.perf_counter()
    with _cpu(torch, seed):
        model = _network(torch, x.shape[1], hidden)
        values = torch.tensor(windows, device="cpu", dtype=torch.float32)
        labels = torch.tensor(training_y, device="cpu", dtype=torch.float32)
        optimizer = torch.optim.Adam(model.parameters(), lr=rate)
        history = []
        model.train()
        for _ in range(epochs):
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.mse_loss(model(values[: len(train)]), labels)
            if not torch.isfinite(loss):
                raise ValueError("nonfinite BiLSTM training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            history.append(float(loss.detach().item()))
        model.eval()
        with torch.no_grad():
            prediction = (
                model(values[len(train) :]).cpu().numpy().astype(float) * target_scale + target_mean
            )
        parameters = _encode_weights(model)
    duration = time.perf_counter() - started
    if not np.isfinite(prediction).all():
        raise ValueError("nonfinite BiLSTM inference")
    state = {
        "implementation": IMPLEMENTATION,
        "architecture": "BIDIRECTIONAL_LSTM_HISTORY_WINDOW",
        "library": "torch",
        "library_version": torch.__version__,
        "configuration": config,
        "features": names,
        "mean": mean.tolist(),
        "scale": scale.tolist(),
        "target_mean": target_mean,
        "target_scale": target_scale,
        "training_data_sha256": _hash(training_identity),
        "parameters": parameters,
        "training_cutoff": times[train_size].isoformat(),
        "max_training_label_end": max(ends[i] for i in train).isoformat(),
        "last_training_decision": times[train[-1]].isoformat(),
        "trading_approved": False,
    }
    model_id = _hash(state)
    rows = [
        {
            "timestamp": times[i].isoformat(),
            "window_start": times[i - window + 1].isoformat(),
            "target": float(y[i]),
            "prediction": float(value),
            "model_id": model_id,
        }
        for i, value in zip(test, prediction, strict=True)
    ]
    residual = y[test] - prediction
    with np.errstate(over="ignore", invalid="ignore"):
        mse = float(np.mean(residual**2))
    if not math.isfinite(mse):
        raise ValueError("BiLSTM metric overflow")
    invariant = {
        "schema_version": "1",
        "state": "AVAILABLE",
        "model": "bilstm",
        "model_id": model_id,
        "model_state": state,
        "source": source,
        "classification": classification,
        "dataset_sha256": _hash(payload["rows"]),
        "as_of": cutoff.isoformat(),
        "configuration": config,
        "predictions": rows,
        "training_loss": history,
        "metrics": {
            "holdout_mse": mse,
            "oos_count": len(test),
            "training_windows": len(train),
            "purged_windows": train_size - window + 1 - len(train),
        },
        "bidirectional_scope": "HISTORICAL_WINDOW_ONLY",
        "economically_validated": False,
        "trading_approved": False,
        "evidence_state": "CALLER_SUPPLIED_RESEARCH",
    }
    result = {
        **invariant,
        "run_id": _hash(invariant),
        "measurements": {"training_and_inference_seconds": duration, "device": "cpu", "threads": 1},
    }
    dumps(result)  # Reject a result outside the declared final wire boundary.
    return result


def predict_sequence(state: dict[str, Any], rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Predict complete historical windows with a verified JSON model state."""
    dumps(state)
    dumps(rows)
    state = json.loads(_bytes(state))
    rows = json.loads(_bytes(rows))
    names, x, times, _, _, _ = _rows(rows, targets=False)
    if names != state.get("features"):
        raise ValueError("inference features do not match model identity/order")
    config = state.get("configuration", {})
    window = _integer(config.get("sequence_length"), "sequence_length", 2, 32)
    if len(x) < window:
        raise ValueError("inference needs a complete historical window")
    mean, scale = (
        np.array(state.get("mean"), dtype=float),
        np.array(state.get("scale"), dtype=float),
    )
    if (
        mean.shape != (len(names),)
        or scale.shape != mean.shape
        or not np.isfinite(mean).all()
        or not np.isfinite(scale).all()
        or np.any(scale <= 0)
    ):
        raise ValueError("invalid registered normalization")
    target_mean, target_scale = (
        _number(state.get("target_mean"), "target_mean"),
        _number(state.get("target_scale"), "target_scale"),
    )
    if target_scale <= 0:
        raise ValueError("invalid registered target normalization")
    with np.errstate(over="ignore", invalid="ignore"):
        scaled = ((x - mean) / scale).astype(np.float32)
    if not np.isfinite(scaled).all():
        raise ValueError("inference float32 normalization overflow")
    windows = np.stack([scaled[i - window + 1 : i + 1] for i in range(window - 1, len(x))])
    torch = _torch()
    with _cpu(torch, 0):
        model = _restore(torch, state)
        with torch.no_grad():
            prediction = (
                model(torch.tensor(windows, device="cpu", dtype=torch.float32))
                .cpu()
                .numpy()
                .astype(float)
            )
    prediction = prediction * target_scale + target_mean
    if not np.isfinite(prediction).all():
        raise ValueError("nonfinite sequence inference")
    return [
        {"timestamp": times[i].isoformat(), "prediction": float(value), "model_id": _hash(state)}
        for i, value in zip(range(window - 1, len(x)), prediction, strict=True)
    ]


class SequenceRegistry:
    """Content-addressed JSON artifacts and tensors; no pickle execution."""

    def __init__(self, directory: str | Path):
        self.directory = Path(directory)

    @staticmethod
    def _identity(value: Any) -> str:
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(c not in "0123456789abcdef" for c in value)
        ):
            raise ValueError("identity must be a lowercase SHA-256")
        return value

    def save(self, artifact: dict[str, Any]) -> str:
        artifact = json.loads(_bytes(artifact))
        model_id, run_id = (
            self._identity(artifact.get("model_id")),
            self._identity(artifact.get("run_id")),
        )
        model = artifact.get("model_state")
        invariant = {
            key: value for key, value in artifact.items() if key not in {"run_id", "measurements"}
        }
        if _hash(model) != model_id or _hash(invariant) != run_id:
            raise ValueError("sequence artifact identity mismatch")
        write_immutable(self.directory / "models" / f"{model_id}.json", _bytes(model))
        # Volatile measurements are separately returned, not part of replay state.
        write_immutable(self.directory / "runs" / f"{run_id}.json", _bytes(invariant))
        return run_id

    def load(self, run_id: str) -> dict[str, Any]:
        run_id = self._identity(run_id)
        content = (self.directory / "runs" / f"{run_id}.json").read_bytes()
        if hashlib.sha256(content).hexdigest() != run_id:
            raise ValueError("sequence run integrity failure")
        result = json.loads(content)
        model_id = self._identity(result["model_id"])
        model = self.load_model(model_id)
        if model != result["model_state"]:
            raise ValueError("sequence model/run provenance mismatch")
        return {**result, "run_id": run_id}

    def load_model(self, model_id: str) -> dict[str, Any]:
        model_id = self._identity(model_id)
        content = (self.directory / "models" / f"{model_id}.json").read_bytes()
        if hashlib.sha256(content).hexdigest() != model_id:
            raise ValueError("sequence model integrity failure")
        return json.loads(content)

    def predict(self, model_id: str, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return predict_sequence(self.load_model(model_id), rows)
