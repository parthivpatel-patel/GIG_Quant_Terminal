"""Bounded state-model research with chronological holdout and durable artifacts.

This workflow fits models on a caller-selected prefix and evaluates the
remaining sequence. It never emits approved trading signals or broker requests.
The caller remains responsible for permissible data and point-in-time universe.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

import numpy as np

from gig.ml.regimes import fit_gaussian_hmm, state_duration_diagnostics
from gig.ml.state_space import kalman_filter, rts_smooth
from gig.ml.volatility import fit_garch11
from gig.timestamps import aware_timestamp

MODEL_IMPLEMENTATION_VERSION = "gig-state-models-1"


def _canonical_json(payload: dict) -> str:
    return json.dumps(payload, allow_nan=False, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _identity(payload: dict) -> str:
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def run_state_model_research(
    returns, timestamps, *, training_end: int, source_id: str, states: int = 2,
    hmm_iterations: int = 100, garch_iterations: int = 500,
) -> dict:
    """Train prefix [0:training_end], evaluate suffix with no parameter refitting.

    Timestamp input strings are preserved verbatim and independently normalized
    for ordering. Aware, strictly increasing timestamps are required. Source_id
    is supplied by the caller; it does not establish independent market provenance.
    Input data is capped at 10000 observations, and response output at the holdout
    length. A SQLite ResearchRegistry can persist and verify the complete artifact.
    """
    x = np.array(returns, dtype=float, copy=True)
    if x.ndim != 1 or not 25 <= len(x) <= 10_000 or not np.isfinite(x).all():
        raise ValueError("research returns must be a finite 1D sequence of 25..10000 observations")
    if isinstance(training_end, bool) or not isinstance(training_end, int) or not 20 <= training_end < len(x):
        raise ValueError("training_end must select >=20 training observations and a nonempty holdout")
    if not isinstance(source_id, str) or not source_id or len(source_id) > 1000:
        raise ValueError("source_id must be a nonempty identifier <=1000 characters")
    if isinstance(states, bool) or not isinstance(states, int) or not 1 <= states <= 8:
        raise ValueError("research workflow supports 1..8 HMM states")
    source_timestamps = list(timestamps)
    if len(source_timestamps) != len(x) or any(not isinstance(value, str) or len(value) > 100 for value in source_timestamps):
        raise ValueError("timestamps must be a matching sequence of timestamp strings <=100 characters")
    normalized_timestamps = []
    previous = None
    for text in source_timestamps:
        try:
            instant = aware_timestamp(text)
            if previous is not None and instant <= previous:
                raise ValueError("timestamps must be strictly chronological")
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("invalid, naive, or nonchronological timestamp") from exc
        normalized_timestamps.append(instant.isoformat())
        previous = instant
    train = x[:training_end]
    hmm = fit_gaussian_hmm(train, states=states, max_iterations=hmm_iterations)
    garch = fit_garch11(train, max_iterations=garch_iterations)
    # Fit identity depends on training data only, so changing holdout cannot
    # change model identity. All fit-affecting options are explicitly included.
    train_data = {
        "source_id": source_id,
        "returns_binary64_le_sha256": hashlib.sha256(train.astype("<f8").tobytes()).hexdigest(),
        "timestamps": source_timestamps[:training_end],
    }
    model_configuration = {
        "implementation_version": MODEL_IMPLEMENTATION_VERSION,
        "states": states, "hmm_iterations": hmm_iterations,
        "garch_iterations": garch_iterations,
        "hmm_variance_floor": 1e-8, "kalman_process_variance_ratio": .01,
    }
    models = {
        "hmm": {
            "initial": hmm.model.initial.tolist(), "transition": hmm.model.transition.tolist(),
            "means": hmm.model.means.tolist(), "variances": hmm.model.variances.tolist(),
            "converged": hmm.converged, "iterations": hmm.iterations, "stop_reason": hmm.stop_reason,
            "training_log_likelihood": hmm.likelihood_history[-1],
            "likelihood_history": list(hmm.likelihood_history),
        },
        "garch": {
            "omega": garch.model.omega, "alpha": garch.model.alpha,
            "beta": garch.model.beta, "mean": garch.model.mean,
            "converged": garch.converged, "iterations": garch.iterations,
            "stop_reason": garch.stop_reason, "training_log_likelihood": garch.log_likelihood,
        },
    }
    train_variance = float(train.var())
    if not np.isfinite(train_variance) or train_variance <= 0:
        raise ValueError("training variance must be finite positive")
    filtered = kalman_filter(
        x[:, None], [[1]], [[1]], [[train_variance * .01]],
        [[train_variance]], [float(train.mean())], [[train_variance]],
    )
    train_filter = kalman_filter(
        train[:, None], [[1]], [[1]], [[train_variance * .01]],
        [[train_variance]], [float(train.mean())], [[train_variance]],
    )
    smooth = rts_smooth(train_filter)
    models["kalman"] = {
        "transition": [[1.0]], "observation": [[1.0]],
        "process_covariance": [[train_variance * .01]],
        "observation_covariance": [[train_variance]],
        "initial_mean": [float(train.mean())], "initial_covariance": [[train_variance]],
        "parameter_method": "DOCUMENTED_TRAINING_VARIANCE_HEURISTIC_NOT_MLE",
    }
    model_id = _identity({"training": train_data, "configuration": model_configuration, "models": models})
    # A full forward-backward invocation also computes offline quantities, but
    # only its prefix-causal forward probabilities are exposed on the holdout.
    inference = hmm.model.infer(x)
    next_variance = float(garch.model.forecast(train, 1)[0])
    holdout_variances = garch.model.conditional_variances(x[training_end:], initial_variance=next_variance)
    holdout_loglik = garch.model.log_likelihood(x[training_end:], initial_variance=next_variance)
    artifact = {
        "schema_version": "gig.research.state-models.v1", "model_id": model_id,
        "configuration": model_configuration, "models": models,
        "provenance": {
            "source_id": source_id, "source_attestation": "CALLER_SUPPLIED_NOT_INDEPENDENTLY_VERIFIED",
            "input_returns_binary64_le_sha256": hashlib.sha256(x.astype("<f8").tobytes()).hexdigest(),
            "source_timestamps": source_timestamps, "utc_timestamps": normalized_timestamps,
            "training_end_exclusive": training_end, "observations": len(x),
        },
        "training_diagnostics": {
            "hmm_durations": state_duration_diagnostics(hmm.model.decode(train), hmm.model.transition),
            "kalman_smoothed_last_mean": float(smooth.mean[-1, 0]),
            "smoothing_scope": "TRAINING_SEQUENCE_ONLY_OFFLINE",
        },
        "holdout": {
            "timestamps": source_timestamps[training_end:],
            "hmm_filtered_probabilities": inference.filtered_probabilities[training_end:].tolist(),
            "hmm_log_likelihood": inference.log_likelihood - hmm.likelihood_history[-1],
            "garch_conditional_variances": holdout_variances.tolist(),
            "garch_log_likelihood": holdout_loglik,
            "kalman_filtered_mean": filtered.filtered_mean[training_end:, 0].tolist(),
            "kalman_filtered_variance": filtered.filtered_covariance[training_end:, 0, 0].tolist(),
        },
        "evidence_state": "REPRODUCIBLE_RESEARCH_NOT_ECONOMICALLY_VALIDATED",
        "operational_readiness": "RESEARCH_ONLY",
        "trading_approval": "NOT_APPROVED",
        "fit_state": "CONVERGED" if hmm.converged and garch.converged else "NOT_CONVERGED",
    }
    artifact["run_id"] = _identity(artifact)
    return artifact


class ResearchRegistry:
    """Append-only content-addressed SQLite artifacts, restart-safe and verified on read."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.path) as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("CREATE TABLE IF NOT EXISTS state_model_runs (run_id TEXT PRIMARY KEY, model_id TEXT NOT NULL, artifact_json TEXT NOT NULL)")

    @staticmethod
    def _verify(artifact: dict) -> str:
        if not isinstance(artifact, dict):
            raise ValueError("artifact must be an object")
        payload = dict(artifact)
        run_id = payload.pop("run_id", None)
        if not isinstance(run_id, str) or run_id != _identity(payload):
            raise ValueError("research artifact identity mismatch")
        if payload.get("schema_version") != "gig.research.state-models.v1" or not isinstance(payload.get("model_id"), str):
            raise ValueError("unsupported research artifact contract")
        return run_id

    def save(self, artifact: dict) -> str:
        run_id = self._verify(artifact)
        encoded = _canonical_json(artifact)
        if len(encoded.encode("utf-8")) > 8_000_000:
            raise ValueError("research artifact exceeds 8MB")
        with sqlite3.connect(self.path) as connection:
            connection.execute("PRAGMA synchronous=FULL")
            connection.execute(
                "INSERT OR IGNORE INTO state_model_runs(run_id,model_id,artifact_json) VALUES (?,?,?)",
                (run_id, artifact["model_id"], encoded),
            )
            saved = connection.execute("SELECT artifact_json FROM state_model_runs WHERE run_id=?", (run_id,)).fetchone()
            if saved is None or saved[0] != encoded:
                raise ValueError("research registry identity conflict")
        return run_id

    def load(self, run_id: str) -> dict:
        if not isinstance(run_id, str) or len(run_id) != 64:
            raise ValueError("run_id must be a SHA-256 identity")
        with sqlite3.connect(self.path) as connection:
            row = connection.execute("SELECT artifact_json FROM state_model_runs WHERE run_id=?", (run_id,)).fetchone()
        if row is None:
            raise KeyError("research run not found")
        artifact = json.loads(row[0])
        if self._verify(artifact) != run_id:
            raise ValueError("research registry identity mismatch")
        return artifact
