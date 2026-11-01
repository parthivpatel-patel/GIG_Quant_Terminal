"""Durable CPU sequence research, with compact immutable HTTP contracts."""

from __future__ import annotations

import hashlib
import json
import math
import sqlite3
from pathlib import Path
from typing import Any

from gig.artifacts import write_immutable
from gig.ml.sequence import SequenceRegistry, train_sequence
from gig.service.contracts import Contract, State
from gig.service.serialization import dumps, public_error
from gig.timestamps import aware_timestamp


class SequenceWorkspace:
    def __init__(self, base_dir: Path):
        self.base_dir = Path(base_dir).absolute()

    def _path(self, name: str) -> Path:
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts or "\\" in name:
            raise ValueError("sequence paths must stay in the fixed workspace")
        result = self.base_dir / relative
        for part in (result, *result.parents):
            if part.is_symlink():
                raise ValueError("sequence state cannot follow symbolic links")
        return result

    def _registry(self) -> SequenceRegistry:
        self._path("registry/models")
        self._path("registry/runs")
        return SequenceRegistry(self._path("registry"))

    @staticmethod
    def _clone(payload: Any) -> dict[str, Any]:
        if not isinstance(payload, dict):
            raise ValueError("sequence request must be an object")
        dumps(payload)
        return json.loads(json.dumps(payload, allow_nan=False, ensure_ascii=False))

    @staticmethod
    def _summary(artifact: dict[str, Any]) -> dict[str, Any]:
        return {
            "title": "CPU bidirectional LSTM research",
            "metrics": {**artifact["metrics"], "trading_approved": False},
            "rows": artifact["predictions"],
            "training_loss": artifact["training_loss"],
            "model_id": artifact["model_id"],
            "run_id": artifact["run_id"],
            "dataset_sha256": artifact["dataset_sha256"],
            "configuration": artifact["configuration"],
            "bidirectional_scope": artifact["bidirectional_scope"],
            "source": artifact["source"],
            "classification": artifact["classification"],
            "trading_approved": False,
            "economically_validated": False,
        }

    @staticmethod
    def _contract(
        data: dict[str, Any] | None,
        *,
        artifact: dict[str, Any],
        scope: str,
        state: State = State.AVAILABLE,
        observed: Any = None,
    ) -> Contract:
        return Contract.admit(
            state,
            data if state not in (State.UNKNOWN, State.UNAVAILABLE, State.ERROR) else None,
            source=artifact["source"],
            classification=artifact["classification"],
            as_of=aware_timestamp(
                observed if observed is not None else artifact["as_of"]
            ).isoformat(),
            scope=scope,
            model_identity=artifact["model_id"],
            model_version=artifact["model_state"]["implementation"],
            source_identity=artifact["dataset_sha256"],
            evidence_state="HASH_VERIFIED_CPU_RESEARCH",
            readiness="RESEARCH_ONLY",
        )

    def train(self, payload: dict[str, Any]) -> Contract:
        request = self._clone(payload)
        artifact = train_sequence(request)
        contract = self._contract(
            self._summary(artifact), artifact=artifact, scope="workspace.sequence.research"
        )
        # Admit the final wire bytes before any state publication.
        registry = self._registry()
        registry.save(artifact)
        digest = hashlib.sha256(contract.wire).hexdigest()
        write_immutable(self._path(f"responses/{digest}.json"), contract.wire)
        with sqlite3.connect(self._path("sequence.sqlite")) as connection:
            connection.execute("PRAGMA synchronous=FULL")
            connection.execute(
                "CREATE TABLE IF NOT EXISTS runs (run_id TEXT PRIMARY KEY, as_of TEXT NOT NULL, response_hash TEXT NOT NULL)"
            )
            previous = connection.execute(
                "SELECT as_of,response_hash FROM runs WHERE run_id=?", (artifact["run_id"],)
            ).fetchone()
            identity = (artifact["as_of"], digest)
            if previous is not None and tuple(previous) != identity:
                raise ValueError("sequence run conflicts with preserved response identity")
            connection.execute(
                "INSERT OR IGNORE INTO runs VALUES(?,?,?)", (artifact["run_id"], *identity)
            )
        return contract

    def predict(self, payload: dict[str, Any]) -> Contract:
        request = self._clone(payload)
        if set(request) != {"model_id", "rows"}:
            raise ValueError("sequence inference requires model_id and rows only")
        registry = self._registry()
        model_id = SequenceRegistry._identity(request["model_id"])
        if not self._path(f"registry/models/{model_id}.json").is_file():
            return Contract.admit(
                State.UNAVAILABLE,
                None,
                source="local-sequence-registry",
                classification="UNAVAILABLE",
                scope="workspace.sequence.inference",
                model_identity=model_id,
                error=public_error("DATA_UNAVAILABLE"),
            )
        state = registry.load_model(model_id)
        predictions = registry.predict(model_id, request["rows"])
        # Registry reload supports retrospective numerical comparison. The
        # terminal inference contract emits only decisions at/after the model's
        # declared out-of-sample boundary, never future-fitted historical scores.
        training_cutoff = aware_timestamp(state["training_cutoff"])
        total_predictions = len(predictions)
        predictions = [
            row for row in predictions if aware_timestamp(row["timestamp"]) >= training_cutoff
        ]
        if not predictions:
            return Contract.admit(
                State.UNAVAILABLE,
                None,
                source="local-sequence-registry",
                classification="UNAVAILABLE",
                scope="workspace.sequence.inference",
                model_identity=model_id,
                error=public_error("DATA_UNAVAILABLE"),
            )
        # Inference is a new caller-supplied dataset, not approved trading use.
        data = {
            "title": "CPU sequence model outputs",
            "metrics": {
                "predictions": len(predictions),
                "excluded_insample_predictions": total_predictions - len(predictions),
                "trading_approved": False,
            },
            "rows": predictions,
            "model_id": model_id,
            "input_sha256": hashlib.sha256(dumps(request["rows"]).encode()).hexdigest(),
            "trading_approved": False,
            "economically_validated": False,
            "bidirectional_scope": "HISTORICAL_WINDOW_ONLY",
        }
        return Contract.admit(
            State.AVAILABLE,
            data,
            source="caller-features-and-local-bilstm",
            classification="MODEL_ESTIMATE",
            as_of=predictions[-1]["timestamp"],
            scope="workspace.sequence.inference",
            model_identity=model_id,
            model_version=state["implementation"],
            source_identity=data["input_sha256"],
            evidence_state="HASH_VERIFIED_CPU_RESEARCH",
            readiness="RESEARCH_ONLY",
        )

    def section(self, section: str, as_of: Any, max_age_seconds: float | None) -> Contract | None:
        if section not in {"models", "signals", "research"}:
            return None
        cutoff = aware_timestamp(as_of)
        if max_age_seconds is not None and (
            isinstance(max_age_seconds, bool)
            or not isinstance(max_age_seconds, (int, float))
            or not math.isfinite(max_age_seconds)
            or not 0 <= max_age_seconds <= 366 * 86400
        ):
            raise ValueError("bounded explicit freshness policy required")
        database = self._path("sequence.sqlite")
        if not database.is_file():
            return None
        selected: tuple[Any, str, str] | None = None
        with sqlite3.connect(database.as_uri() + "?mode=ro", uri=True) as connection:
            for run_id, timestamp, digest in connection.execute(
                "SELECT run_id,as_of,response_hash FROM runs ORDER BY rowid"
            ):
                observed = aware_timestamp(timestamp)
                if observed <= cutoff and (selected is None or observed >= selected[0]):
                    selected = (observed, run_id, digest)
        if selected is None:
            return None
        observed, run_id, digest = selected
        SequenceRegistry._identity(digest)
        wire = self._path(f"responses/{digest}.json").read_bytes()
        if hashlib.sha256(wire).hexdigest() != digest:
            raise ValueError("sequence response integrity failure")
        saved = Contract(wire).to_dict()
        artifact = self._registry().load(run_id)
        if saved["data"] != self._summary(artifact) or artifact["as_of"] != observed.isoformat():
            raise ValueError("sequence registry differs from preserved response")
        state = (
            State.UNKNOWN
            if max_age_seconds is None
            else (
                State.STALE
                if (cutoff - observed).total_seconds() > max_age_seconds
                else State.AVAILABLE
            )
        )
        if section == "models":
            data = {
                "title": "Registered CPU BiLSTM",
                "metrics": {"trained_models": 1, "trading_approved": False},
                "rows": [
                    {
                        "model": "bilstm",
                        "trained": True,
                        "model_id": artifact["model_id"],
                        "run_id": artifact["run_id"],
                        "library": "torch",
                        "library_version": artifact["model_state"]["library_version"],
                        "device": "cpu",
                        "trading_approved": False,
                        "training_cutoff": artifact["model_state"]["training_cutoff"],
                    }
                ],
            }
        else:
            data = self._summary(artifact)
            if section == "signals":
                data["title"] = "Unapproved CPU sequence research outputs"
        return self._contract(
            data, artifact=artifact, scope=f"workspace.sequence.{section}", state=state
        )
