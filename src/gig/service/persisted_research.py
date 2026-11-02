"""Hash-verified persisted-data CPU research and saved-state inference."""

from __future__ import annotations

import hashlib
import math
import sqlite3
from pathlib import Path
from typing import Any

from gig.artifacts import write_immutable
from gig.research.model_pipeline import (
    IMPLEMENTATION,
    configuration,
    fit_dataset,
    identity,
    saved_predict,
    sha256,
)
from gig.service.contracts import Contract, State
from gig.service.serialization import dumps, loads, public_error
from gig.timestamps import aware_timestamp


class PersistedResearchWorkspace:
    def __init__(self, base_dir: Path):
        self.base_dir = Path(base_dir).absolute()

    def _path(self, name: str) -> Path:
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts or "\\" in name:
            raise ValueError("research paths must remain in the fixed workspace")
        path = self.base_dir / "persisted-research" / relative
        if any(p.is_symlink() for p in (path, *path.parents)):
            raise ValueError("persisted research paths cannot be symbolic links")
        return path

    def _dataset(self, dataset_id: str) -> dict[str, Any]:
        from gig.service.dataset_workspace import PersistedDatasetWorkspace

        return PersistedDatasetWorkspace(self.base_dir).load(sha256(dataset_id))

    def _model(self, model_id: str) -> dict[str, Any]:
        model_id = sha256(model_id)
        wire = self._path(f"models/{model_id}.json").read_bytes()
        if hashlib.sha256(wire).hexdigest() != model_id:
            raise ValueError("persisted model integrity mismatch")
        state = loads(wire)
        if state.get("implementation") != IMPLEMENTATION:
            raise ValueError("unsupported persisted model state")
        return state

    @staticmethod
    def _summary(artifact: dict[str, Any], section: str = "research") -> dict[str, Any]:
        data = {
            "title": "Persisted-data purged CPU research",
            "run_id": artifact["run_id"], "dataset_id": artifact["dataset_id"],
            "model": artifact["model"], "model_ids": artifact["model_ids"],
            "as_of": artifact["as_of"], "metrics": artifact["metrics"],
            "rows": artifact["rows"][:500], "rows_total": len(artifact["rows"]),
            "rows_complete": len(artifact["rows"]) <= 500,
            "latest_scores": artifact["latest_scores"], "folds": artifact["folds"],
            "prediction_kind": artifact["prediction_kind"],
            "usable_for_portfolio": artifact["usable_for_portfolio"],
            "input_classification": artifact["input_classification"],
            "dataset_diagnostics": artifact["dataset_diagnostics"],
            "trading_approved": False, "economically_validated": False,
            "notes": ["Models and features are reconstructed historical research, not historically published model availability.",
                      "No uncertainty diagnostic is an approved trading signal or proof of economic value."]}
        if section == "models":
            data["title"] = "Registered persisted-data CPU models"
            data["rows"] = [{**fold, "trained": True, "model": artifact["model"],
                             "prediction_kind": artifact["prediction_kind"], "trading_approved": False}
                            for fold in artifact["folds"]]
            data["rows_total"] = len(data["rows"])
            data["rows_complete"] = True
        elif section == "signals":
            data["title"] = "Unapproved persisted-data model estimates"
            data["rows"] = artifact["latest_scores"]
            data["rows_total"] = len(data["rows"])
            data["rows_complete"] = True
        return data

    @staticmethod
    def _contract(artifact: dict[str, Any], section: str, *, state: State = State.AVAILABLE) -> Contract:
        return Contract.admit(
            state, PersistedResearchWorkspace._summary(artifact, section)
            if state in (State.AVAILABLE, State.STALE) else None,
            source=artifact["source"], classification="MODEL_ESTIMATE",
            as_of=artifact["as_of"], source_identity=artifact["dataset_id"],
            model_identity=artifact["run_id"], model_version=IMPLEMENTATION,
            evidence_state="HASH_VERIFIED_PERSISTED_DATA_RESEARCH", readiness="RESEARCH_ONLY",
            scope=f"workspace.persisted.{section}")

    def run(self, payload: dict[str, Any]) -> Contract:
        request = loads(dumps(payload))
        config = configuration(request)
        try:
            dataset = self._dataset(request["dataset_id"])
        except FileNotFoundError:
            return self._missing()
        artifact, models = fit_dataset(dataset, config)
        contract = self._contract(artifact, "research")
        # Both full durable bytes and compact public contract are admitted first.
        encoded = dumps({k: v for k, v in artifact.items() if k != "run_id"}).encode()
        model_bytes = {key: dumps(value).encode() for key, value in models.items()}
        for model_id, content in model_bytes.items():
            if hashlib.sha256(content).hexdigest() != model_id:
                raise ValueError("model identity changed before publication")
        for model_id, content in model_bytes.items():
            write_immutable(self._path(f"models/{model_id}.json"), content)
        write_immutable(self._path(f"runs/{artifact['run_id']}.json"), encoded)
        digest = hashlib.sha256(contract.wire).hexdigest()
        write_immutable(self._path(f"responses/{digest}.json"), contract.wire)
        with sqlite3.connect(self._path("research.sqlite")) as connection:
            connection.execute("PRAGMA synchronous=FULL")
            connection.execute("CREATE TABLE IF NOT EXISTS runs (run_id TEXT PRIMARY KEY, as_of TEXT NOT NULL, response_hash TEXT NOT NULL)")
            previous = connection.execute("SELECT as_of,response_hash FROM runs WHERE run_id=?", (artifact["run_id"],)).fetchone()
            values = (artifact["as_of"], digest)
            if previous is not None and tuple(previous) != values:
                raise ValueError("persisted run index conflicts with immutable response")
            connection.execute("INSERT OR IGNORE INTO runs VALUES(?,?,?)", (artifact["run_id"], *values))
        return contract

    def load(self, run_id: str) -> dict[str, Any]:
        run_id = sha256(run_id)
        content = self._path(f"runs/{run_id}.json").read_bytes()
        if hashlib.sha256(content).hexdigest() != run_id:
            raise ValueError("persisted research run integrity mismatch")
        artifact = loads(content)
        if artifact.get("schema_version") != IMPLEMENTATION:
            raise ValueError("unsupported persisted research artifact")
        # Dataset and every fitted model remain verified on each read/restart.
        dataset = self._dataset(artifact["dataset_id"])
        if dataset["as_of"] != artifact["as_of"] or dataset["classification"] != artifact["input_classification"]:
            raise ValueError("persisted research provenance differs from admitted dataset")
        for model_id in artifact["model_ids"]:
            self._model(model_id)
        for row in artifact["rows"]:
            if row["model_id"] not in artifact["model_ids"]:
                raise ValueError("prediction references an unregistered model")
        return {**artifact, "run_id": run_id}

    def predict(self, payload: dict[str, Any]) -> Contract:
        request = loads(dumps(payload))
        if not isinstance(request, dict) or set(request) != {"run_id", "dataset_id"}:
            raise ValueError("saved inference requires run_id and admitted dataset_id only")
        try:
            artifact = self.load(sha256(request["run_id"]))
            dataset = self._dataset(sha256(request["dataset_id"]))
        except FileNotFoundError:
            return self._missing()
        if dataset["source"] != artifact["source"] or dataset["classification"] != artifact["input_classification"]:
            raise ValueError("saved model inference cannot silently change source classification")
        # Choose the latest fitted model per instrument; prediction excludes its
        # in-sample decisions and retains the historical training cutoff.
        states = [self._model(value) for value in artifact["model_ids"]]
        selected: dict[str, dict[str, Any]] = {}
        for state in states:
            instrument = state["instrument_id"]
            if instrument not in selected or aware_timestamp(state["training_cutoff"]) > aware_timestamp(selected[instrument]["training_cutoff"]):
                selected[instrument] = state
        rows = []
        for instrument in sorted(selected):
            rows.extend(saved_predict(selected[instrument], dataset))
        rows.sort(key=lambda row: (aware_timestamp(row["timestamp"]), row["instrument_id"]))
        if not rows:
            return Contract.admit(State.UNAVAILABLE, None, source=artifact["source"],
                                  classification="UNAVAILABLE", scope="workspace.persisted.inference",
                                  error=public_error("DATA_UNAVAILABLE"))
        data = {"title": "Saved CPU model inference on admitted observations",
                "run_id": artifact["run_id"], "dataset_id": dataset["dataset_id"],
                "rows": rows[:500], "rows_total": len(rows), "rows_complete": len(rows) <= 500,
                "metrics": {"predictions": len(rows), "training_performed": False},
                "prediction_kind": artifact["prediction_kind"],
                "input_classification": dataset["classification"],
                "model_ids": sorted(identity(v) for v in selected.values()),
                "usable_for_portfolio": artifact["usable_for_portfolio"],
                "trading_approved": False, "economically_validated": False}
        return Contract.admit(State.AVAILABLE, data, source=dataset["source"],
                              classification="MODEL_ESTIMATE", as_of=dataset["as_of"],
                              scope="workspace.persisted.inference", source_identity=dataset["dataset_id"],
                              model_identity=artifact["run_id"], model_version=IMPLEMENTATION,
                              evidence_state="HASH_VERIFIED_PERSISTED_DATA_RESEARCH", readiness="RESEARCH_ONLY")

    @staticmethod
    def _missing() -> Contract:
        return Contract.admit(State.UNAVAILABLE, None, source="persisted-data-cpu-research",
                              classification="UNAVAILABLE", scope="workspace.persisted.research",
                              error=public_error("DATA_UNAVAILABLE"))

    def section(self, section: str, as_of: Any, max_age_seconds: float | None) -> Contract | None:
        if section not in {"research", "models", "signals"}:
            return None
        cutoff = aware_timestamp(as_of)
        if max_age_seconds is not None and (isinstance(max_age_seconds, bool)
                or not isinstance(max_age_seconds, (int, float))
                or not math.isfinite(max_age_seconds) or not 0 <= max_age_seconds <= 366*86400):
            raise ValueError("bounded explicit freshness policy required")
        path = self._path("research.sqlite")
        if not path.is_file():
            return None
        latest: tuple[Any, str, str] | None = None
        with sqlite3.connect(path.as_uri()+"?mode=ro", uri=True) as connection:
            for run_id, stamp, digest in connection.execute("SELECT run_id,as_of,response_hash FROM runs ORDER BY rowid"):
                observed = aware_timestamp(stamp)
                if observed <= cutoff and (latest is None or observed >= latest[0]):
                    latest = (observed, run_id, digest)
        if latest is None:
            return None
        observed, run_id, digest = latest
        artifact = self.load(run_id)
        sha256(digest)
        wire = self._path(f"responses/{digest}.json").read_bytes()
        if hashlib.sha256(wire).hexdigest() != digest or artifact["as_of"] != observed.isoformat():
            raise ValueError("persisted research response/index integrity mismatch")
        if Contract(wire).to_dict() != self._contract(artifact, "research").to_dict():
            raise ValueError("persisted research response differs from hashed artifact")
        state = State.UNKNOWN if max_age_seconds is None else State.STALE if (cutoff-observed).total_seconds() > max_age_seconds else State.AVAILABLE
        return self._contract(artifact, section, state=state)
