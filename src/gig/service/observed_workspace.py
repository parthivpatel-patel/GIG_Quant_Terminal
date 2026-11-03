"""Fixed-source observed reference evidence and honest chronological research."""
from __future__ import annotations

import hashlib
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

from gig.artifacts import write_immutable
from gig.data.public_reference import admit_fixture, fetch_reference
from gig.research.reference_research import VERSION, configuration, digest, fit_reference
from gig.service.contracts import Contract, State
from gig.service.serialization import dumps, loads, public_error
from gig.service.workspace import TerminalWorkspace, _age, _fields
from gig.timestamps import aware_timestamp


def _identity(value):
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError("canonical observed reference identity required")
    return value


class ObservedWorkspace:
    def __init__(self, base_dir: Path):
        self.workspace = TerminalWorkspace(base_dir)

    def _path(self, name):
        return self.workspace._path("observed-reference/"+name)

    @staticmethod
    def _missing():
        return Contract.admit(State.UNAVAILABLE, None, source="ecb-reference-research", classification="UNAVAILABLE",
                              scope="workspace.observed-reference", error=public_error("DATA_UNAVAILABLE"))

    @staticmethod
    def _contract(document, *, state=State.AVAILABLE, research=False):
        return Contract.admit(state, {"title": "Acquired ECB reference quote research" if research else "Admitted daily reference observations", **document}
            if state in {State.AVAILABLE, State.STALE} else None,
            source="ECB_REFERENCE_FX" if document.get("classification", document.get("input_classification")) == "HISTORICAL" else "authored-reference-fixture",
            classification="MODEL_ESTIMATE" if research else document["classification"],
            as_of=document["as_of"] if research else document["acquired_at"], scope="workspace.observed-reference",
            source_identity=document["snapshot_id"], model_identity=document.get("evaluation_id", document.get("report_id")), model_version=document["schema_version"] if research else None,
            evidence_state="OBSERVATION_ORDER_RECONSTRUCTION_NOT_HISTORICAL_POINT_IN_TIME" if research else document["source_authenticity"], readiness="RESEARCH_ONLY")

    def _index(self, kind, identity, as_of):
        path = self._path("reference.sqlite")
        with sqlite3.connect(path) as connection:
            connection.execute("PRAGMA synchronous=FULL")
            connection.execute("CREATE TABLE IF NOT EXISTS entries(kind TEXT,identity TEXT,as_of TEXT,PRIMARY KEY(kind,identity))")
            previous = connection.execute("SELECT as_of FROM entries WHERE kind=? AND identity=?", (kind, identity)).fetchone()
            if previous is not None and previous[0] != as_of:
                raise ValueError("reference index conflict")
            if previous is None and connection.execute("SELECT COUNT(*) FROM entries").fetchone()[0] >= 1000:
                raise ValueError("reference evidence retention bound exceeded")
            connection.execute("INSERT OR IGNORE INTO entries VALUES(?,?,?)", (kind, identity, as_of))

    def _admit(self, document, raw):
        identity = digest(document)
        document = {**document, "snapshot_id": identity}
        contract = self._contract(document)
        write_immutable(self._path("raw/"+document["raw_sha256"]+".bin"), raw)
        write_immutable(self._path("snapshots/"+identity+".json"), dumps({k: v for k, v in document.items() if k != "snapshot_id"}).encode())
        self._index("snapshot", identity, document["acquired_at"])
        return contract

    def snapshot(self, snapshot_id):
        identity = _identity(snapshot_id)
        wire = self._path("snapshots/"+identity+".json").read_bytes()
        document = loads(wire)
        if hashlib.sha256(wire).hexdigest() != identity or dumps(document).encode() != wire:
            raise ValueError("reference snapshot integrity failure")
        raw = self._path("raw/"+_identity(document["raw_sha256"])+".bin").read_bytes()
        if hashlib.sha256(raw).hexdigest() != document["raw_sha256"]:
            raise ValueError("reference source evidence integrity failure")
        self._verify_index("snapshot", identity, document["acquired_at"])
        return {**document, "snapshot_id": identity}

    def _verify_index(self, kind, identity, at):
        path = self._path("reference.sqlite")
        if not path.is_file():
            raise FileNotFoundError("reference index unavailable")
        with sqlite3.connect(path.as_uri()+"?mode=ro", uri=True) as connection:
            row = connection.execute("SELECT as_of FROM entries WHERE kind=? AND identity=?", (kind, identity)).fetchone()
        if row is None or row[0] != at:
            raise ValueError("reference index differs from immutable evidence")

    def research(self, payload):
        config = configuration(payload)
        try:
            snapshot = self.snapshot(config["snapshot_id"])
        except FileNotFoundError:
            return self._missing()
        body, models = fit_reference(snapshot, config)
        identity = digest(body)
        result = self._contract({**body, "report_id": identity}, research=True)
        for model_id, model in models.items():
            if digest(model) != model_id:
                raise ValueError("reference model identity failure")
            write_immutable(self._path("models/"+model_id+".json"), dumps(model).encode())
        write_immutable(self._path("reports/"+identity+".json"), dumps(body).encode())
        self._index("report", identity, body["as_of"])
        return result

    def load(self, report_id):
        identity = _identity(report_id)
        wire = self._path("reports/"+identity+".json").read_bytes()
        body = loads(wire)
        if hashlib.sha256(wire).hexdigest() != identity or dumps(body).encode() != wire or body.get("schema_version") != VERSION:
            raise ValueError("reference research integrity failure")
        self._verify_index("report", identity, body["as_of"])
        snapshot = self.snapshot(body["snapshot_id"])
        if snapshot["acquired_at"] != body["actual_acquired_at"]:
            raise ValueError("research acquisition provenance differs")
        for model_id in body["model_ids"]:
            wire = self._path("models/"+_identity(model_id)+".json").read_bytes()
            if hashlib.sha256(wire).hexdigest() != model_id:
                raise ValueError("saved reference model integrity failure")
        return {**body, "report_id": identity}

    def run(self, payload):
        if not isinstance(payload, dict) or "operation" not in payload:
            raise ValueError("explicit reference operation required")
        operation = payload["operation"]
        request = {k: v for k, v in payload.items() if k != "operation"}
        if operation == "admit":
            return self._admit(*admit_fixture(request))
        if operation == "fetch":
            fields = {"provider", "currencies", "start", "end"}
            _fields(request, fields, fields)
            return self._admit(*fetch_reference(request))
        if operation == "research":
            return self.research(request)
        if operation == "evaluate":
            from gig.research.reference_evaluation import configuration, evaluate_reference
            config = configuration(request)
            try:
                report = self.load(config["report_id"])
            except FileNotFoundError:
                return self._missing()
            models = {identity: loads(self._path("models/"+identity+".json").read_bytes()) for identity in report["model_ids"]}
            body = evaluate_reference(report, models, config, datetime.now(UTC).isoformat(),
                                      snapshot=self.snapshot(report["snapshot_id"]))
            identity = digest(body)
            result = self._contract({**body, "evaluation_id": identity}, research=True)
            write_immutable(self._path("evaluations/"+identity+".json"), dumps(body).encode())
            self._index("evaluation", identity, body["as_of"])
            return result
        if operation == "evaluation-replay":
            _fields(request, {"evaluation_id"}, {"evaluation_id"})
            try:
                return self._contract(self.load_evaluation(request["evaluation_id"]), research=True)
            except FileNotFoundError:
                return self._missing()
        if operation == "replay":
            _fields(request, {"report_id"}, {"report_id"})
            try:
                return self._contract(self.load(request["report_id"]), research=True)
            except FileNotFoundError:
                return self._missing()
        raise ValueError("unsupported reference operation")

    def load_evaluation(self, evaluation_id):
        from gig.research.reference_evaluation import (
            SUPPORTED_VERSIONS,
            configuration,
            evaluate_reference,
        )
        identity = _identity(evaluation_id)
        wire = self._path("evaluations/"+identity+".json").read_bytes()
        body = loads(wire)
        if digest(body) != identity or dumps(body).encode() != wire or body.get("schema_version") not in SUPPORTED_VERSIONS:
            raise ValueError("reference evaluation integrity failure")
        self._verify_index("evaluation", identity, body["as_of"])
        report = self.load(body["report_id"])
        models = {key: loads(self._path("models/"+key+".json").read_bytes()) for key in report["model_ids"]}
        if digest(evaluate_reference(report, models, configuration(body["request"]), body["as_of"],
                                     schema_version=body["schema_version"], snapshot=self.snapshot(report["snapshot_id"]))) != identity:
            raise ValueError("reference evaluation replay differs")
        return {**body, "evaluation_id": identity}

    def section(self, section, as_of, max_age_seconds):
        if section not in {"markets", "research", "models", "signals", "operations"}:
            return None
        cutoff, maximum = _age(as_of, max_age_seconds)
        path = self._path("reference.sqlite")
        if not path.is_file():
            return None
        kind = "snapshot" if section in {"markets", "operations"} else "report"
        kinds = (kind,) if kind == "snapshot" else ("report", "evaluation")
        with sqlite3.connect(path.as_uri()+"?mode=ro", uri=True) as connection:
            entries = connection.execute("SELECT kind,identity,as_of FROM entries WHERE kind IN ("+",".join("?" for _ in kinds)+") LIMIT 1001", kinds).fetchall()
        if len(entries) > 1000:
            raise ValueError("reference history read bound exceeded")
        eligible = [(aware_timestamp(stamp), identity, entry_kind) for entry_kind, identity, stamp in entries if aware_timestamp(stamp) <= cutoff]
        if not eligible:
            return None
        observed, identity, kind = max(eligible)
        state = State.UNKNOWN if maximum is None else State.STALE if (cutoff.value-observed.value)/1e9 > maximum else State.AVAILABLE
        document = self.snapshot(identity) if kind == "snapshot" else self.load_evaluation(identity) if kind == "evaluation" else self.load(identity)
        return self._contract(document, state=state, research=kind != "snapshot")
