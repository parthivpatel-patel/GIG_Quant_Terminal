"""Immutable, replayable economic diagnostics over admitted research runs."""
from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

from gig.artifacts import write_immutable
from gig.research.economic_validation import VERSION, configuration, evaluate_runs
from gig.research.model_pipeline import sha256
from gig.research.persisted_data import _read_universes, digest, fixed_path
from gig.service.contracts import Contract, State
from gig.service.dataset_workspace import PersistedDatasetWorkspace
from gig.service.pairs_workspace import PairsResearchWorkspace
from gig.service.persisted_research import PersistedResearchWorkspace
from gig.service.serialization import dumps, loads, public_error
from gig.service.workspace import _age, _fields
from gig.timestamps import aware_timestamp


class EconomicResearchWorkspace:
    def __init__(self, base_dir: Path):
        self.base_dir = Path(base_dir).absolute()

    def _path(self, name: str) -> Path:
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts or "\\" in name:
            raise ValueError("fixed economic evaluation paths required")
        return fixed_path(self.base_dir, "economic-research/"+name)

    @staticmethod
    def _missing():
        return Contract.admit(State.UNAVAILABLE, None, source="persisted-economic-research",
            classification="UNAVAILABLE", scope="workspace.economic-research",
            error=public_error("DATA_UNAVAILABLE"))

    @staticmethod
    def _contract(document, state=State.AVAILABLE, section="research"):
        return Contract.admit(state,
            {"title": "Economic research diagnostics · value not established", **document}
            if state in {State.AVAILABLE, State.STALE} else None,
            source=document["source"], classification="MODEL_ESTIMATE", as_of=document["as_of"],
            source_identity=document["dataset_id"], model_identity=document["report_id"],
            model_version=VERSION, evidence_state="HASH_VERIFIED_RECONSTRUCTED_OOS_RESEARCH",
            readiness="RESEARCH_ONLY", scope="workspace.economic-research."+section)

    def _evaluate(self, request):
        workspace = (PersistedResearchWorkspace(self.base_dir) if request["kind"] == "CPU"
                     else PairsResearchWorkspace(self.base_dir))
        runs = [workspace.load(value) for value in request["run_ids"]]
        if len({run["dataset_id"] for run in runs}) != 1:
            raise ValueError("economic candidates must share one admitted dataset identity")
        dataset = PersistedDatasetWorkspace(self.base_dir).load(runs[0]["dataset_id"])
        hashes = set(dataset["provenance"]["universe_snapshot_hashes"])
        universes = [row for row in _read_universes(self.base_dir, dataset["request"])
                     if row["source_sha256"] in hashes]
        if {row["source_sha256"] for row in universes} != hashes:
            raise ValueError("preserved dataset universe evidence is unavailable")
        return evaluate_runs(runs, dataset, universes, request)

    def run(self, payload: dict[str, Any]) -> Contract:
        request = configuration(payload)
        try:
            body = self._evaluate(request)
        except FileNotFoundError:
            return self._missing()
        identity = digest(body)
        document = {**body, "report_id": identity}
        result = self._contract(document)
        index = self._path("reports.sqlite")
        index.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(index) as connection:
            connection.execute("PRAGMA synchronous=FULL")
            connection.execute("BEGIN IMMEDIATE")
            connection.execute("CREATE TABLE IF NOT EXISTS reports(report_id TEXT PRIMARY KEY,as_of TEXT NOT NULL)")
            previous = connection.execute("SELECT as_of FROM reports WHERE report_id=?", (identity,)).fetchone()
            if previous is not None and previous[0] != body["as_of"]:
                raise ValueError("economic report index conflict")
            if previous is None and connection.execute("SELECT COUNT(*) FROM reports").fetchone()[0] >= 1000:
                raise ValueError("economic report retention bound exceeded")
            write_immutable(self._path("reports/"+identity+".json"), dumps(document).encode())
            connection.execute("INSERT OR IGNORE INTO reports VALUES(?,?)", (identity, body["as_of"]))
        return result

    def load(self, report_id: str) -> dict[str, Any]:
        identity = sha256(report_id)
        content = self._path("reports/"+identity+".json").read_bytes()
        document = loads(content)
        body = {k: v for k, v in document.items() if k != "report_id"}
        if (document.get("report_id") != identity or digest(body) != identity
                or document.get("schema_version") != VERSION or dumps(document).encode() != content):
            raise ValueError("economic report integrity mismatch")
        with sqlite3.connect(self._path("reports.sqlite").as_uri()+"?mode=ro", uri=True) as connection:
            row = connection.execute("SELECT as_of FROM reports WHERE report_id=?", (identity,)).fetchone()
        if row is None or row[0] != document["as_of"]:
            raise ValueError("economic report index differs from immutable evidence")
        if configuration(document["request"]) != document["request"] or self._evaluate(document["request"]) != body:
            raise ValueError("economic deterministic replay differs from source evidence")
        return document

    def replay(self, payload: dict[str, Any]) -> Contract:
        _fields(payload, {"report_id"}, {"report_id"})
        try:
            document = self.load(payload["report_id"])
        except FileNotFoundError:
            return self._missing()
        return self._contract(document)

    def section(self, section: str, as_of: Any, max_age_seconds: float | None) -> Contract | None:
        if section not in {"research", "models", "stat-arb", "signals"}:
            return None
        cutoff, maximum = _age(as_of, max_age_seconds)
        path = self._path("reports.sqlite")
        if not path.is_file():
            return None
        with sqlite3.connect(path.as_uri()+"?mode=ro", uri=True) as connection:
            rows = connection.execute("SELECT report_id,as_of FROM reports ORDER BY rowid LIMIT 1001").fetchall()
        if len(rows) > 1000:
            raise ValueError("economic report read bound exceeded")
        choices = [(aware_timestamp(stamp), number, value) for number, (value, stamp) in enumerate(rows)
                   if aware_timestamp(stamp) <= cutoff]
        if not choices:
            return None
        stamp, _, identity = max(choices)
        document = self.load(identity)
        if aware_timestamp(document["as_of"]) != stamp:
            raise ValueError("economic report index time differs")
        state = State.UNKNOWN if maximum is None else State.STALE if (cutoff-stamp).total_seconds() > maximum else State.AVAILABLE
        return self._contract(document, state, section)
