"""Fixed-path durable point-in-time dataset publication and immutable replay."""

from __future__ import annotations

import math
import re
import sqlite3
from pathlib import Path
from typing import Any

from gig.artifacts import write_immutable
from gig.research.persisted_data import (
    IMPLEMENTATION,
    DatasetUnavailable,
    build_document,
    digest,
    fixed_path,
)
from gig.service.contracts import Contract, State
from gig.service.serialization import dumps, loads, public_error
from gig.timestamps import aware_timestamp


class PersistedDatasetWorkspace:
    def __init__(self, base_dir: Path):
        self.base_dir = Path(base_dir).absolute()

    def _path(self, relative: str) -> Path:
        return fixed_path(self.base_dir, "research-datasets/" + relative)

    @staticmethod
    def _identity(value: Any) -> str:
        if not isinstance(value, str) or re.fullmatch(r"[a-f0-9]{64}", value) is None:
            raise ValueError("a canonical dataset identity is required")
        return value

    @staticmethod
    def _contract(document: dict[str, Any], *, state: State = State.AVAILABLE,
                  section: str = "research") -> Contract:
        data = {"title": "Persisted point-in-time research dataset", **document,
                "metrics": {"training_rows": document["diagnostics"]["training_rows"],
                            "panel_rows": document["diagnostics"]["panel_rows"],
                            "missing_values_imputed": 0, "trading_approved": False}}
        if section == "markets":
            data["rows"] = document["panel_rows"]
        elif section == "operations":
            data["rows"] = [{"diagnostic": k, "value": v} for k, v in document["diagnostics"].items()]
        return Contract.admit(state, data if state in {State.AVAILABLE, State.STALE} else None,
                              source=document["source"], classification=document["classification"],
                              as_of=document["as_of"], scope=f"workspace.persisted-dataset.{section}",
                              source_identity=document["dataset_id"], model_version=IMPLEMENTATION,
                              evidence_state="PERSISTED_PIT_CALLER_EVIDENCE", readiness="RESEARCH_ONLY")

    def build(self, payload: dict[str, Any]) -> Contract:
        try:
            document = build_document(self.base_dir, payload)
        except DatasetUnavailable:
            return Contract.admit(State.UNAVAILABLE, None, source="persisted-point-in-time-data",
                                  classification="UNAVAILABLE", scope="workspace.persisted-dataset.research",
                                  error=public_error("DATA_UNAVAILABLE"))
        result = self._contract(document)
        path = self._path(f"datasets/{document['dataset_id']}.json")
        index = self._path("datasets.sqlite")
        encoded = dumps(document).encode()
        # Verify final response bytes before publication; a killed writer can
        # leave only an unindexed immutable file, never a partial admitted run.
        write_immutable(path, encoded)
        with sqlite3.connect(index) as connection:
            connection.execute("PRAGMA synchronous=FULL")
            connection.execute("CREATE TABLE IF NOT EXISTS datasets (dataset_id TEXT PRIMARY KEY,as_of TEXT NOT NULL)")
            prior = connection.execute("SELECT as_of FROM datasets WHERE dataset_id=?", (document["dataset_id"],)).fetchone()
            if prior is not None and prior[0] != document["as_of"]:
                raise ValueError("dataset index differs from preserved evidence")
            count = connection.execute("SELECT COUNT(*) FROM datasets").fetchone()[0]
            if prior is None and count >= 1000:
                raise ValueError("dataset registry exceeds retention bound")
            connection.execute("INSERT OR IGNORE INTO datasets VALUES(?,?)", (document["dataset_id"], document["as_of"]))
        return result

    def load(self, dataset_id: str) -> dict[str, Any]:
        dataset_id = self._identity(dataset_id)
        index = self._path("datasets.sqlite")
        path = self._path(f"datasets/{dataset_id}.json")
        if not index.is_file() or not path.is_file():
            raise FileNotFoundError("persisted research dataset is unavailable")
        with sqlite3.connect(index.as_uri() + "?mode=ro", uri=True) as connection:
            row = connection.execute("SELECT as_of FROM datasets WHERE dataset_id=?", (dataset_id,)).fetchone()
        if row is None:
            raise FileNotFoundError("persisted research dataset is not published")
        encoded = path.read_bytes()
        document = loads(encoded)
        if not isinstance(document, dict) or document.get("dataset_id") != dataset_id:
            raise ValueError("dataset source identity mismatch")
        body = {k: v for k, v in document.items() if k != "dataset_id"}
        if digest(body) != dataset_id or dumps(document).encode() != encoded:
            raise ValueError("persisted dataset integrity failure")
        if document["schema_version"] != IMPLEMENTATION or document["as_of"] != row[0]:
            raise ValueError("dataset index differs from immutable source")
        self._contract(document)
        return document

    def section(self, section: str, as_of: Any, max_age_seconds: float | None) -> Contract | None:
        if section not in {"research", "markets", "operations"}:
            return None
        cutoff = aware_timestamp(as_of)
        if max_age_seconds is not None and (isinstance(max_age_seconds, bool)
                or not isinstance(max_age_seconds, (int, float)) or not math.isfinite(max_age_seconds)
                or not 0 <= max_age_seconds <= 366 * 86400):
            raise ValueError("bounded explicit dataset freshness policy required")
        index = self._path("datasets.sqlite")
        if not index.is_file():
            return None
        with sqlite3.connect(index.as_uri() + "?mode=ro", uri=True) as connection:
            entries = connection.execute("SELECT dataset_id,as_of FROM datasets LIMIT 1001").fetchall()
        if len(entries) > 1000:
            raise ValueError("dataset registry exceeds bounded read size")
        candidates = [(aware_timestamp(stamp), identity) for identity, stamp in entries
                      if aware_timestamp(stamp) <= cutoff]
        if not candidates:
            return None
        observed, identity = max(candidates)
        document = self.load(identity)
        state = State.UNKNOWN if max_age_seconds is None else State.STALE if (
            cutoff - observed).total_seconds() > max_age_seconds else State.AVAILABLE
        return self._contract(document, state=state, section=section)
