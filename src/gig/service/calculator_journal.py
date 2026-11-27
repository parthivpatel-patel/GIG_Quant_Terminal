"""Content-addressed journal for the stateless research calculators.

Saving is explicit. The record is the calculator kind and the exact inputs.
Replay runs the calculator again and refuses the record if the result differs.
Nothing here reads an account, a feed, or an order.
"""
from __future__ import annotations

import hashlib
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Any

from gig.service.contracts import Contract, State
from gig.service.quant_analytics import ANALYSES, run
from gig.service.serialization import dumps, loads, public_error
from gig.service.workspace import TerminalWorkspace, _fields

IMPLEMENTATION = "gig.calculator-journal.v1"
MAX_RECORDS = 128
SCHEMA = """
CREATE TABLE IF NOT EXISTS records(
    id TEXT PRIMARY KEY,
    kind TEXT NOT NULL,
    body TEXT NOT NULL
)
"""


def _identity(value: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError("calculator record identity must be a lowercase SHA-256")
    return value


class CalculatorJournal:
    def __init__(self, directory: Path):
        self.path = TerminalWorkspace(directory)._path("calculator-journal")/"journal.sqlite"

    def _connect(self, *, write: bool = False) -> sqlite3.Connection:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        db = sqlite3.connect(self.path)
        db.execute("PRAGMA synchronous=FULL")
        if write:
            db.execute("PRAGMA journal_mode=DELETE")
            db.executescript(SCHEMA)
        return db

    def save(self, kind: str, inputs: dict[str, Any]) -> dict[str, Any]:
        if kind not in ANALYSES:
            raise ValueError("unsupported calculator")
        if not isinstance(inputs, dict):
            raise ValueError("calculator inputs must be an object")
        result = run(kind, inputs)
        body = {"implementation": IMPLEMENTATION, "operation": "calculator", "kind": kind, "inputs": inputs,
                "result": result, "classification": "MODEL_ESTIMATE", "trading_approved": False,
                "generative_state": "UNAVAILABLE", "persistence": "LOCAL_CONTENT_ADDRESSED_RECORD"}
        wire = dumps(body)
        identity = hashlib.sha256(wire.encode()).hexdigest()
        with closing(self._connect(write=True)) as db:
            db.execute("BEGIN IMMEDIATE")
            try:
                stored = db.execute("SELECT body FROM records WHERE id=?", (identity,)).fetchone()
                if stored is None:
                    count = db.execute("SELECT COUNT(*) FROM records").fetchone()[0]
                    if count >= MAX_RECORDS:
                        raise ValueError("calculator journal capacity exhausted")
                    db.execute("INSERT INTO records VALUES(?,?,?)", (identity, kind, wire))
                elif stored[0] != wire:
                    raise ValueError("calculator record conflicts with immutable content")
                db.execute("COMMIT")
            except Exception:
                db.execute("ROLLBACK")
                raise
        return {**body, "record_id": identity}

    def replay(self, record_id: str) -> dict[str, Any]:
        identity = _identity(record_id)
        if not self.path.exists():
            raise FileNotFoundError("calculator record unavailable")
        try:
            with closing(sqlite3.connect(self.path.as_uri()+"?mode=ro", uri=True)) as db:
                row = db.execute("SELECT body FROM records WHERE id=?", (identity,)).fetchone()
        except sqlite3.OperationalError:
            row = None
        if row is None:
            raise FileNotFoundError("calculator record unavailable")
        body = loads(row[0])
        if dumps(body) != row[0] or hashlib.sha256(row[0].encode()).hexdigest() != identity \
                or body.get("implementation") != IMPLEMENTATION:
            raise ValueError("calculator record integrity failed")
        again = run(body["kind"], body["inputs"])
        if dumps(again) != dumps(body["result"]):
            raise ValueError("calculator replay changed")
        return {**body, "record_id": identity, "replay_state": "REPRODUCED_FROM_PINNED_INPUTS"}


class CalculatorJournalWorkspace:
    def __init__(self, directory: Path):
        self.journal = CalculatorJournal(directory)

    @staticmethod
    def _contract(data, *, state=State.AVAILABLE):
        return Contract.admit(state, data, source="user-supplied-research-inputs",
                              classification="MODEL_ESTIMATE" if data else "UNAVAILABLE",
                              scope="workspace.calculator-journal", readiness="RESEARCH_ONLY",
                              evidence_state="RESEARCH_OUTPUT",
                              error=public_error("DATA_UNAVAILABLE") if state == State.UNAVAILABLE else None)

    def run(self, payload: dict[str, Any]) -> Contract:
        if not isinstance(payload, dict) or payload.get("operation") not in ("save", "replay"):
            raise ValueError("calculator journal operation must be save or replay")
        try:
            if payload["operation"] == "save":
                fields = _fields(payload, {"operation", "kind", "inputs"}, {"operation", "kind", "inputs"})
                if not isinstance(fields["kind"], str):
                    raise ValueError("calculator kind must be a string")
                return self._contract(self.journal.save(fields["kind"], fields["inputs"]))
            fields = _fields(payload, {"operation", "record_id"}, {"operation", "record_id"})
            return self._contract(self.journal.replay(fields["record_id"]))
        except FileNotFoundError:
            return self._contract(None, state=State.UNAVAILABLE)
