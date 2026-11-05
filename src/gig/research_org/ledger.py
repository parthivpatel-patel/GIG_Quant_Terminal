"""Restart-safe ledger. Reads do not create the database."""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
  run_id TEXT PRIMARY KEY, hypothesis_id TEXT NOT NULL, config_hash TEXT NOT NULL,
  started_at TEXT NOT NULL, finished_at TEXT NOT NULL, payload TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS audit (
  id INTEGER PRIMARY KEY, at TEXT NOT NULL, actor TEXT NOT NULL,
  action TEXT NOT NULL, detail TEXT NOT NULL);
"""
MAX_RUNS = 200


class ResearchLedger:
    def __init__(self, directory: Path):
        self.directory = directory
        self.path = directory / "research-org.sqlite3"
        self._lock = threading.Lock()
        self._ready = False

    def _connect(self) -> sqlite3.Connection:
        self.directory.mkdir(parents=True, exist_ok=True)
        db = sqlite3.connect(self.path)
        db.row_factory = sqlite3.Row
        db.executescript(_SCHEMA)
        self._ready = True
        return db

    def _read(self) -> sqlite3.Connection | None:
        if not self.path.is_file():
            return None
        db = sqlite3.connect(f"file:{self.path}?mode=ro", uri=True)
        db.row_factory = sqlite3.Row
        return db

    def status(self) -> dict[str, Any]:
        with self._lock:
            db = self._read()
            if db is None:
                return {"run_count": 0, "runs": [], "audit": []}
            try:
                runs = [dict(row) for row in db.execute(
                    "SELECT run_id, hypothesis_id, config_hash, started_at, finished_at, payload "
                    "FROM runs ORDER BY finished_at DESC LIMIT 20")]
                audit = [dict(row) for row in db.execute(
                    "SELECT at, actor, action, detail FROM audit ORDER BY id DESC LIMIT 20")]
                count = db.execute("SELECT COUNT(*) FROM runs").fetchone()[0]
            finally:
                db.close()
        for row in runs:
            row["report"] = json.loads(row.pop("payload"))
        return {"run_count": count, "runs": runs, "audit": audit}

    def find(self, config_hash: str) -> dict[str, Any] | None:
        with self._lock:
            db = self._read()
            if db is None:
                return None
            try:
                row = db.execute("SELECT payload, finished_at FROM runs WHERE config_hash = ?",
                                 (config_hash,)).fetchone()
            finally:
                db.close()
        if row is None:
            return None
        return {"finished_at": row["finished_at"], "report": json.loads(row["payload"])}

    def count(self) -> int:
        return int(self.status()["run_count"])

    def append(self, config_hash: str, report: dict[str, Any], actor: str) -> str:
        with self._lock:
            if self.path.is_file():
                probe = sqlite3.connect(f"file:{self.path}?mode=ro", uri=True)
                try:
                    existing = probe.execute("SELECT COUNT(*) FROM runs").fetchone()[0]
                except sqlite3.OperationalError:
                    existing = 0
                finally:
                    probe.close()
            else:
                existing = 0
            if existing >= MAX_RUNS:
                raise RuntimeError("research-org run budget is exhausted")
            now = datetime.now(UTC).isoformat()
            db = self._connect()
            try:
                db.execute(
                    "INSERT INTO runs (run_id, hypothesis_id, config_hash, started_at, finished_at, payload) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (config_hash, report["hypothesis_id"], config_hash, now, now, json.dumps(report, sort_keys=True)))
                db.execute("INSERT INTO audit (at, actor, action, detail) VALUES (?, ?, ?, ?)",
                           (now, actor, "run", report["verdict"]))
                db.commit()
            finally:
                db.close()
        return config_hash

    def note(self, actor: str, action: str, detail: str) -> None:
        with self._lock:
            now = datetime.now(UTC).isoformat()
            db = self._connect()
            try:
                db.execute("INSERT INTO audit (at, actor, action, detail) VALUES (?, ?, ?, ?)",
                           (now, actor, action, detail))
                db.commit()
            finally:
                db.close()
