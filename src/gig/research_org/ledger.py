"""Restart-safe ledger. Reads do not create the database."""

from __future__ import annotations

import json
import sqlite3
import threading
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
  run_id TEXT PRIMARY KEY, hypothesis_id TEXT NOT NULL, config_hash TEXT NOT NULL,
  started_at TEXT NOT NULL, finished_at TEXT NOT NULL, payload TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS audit (
  id INTEGER PRIMARY KEY, at TEXT NOT NULL, actor TEXT NOT NULL,
  action TEXT NOT NULL, detail TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS jobs (
  job_id TEXT PRIMARY KEY, hypothesis_id TEXT NOT NULL, status TEXT NOT NULL,
  attempts INTEGER NOT NULL, error TEXT, seq INTEGER NOT NULL, updated_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS objectives (
  id INTEGER PRIMARY KEY, at TEXT NOT NULL, text TEXT NOT NULL,
  disposition TEXT NOT NULL, detail TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS scheduler (
  id INTEGER PRIMARY KEY CHECK (id = 1),
  desired TEXT NOT NULL, cadence_seconds REAL NOT NULL,
  lease_owner TEXT, lease_until TEXT, candidates INTEGER NOT NULL,
  reason TEXT, updated_at TEXT NOT NULL);
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
                return {"run_count": 0, "runs": [], "audit": [], "objectives": []}
            try:
                runs = [dict(row) for row in db.execute(
                    "SELECT run_id, hypothesis_id, config_hash, started_at, finished_at, payload "
                    "FROM runs ORDER BY finished_at DESC LIMIT 20")]
                audit = [dict(row) for row in db.execute(
                    "SELECT at, actor, action, detail FROM audit ORDER BY id DESC LIMIT 20")]
                count = db.execute("SELECT COUNT(*) FROM runs").fetchone()[0]
                objectives = _objectives(db)
            finally:
                db.close()
        for row in runs:
            row["report"] = json.loads(row.pop("payload"))
        return {"run_count": count, "runs": runs, "audit": audit, "objectives": objectives}

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
                    (config_hash, report["hypothesis_id"], config_hash, now, now,
                     json.dumps(report, sort_keys=True, allow_nan=False)))
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

    def jobs(self) -> list[dict[str, Any]]:
        with self._lock:
            db = self._read()
            if db is None:
                return []
            try:
                try:
                    rows = db.execute(
                        "SELECT job_id, hypothesis_id, status, attempts, error, seq FROM jobs ORDER BY seq"
                    ).fetchall()
                except sqlite3.OperationalError:
                    return []
            finally:
                db.close()
        return [dict(row) for row in rows]

    def recover_and_claim(self, approved: Sequence[dict[str, Any]]) -> dict[str, Any] | None:
        """Return a crashed job to the queue, then claim the earliest queued job."""
        with self._lock:
            now = datetime.now(UTC).isoformat()
            db = self._connect()
            try:
                db.execute("BEGIN IMMEDIATE")
                db.execute("UPDATE jobs SET status = 'QUEUED', updated_at = ? WHERE status = 'RUNNING'", (now,))
                for job in approved:
                    db.execute(
                        "INSERT OR IGNORE INTO jobs (job_id, hypothesis_id, status, attempts, error, seq, updated_at) "
                        "VALUES (?, ?, 'QUEUED', 0, NULL, ?, ?)",
                        (job["job_id"], job["hypothesis_id"], int(job["seq"]), now))
                row = db.execute(
                    "SELECT job_id, hypothesis_id, attempts, seq FROM jobs WHERE status = 'QUEUED' ORDER BY seq LIMIT 1"
                ).fetchone()
                if row is None:
                    db.commit()
                    return None
                attempts = int(row["attempts"]) + 1
                db.execute(
                    "UPDATE jobs SET status = 'RUNNING', attempts = ?, updated_at = ? WHERE job_id = ?",
                    (attempts, now, row["job_id"]))
                match = next((item for item in approved if item["job_id"] == row["job_id"]), None)
                runner = None if match is None else match.get("runner")
                db.execute("INSERT INTO audit (at, actor, action, detail) VALUES (?, ?, ?, ?)",
                           (now, "orchestrator", "claim", row["job_id"]))
                db.commit()
            finally:
                db.close()
        return {"job_id": row["job_id"], "hypothesis_id": row["hypothesis_id"],
                "attempts": attempts, "runner": runner}

    def complete_job(self, job_id: str, state: str, error: str | None) -> None:
        if state not in {"DONE", "QUEUED", "FAILED", "CANCELLED"}:
            raise ValueError("job state")
        with self._lock:
            now = datetime.now(UTC).isoformat()
            db = self._connect()
            try:
                db.execute("UPDATE jobs SET status = ?, error = ?, updated_at = ? WHERE job_id = ?",
                           (state, error, now, job_id))
                db.execute("INSERT INTO audit (at, actor, action, detail) VALUES (?, ?, ?, ?)",
                           (now, "orchestrator", state.lower(), job_id))
                db.commit()
            finally:
                db.close()

    def ensure_job(self, job: dict[str, Any]) -> None:
        with self._lock:
            now = datetime.now(UTC).isoformat()
            db = self._connect()
            try:
                db.execute(
                    "INSERT OR IGNORE INTO jobs (job_id, hypothesis_id, status, attempts, error, seq, updated_at) "
                    "VALUES (?, ?, 'QUEUED', 0, NULL, ?, ?)",
                    (job["job_id"], job["hypothesis_id"], int(job["seq"]), now))
                db.commit()
            finally:
                db.close()

    def cancel(self, job_id: str) -> bool:
        """Cancel a job that has not started. A running or finished job stays as it is."""
        with self._lock:
            now = datetime.now(UTC).isoformat()
            db = self._connect()
            try:
                db.execute("BEGIN IMMEDIATE")
                row = db.execute("SELECT status FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
                if row is None or row["status"] != "QUEUED":
                    db.commit()
                    return False
                db.execute("UPDATE jobs SET status = 'CANCELLED', updated_at = ? WHERE job_id = ?",
                           (now, job_id))
                db.execute("INSERT INTO audit (at, actor, action, detail) VALUES (?, ?, ?, ?)",
                           (now, "orchestrator", "cancel", job_id))
                db.commit()
            finally:
                db.close()
        return True

    def record_objective(self, text: str, disposition: str, detail: str) -> None:
        with self._lock:
            now = datetime.now(UTC).isoformat()
            db = self._connect()
            try:
                db.execute(
                    "INSERT INTO objectives (at, text, disposition, detail) VALUES (?, ?, ?, ?)",
                    (now, text, disposition, detail))
                db.execute("INSERT INTO audit (at, actor, action, detail) VALUES (?, ?, ?, ?)",
                           (now, "director", disposition, detail))
                db.commit()
            finally:
                db.close()

    def scheduler_state(self) -> dict[str, Any]:
        with self._lock:
            db = self._read()
            if db is None:
                return _scheduler_default()
            try:
                try:
                    row = db.execute(
                        "SELECT desired, cadence_seconds, lease_owner, lease_until, candidates, reason "
                        "FROM scheduler WHERE id = 1"
                    ).fetchone()
                except sqlite3.OperationalError:
                    return _scheduler_default()
            finally:
                db.close()
        if row is None:
            return _scheduler_default()
        return {"desired": row["desired"], "cadence_seconds": float(row["cadence_seconds"]),
                "lease_owner": row["lease_owner"], "lease_until": row["lease_until"],
                "candidates": int(row["candidates"]), "reason": row["reason"]}

    def set_desired(self, desired: str, cadence_seconds: float | None = None, reason: str | None = None) -> None:
        if desired not in {"STOPPED", "PAUSED", "RUNNING"}:
            raise ValueError("scheduler state")
        with self._lock:
            now = datetime.now(UTC).isoformat()
            db = self._connect()
            try:
                current = db.execute("SELECT cadence_seconds FROM scheduler WHERE id = 1").fetchone()
                cadence = float(cadence_seconds) if cadence_seconds is not None else (
                    float(current["cadence_seconds"]) if current else 5.0)
                db.execute(
                    "INSERT INTO scheduler (id, desired, cadence_seconds, lease_owner, lease_until, candidates, reason, updated_at) "
                    "VALUES (1, ?, ?, NULL, NULL, 0, ?, ?) "
                    "ON CONFLICT(id) DO UPDATE SET desired = excluded.desired, "
                    "cadence_seconds = excluded.cadence_seconds, reason = excluded.reason, updated_at = excluded.updated_at",
                    (desired, cadence, reason, now))
                db.execute("INSERT INTO audit (at, actor, action, detail) VALUES (?, ?, ?, ?)",
                           (now, "scheduler", desired.lower(), reason or ""))
                db.commit()
            finally:
                db.close()

    def lease_blocks(self, owner: str | None) -> bool:
        state = self.scheduler_state()
        until = state.get("lease_until")
        holder = state.get("lease_owner")
        if not until or not holder:
            return False
        if until <= datetime.now(UTC).isoformat():
            return False
        return holder != owner

    def acquire_lease(self, owner: str, seconds: float) -> bool:
        with self._lock:
            now = datetime.now(UTC)
            db = self._connect()
            try:
                db.execute("BEGIN IMMEDIATE")
                row = db.execute("SELECT lease_owner, lease_until FROM scheduler WHERE id = 1").fetchone()
                if (row and row["lease_until"] and row["lease_owner"] not in {None, owner}
                        and row["lease_until"] > now.isoformat()):
                    db.commit()
                    return False
                until = (now + timedelta(seconds=seconds)).isoformat()
                if row is None:
                    db.execute(
                        "INSERT INTO scheduler (id, desired, cadence_seconds, lease_owner, lease_until, candidates, reason, updated_at) "
                        "VALUES (1, 'RUNNING', 5, ?, ?, 0, NULL, ?)",
                        (owner, until, now.isoformat()))
                else:
                    db.execute("UPDATE scheduler SET lease_owner = ?, lease_until = ?, updated_at = ? WHERE id = 1",
                               (owner, until, now.isoformat()))
                db.commit()
            finally:
                db.close()
        return True

    def expire_lease(self) -> None:
        """Mark the lease as already over. Used to simulate a dead worker's clock."""
        with self._lock:
            past = (datetime.now(UTC) - timedelta(seconds=5)).isoformat()
            db = self._connect()
            try:
                db.execute("UPDATE scheduler SET lease_until = ? WHERE id = 1", (past,))
                db.commit()
            finally:
                db.close()

    def release_lease(self, owner: str) -> None:
        with self._lock:
            now = datetime.now(UTC).isoformat()
            db = self._connect()
            try:
                db.execute(
                    "UPDATE scheduler SET lease_until = ?, updated_at = ? WHERE id = 1 AND lease_owner = ?",
                    (now, now, owner))
                db.commit()
            finally:
                db.close()

    def find_hypothesis(self, hypothesis_id: str) -> bool:
        with self._lock:
            db = self._read()
            if db is None:
                return False
            try:
                try:
                    row = db.execute("SELECT 1 FROM runs WHERE hypothesis_id = ? LIMIT 1",
                                     (hypothesis_id,)).fetchone()
                except sqlite3.OperationalError:
                    return False
            finally:
                db.close()
        return row is not None

    def mark_candidate(self) -> None:
        with self._lock:
            now = datetime.now(UTC).isoformat()
            db = self._connect()
            try:
                db.execute(
                    "INSERT INTO scheduler (id, desired, cadence_seconds, lease_owner, lease_until, candidates, reason, updated_at) "
                    "VALUES (1, 'STOPPED', 5, NULL, NULL, 1, NULL, ?) "
                    "ON CONFLICT(id) DO UPDATE SET candidates = candidates + 1, updated_at = excluded.updated_at",
                    (now,))
                db.commit()
            finally:
                db.close()


def _scheduler_default() -> dict[str, Any]:
    return {"desired": "STOPPED", "cadence_seconds": 5.0, "lease_owner": None,
            "lease_until": None, "candidates": 0, "reason": None}


def _objectives(db: sqlite3.Connection) -> list[dict[str, Any]]:
    try:
        rows = db.execute(
            "SELECT at, text, disposition, detail FROM objectives ORDER BY id DESC LIMIT 10"
        ).fetchall()
    except sqlite3.OperationalError:
        return []
    return [dict(row) for row in rows]
