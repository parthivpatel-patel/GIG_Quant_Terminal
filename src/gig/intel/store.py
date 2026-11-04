"""Restart-safe event store with revision history, deduplication and source health.

Every fetched or admitted record is appended as an immutable revision keyed by
``(event_id, revision_sha256)``. Re-fetching identical content is a no-op; a
changed provider record becomes a new revision. Views at a cutoff only see
revisions acquired at or before that cutoff, so replay cannot see the future.
"""
from __future__ import annotations

import json
import sqlite3
import threading
from collections.abc import Iterable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from gig.intel.events import canonical_hash
from gig.intel.sources import SOURCES
from gig.timestamps import aware_timestamp

SEMANTICS = ("LIVE", "DELAYED", "HISTORICAL", "CACHED", "DEMONSTRATION", "UNAVAILABLE")
_SCHEMA = """
CREATE TABLE IF NOT EXISTS revisions (
  event_id TEXT NOT NULL, revision_sha256 TEXT NOT NULL, provider TEXT NOT NULL,
  classification TEXT NOT NULL, observed_at TEXT NOT NULL, acquired_at TEXT NOT NULL,
  acquired_ns INTEGER NOT NULL, ingested_at TEXT NOT NULL, record TEXT NOT NULL,
  PRIMARY KEY (event_id, revision_sha256));
CREATE INDEX IF NOT EXISTS revisions_acquired ON revisions(acquired_ns);
CREATE TABLE IF NOT EXISTS fetches (
  source_id TEXT NOT NULL, attempted_at TEXT NOT NULL, attempted_ns INTEGER NOT NULL,
  ok INTEGER NOT NULL, http_status INTEGER, error_code TEXT, raw_sha256 TEXT,
  received INTEGER NOT NULL, new_revisions INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS investigations (
  investigation_id TEXT PRIMARY KEY, created_at TEXT NOT NULL, name TEXT NOT NULL, body TEXT NOT NULL);
"""


def _ns(value: Any) -> int:
    return int(aware_timestamp(value).value)


class IntelStore:
    """Reads never create state: the directory and schema appear on the first write."""

    def __init__(self, directory: Path):
        self.directory = directory
        self.path = directory / "intel.sqlite3"
        self._lock = threading.Lock()
        self._ready = False

    def _connect(self) -> sqlite3.Connection:
        if not self._ready:
            self.directory.mkdir(parents=True, exist_ok=True)
        db = sqlite3.connect(self.path, timeout=10, isolation_level=None)
        db.execute("PRAGMA journal_mode=WAL")
        db.execute("PRAGMA synchronous=FULL")
        if not self._ready:
            db.executescript(_SCHEMA)
            self._ready = True
        return db

    def _rows(self, sql: str, params: tuple[Any, ...]) -> list[tuple[Any, ...]]:
        if not self._ready and not self.path.exists():
            return []
        with self._connect() as db:
            return db.execute(sql, params).fetchall()

    def append(self, records: Iterable[dict[str, Any]], *, classification: str, ingested_at: str) -> int:
        """Append revisions atomically; returns the count of new revisions."""
        rows = []
        for record in records:
            record = {**record, "classification": classification}
            rows.append((record["event_id"], record["revision_sha256"], record["provider"], classification,
                         record["observed_at"], record["acquired_at"], _ns(record["acquired_at"]),
                         ingested_at, json.dumps(record, sort_keys=True)))
        with self._lock, self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            before = db.total_changes
            db.executemany("INSERT OR IGNORE INTO revisions VALUES (?,?,?,?,?,?,?,?,?)", rows)
            added = db.total_changes - before
            db.execute("COMMIT")
        return added

    def log_fetch(self, source_id: str, *, attempted_at: str, ok: bool, http_status: int | None,
                  error_code: str | None, raw_sha256: str | None, received: int, new_revisions: int) -> None:
        with self._lock, self._connect() as db:
            db.execute("INSERT INTO fetches VALUES (?,?,?,?,?,?,?,?,?)",
                       (source_id, attempted_at, _ns(attempted_at), int(ok), http_status, error_code,
                        raw_sha256, received, new_revisions))

    def fetch_history(self, source_id: str, limit: int = 20) -> list[dict[str, Any]]:
        rows = self._rows("SELECT attempted_at, ok, http_status, error_code, raw_sha256, received, new_revisions "
                          "FROM fetches WHERE source_id=? ORDER BY attempted_ns DESC LIMIT ?", (source_id, limit))
        keys = ("attempted_at", "ok", "http_status", "error_code", "raw_sha256", "received", "new_revisions")
        return [dict(zip(keys, (*r[:1], bool(r[1]), *r[2:]), strict=True)) for r in rows]

    def next_allowed(self, source_id: str) -> datetime | None:
        """Bounded refresh interval with exponential backoff after consecutive failures."""
        history = self.fetch_history(source_id, 8)
        if not history:
            return None
        interval = SOURCES[source_id]["min_refresh_seconds"]
        failures = 0
        for item in history:
            if item["ok"]:
                break
            failures += 1
        delay = min(6 * 3600, interval * (2 ** failures))
        return aware_timestamp(history[0]["attempted_at"]).to_pydatetime() + timedelta(seconds=delay)

    def semantics(self, source_id: str, *, now: datetime, as_of: datetime) -> dict[str, Any]:
        source = SOURCES[source_id]
        history = [h for h in self.fetch_history(source_id, 50) if aware_timestamp(h["attempted_at"]) <= as_of]
        successes = [h for h in history if h["ok"]]
        state: str
        if (now - as_of) > timedelta(hours=1):
            state = "HISTORICAL" if successes else "UNAVAILABLE"
        elif not successes:
            state = "UNAVAILABLE"
        else:
            age = (as_of - aware_timestamp(successes[0]["attempted_at"]).to_pydatetime()).total_seconds()
            failing = bool(history) and not history[0]["ok"]
            state = "CACHED" if failing or age > source["stale_after_seconds"] else source["semantics"]
        return {"source_id": source_id, "semantics": state,
                "last_success_at": successes[0]["attempted_at"] if successes else None,
                "last_attempt_at": history[0]["attempted_at"] if history else None,
                "last_error": next((h["error_code"] for h in history if not h["ok"]), None) if history and not history[0]["ok"] else None,
                "consecutive_failures": next((i for i, h in enumerate(history) if h["ok"]), len(history)),
                "next_allowed_at": (lambda t: None if t is None else t.isoformat())(self.next_allowed(source_id)),
                "note": source.get("semantics_note")}

    def events(self, *, as_of: datetime, layers: set[str] | None = None, categories: set[str] | None = None,
               since: datetime | None = None, query: str | None = None, limit: int = 1000) -> list[dict[str, Any]]:
        rows = self._rows("SELECT event_id, record, acquired_ns FROM revisions WHERE acquired_ns <= ? ORDER BY acquired_ns",
                          (_ns(as_of),))
        latest: dict[str, dict[str, Any]] = {}
        counts: dict[str, int] = {}
        for event_id, raw, _ in rows:
            latest[event_id] = json.loads(raw)
            counts[event_id] = counts.get(event_id, 0) + 1
        out = []
        needle = (query or "").lower()
        for event_id, record in latest.items():
            if layers and record.get("layer") not in layers:
                continue
            if categories and record.get("category") not in categories:
                continue
            if since and aware_timestamp(record["observed_at"]) < since:
                continue
            if aware_timestamp(record["observed_at"]) > as_of:
                continue
            if needle and needle not in (record.get("title") or "").lower():
                continue
            out.append({**record, "revision_count": counts[event_id]})
        out.sort(key=lambda r: r["observed_at"], reverse=True)
        return out[:limit]

    def save_investigation(self, name: str, body: dict[str, Any]) -> dict[str, Any]:
        created = datetime.now(UTC).isoformat()
        identity = canonical_hash({"name": name, "body": body})
        with self._lock, self._connect() as db:
            db.execute("INSERT OR IGNORE INTO investigations VALUES (?,?,?,?)",
                       (identity, created, name, json.dumps(body, sort_keys=True)))
        return {"investigation_id": identity, "name": name}

    def investigations(self, limit: int = 50) -> list[dict[str, Any]]:
        rows = self._rows("SELECT investigation_id, created_at, name, body FROM investigations "
                          "ORDER BY created_at DESC LIMIT ?", (limit,))
        return [{"investigation_id": r[0], "created_at": r[1], "name": r[2], "body": json.loads(r[3])} for r in rows]
