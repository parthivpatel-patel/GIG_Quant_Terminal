"""Immutable OHLCV admission and revision-aware point-in-time storage.

This module admits caller-provided observations; it never contacts a provider.
Timestamps are timezone-aware ISO 8601 strings, with pandas nanoseconds retained.
``available_at`` records when a revision became knowable, independently of the
observation timestamp. Corporate-action treatment is declared, never guessed.
"""

from __future__ import annotations

import hashlib
import json
import math
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd

from gig.artifacts import write_immutable
from gig.timestamps import aware_timestamp

CLASSIFICATIONS = frozenset({"REAL_TIME", "DELAYED", "SNAPSHOT", "HISTORICAL", "SYNTHETIC"})
ADJUSTMENTS = frozenset({"RAW", "SPLIT_ADJUSTED", "TOTAL_RETURN_ADJUSTED"})
_MAX_ROWS = 100_000


def _timestamp(value: Any) -> str:
    return aware_timestamp(value).isoformat()


def _canonical(value: Any) -> bytes:
    return json.dumps(value, allow_nan=False, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":")).encode("utf-8")


def _finite(value: Any, name: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    try:
        number = float(value)
    except (ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not math.isfinite(number) or (positive and number <= 0):
        raise ValueError(f"{name} must be finite" + (" and positive" if positive else ""))
    return number


@dataclass(frozen=True, slots=True)
class DataSnapshot:
    """Canonical bytes prevent mutations of admitted source data or returned rows."""

    payload: bytes
    sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.payload, bytes) or hashlib.sha256(self.payload).hexdigest() != self.sha256:
            raise ValueError("snapshot hash mismatch")

    def as_dict(self) -> dict[str, Any]:
        return json.loads(self.payload)

    @property
    def rows(self) -> list[dict[str, Any]]:
        return self.as_dict()["rows"]

    def save(self, directory: str | Path) -> Path:
        """Content-addressed immutable file; a conflicting existing object fails."""
        path = Path(directory) / f"{self.sha256}.json"
        return write_immutable(path, self.payload)

    @classmethod
    def load(cls, path: str | Path) -> DataSnapshot:
        source = Path(path)
        payload = source.read_bytes()
        return cls(payload, source.stem)


def admit_bars(
    records: Iterable[Mapping[str, Any]], *, source: str, classification: str,
    corporate_action_policy: str, currency: str = "USD",
) -> DataSnapshot:
    """Admit bounded long-form bars with explicit point-in-time provenance.

    Required keys: instrument_id, timestamp, available_at, open, high, low,
    close, volume. Instrument IDs must already be stable provider/domain IDs;
    a bare ticker is allowed but must not be mistaken for permanent identity.
    ``available_at`` means provider publication/observation knowledge time.
    A bar must be timestamped at its closing observation, not session open.
    """
    if not isinstance(source, str) or not source.strip() or len(source) > 512:
        raise ValueError("source identity is required and bounded")
    if classification not in CLASSIFICATIONS:
        raise ValueError("explicit supported data classification required")
    if corporate_action_policy not in ADJUSTMENTS:
        raise ValueError("corporate-action treatment must be declared")
    if not isinstance(currency, str) or len(currency) != 3 or not currency.isalpha():
        raise ValueError("currency must be a three-letter code")
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    required = {"instrument_id", "timestamp", "available_at", "open", "high", "low", "close", "volume"}
    for raw in records:
        if len(rows) >= _MAX_ROWS:
            raise ValueError("bar admission row bound exceeded")
        if not isinstance(raw, Mapping) or not required.issubset(raw):
            raise ValueError("bar is missing required fields")
        if set(raw) - required:
            raise ValueError("unsupported bar field: preserve extra source evidence separately")
        instrument = raw["instrument_id"]
        if not isinstance(instrument, str) or not instrument.strip() or len(instrument) > 256:
            raise ValueError("instrument identity must be a bounded nonempty string")
        event, available = _timestamp(raw["timestamp"]), _timestamp(raw["available_at"])
        if pd.Timestamp(available) < pd.Timestamp(event):
            raise ValueError("bar cannot be available before its closing observation")
        identity = (instrument, event, available)
        if identity in seen:
            raise ValueError("duplicate instrument/event/revision identity")
        seen.add(identity)
        prices = {name: _finite(raw[name], name, positive=True) for name in ("open", "high", "low", "close")}
        if prices["high"] < max(prices["open"], prices["close"], prices["low"]):
            raise ValueError("high must cover all bar prices")
        if prices["low"] > min(prices["open"], prices["close"], prices["high"]):
            raise ValueError("low must cover all bar prices")
        volume = raw["volume"]
        if isinstance(volume, bool) or not isinstance(volume, int) or not 0 <= volume <= 2**63 - 1:
            raise ValueError("equity volume must be a nonnegative signed-64-bit integer")
        # Volume is encoded as text to avoid binary64 rounding in browser clients.
        rows.append({"instrument_id": instrument, "timestamp": event, "available_at": available,
                     **prices, "volume": str(volume)})
    rows.sort(key=lambda row: (pd.Timestamp(row["timestamp"]), row["instrument_id"], pd.Timestamp(row["available_at"])))
    payload = _canonical({"schema_version": "1", "source": source, "classification": classification,
                          "corporate_action_policy": corporate_action_policy, "currency": currency.upper(),
                          "timestamp_semantics": "closing_observation", "volume_encoding": "integer_string", "rows": rows})
    return DataSnapshot(payload, hashlib.sha256(payload).hexdigest())


class PointInTimeStore:
    """SQLite immutable-revision store, atomic batches, idempotent restart replay.

    A conflicting payload for an existing identity fails; corrected prices need
    a later knowledge timestamp. Source and adjustment policy remain separate.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute("""CREATE TABLE IF NOT EXISTS bar_revisions (
                source TEXT NOT NULL, classification TEXT NOT NULL, adjustment TEXT NOT NULL,
                currency TEXT NOT NULL, instrument TEXT NOT NULL, event TEXT NOT NULL,
                available TEXT NOT NULL, row_json TEXT NOT NULL, snapshot_hash TEXT NOT NULL,
                PRIMARY KEY (source, classification, adjustment, currency, instrument, event, available))""")
            connection.execute("""CREATE TABLE IF NOT EXISTS bar_snapshot_evidence (
                snapshot_hash TEXT PRIMARY KEY, payload BLOB NOT NULL)""")

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(str(self.path), timeout=10)
        connection.execute("PRAGMA synchronous=FULL")
        return connection

    def ingest(self, snapshot: DataSnapshot) -> dict[str, int]:
        data = snapshot.as_dict()
        # Re-admit canonical rows rather than trusting a hash alone as validation.
        raw = [{**row, "volume": int(row["volume"])} for row in data["rows"]]
        checked = admit_bars(raw, source=data["source"], classification=data["classification"],
                             corporate_action_policy=data["corporate_action_policy"], currency=data["currency"])
        if checked.payload != snapshot.payload:
            raise ValueError("noncanonical or invalid snapshot")
        added = replayed = 0
        with self._connect() as connection:
            prior = connection.execute(
                "SELECT payload FROM bar_snapshot_evidence WHERE snapshot_hash=?",
                (snapshot.sha256,),
            ).fetchone()
            if prior is not None and bytes(prior[0]) != snapshot.payload:
                raise ValueError("source snapshot evidence failed hash verification")
            connection.execute("INSERT OR IGNORE INTO bar_snapshot_evidence VALUES (?,?)",
                               (snapshot.sha256, snapshot.payload))
            for row in data["rows"]:
                key = (data["source"], data["classification"], data["corporate_action_policy"], data["currency"],
                       row["instrument_id"], row["timestamp"], row["available_at"])
                encoded = _canonical(row).decode("utf-8")
                existing = connection.execute("SELECT row_json FROM bar_revisions WHERE source=? AND classification=? AND adjustment=? AND currency=? AND instrument=? AND event=? AND available=?", key).fetchone()
                if existing:
                    if existing[0] != encoded:
                        raise ValueError("conflicting revision identity; use a later available_at")
                    replayed += 1
                else:
                    connection.execute("INSERT INTO bar_revisions VALUES (?,?,?,?,?,?,?,?,?)", (*key, encoded, snapshot.sha256))
                    added += 1
        return {"inserted": added, "replayed": replayed}

    def snapshot(self, as_of: Any, *, source: str | None = None, max_age_seconds: float | None = None) -> dict[str, Any]:
        cutoff = pd.Timestamp(_timestamp(as_of))
        if max_age_seconds is not None and (isinstance(max_age_seconds, bool)
                or not isinstance(max_age_seconds, (int, float))
                or not math.isfinite(max_age_seconds) or max_age_seconds < 0):
            raise ValueError("max_age_seconds must be finite and nonnegative")
        query = "SELECT source,classification,adjustment,currency,row_json,snapshot_hash,instrument,event,available FROM bar_revisions"
        params: tuple[str, ...] = ()
        if source is not None:
            query += " WHERE source=?"
            params = (source,)
        latest: dict[tuple[str, ...], tuple[Any, ...]] = {}
        with self._connect() as connection:
            anchors: dict[str, tuple[tuple[str, ...], set[bytes]]] = {}
            # Do timestamp comparisons in pandas: mixed ISO fractional precision
            # is not chronologically sortable with SQLite TEXT comparisons.
            for record in connection.execute(query, params):
                row = json.loads(record[4])
                digest = record[5]
                if digest not in anchors:
                    stored = connection.execute(
                        "SELECT payload FROM bar_snapshot_evidence WHERE snapshot_hash=?", (digest,)
                    ).fetchone()
                    if stored is None:
                        raise ValueError("source snapshot evidence absent; reingest original admitted snapshot")
                    evidence = DataSnapshot(bytes(stored[0]), digest).as_dict()
                    metadata = (evidence["source"], evidence["classification"],
                                evidence["corporate_action_policy"], evidence["currency"])
                    anchors[digest] = (metadata, {_canonical(item) for item in evidence["rows"]})
                metadata, source_rows = anchors[digest]
                if (tuple(record[:4]) != metadata or _canonical(row) not in source_rows
                        or tuple(record[6:]) != (row["instrument_id"], row["timestamp"], row["available_at"])):
                    raise ValueError("stored revision does not match source snapshot evidence")
                if pd.Timestamp(row["available_at"]) > cutoff or pd.Timestamp(row["timestamp"]) > cutoff:
                    continue
                key = (*record[:4], row["instrument_id"], row["timestamp"])
                prior = latest.get(key)
                if prior is None or pd.Timestamp(row["available_at"]) > pd.Timestamp(prior[4]["available_at"]):
                    latest[key] = (*record[:4], row, record[5])
                if len(latest) > _MAX_ROWS:
                    raise ValueError("snapshot row bound exceeded; select a narrower source archive")
        observations = []
        for record in latest.values():
            row = record[4]
            observations.append({**row, "source": record[0], "classification": record[1],
                                 "corporate_action_policy": record[2], "currency": record[3],
                                 "snapshot_hash": record[5]})
        observations.sort(key=lambda row: (pd.Timestamp(row["timestamp"]), row["source"], row["instrument_id"]))
        age = max((cutoff - pd.Timestamp(row["timestamp"])).total_seconds() for row in observations) if observations else None
        # Freshness is assessed on each instrument/source latest bar, not the
        # oldest historical observation retained in the returned snapshot.
        newest: dict[tuple[str, ...], pd.Timestamp] = {}
        for row in observations:
            key = (row["source"], row["classification"], row["corporate_action_policy"],
                   row["currency"], row["instrument_id"])
            newest[key] = max(newest.get(key, pd.Timestamp(row["timestamp"])), pd.Timestamp(row["timestamp"]))
        latest_age = max((cutoff - timestamp).total_seconds() for timestamp in newest.values()) if newest else None
        if not observations:
            state = "UNAVAILABLE"
        elif max_age_seconds is None:
            state = "UNKNOWN"
        elif latest_age is not None and latest_age > max_age_seconds:
            state = "STALE"
        else:
            state = "AVAILABLE"
        return {"state": state, "as_of": cutoff.isoformat(), "rows": observations,
                "latest_instrument_max_age_seconds": latest_age,
                "oldest_observation_age_seconds": age,
                "freshness_policy_seconds": max_age_seconds,
                "provenance_hashes": sorted({row["snapshot_hash"] for row in observations})}
