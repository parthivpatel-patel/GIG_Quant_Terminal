"""Atomic immutable document revisions, point-in-time retrieval and cited replay."""
from __future__ import annotations

import hashlib
import re
import sqlite3
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from gig.financial_intelligence.retrieval import IMPLEMENTATION, rank
from gig.service.serialization import dumps, loads
from gig.timestamps import aware_timestamp

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents(id TEXT PRIMARY KEY, revision_key TEXT UNIQUE NOT NULL,
 logical_id TEXT NOT NULL, known_ns INTEGER NOT NULL, descriptor TEXT NOT NULL, raw BLOB NOT NULL);
CREATE TABLE IF NOT EXISTS passages(document_id TEXT NOT NULL, ordinal INTEGER NOT NULL,
 body TEXT NOT NULL, PRIMARY KEY(document_id,ordinal));
CREATE TABLE IF NOT EXISTS snapshots(id TEXT PRIMARY KEY, body TEXT NOT NULL);
"""
NORMALIZER = "gig.financial-documents.v1"


def digest(value: Any):
    return hashlib.sha256(dumps(value).encode()).hexdigest()


def _sha(value):
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError("financial identity requires lowercase SHA256")
    return value


class FinancialCorpus:
    def __init__(self, directory: Path, *, clock=None):
        self.directory = Path(directory).absolute()
        self.path = self.directory/"documents.sqlite"
        self.clock = clock or (lambda: datetime.now(UTC).isoformat())
        self._safe_path()

    def _safe_path(self):
        if any(p.is_symlink() for p in (self.path, *self.path.parents)):
            raise ValueError("financial workspace cannot follow symlinks")

    def _connect(self, *, write=False):
        self._safe_path()
        if write:
            self.directory.mkdir(parents=True, exist_ok=True)
            connection = sqlite3.connect(self.path, timeout=10, isolation_level=None)
            connection.execute("PRAGMA synchronous=FULL")
            connection.executescript(SCHEMA)
            return connection
        return sqlite3.connect(self.path.as_uri()+"?mode=ro", uri=True, timeout=10)

    @staticmethod
    def _insert_passages(connection, document_id, passages):
        connection.executemany("INSERT INTO passages VALUES(?,?,?)",
            [(document_id, i, dumps(p)) for i, p in enumerate(passages)])

    def admit(self, raw: bytes, metadata: dict, passages: list):
        if not isinstance(raw, bytes) or not 1 <= len(raw) <= 524288 or not 1 <= len(passages) <= 512:
            raise ValueError("financial document budget exceeded")
        if metadata["classification"] not in {"SYNTHETIC", "HISTORICAL"}:
            raise ValueError("explicit document provenance required")
        if metadata["raw_sha256"] != hashlib.sha256(raw).hexdigest():
            raise ValueError("financial source bytes changed")
        encoded = []
        for passage in passages:
            if (not isinstance(passage["text"], str) or not passage["text"].strip() or len(passage["text"]) > 4096
                    or type(passage["source_start"]) is not int or type(passage["source_end"]) is not int
                    or not 0 <= passage["source_start"] < passage["source_end"] <= len(raw)):
                raise ValueError("bounded cited passage required")
            encoded.append(loads(dumps(passage)))
        # Validate token limits before any persistence mutation.
        rank([dict(p, document_id="0"*64, ordinal=i) for i, p in enumerate(encoded)], "validation", 1)
        key = digest({"metadata": metadata, "passage_hashes": [digest(p) for p in encoded], "normalizer": NORMALIZER})
        logical_id = digest({"namespace": metadata["namespace"], "source_url": metadata["source_url"]})
        now = aware_timestamp(self.clock())
        if metadata.get("source_acquired_at") and now < aware_timestamp(metadata["source_acquired_at"]):
            raise ValueError("document cannot precede verified source acquisition")
        with closing(self._connect(write=True)) as db:
            db.execute("BEGIN IMMEDIATE")
            try:
                previous = db.execute("SELECT id FROM documents WHERE revision_key=?", (key,)).fetchone()
                if previous:
                    db.execute("COMMIT")
                    return self.document(previous[0])["metadata"]
                count, latest = db.execute("SELECT COUNT(*), MAX(known_ns) FROM documents").fetchone()
                passage_count = db.execute("SELECT COUNT(*) FROM passages").fetchone()[0]
                if count >= 32 or passage_count+len(passages) > 4096:
                    raise ValueError("durable financial corpus capacity exhausted")
                if latest is not None and now.value <= latest:
                    raise ValueError("new financial evidence requires an advancing acquisition clock")
                descriptor = {**metadata, "known_at": now.isoformat(), "normalizer": NORMALIZER,
                              "passage_hashes": [digest(p) for p in encoded]}
                identity = digest(descriptor)
                db.execute("INSERT INTO documents VALUES(?,?,?,?,?,?)",
                    (identity, key, logical_id, int(now.value), dumps(descriptor), raw))
                self._insert_passages(db, identity, encoded)
                db.execute("COMMIT")
            except Exception:
                db.execute("ROLLBACK")
                raise
        return self.document(identity)["metadata"]

    def admit_authored(self, payload):
        if (not isinstance(payload, dict) or set(payload) != {"source_id", "title", "text", "license"}
                or not isinstance(payload["source_id"], str) or not re.fullmatch(r"[a-z0-9_-]{1,80}", payload["source_id"])
                or not isinstance(payload["title"], str) or not 1 <= len(payload["title"]) <= 256
                or not isinstance(payload["text"], str) or not payload["text"].strip()
                or payload["license"] != "GIG-AUTHORED"):
            raise ValueError("explicit bounded authored text and authorship attestation required")
        raw = payload["text"].encode("utf-8")
        if len(raw) > 32768:
            raise ValueError("authored financial text budget exceeded")
        passages = [{"text": m.group().decode("utf-8"), "source_start": m.start(), "source_end": m.end(),
                     "observation_date": None}
                    for m in re.finditer(rb"[^\r\n]+", raw) if m.group().strip()]
        return self.admit(raw, {"namespace": "CALLER_AUTHORED", "title": payload["title"],
             "source_url": "urn:gig:authored:"+payload["source_id"], "license": "GIG-AUTHORED",
             "license_note": "Caller attestation, not independently verified third-party rights",
             "source_acquired_at": None, "publication_time": None, "publication_state": "UNKNOWN",
             "classification": "SYNTHETIC", "normalization": "EXACT_UTF8_LINES", "raw_sha256": hashlib.sha256(raw).hexdigest()}, passages)

    def document(self, document_id):
        document_id = _sha(document_id)
        if not self.path.exists():
            raise FileNotFoundError("financial document unavailable")
        with closing(self._connect()) as db:
            row = db.execute("SELECT descriptor,raw FROM documents WHERE id=?", (document_id,)).fetchone()
            records = db.execute("SELECT ordinal,body FROM passages WHERE document_id=? ORDER BY ordinal", (document_id,)).fetchall()
        if row is None:
            raise FileNotFoundError("financial document unavailable")
        descriptor, raw = loads(row[0]), row[1]
        if digest(descriptor) != document_id or hashlib.sha256(raw).hexdigest() != descriptor["raw_sha256"]:
            raise ValueError("financial document integrity failed")
        if len(records) != len(descriptor["passage_hashes"]):
            raise ValueError("financial passage completeness failed")
        passages = []
        for ordinal, wire in records:
            p = loads(wire)
            if ordinal != len(passages) or digest(p) != descriptor["passage_hashes"][ordinal]:
                raise ValueError("financial passage integrity failed")
            excerpt = raw[p["source_start"]:p["source_end"]].decode("utf-8")
            if descriptor["normalization"] == "EXACT_UTF8_LINES" and excerpt != p["text"]:
                raise ValueError("authored citation text differs from original bytes")
            passages.append({**p, "document_id": document_id, "ordinal": ordinal,
                "passage_id": digest({"document_id": document_id, "ordinal": ordinal, "passage_hash": digest(p)}),
                "source_url": descriptor["source_url"], "raw_sha256": descriptor["raw_sha256"],
                "source_excerpt": excerpt, "normalization": descriptor["normalization"],
                "classification": descriptor["classification"], "known_at": descriptor["known_at"]})
        return {"metadata": {**descriptor, "document_id": document_id}, "passages": passages}

    def _visible_ids(self, at):
        if not self.path.exists():
            return []
        with closing(self._connect()) as db:
            rows = db.execute("SELECT id,logical_id,known_ns FROM documents WHERE known_ns<=? ORDER BY known_ns,id", (int(at.value),)).fetchall()
        latest = {}
        for identity, logical, _ in rows:
            latest[logical] = identity
        return sorted(latest.values())

    def search(self, query, as_of, *, limit=5, expand=False, answer=False, document_ids=None, save=True):
        at = aware_timestamp(as_of)
        if type(answer) is not bool:
            raise ValueError("exact extractive mode required")
        rank([], query, limit, expand=expand)
        ids = self._visible_ids(at) if document_ids is None else document_ids
        if not ids:
            return None
        passages = []
        for identity in ids:
            document = self.document(identity)
            if aware_timestamp(document["metadata"]["known_at"]) > at:
                raise ValueError("future financial evidence cannot enter retrieval")
            passages.extend(document["passages"])
        results, terms = rank(passages, query, limit, expand=expand)
        body = {"implementation": IMPLEMENTATION, "query": query, "as_of": at.isoformat(),
                "limit": limit, "expand": expand, "expanded_terms": terms,
                "corpus_document_ids": ids, "corpus_identity": digest(ids), "rows": results,
                "answer": results[0]["text"] if answer and results else None,
                "answer_kind": "EXTRACTIVE_PASSAGE_ONLY" if answer else "NOT_REQUESTED",
                "answer_state": "AVAILABLE" if answer and results else "UNAVAILABLE",
                "semantic_state": "UNAVAILABLE", "generative_state": "UNAVAILABLE",
                "monitoring_state": "NOT_RUNNING", "trading_approved": False,
                "classification": "SYNTHETIC" if any(p["classification"] == "SYNTHETIC" for p in passages) else "HISTORICAL"}
        identity = digest(body)
        if save:
            with closing(self._connect(write=True)) as db:
                db.execute("INSERT OR IGNORE INTO snapshots VALUES(?,?)", (identity, dumps(body)))
                existing = db.execute("SELECT body FROM snapshots WHERE id=?", (identity,)).fetchone()
                if existing is None or existing[0] != dumps(body):
                    raise ValueError("financial snapshot conflicts with immutable content")
        return {**body, "snapshot_id": identity}

    def replay(self, snapshot_id):
        identity = _sha(snapshot_id)
        if not self.path.exists():
            raise FileNotFoundError("financial snapshot unavailable")
        with closing(self._connect()) as db:
            row = db.execute("SELECT body FROM snapshots WHERE id=?", (identity,)).fetchone()
        if row is None:
            raise FileNotFoundError("financial snapshot unavailable")
        body = loads(row[0])
        if digest(body) != identity or body["implementation"] != IMPLEMENTATION:
            raise ValueError("financial snapshot integrity failed")
        reproduced = self.search(body["query"], body["as_of"], limit=body["limit"], expand=body["expand"],
            answer=body["answer_kind"] == "EXTRACTIVE_PASSAGE_ONLY", document_ids=body["corpus_document_ids"], save=False)
        if reproduced is None or reproduced["snapshot_id"] != identity:
            raise ValueError("financial retrieval replay changed")
        return reproduced
