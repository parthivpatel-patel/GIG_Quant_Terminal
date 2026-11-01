"""Durable caller-supplied local research workspace; never a brokerage client.

Integrity hashes establish admitted content identity, not vendor authenticity.
Reading a missing workspace never creates directories or SQLite databases.
"""

from __future__ import annotations

import hashlib
import math
import sqlite3
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from gig.artifacts import write_immutable
from gig.data.validated import CLASSIFICATIONS, PointInTimeStore, admit_bars
from gig.execution.oms import OfflineOMS, _decimal
from gig.execution.simulator import simulate_signals
from gig.nlp.workflow import (
    ArticleEvidence,
    NewsStore,
    admit_article,
    analyze_articles,
    sentiment_timeline,
)
from gig.research.workflow import ResearchWorkflow
from gig.service.contracts import Contract, State
from gig.service.serialization import dumps, loads, public_error
from gig.timestamps import aware_timestamp

EVIDENCE = "CALLER_SUPPLIED_NOT_VENDOR_VERIFIED"
SUPPORTED = frozenset(
    {
        "command-center",
        "markets",
        "signals",
        "portfolio",
        "risk",
        "stat-arb",
        "news",
        "research",
        "execution",
        "models",
    }
)


def _fields(payload: Any, allowed: set[str], required: set[str]) -> dict[str, Any]:
    if not isinstance(payload, dict) or set(payload) - allowed or not required <= set(payload):
        raise ValueError("unsupported or missing workspace request fields")
    return payload


def _request_snapshot(value: Any) -> dict[str, Any]:
    """Detach bounded JSON request values without coercing exact integers.

    A JSON re-dump/re-parse would replace large integers with wire tags. This
    copy retains their admitted Python value while forbidding custom objects.
    """
    dumps(value)

    def clone(item: Any) -> Any:
        if item is None or type(item) in (bool, int, float, str):
            return item
        if type(item) is list:
            return [clone(child) for child in item]
        if type(item) is dict and all(type(key) is str for key in item):
            return {key: clone(child) for key, child in item.items()}
        raise ValueError("workspace requests require JSON values")

    if type(value) is not dict:
        raise ValueError("workspace requests require an object")
    return clone(value)


def _number(value: Any, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("finite numeric input required")
    try:
        result = float(value)
    except (ValueError, OverflowError):
        raise ValueError("finite numeric input required") from None
    if not math.isfinite(result) or (positive and result <= 0):
        raise ValueError("numeric input is outside its declared domain")
    return result


def _text(value: Any, *, bound: int = 512) -> str:
    if (
        not isinstance(value, str)
        or not value.strip()
        or len(value) > bound
        or any(ord(char) < 32 or 0xD800 <= ord(char) <= 0xDFFF for char in value)
    ):
        raise ValueError("bounded nonempty identity text required")
    return value


def _classification(value: Any) -> str:
    if not isinstance(value, str) or value not in CLASSIFICATIONS:
        raise ValueError("explicit supported source classification required")
    return str(value)


def _currency(value: Any) -> str:
    if not isinstance(value, str) or len(value) != 3 or not value.isascii() or not value.isalpha():
        raise ValueError("currency requires three ASCII letters")
    return value.upper()


def _age(as_of: Any, maximum: Any) -> tuple[pd.Timestamp, float | None]:
    cutoff = aware_timestamp(as_of)
    if maximum is None:
        return cutoff, None
    age = _number(maximum)
    if not 0 <= age <= 366 * 86400:
        raise ValueError("freshness bound must lie between zero and 366 days")
    return cutoff, age


def _state(cutoff: pd.Timestamp, observed: pd.Timestamp, maximum: float | None) -> State:
    if observed > cutoff:
        raise ValueError("result cannot be known at requested point in time")
    if maximum is None:
        return State.UNKNOWN
    return State.STALE if (cutoff - observed).total_seconds() > maximum else State.AVAILABLE


class TerminalWorkspace:
    """One fixed application-owned directory; payloads contain no filesystem paths."""

    def __init__(self, base_dir: Path) -> None:
        self.base_dir = Path(base_dir).absolute()

    def _path(self, name: str) -> Path:
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts or "\\" in name:
            raise ValueError("workspace state path must remain inside its fixed directory")
        path = self.base_dir / name
        # Application paths are fixed or validated content hashes, never caller
        # filenames. Existing symlinks fail rather than redirecting state writes.
        for component in (
            self.base_dir,
            *self.base_dir.parents,
            *path.relative_to(self.base_dir).parents,
        ):
            check = component if component.is_absolute() else self.base_dir / component
            if check.is_symlink():
                raise ValueError("workspace state paths cannot be symbolic links")
        if path.is_symlink():
            raise ValueError("workspace state paths cannot be symbolic links")
        return path

    def _ensure(self) -> None:
        self._path("workspace.sqlite")
        self.base_dir.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self._path("workspace.sqlite")) as connection:
            connection.execute("PRAGMA synchronous=FULL")
            connection.execute(
                "CREATE TABLE IF NOT EXISTS receipts (digest TEXT PRIMARY KEY, kind TEXT NOT NULL, metadata_json TEXT NOT NULL)"
            )
            connection.execute(
                "CREATE TABLE IF NOT EXISTS runs (identity TEXT PRIMARY KEY, kind TEXT NOT NULL, as_of TEXT NOT NULL, source TEXT NOT NULL, classification TEXT NOT NULL, currency TEXT, receipt_hash TEXT NOT NULL, manifest_hash TEXT NOT NULL)"
            )

    def _receipt(self, kind: str, payload: dict[str, Any], metadata: dict[str, Any]) -> str:
        encoded = dumps({"schema_version": "1", "kind": kind, "payload": payload}).encode()
        digest = hashlib.sha256(encoded).hexdigest()
        self._ensure()
        write_immutable(self._path(f"receipts/{digest}.json"), encoded)
        with sqlite3.connect(self._path("workspace.sqlite")) as connection:
            connection.execute("PRAGMA synchronous=FULL")
            connection.execute(
                "INSERT OR IGNORE INTO receipts VALUES (?,?,?)", (digest, kind, dumps(metadata))
            )
        return digest

    def _contract(
        self,
        data: dict[str, Any] | None,
        *,
        scope: str,
        state: State = State.AVAILABLE,
        source: str = "local-workspace",
        classification: str = "MODEL_ESTIMATE",
        as_of: Any | None = None,
        source_identity: str | None = None,
        model_identity: str | None = None,
    ) -> Contract:
        if state in (State.UNKNOWN, State.UNAVAILABLE, State.ERROR):
            data = None
        return Contract.admit(
            state,
            data,
            source=source,
            classification=classification,
            scope=f"workspace.{scope}",
            as_of=None if as_of is None else aware_timestamp(as_of).isoformat(),
            source_identity=source_identity,
            model_identity=model_identity,
            evidence_state=EVIDENCE,
            readiness="RESEARCH_ONLY",
            error=(
                public_error("DATA_UNAVAILABLE")
                if state == State.UNAVAILABLE
                else public_error("INVALID_REQUEST")
                if state == State.ERROR
                else None
            ),
        )

    def _missing(self, section: str, *, unknown: bool = False) -> Contract:
        return self._contract(
            None,
            scope=section,
            state=State.UNKNOWN if unknown else State.UNAVAILABLE,
            classification="UNKNOWN" if unknown else "UNAVAILABLE",
        )

    def ingest_bars(self, payload: dict[str, Any]) -> Contract:
        payload = _request_snapshot(payload)
        _fields(
            payload,
            {"rows", "source", "classification", "corporate_action_policy", "currency"},
            {"rows", "source", "classification", "corporate_action_policy", "currency"},
        )
        rows = payload["rows"]
        if not isinstance(rows, list) or not 1 <= len(rows) <= 2000:
            raise ValueError("bar request requires between one and 2000 rows")
        snapshot = admit_bars(
            rows,
            source=payload["source"],
            classification=payload["classification"],
            corporate_action_policy=payload["corporate_action_policy"],
            currency=_currency(payload["currency"]),
        )
        if len({row["instrument_id"] for row in snapshot.rows}) > 32:
            raise ValueError("workspace bar request exceeds 32 instruments")
        # Check the receipt can cross the boundary before any state is changed.
        dumps(payload)
        self._ensure()
        snapshot.save(self._path("snapshots"))
        result = PointInTimeStore(self._path("bars.sqlite")).ingest(snapshot)
        receipt = self._receipt("bars", payload, {"snapshot_hash": snapshot.sha256})
        observed = max(aware_timestamp(row["available_at"]) for row in snapshot.rows)
        return self._contract(
            {
                "title": "Admitted bars",
                "metrics": {**result, "rows": len(rows)},
                "rows": [],
                "snapshot_hash": snapshot.sha256,
                "source_receipt_sha256": receipt,
                "classification_is_caller_asserted": True,
            },
            scope="bars",
            source=payload["source"],
            classification=payload["classification"],
            as_of=observed,
            source_identity=snapshot.sha256,
        )

    def ingest_news(self, payload: dict[str, Any]) -> Contract:
        payload = _request_snapshot(payload)
        _fields(payload, {"articles", "as_of", "aliases"}, {"articles", "as_of"})
        articles = payload["articles"]
        if not isinstance(articles, list) or not 1 <= len(articles) <= 100:
            raise ValueError("news request requires between one and 100 articles")
        cutoff = aware_timestamp(payload["as_of"])
        aliases = self._aliases(payload.get("aliases", {}))
        evidence = [admit_article(article) for article in articles]
        dumps(payload)
        # News explanations can be much larger than the original source text.
        # Validate the complete candidate response, not only its input schema,
        # before creating a store or publishing any immutable evidence.
        revisions = self._news_revisions(evidence)
        receipt = hashlib.sha256(
            dumps({"schema_version": "1", "kind": "news", "payload": payload}).encode()
        ).hexdigest()
        future_cutoff = max(
            [cutoff, *(aware_timestamp(item.as_dict()["available_at"]) for item in revisions)]
        )
        result = self._news_analysis(revisions, cutoff, aliases)
        checks = [result]
        if future_cutoff != cutoff:
            checks.append(self._news_analysis(revisions, future_cutoff, aliases))
        for candidate in checks:
            self._contract(
                self._news_data(candidate, {"inserted": len(articles), "replayed": len(articles)}, receipt),
                scope="news",
                source="caller-supplied-news",
                classification="MODEL_ESTIMATE",
                as_of=cutoff,
                source_identity=receipt,
            )
        self._ensure()
        store = NewsStore(self._path("news.sqlite"))
        counts = store.ingest(articles)
        receipt = self._receipt(
            "news", payload, {"article_hashes": [item.evidence_hash for item in evidence]}
        )
        self._save_run(
            receipt,
            "news",
            cutoff,
            "caller-supplied-news",
            "MODEL_ESTIMATE",
            None,
            receipt,
            {"aliases": aliases},
        )
        return self._contract(
            self._news_data(result, counts, receipt),
            scope="news",
            source="caller-supplied-news",
            classification="MODEL_ESTIMATE",
            as_of=cutoff,
            source_identity=receipt,
        )

    def _news_revisions(self, additional: list[ArticleEvidence]) -> list[ArticleEvidence]:
        """Read and verify source evidence without creating or changing a store."""
        revisions: dict[tuple[str, str], ArticleEvidence] = {}
        path = self._path("news.sqlite")
        if path.is_file():
            with sqlite3.connect(path.as_uri() + "?mode=ro", uri=True) as connection:
                for digest, identity, available, payload in connection.execute(
                    "SELECT evidence_hash,identity,available_at,payload FROM articles"
                ):
                    item = ArticleEvidence(bytes(payload), digest)
                    decoded = item.as_dict()
                    if (decoded["identity"], decoded["available_at"]) != (identity, available):
                        raise ValueError("article metadata does not match source evidence")
                    revisions[(identity, available)] = item
        for item in additional:
            row = item.as_dict()
            key = (row["identity"], row["available_at"])
            previous = revisions.get(key)
            if previous is not None and previous.payload != item.payload:
                raise ValueError("conflicting article revision; preserve later knowledge timestamp")
            revisions[key] = item
        return list(revisions.values())

    @staticmethod
    def _news_analysis(
        revisions: list[ArticleEvidence], cutoff: pd.Timestamp, aliases: dict[str, str]
    ) -> dict[str, Any]:
        latest: dict[str, ArticleEvidence] = {}
        for item in revisions:
            row = item.as_dict()
            available = aware_timestamp(row["available_at"])
            if available > cutoff:
                continue
            previous = latest.get(row["identity"])
            if previous is None or available > aware_timestamp(previous.as_dict()["available_at"]):
                latest[row["identity"]] = item
        return analyze_articles(latest.values(), as_of=cutoff, aliases=aliases)

    @staticmethod
    def _news_data(result: dict[str, Any], counts: dict[str, int], receipt: str) -> dict[str, Any]:
        return {
            "title": "Caller-supplied news",
            "metrics": {**counts, "visible_articles": len(result["articles"])},
            "rows": result["articles"],
            "analysis": {key: value for key, value in result.items() if key != "articles"},
            "source_receipt_sha256": receipt,
        }

    @staticmethod
    def _aliases(value: Any) -> dict[str, str]:
        if not isinstance(value, dict) or len(value) > 100:
            raise ValueError("alias mapping exceeds bounds")
        return {_text(key, bound=128): _text(symbol, bound=128) for key, symbol in value.items()}

    def run_research(self, payload: dict[str, Any]) -> Contract:
        payload = _request_snapshot(payload)
        _fields(
            payload,
            {
                "rows",
                "source",
                "classification",
                "model",
                "min_train",
                "test_size",
                "embargo",
                "ridge_alpha",
                "seed",
                "as_of",
            },
            {"rows", "source", "classification"},
        )
        rows = payload["rows"]
        if not isinstance(rows, list) or not 20 <= len(rows) <= 1000:
            raise ValueError("research request requires 20 to 1000 rows")
        source, classification = (
            _text(payload["source"]),
            _classification(payload["classification"]),
        )
        names: list[str] | None = None
        timestamps, values, targets, label_ends, available = [], [], [], [], []
        for row in rows:
            _fields(
                row,
                {"timestamp", "features", "target", "label_end", "available_at"},
                {"timestamp", "features", "target", "label_end", "available_at"},
            )
            features = row["features"]
            if not isinstance(features, dict) or not 1 <= len(features) <= 16:
                raise ValueError("each research row requires one to 16 named features")
            keys = sorted(_text(name, bound=64) for name in features)
            if names is not None and names != keys:
                raise ValueError("feature identity must match in every research row")
            names = keys
            timestamps.append(aware_timestamp(row["timestamp"]))
            label_ends.append(aware_timestamp(row["label_end"]))
            available.append(aware_timestamp(row["available_at"]))
            values.append([_number(features[name]) for name in names])
            targets.append(_number(row["target"]))
        index = pd.DatetimeIndex(timestamps)
        if not index.is_unique or not index.is_monotonic_increasing:
            raise ValueError("research timestamps must increase strictly")
        if any(
            end < event or knowledge > event
            for end, knowledge, event in zip(label_ends, available, timestamps, strict=True)
        ):
            raise ValueError("research label or feature availability permits look-ahead")
        observed = aware_timestamp(payload["as_of"]) if "as_of" in payload else max(label_ends)
        if any(end > observed for end in label_ends):
            raise ValueError("research labels are unavailable at requested cutoff")
        parameters = {
            "model": payload.get("model", "ridge"),
            "min_train": payload.get("min_train", 10),
            "test_size": payload.get("test_size", 5),
            "embargo": payload.get("embargo", 1),
            "ridge_alpha": payload.get("ridge_alpha", 1.0),
            "seed": payload.get("seed", 42),
        }
        for name, minimum in (("min_train", 10), ("test_size", 1), ("embargo", 0)):
            value = parameters[name]
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or not minimum <= value <= 1000
            ):
                raise ValueError("invalid bounded walk-forward parameter")
        folds = (len(rows) - parameters["min_train"] - parameters["embargo"]) // parameters[
            "test_size"
        ]
        if not 1 <= folds <= 20:
            raise ValueError("research request must produce one to 20 complete folds")
        _number(parameters["ridge_alpha"], positive=True)
        if parameters["model"] not in {"ridge", "lightgbm"}:
            raise ValueError("supported models are ridge and lightgbm")
        if (
            isinstance(parameters["seed"], bool)
            or not isinstance(parameters["seed"], int)
            or not 0 <= parameters["seed"] < 2**31
        ):
            raise ValueError("research seed must be a bounded integer")
        dumps(payload)
        # Validate completely before creating workspace state. Model availability
        # is checked by ResearchWorkflow and is never replaced by a baseline.
        self._ensure()
        result = ResearchWorkflow(self._path("research")).run(
            pd.DataFrame(values, index=index, columns=names),
            pd.Series(targets, index=index),
            label_end=pd.Series(label_ends, index=index),
            feature_available_at=pd.Series(available, index=index),
            **parameters,
        )
        receipt = self._receipt("research", payload, {"run_id": result["run_id"]})
        model_ids = sorted({fold["model_id"] for fold in result["folds"]})
        summary = {
            "title": "Purged walk-forward research",
            "metrics": result["metrics"],
            "rows": result["predictions"],
            "folds": result["folds"],
            "run_id": result["run_id"],
            "dataset_hash": result["dataset_hash"],
            "config_hash": result["config_hash"],
            "model_ids": model_ids,
            "source_receipt_sha256": receipt,
            "trading_approved": False,
            "economically_validated": False,
        }
        workspace_identity = hashlib.sha256(
            dumps({"run_id": result["run_id"], "receipt": receipt}).encode()
        ).hexdigest()
        self._save_run(
            workspace_identity, "research", observed, source, classification, None, receipt, summary
        )
        return self._contract(
            summary,
            scope="research",
            source=source,
            classification=classification,
            as_of=observed,
            source_identity=receipt,
            model_identity=result["run_id"],
        )

    def simulate(self, payload: dict[str, Any]) -> Contract:
        payload = _request_snapshot(payload)
        _fields(
            payload,
            {
                "scores",
                "prices",
                "sectors",
                "run_id",
                "asof",
                "now",
                "nav",
                "n_long",
                "n_short",
                "source_id",
                "classification",
                "currency",
            },
            {"scores", "prices", "sectors", "run_id", "asof", "now", "source_id", "classification"},
        )
        source, classification = (
            _text(payload["source_id"], bound=128),
            _classification(payload["classification"]),
        )
        if classification not in {"SYNTHETIC", "HISTORICAL"}:
            raise ValueError("offline simulation supports synthetic or historical inputs only")
        run_id = _text(payload["run_id"], bound=128)
        timestamps: list[datetime] = []
        for name in ("asof", "now"):
            stamp = aware_timestamp(payload[name])
            if stamp.nanosecond:
                raise ValueError("offline OMS timestamp precision is limited to microseconds")
            timestamps.append(stamp.to_pydatetime())
        if timestamps[0] > timestamps[1]:
            raise ValueError("simulation price timestamp cannot be in the future")
        scores, prices, sectors = payload["scores"], payload["prices"], payload["sectors"]
        if (
            not isinstance(scores, dict)
            or not isinstance(prices, dict)
            or not isinstance(sectors, dict)
        ):
            raise ValueError("simulation requires identity-keyed mappings")
        if not 2 <= len(scores) <= 32 or not 2 <= len(prices) <= 32 or len(sectors) > 32:
            raise ValueError("simulation requires two to 32 instruments")
        for mapping in (scores, prices, sectors):
            for name in mapping:
                _text(name, bound=128)
        for value in scores.values():
            _number(value)
        for value in prices.values():
            _number(value, positive=True)
            _decimal(str(value))
        for value in sectors.values():
            _text(value, bound=128)
        if set(scores) - prices.keys():
            raise ValueError("each signal requires a reference price")
        if set(scores) - sectors.keys():
            raise ValueError("each signal requires explicit sector identity")
        currency = _currency(payload["currency"]) if "currency" in payload else None
        nav = _number(payload.get("nav", 1_000_000.0), positive=True)
        if not 1000 <= nav <= 1e12:
            raise ValueError("simulated NAV must lie between 1000 and 1e12 caller currency units")
        for name in ("n_long", "n_short"):
            value = payload.get(name, 1)
            if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 16:
                raise ValueError("invalid portfolio side count")
        dumps(payload)
        self._ensure()
        with OfflineOMS(self._path("execution.sqlite")) as oms:
            result = simulate_signals(
                scores,
                prices,
                sectors,
                oms=oms,
                run_id=run_id,
                asof=timestamps[0],
                now=timestamps[1],
                nav=nav,
                n_long=payload.get("n_long", 1),
                n_short=payload.get("n_short", 1),
                source_id=source,
                data_classification=classification,
            )
        receipt = self._receipt("simulation", payload, {"run_id": run_id})
        summary = {
            "title": "Offline execution rehearsal",
            "metrics": {
                "state": result["state"],
                "broker_connected": False,
                "new_simulated_fills": len(result["orders"]),
                "replayed": result.get("replayed", False),
            },
            "rows": result["orders"],
            "result": result,
            "valuation_inputs": {"prices": dict(prices), "nav": nav, "currency": currency},
            "source_receipt_sha256": receipt,
            "trading_approved": False,
        }
        identity = hashlib.sha256(
            dumps({"run_id": run_id, "receipt": receipt}).encode()
        ).hexdigest()
        # Replaying the same admitted request must not overwrite its first
        # report with a different 'new fills=0' response. The replay response is
        # returned to the caller; the original durable evidence stays intact.
        with sqlite3.connect(self._path("workspace.sqlite")) as connection:
            already_recorded = connection.execute(
                "SELECT 1 FROM runs WHERE identity=?", (identity,)
            ).fetchone()
        if not already_recorded:
            self._save_run(
                identity,
                "execution",
                timestamps[1],
                source,
                classification,
                currency,
                receipt,
                summary,
            )
        return self._contract(
            summary,
            scope="execution",
            state=State(result["state"]),
            source=source,
            classification=classification,
            as_of=timestamps[0],
            source_identity=receipt,
        )

    def _save_run(
        self,
        identity: str,
        kind: str,
        observed: Any,
        source: str,
        classification: str,
        currency: str | None,
        receipt: str,
        summary: dict[str, Any],
    ) -> None:
        manifest = {
            "schema_version": "1",
            "identity": identity,
            "kind": kind,
            "as_of": aware_timestamp(observed).isoformat(),
            "source": source,
            "classification": classification,
            "currency": currency,
            "source_receipt_sha256": receipt,
            "evidence_state": EVIDENCE,
            "data": summary,
        }
        content = dumps(manifest).encode()
        digest = hashlib.sha256(content).hexdigest()
        write_immutable(self._path(f"manifests/{digest}.json"), content)
        self._ensure()
        with sqlite3.connect(self._path("workspace.sqlite")) as connection:
            connection.execute("PRAGMA synchronous=FULL")
            existing = connection.execute(
                "SELECT manifest_hash FROM runs WHERE identity=?", (identity,)
            ).fetchone()
            if existing is not None and existing[0] != digest:
                raise ValueError("workspace run identity conflicts with preserved evidence")
            connection.execute(
                "INSERT OR IGNORE INTO runs VALUES (?,?,?,?,?,?,?,?)",
                (
                    identity,
                    kind,
                    manifest["as_of"],
                    source,
                    classification,
                    currency,
                    receipt,
                    digest,
                ),
            )

    def _latest_run(self, kind: str, cutoff: pd.Timestamp) -> dict[str, Any] | None:
        path = self._path("workspace.sqlite")
        if not path.is_file():
            return None
        latest = None
        with sqlite3.connect(path.as_uri() + "?mode=ro", uri=True) as connection:
            for record in connection.execute(
                "SELECT * FROM runs WHERE kind=? ORDER BY rowid", (kind,)
            ):
                observed = aware_timestamp(record[2])
                if observed > cutoff or (
                    latest is not None and observed < aware_timestamp(latest["as_of"])
                ):
                    continue
                raw = self._path(f"manifests/{record[7]}.json").read_bytes()
                if hashlib.sha256(raw).hexdigest() != record[7]:
                    raise ValueError("workspace run manifest failed integrity verification")
                manifest = loads(raw)
                expected = (
                    manifest["identity"],
                    manifest["kind"],
                    manifest["as_of"],
                    manifest["source"],
                    manifest["classification"],
                    manifest["currency"],
                    manifest["source_receipt_sha256"],
                )
                if tuple(record[:7]) != expected:
                    raise ValueError("workspace run metadata differs from immutable manifest")
                receipt = self._path(f"receipts/{record[6]}.json").read_bytes()
                if hashlib.sha256(receipt).hexdigest() != record[6]:
                    raise ValueError("workspace source receipt failed integrity verification")
                if kind == "research":
                    run_id = manifest["data"]["run_id"]
                    if (
                        hashlib.sha256(
                            self._path(f"research/{run_id}.json").read_bytes()
                        ).hexdigest()
                        != run_id
                    ):
                        raise ValueError("research artifact failed identity verification")
                    for model_id in manifest["data"]["model_ids"]:
                        if (
                            hashlib.sha256(
                                self._path(f"research/models/{model_id}.json").read_bytes()
                            ).hexdigest()
                            != model_id
                        ):
                            raise ValueError(
                                "registered research model failed identity verification"
                            )
                latest = manifest
        return latest

    def _bars(self, cutoff: pd.Timestamp, maximum: float | None) -> dict[str, Any] | None:
        path = self._path("bars.sqlite")
        if not path.is_file():
            return None
        # Existing store only; missing paths are guarded before constructor.
        return PointInTimeStore(path).snapshot(cutoff, max_age_seconds=maximum)

    def section(self, section: str, as_of: Any, max_age_seconds: float | None) -> Contract | None:
        if section not in SUPPORTED:
            return None
        cutoff, maximum = _age(as_of, max_age_seconds)
        if section in ("research", "execution", "models"):
            manifest = self._latest_run("research" if section == "models" else section, cutoff)
            if manifest is None:
                return None if section == "models" else self._missing(section)
            observed = aware_timestamp(manifest["as_of"])
            if section == "execution":
                observed = aware_timestamp(manifest["data"]["result"]["asof"])
            result_state = _state(cutoff, observed, maximum)
            if section == "execution" and manifest["data"]["result"]["state"] == "ERROR":
                result_state = State.ERROR
            elif section == "execution" and manifest["data"]["result"]["state"] == "STALE":
                result_state = State.STALE
            data = manifest["data"]
            if section == "models":
                model_rows = []
                for identity in data["model_ids"]:
                    model_state = loads(self._path(f"research/models/{identity}.json").read_bytes())
                    model_rows.append({
                        "model_id": identity, "model": model_state["model"],
                        "training_end": model_state["training_end"],
                        "max_training_label_end": model_state["max_training_label_end"],
                        "trained": True, "trading_approved": False,
                        "library_version": model_state.get("library_version"),
                    })
                data = {"title": "Registered research models", "rows": model_rows,
                        "metrics": {"registered_models": len(model_rows), "trading_approved": False},
                        "run_id": manifest["data"]["run_id"],
                        "description": "Hash-verified models from the latest knowable purged research run; training does not establish economic validation."}
            return self._contract(
                data,
                scope=section,
                state=result_state,
                source=manifest["source"],
                classification=manifest["classification"],
                as_of=observed,
                source_identity=manifest["source_receipt_sha256"],
                model_identity=manifest["data"]["run_id"] if section != "execution" else None,
            )
        if section == "news":
            path = self._path("news.sqlite")
            if not path.is_file():
                return self._missing(section)
            news_manifest = self._latest_run("news", cutoff)
            news = NewsStore(path).snapshot(
                cutoff, aliases={} if news_manifest is None else news_manifest["data"]["aliases"]
            )
            if not news["articles"]:
                return self._missing(section)
            observed = max(aware_timestamp(row["available_at"]) for row in news["articles"])
            return self._contract(
                {
                    "title": "Local financial news",
                    "metrics": {
                        "articles": len(news["articles"]),
                        "mean_lexicon_sentiment": news["mean_sentiment"],
                    },
                    "rows": news["articles"],
                    "sentiment_timeline": sentiment_timeline(news),
                },
                scope=section,
                state=_state(cutoff, observed, maximum),
                source="caller-supplied-news",
                classification="MODEL_ESTIMATE",
                as_of=observed,
            )
        snapshot = self._bars(cutoff, maximum)
        if snapshot is None or not snapshot["rows"]:
            return self._missing(section)
        rows = snapshot["rows"]
        metadata = {
            (row["source"], row["classification"], row["corporate_action_policy"], row["currency"])
            for row in rows
        }
        if len(metadata) != 1:
            return self._missing(section, unknown=True)
        source, classification, adjustment, currency = next(iter(metadata))
        newest: dict[str, dict[str, Any]] = {}
        for row in rows:
            if row["instrument_id"] not in newest or aware_timestamp(
                row["timestamp"]
            ) > aware_timestamp(newest[row["instrument_id"]]["timestamp"]):
                newest[row["instrument_id"]] = row
        if len(newest) > 32:
            return self._missing(section, unknown=True)
        observed = min(aware_timestamp(row["timestamp"]) for row in newest.values())
        state = _state(cutoff, observed, maximum)
        hashes = sorted(snapshot["provenance_hashes"])
        provenance = hashlib.sha256(b"gig.workspace.snapshot-hashes.v1\0")
        for digest in hashes:
            provenance.update(digest.encode("ascii") + b"\n")
        data = {
            "title": section.replace("-", " ").title(),
            "metrics": {},
            "rows": [],
            "currency": currency,
            "corporate_action_policy": adjustment,
            "source_snapshot_hashes": hashes[:256],
            "source_snapshot_hash_count": len(hashes),
            "source_snapshot_hashes_complete": len(hashes) <= 256,
            "source_snapshot_manifest_sha256": provenance.hexdigest(),
            "classification_is_caller_asserted": True,
            "trading_approved": False,
        }
        if section in ("command-center", "markets"):
            data["rows"] = [dict(newest[instrument]) for instrument in sorted(newest)]
            data["metrics"] = {
                "instruments": len(newest),
                "admitted_revisions_visible": len(rows),
                "latest_max_age_seconds": snapshot["latest_instrument_max_age_seconds"],
                "broker_connected": False,
            }
        else:
            if len(rows) > 32_000:
                return self._missing(section, unknown=True)
            panel = pd.DataFrame(rows).pivot(
                index="timestamp", columns="instrument_id", values="close"
            )
            panel.index = pd.DatetimeIndex([aware_timestamp(value) for value in panel.index])
            panel = panel.sort_index().sort_index(axis=1)
            if panel.isna().any().any() or len(panel.columns) > 32:
                return self._missing(section, unknown=True)
            if section == "signals":
                if len(panel) < 21:
                    return self._missing(section)
                changes = panel.pct_change(fill_method=None).iloc[1:]
                momentum = panel.iloc[-1] / panel.iloc[-21] - 1
                reversal = -(panel.iloc[-1] / panel.iloc[-6] - 1)
                volatility = changes.iloc[-20:].std(ddof=1)
                ranks = momentum.rank(method="average", pct=True)
                data["rows"] = [
                    {
                        "instrument_id": name,
                        "momentum_20": float(momentum[name]),
                        "reversal_5": float(reversal[name]),
                        "volatility_20": float(volatility[name]),
                        "research_rank": float(ranks[name]),
                        "trading_approved": False,
                    }
                    for name in panel.columns
                ]
                data["metrics"] = {"lookback_observations": 20, "instruments": len(panel.columns)}
                data["chart"] = {
                    "kind": "factor-space",
                    "title": "Admitted instrument factor observations",
                    "axes": [
                        {
                            "key": "momentum_20",
                            "label": "Momentum, 20 observations",
                            "unit": "arithmetic return",
                        },
                        {
                            "key": "reversal_5",
                            "label": "Reversal, 5 observations",
                            "unit": "negative arithmetic return",
                        },
                        {
                            "key": "volatility_20",
                            "label": "Volatility, 20 observations",
                            "unit": "return std per observation",
                        },
                    ],
                    "points": [{"symbol": row["instrument_id"], **row} for row in data["rows"]],
                }
            elif section == "stat-arb":
                if len(panel.columns) < 2 or len(panel) < 10:
                    return self._missing(section)
                if len(panel) > 2000:
                    return self._missing(section, unknown=True)
                from gig.research.stat_arb import (
                    evaluate_pair,
                    fit_pair,
                    mean_reversion_diagnostics,
                )

                names = list(panel.columns[:2])
                training_rows = max(5, 2 * len(panel) // 3)
                try:
                    model = fit_pair(
                        panel[names[0]],
                        panel[names[1]],
                        dependent=names[0],
                        independent=names[1],
                        training_rows=training_rows,
                    )
                    evaluation = evaluate_pair(
                        model,
                        panel[names[0]].iloc[training_rows:],
                        panel[names[1]].iloc[training_rows:],
                    )
                    diagnostics = (
                        mean_reversion_diagnostics(evaluation["spread"])
                        if len(evaluation["spread"]) >= 5
                        else None
                    )
                except ValueError:
                    return self._missing(section, unknown=True)
                data["metrics"] = {
                    "dependent": names[0],
                    "independent": names[1],
                    "training_rows": training_rows,
                    "hedge_elasticity": model.beta,
                    "stationarity_tested": False,
                }
                data["model"] = asdict(model)
                data["diagnostics"] = (
                    None
                    if diagnostics is None
                    else {
                        key: value for key, value in diagnostics.items() if key != "half_life_years"
                    }
                )
                data["description"] = (
                    "First two sorted instrument identities; descriptive OLS research, no cointegration or trading claim."
                )
                data["rows"] = [
                    {
                        "timestamp": panel.index[training_rows + i].isoformat(),
                        "spread": float(value),
                        "zscore": None
                        if evaluation["zscore"] is None
                        else float(evaluation["zscore"][i]),
                    }
                    for i, value in enumerate(evaluation["spread"])
                ]
            else:
                manifest = self._latest_run("execution", cutoff)
                if manifest is None or manifest["data"]["result"]["state"] != "AVAILABLE":
                    return self._missing(section)
                if (manifest["source"], manifest["classification"], manifest["currency"]) != (
                    source,
                    classification,
                    currency,
                ):
                    return self._missing(section, unknown=True)
                result, inputs = manifest["data"]["result"], manifest["data"]["valuation_inputs"]
                positions = result["positions"]
                oms_path = self._path("execution.sqlite")
                if not oms_path.is_file() or not positions:
                    return self._missing(section)
                with sqlite3.connect(oms_path.as_uri() + "?mode=ro", uri=True) as connection:
                    working = connection.execute(
                        "SELECT 1 FROM orders WHERE state IN ('ACKNOWLEDGED','PARTIALLY_FILLED') LIMIT 1"
                    ).fetchone()
                    incomplete = connection.execute(
                        "SELECT 1 FROM simulation_runs WHERE state='RUNNING' LIMIT 1"
                    ).fetchone()
                    actual_positions = dict(
                        connection.execute("SELECT symbol,quantity FROM positions")
                    )
                if working or incomplete or actual_positions != positions:
                    return self._missing(section, unknown=True)
                if (
                    set(positions) - newest.keys()
                    or len({row["timestamp"] for row in newest.values()}) != 1
                ):
                    return self._missing(section, unknown=True)
                symbols = sorted(positions)
                quantities = np.array([float(positions[name]) for name in symbols])
                prices = np.array([newest[name]["close"] for name in symbols])
                previous = np.array([inputs["prices"][name] for name in symbols])
                nav = float(inputs["nav"] + quantities @ (prices - previous))
                if not math.isfinite(nav) or nav <= 0:
                    return self._missing(section, unknown=True)
                weights = quantities * prices / nav
                data["rows"] = [
                    {
                        "instrument_id": name,
                        "simulated_quantity": positions[name],
                        "marked_price": float(prices[i]),
                        "marked_value": float(quantities[i] * prices[i]),
                        "weight": float(weights[i]),
                        "execution_classification": "SIMULATED",
                    }
                    for i, name in enumerate(symbols)
                ]
                data["metrics"] = {
                    "simulated_marked_nav": nav,
                    "gross_exposure": float(np.abs(weights).sum()),
                    "net_exposure": float(weights.sum()),
                    "broker_connected": False,
                }
                if section == "risk":
                    if len(panel) < 21:
                        return self._missing(section)
                    from gig.risk.analytics import portfolio_risk

                    risk = portfolio_risk(
                        panel[symbols].pct_change(fill_method=None).iloc[1:].to_numpy(),
                        weights,
                        annualization=1,
                    )
                    data["risk"] = risk
                    data["metrics"]["return_volatility_per_observation"] = risk[
                        "annualized_volatility"
                    ]
                    data["description"] = (
                        "Constant current simulated weights on aligned historical returns; annualization factor one, no realized brokerage performance."
                    )
        return self._contract(
            data,
            scope=section,
            state=state,
            source=source,
            classification=classification,
            as_of=observed,
            source_identity=provenance.hexdigest(),
        )
