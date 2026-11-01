"""Explicit local FinBERT research annotations with durable source evidence.

Inference requires a fixed application-selected checkpoint. Requests cannot
select a path, download weights, connect to a provider, or approve trading use.
"""

from __future__ import annotations

import hashlib
import json
import math
import sqlite3
from pathlib import Path
from typing import Any

from gig.artifacts import write_immutable
from gig.nlp.workflow import LocalFinBERT, NewsStore, analyze_articles, sentiment_timeline
from gig.service.contracts import Contract, State
from gig.service.serialization import dumps
from gig.timestamps import aware_timestamp


class SentimentWorkspace:
    def __init__(self, base_dir: Path, model_dir: Path | None):
        self.base_dir = Path(base_dir).absolute()
        self.model_dir = None if model_dir is None else Path(model_dir).absolute()
        self._model: LocalFinBERT | None = None

    def _path(self, name: str) -> Path:
        path = self.base_dir / name
        if Path(name).is_absolute() or ".." in Path(name).parts or "\\" in name:
            raise ValueError("fixed local sentiment paths required")
        if any(part.is_symlink() for part in (path, *path.parents)):
            raise ValueError("sentiment state cannot follow symbolic links")
        return path

    def _load(self) -> LocalFinBERT:
        if self.model_dir is None:
            raise ImportError("No local verified FinBERT checkpoint is configured")
        if self._model is None:
            self._model = LocalFinBERT(self.model_dir)
        return self._model

    @staticmethod
    def _envelope(
        data: dict[str, Any],
        *,
        observed: str,
        model: dict[str, Any],
        state: State = State.AVAILABLE,
        scope: str = "workspace.finbert",
    ) -> Contract:
        return Contract.admit(
            state,
            data if state != State.UNKNOWN else None,
            source="caller-news-and-local-finbert",
            classification="MODEL_ESTIMATE",
            as_of=observed,
            scope=scope,
            model_identity=model["model_manifest_hash"],
            model_version=model["revision"],
            evidence_state="CALLER_EVIDENCE_AND_HASH_VERIFIED_LOCAL_MODEL",
            readiness="NONCOMMERCIAL_RESEARCH_ONLY",
        )

    def analyze(self, payload: dict[str, Any]) -> Contract:
        if (
            not isinstance(payload, dict)
            or set(payload) - {"articles", "as_of", "aliases"}
            or not {"articles", "as_of"} <= set(payload)
        ):
            raise ValueError("unsupported or missing sentiment request fields")
        # Plain JSON clone retains exact integers and source strings; no scalar
        # tags are re-admitted as if they were original field values.
        dumps(payload)
        request = json.loads(json.dumps(payload, allow_nan=False, ensure_ascii=False))
        articles = request["articles"]
        if not isinstance(articles, list) or not 1 <= len(articles) <= 20:
            raise ValueError("FinBERT request requires one to twenty articles")
        cutoff = aware_timestamp(request["as_of"])
        aliases = request.get("aliases", {})
        if (
            not isinstance(aliases, dict)
            or len(aliases) > 100
            or any(
                not isinstance(k, str)
                or not isinstance(v, str)
                or not k
                or not v
                or len(k) > 128
                or len(v) > 128
                for k, v in aliases.items()
            )
        ):
            raise ValueError("bounded alias mapping required")
        # Validate source before loading a 400MB model or changing state.
        admitted = analyze_articles(articles, as_of=cutoff, aliases=aliases)
        if not admitted["articles"]:
            raise ValueError("no articles are knowable at the requested cutoff")
        model = self._load()
        analysis = analyze_articles(articles, as_of=cutoff, aliases=aliases, finbert=model)
        observed = max(
            aware_timestamp(row["available_at"]) for row in analysis["articles"]
        ).isoformat()
        metadata = {
            "model_id": "ProsusAI/finbert",
            "model_manifest_hash": model.identity,
            "revision": model.revision,
            "license": model.license,
            "attribution": model.attribution,
            "license_url": model.license_url,
            "source_metadata_url": model.source_metadata_url,
            "device": "cpu",
            "trained": True,
            "load_verified": True,
            "commercial_use_permitted": False,
            "trading_approved": False,
            "evaluation_scope": "RETROSPECTIVE_NONCOMMERCIAL_RESEARCH",
            "historical_model_availability": "UNVERIFIED",
            "model_training_data_cutoff": None,
        }
        data = {
            "title": "Local FinBERT news research",
            "metrics": {
                "articles": len(analysis["articles"]),
                "mean_finbert_sentiment": analysis["mean_sentiment"],
                "broker_connected": False,
                "trading_approved": False,
            },
            "rows": analysis["articles"],
            "sentiment_timeline": sentiment_timeline(analysis),
            "model": metadata,
            "analysis_as_of": cutoff.isoformat(),
            "input_sha256": hashlib.sha256(dumps(request).encode()).hexdigest(),
        }
        contract = self._envelope(data, observed=observed, model=metadata)
        # The complete response is admitted before any SQLite or file mutation.
        directory = self._path("finbert-runs")
        digest = hashlib.sha256(contract.wire).hexdigest()
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self._path("finbert-news.sqlite")
        NewsStore(self._path("finbert-news.sqlite")).ingest(articles)
        write_immutable(directory / f"{digest}.json", contract.wire)
        with sqlite3.connect(self._path("finbert-runs.sqlite")) as connection:
            connection.execute("PRAGMA synchronous=FULL")
            connection.execute(
                "CREATE TABLE IF NOT EXISTS runs (digest TEXT PRIMARY KEY, cutoff TEXT NOT NULL)"
            )
            connection.execute(
                "INSERT OR IGNORE INTO runs VALUES (?,?)", (digest, cutoff.isoformat())
            )
        return contract

    def section(self, section: str, as_of: Any, max_age_seconds: float | None) -> Contract | None:
        if section not in {"news", "models"}:
            return None
        cutoff = aware_timestamp(as_of)
        if max_age_seconds is not None and (
            isinstance(max_age_seconds, bool)
            or not isinstance(max_age_seconds, (float, int))
            or not math.isfinite(max_age_seconds)
            or not 0 <= max_age_seconds <= 366 * 86400
        ):
            raise ValueError("bounded explicit freshness policy required")
        database = self._path("finbert-runs.sqlite")
        if not database.exists():
            return None
        best = None
        with sqlite3.connect(database.as_uri() + "?mode=ro", uri=True) as connection:
            for digest, recorded_cutoff in connection.execute(
                "SELECT digest,cutoff FROM runs ORDER BY rowid"
            ):
                recorded = aware_timestamp(recorded_cutoff)
                if recorded > cutoff or (best is not None and recorded < best[0]):
                    continue
                if (
                    not isinstance(digest, str)
                    or len(digest) != 64
                    or any(c not in "0123456789abcdef" for c in digest)
                ):
                    raise ValueError("invalid local sentiment artifact identity")
                content = self._path(f"finbert-runs/{digest}.json").read_bytes()
                if hashlib.sha256(content).hexdigest() != digest:
                    raise ValueError("local sentiment artifact hash mismatch")
                contract = Contract(content)
                saved = contract.to_dict()
                if saved["data"].get("analysis_as_of") != recorded.isoformat():
                    raise ValueError("local sentiment cutoff differs from immutable artifact")
                if aware_timestamp(saved["provenance"]["as_of"]) > cutoff:
                    continue
                best = (recorded, saved)
        if best is None:
            return None
        saved = best[1]
        data = saved["data"]
        observed = aware_timestamp(saved["provenance"]["as_of"])
        state = (
            State.UNKNOWN
            if max_age_seconds is None
            else (
                State.STALE
                if (cutoff - observed).total_seconds() > max_age_seconds
                else State.AVAILABLE
            )
        )
        if section == "models":
            data = {
                "title": "Verified local FinBERT model",
                "metrics": {"verified_models": 1, "trading_approved": False},
                "rows": [data["model"]],
                "model": data["model"],
            }
        return self._envelope(
            data,
            observed=observed.isoformat(),
            model=data["model"],
            state=state,
            scope=f"workspace.finbert.{section}",
        )
