"""Durable hash-verified statistical-arbitrage experiments; no broker authority."""
from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path
from typing import Any

from gig.artifacts import write_immutable
from gig.research.model_pipeline import sha256
from gig.research.persisted_data import digest, fixed_path
from gig.research.persisted_pairs import IMPLEMENTATION, configuration, fit_pairs
from gig.service.contracts import Contract, State
from gig.service.dataset_workspace import PersistedDatasetWorkspace
from gig.service.serialization import dumps, loads, public_error
from gig.service.workspace import _age
from gig.timestamps import aware_timestamp


class PairsResearchWorkspace:
    def __init__(self, base_dir: Path):
        self.base_dir = Path(base_dir).absolute()

    def _path(self, name):
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts or "\\" in name:
            raise ValueError("fixed pairs research paths required")
        return fixed_path(self.base_dir, "pairs-research/" + name)

    @staticmethod
    def _contract(artifact, section="stat-arb", state=State.AVAILABLE):
        data = {"title": "Persisted point-in-time statistical arbitrage", **artifact,
                "rows": artifact["rows"][:500], "rows_total": len(artifact["rows"]),
                "rows_complete": len(artifact["rows"]) <= 500,
                "notes": ["Training-only Engle–Granger screens use Bonferroni correction across all training-eligible pairs.",
                          "Residual ADF is descriptive; post-selection inference and economic value are not validated.",
                          "Kalman innovations are causal. Regime labels are trailing-volatility heuristics, not HMM outputs.",
                          "Fractional-weight backtest assumptions do not establish executable prices, fill quality or cash/share reconciliation."]}
        if section == "models":
            data["rows"] = [dict(row, fold=fold["fold"], training_cutoff=fold["training_cutoff"])
                            for fold in artifact["folds"] for row in fold["screening"]]
        if section in {"signals", "portfolio", "risk"}:
            data["rows"] = artifact["proposals"]
        data["rows_total"] = len(data["rows"])
        data["rows_complete"] = True
        return Contract.admit(state, data if state in {State.AVAILABLE, State.STALE} else None,
            source=artifact["source"], classification="MODEL_ESTIMATE", as_of=artifact["as_of"],
            source_identity=artifact["dataset_id"], model_identity=artifact["run_id"],
            model_version=IMPLEMENTATION, evidence_state="HASH_VERIFIED_PERSISTED_PAIRS_RESEARCH",
            readiness="RESEARCH_ONLY", scope="workspace.pairs." + section)

    @staticmethod
    def _missing():
        return Contract.admit(State.UNAVAILABLE, None, source="persisted-pairs-research",
            classification="UNAVAILABLE", error=public_error("DATA_UNAVAILABLE"), scope="workspace.pairs.stat-arb")

    def run(self, payload: dict[str, Any]) -> Contract:
        config = configuration(payload)
        try:
            dataset = PersistedDatasetWorkspace(self.base_dir).load(sha256(payload["dataset_id"]))
        except FileNotFoundError:
            return self._missing()
        artifact, models = fit_pairs(dataset, config)
        contract = self._contract(artifact)
        encoded = dumps({k: v for k, v in artifact.items() if k != "run_id"}).encode()
        # Complete admission/hash validation precedes every mutable publication.
        if hashlib.sha256(encoded).hexdigest() != artifact["run_id"]:
            raise ValueError("pair run identity mismatch")
        for model_id, value in models.items():
            if digest(value) != model_id:
                raise ValueError("pair model identity mismatch")
        for model_id, value in models.items():
            write_immutable(self._path("models/" + model_id + ".json"), dumps(value).encode())
        write_immutable(self._path("runs/" + artifact["run_id"] + ".json"), encoded)
        with sqlite3.connect(self._path("pairs.sqlite")) as connection:
            connection.execute("PRAGMA synchronous=FULL")
            connection.execute("CREATE TABLE IF NOT EXISTS runs(run_id TEXT PRIMARY KEY,as_of TEXT NOT NULL)")
            previous = connection.execute("SELECT as_of FROM runs WHERE run_id=?", (artifact["run_id"],)).fetchone()
            if previous is not None and previous[0] != artifact["as_of"]:
                raise ValueError("pair run index conflict")
            if previous is None and connection.execute("SELECT COUNT(*) FROM runs").fetchone()[0] >= 1000:
                raise ValueError("pairs experiment retention bound exceeded")
            connection.execute("INSERT OR IGNORE INTO runs VALUES(?,?)", (artifact["run_id"], artifact["as_of"]))
        return contract

    def load(self, run_id: str) -> dict[str, Any]:
        run_id = sha256(run_id)
        path = self._path("pairs.sqlite")
        if not path.is_file():
            raise FileNotFoundError("pair run unavailable")
        with sqlite3.connect(path.as_uri()+"?mode=ro", uri=True) as connection:
            entry = connection.execute("SELECT as_of FROM runs WHERE run_id=?", (run_id,)).fetchone()
        if entry is None:
            raise FileNotFoundError("pair run not published")
        content = self._path("runs/" + run_id + ".json").read_bytes()
        body = loads(content)
        if (not isinstance(body, dict) or hashlib.sha256(content).hexdigest() != run_id
                or body.get("schema_version") != IMPLEMENTATION or dumps(body).encode() != content
                or body["as_of"] != entry[0]):
            raise ValueError("pair run evidence/index integrity failure")
        dataset = PersistedDatasetWorkspace(self.base_dir).load(body["dataset_id"])
        if (dataset["as_of"], dataset["source"], dataset["classification"]) != (
                body["as_of"], body["source"], body["input_classification"]):
            raise ValueError("pair run differs from admitted dataset")
        for model_id in body["model_ids"]:
            sha256(model_id)
            encoded = self._path("models/" + model_id + ".json").read_bytes()
            model = loads(encoded)
            if (hashlib.sha256(encoded).hexdigest() != model_id or model.get("implementation") != IMPLEMENTATION
                    or dumps(model).encode() != encoded):
                raise ValueError("pair model integrity failure")
        for row in body["rows"]:
            if row["model_id"] is not None and row["model_id"] not in body["model_ids"]:
                raise ValueError("unregistered pair model in saved decisions")
        return {"run_id": run_id, **body}

    def replay(self, payload: dict[str, Any]) -> Contract:
        if not isinstance(payload, dict) or set(payload) != {"run_id"}:
            raise ValueError("pair replay requires only a saved run_id")
        try:
            artifact = self.load(sha256(payload["run_id"]))
        except FileNotFoundError:
            return self._missing()
        return self._contract(artifact)

    def section(self, section, as_of, max_age_seconds):
        if section not in {"stat-arb", "research", "models", "signals", "portfolio", "risk"}:
            return None
        cutoff, maximum = _age(as_of, max_age_seconds)
        path = self._path("pairs.sqlite")
        if not path.is_file():
            return None
        with sqlite3.connect(path.as_uri()+"?mode=ro", uri=True) as connection:
            entries = list(connection.execute("SELECT run_id,as_of FROM runs LIMIT 1001"))
        if len(entries) > 1000:
            raise ValueError("bounded pair registry read exceeded")
        eligible = [(aware_timestamp(stamp), identity) for identity, stamp in entries
                    if aware_timestamp(stamp) <= cutoff]
        if not eligible:
            return None
        observed, identity = max(eligible)
        state = State.UNKNOWN if maximum is None else State.STALE if (cutoff-observed).total_seconds() > maximum else State.AVAILABLE
        return self._contract(self.load(identity), section, state)
