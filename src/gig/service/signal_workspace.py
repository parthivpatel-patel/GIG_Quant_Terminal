"""Knowledge-time OOS IC diagnostics and reproducible unapproved score ensembles."""
from __future__ import annotations

import hashlib
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from gig.artifacts import write_immutable
from gig.research.model_pipeline import integer, sha256
from gig.service.contracts import Contract, State
from gig.service.dataset_workspace import PersistedDatasetWorkspace
from gig.service.persisted_research import PersistedResearchWorkspace
from gig.service.serialization import dumps, loads, public_error
from gig.service.workspace import TerminalWorkspace, _age, _fields
from gig.timestamps import aware_timestamp

IMPLEMENTATION = "pit-oos-rank-ic-v1"


def _hash(value):
    return hashlib.sha256(dumps(value).encode()).hexdigest()


def score_runs(runs, dataset, request):
    """Per-date rank IC; only labels strictly known before the selected decision.

    Two-name IC is allowed but explicitly disclosed as a degenerate +/-1 rank
    statistic. It has no significance, probability, or calibrated-confidence claim.
    Negative historical IC never causes an automatic sign reversal.
    """
    now = aware_timestamp(request["now"])
    latest_sets = [{r["timestamp"] for r in run["rows"] if aware_timestamp(r["timestamp"]) <= now}
                   for run in runs]
    common = set.intersection(*latest_sets)
    if not common:
        raise ValueError("model runs have no common out-of-sample decision")
    decision = max(common, key=aware_timestamp)
    cutoff = aware_timestamp(decision)
    labels = {(r["instrument_id"], r["timestamp"]): r for r in dataset["rows"]}
    diagnostics, latest = [], []
    for run in runs:
        by_time = defaultdict(list)
        for row in run["rows"]:
            if aware_timestamp(row["timestamp"]) < cutoff:
                by_time[row["timestamp"]].append(row)
        observations, evidence = [], []
        for stamp in sorted(by_time, key=aware_timestamp):
            predictions, targets, known_rows = [], [], []
            for row in by_time[stamp]:
                source = labels.get((row["instrument_id"], stamp))
                if source is None:
                    continue
                vintages = source.get("label_vintages", [{"target": source["target"],
                    "label_end": source["label_end"], "knowledge_at": source["label_end"]}])
                eligible = [v for v in vintages if aware_timestamp(v["knowledge_at"]) < cutoff
                            and aware_timestamp(v["label_end"]) < cutoff]
                if not eligible or row.get("fit_state", "CONVERGED") != "CONVERGED":
                    continue
                vintage = max(eligible, key=lambda v: (aware_timestamp(v["knowledge_at"]), aware_timestamp(v["label_end"])))
                predictions.append(row["prediction"])
                targets.append(vintage["target"])
                known_rows.append({"instrument_id": row["instrument_id"], "timestamp": stamp,
                    "prediction": row["prediction"], "model_id": row["model_id"], "label": vintage})
            if len(predictions) < 2 or len(set(predictions)) < 2 or len(set(targets)) < 2:
                continue
            a, b = pd.Series(predictions).rank(), pd.Series(targets).rank()
            ic = float(np.corrcoef(a, b)[0, 1])
            if np.isfinite(ic):
                observations.append({"timestamp": stamp, "ic": ic, "instruments": len(predictions)})
                evidence.append(known_rows)
        observations, evidence = observations[-request["lookback"]:], evidence[-request["lookback"]:]
        n = len(observations)
        mean = float(np.mean([r["ic"] for r in observations])) if n else None
        sufficient = n >= request["min_ic_observations"]
        weight = max(0., mean or 0.) * n / (n + 20.) if sufficient else 0.
        selected = [r for r in run["rows"] if r["timestamp"] == decision]
        if any(r.get("fit_state", "CONVERGED") != "CONVERGED" for r in selected):
            weight = 0.
        latest.append(selected)
        diagnostics.append({"run_id": run["run_id"], "model": run["model"],
            "prediction_kind": run["prediction_kind"], "ic_state": "AVAILABLE" if sufficient else "UNKNOWN",
            "mean_oos_rank_ic": mean, "observations": n, "dates": observations,
            "selected_label_evidence_hash": _hash(evidence),
            "two_instrument_dates": sum(r["instruments"] == 2 for r in observations),
            "raw_adaptive_weight": weight, "adaptive_weight": 0.,
            "confidence": {"state": "UNKNOWN", "value": None, "method": "NO_PROBABILITY_CALIBRATION"}})
    total = sum(d["raw_adaptive_weight"] for d in diagnostics)
    symbols = sorted(set.intersection(*[{r["instrument_id"] for r in values} for values in latest]))
    rows = []
    if total > 0 and len(symbols) >= 2:
        blended = {symbol: 0. for symbol in symbols}
        for diagnostic, selected in zip(diagnostics, latest, strict=True):
            diagnostic["adaptive_weight"] = diagnostic["raw_adaptive_weight"] / total
            scores = pd.Series({r["instrument_id"]: r["prediction"] for r in selected}).reindex(symbols)
            rank = scores.rank(method="average")
            centered = (rank - rank.mean()) / max(1., float(rank.max() - rank.min()))
            for symbol in symbols:
                blended[symbol] += diagnostic["adaptive_weight"] * float(centered[symbol])
        rows = [{"instrument_id": s, "timestamp": decision, "prediction": blended[s],
                 "prediction_kind": "IC_WEIGHTED_CENTERED_RANK_SCORE", "trading_approved": False,
                 "confidence": {"state": "UNKNOWN", "value": None},
                 "uncertainty": {"state": "NOT_CALIBRATED", "value": None}} for s in symbols]
    usable = bool(rows and max(r["prediction"] for r in rows) > min(r["prediction"] for r in rows))
    return {"schema_version": IMPLEMENTATION, "request": request, "as_of": now.isoformat(),
        "dataset_id": dataset["dataset_id"], "source": dataset["source"],
        "input_classification": dataset["classification"], "signal_as_of": decision,
        "run_ids": [r["run_id"] for r in runs], "model_ids": sorted({m for r in runs for m in r["model_ids"]}),
        "rows": rows, "diagnostics": diagnostics, "scoring_state": "AVAILABLE" if usable else "UNKNOWN",
        "usable_for_portfolio": usable, "trading_approved": False, "economically_validated": False,
        "metrics": {"eligible_models": sum(d["adaptive_weight"] > 0 for d in diagnostics),
                    "instruments": len(rows), "trading_approved": False},
        "method": "MEAN_PRIOR_OOS_CS_SPEARMAN_POSITIVE_PART_SHRINK_N_OVER_N_PLUS_20",
        "library_versions": {"numpy": np.__version__, "pandas": pd.__version__},
        "notes": ["Latest-decision outcome and later-known label revisions are excluded from IC adaptation.",
                  "Two-name rank IC takes +/-1; neither IC nor shrinkage is calibrated confidence or economic validation.",
                  "Nonpositive or insufficient IC gets zero ensemble weight; no automatic sign reversal."]}


class SignalWorkspace:
    def __init__(self, base_dir: Path):
        self.workspace = TerminalWorkspace(base_dir)
        self.base_dir = self.workspace.base_dir

    def _path(self, name):
        return self.workspace._path("research-signals/" + name)

    @staticmethod
    def _contract(document, state=State.AVAILABLE):
        return Contract.admit(state, {**document, "title": "Unapproved IC-aware research ensemble"}
            if state in {State.AVAILABLE, State.STALE} else None, source=document["source"],
            classification="MODEL_ESTIMATE", as_of=document["as_of"], scope="workspace.research-signals",
            source_identity=document["dataset_id"], model_identity=document["signal_id"],
            model_version=IMPLEMENTATION, evidence_state="HASH_VERIFIED_PIT_OOS_RESEARCH", readiness="RESEARCH_ONLY")

    def run(self, payload: dict[str, Any]) -> Contract:
        _fields(payload, {"run_ids", "now", "lookback", "min_ic_observations"}, {"run_ids", "now"})
        ids = payload["run_ids"]
        if not isinstance(ids, list) or not 1 <= len(ids) <= 8 or len(set(ids)) != len(ids):
            raise ValueError("one to eight distinct saved run identities required")
        request = {"run_ids": sorted(sha256(i) for i in ids), "now": aware_timestamp(payload["now"]).isoformat(),
            "lookback": integer(payload.get("lookback", 20), 2, 200, "lookback"),
            "min_ic_observations": integer(payload.get("min_ic_observations", 5), 2, 200, "min_ic_observations")}
        if request["min_ic_observations"] > request["lookback"]:
            raise ValueError("minimum IC observations exceeds the selected lookback")
        try:
            runs = [PersistedResearchWorkspace(self.base_dir).load(i) for i in request["run_ids"]]
            if len({r["dataset_id"] for r in runs}) != 1:
                raise ValueError("ensemble models must share one admitted dataset identity")
            dataset = PersistedDatasetWorkspace(self.base_dir).load(runs[0]["dataset_id"])
        except FileNotFoundError:
            return Contract.admit(State.UNAVAILABLE, None, source="persisted-research-signals", classification="UNAVAILABLE",
                scope="workspace.research-signals", error=public_error("DATA_UNAVAILABLE"))
        if any(not r["usable_for_portfolio"] or aware_timestamp(r["as_of"]) > aware_timestamp(request["now"]) for r in runs):
            raise ValueError("run is unavailable at this time or is not an eligible forecasting/ranking model")
        body = score_runs(runs, dataset, request)
        identity = _hash(body)
        document = {**body, "signal_id": identity}
        result = self._contract(document)
        self._path("runs").mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self._path("signals.sqlite")) as connection:
            connection.execute("PRAGMA synchronous=FULL")
            connection.execute("CREATE TABLE IF NOT EXISTS signals(identity TEXT PRIMARY KEY,as_of TEXT NOT NULL)")
            count = connection.execute("SELECT COUNT(*) FROM signals").fetchone()[0]
            old = connection.execute("SELECT as_of FROM signals WHERE identity=?", (identity,)).fetchone()
            if old is not None and old[0] != document["as_of"]:
                raise ValueError("signal index conflicts with immutable evidence")
            if old is None and count >= 1000:
                raise ValueError("signal registry exceeds retention bound")
            write_immutable(self._path("runs/" + identity + ".json"), dumps(document).encode())
            connection.execute("INSERT OR IGNORE INTO signals VALUES(?,?)", (identity, document["as_of"]))
        return result

    def load(self, signal_id: str) -> dict[str, Any]:
        identity = sha256(signal_id)
        document = loads(self._path("runs/" + identity + ".json").read_bytes())
        body = {k: v for k, v in document.items() if k != "signal_id"}
        if document.get("signal_id") != identity or _hash(body) != identity or body.get("schema_version") != IMPLEMENTATION:
            raise ValueError("signal artifact integrity mismatch")
        with sqlite3.connect(self._path("signals.sqlite").as_uri() + "?mode=ro", uri=True) as connection:
            row = connection.execute("SELECT as_of FROM signals WHERE identity=?", (identity,)).fetchone()
        if row is None or row[0] != document["as_of"]:
            raise ValueError("signal index differs from preserved artifact")
        dataset = PersistedDatasetWorkspace(self.base_dir).load(document["dataset_id"])
        runs = [PersistedResearchWorkspace(self.base_dir).load(i) for i in document["run_ids"]]
        if score_runs(runs, dataset, document["request"]) != body:
            raise ValueError("signal evidence differs from deterministic replay")
        return document

    def replay(self, payload):
        _fields(payload, {"signal_id"}, {"signal_id"})
        try:
            document = self.load(payload["signal_id"])
        except FileNotFoundError:
            return Contract.admit(State.UNAVAILABLE, None, source="persisted-research-signals", classification="UNAVAILABLE",
                scope="workspace.research-signals", error=public_error("DATA_UNAVAILABLE"))
        return self._contract(document)

    def section(self, section, as_of, maximum):
        if section not in {"signals", "models", "research", "portfolio", "risk"}:
            return None
        at, maximum = _age(as_of, maximum)
        path = self._path("signals.sqlite")
        if not path.is_file():
            return None
        with sqlite3.connect(path.as_uri() + "?mode=ro", uri=True) as connection:
            entries = connection.execute("SELECT identity,as_of FROM signals LIMIT 1001").fetchall()
        if len(entries) > 1000:
            raise ValueError("bounded signal registry read required")
        visible = [(aware_timestamp(stamp), i) for i, stamp in entries if aware_timestamp(stamp) <= at]
        if not visible:
            return None
        stamp, identity = max(visible)
        document = self.load(identity)
        if aware_timestamp(document["as_of"]) != stamp:
            raise ValueError("signal index time differs from evidence")
        state = State.UNKNOWN if maximum is None else State.STALE if (at-stamp).total_seconds() > maximum else State.AVAILABLE
        return self._contract(document, state)
