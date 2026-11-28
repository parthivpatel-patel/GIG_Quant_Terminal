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
from gig.research.factor_screen import FACTOR_SCHEMA, score_factor_dataset
from gig.research.model_pipeline import integer, sha256
from gig.research.workflow import drift_report
from gig.service.contracts import Contract, State
from gig.service.dataset_workspace import PersistedDatasetWorkspace
from gig.service.persisted_research import PersistedResearchWorkspace
from gig.service.serialization import dumps, loads, public_error
from gig.service.workspace import TerminalWorkspace, _age, _fields
from gig.timestamps import aware_timestamp

IMPLEMENTATION = "pit-oos-rank-ic-v1"
PAIRS_SCHEMA = "pair-weight-signal-v1"
SCHEMAS = frozenset({IMPLEMENTATION, FACTOR_SCHEMA, PAIRS_SCHEMA})
PREDICTION_KIND_PAIRS = "UNAPPROVED_PAIR_TARGET_WEIGHT_SCORE"


def _hash(value):
    return hashlib.sha256(dumps(value).encode()).hexdigest()


def score_pair_run(pair_run: dict[str, Any], request: dict[str, Any]) -> dict[str, Any]:
    """Collapse latest available pair proposals into centered research scores."""
    now = aware_timestamp(request["now"])
    proposals = [
        proposal for proposal in pair_run.get("proposals", [])
        if proposal.get("state") == "AVAILABLE"
        and aware_timestamp(proposal["timestamp"]) <= now
        and isinstance(proposal.get("weights"), dict)
    ]
    if not proposals:
        raise ValueError("no available pair proposals at or before now")
    latest = max(aware_timestamp(proposal["timestamp"]) for proposal in proposals)
    weights: dict[str, float] = {}
    for proposal in proposals:
        if aware_timestamp(proposal["timestamp"]) != latest:
            continue
        for symbol, weight in proposal["weights"].items():
            value = float(weight)
            if not np.isfinite(value):
                raise ValueError("pair proposal weights must be finite")
            weights[symbol] = weights.get(symbol, 0.0) + value
    if len(weights) < 2:
        raise ValueError("pair proposals must cover at least two instruments")
    series = pd.Series(weights, dtype=float).sort_index()
    if series.nunique(dropna=False) < 2:
        raise ValueError("pair weights lack cross-sectional variation")
    centered = series - float(series.mean())
    decision = latest.isoformat()
    rows = [{
        "instrument_id": symbol,
        "timestamp": decision,
        "prediction": float(centered[symbol]),
        "prediction_kind": PREDICTION_KIND_PAIRS,
        "trading_approved": False,
        "confidence": {"state": "UNKNOWN", "value": None},
        "uncertainty": {"state": "NOT_CALIBRATED", "value": None},
    } for symbol in centered.index]
    usable = bool(rows and max(r["prediction"] for r in rows) > min(r["prediction"] for r in rows))
    return {
        "schema_version": PAIRS_SCHEMA,
        "request": request,
        "as_of": now.isoformat(),
        "dataset_id": pair_run["dataset_id"],
        "source": pair_run["source"],
        "input_classification": pair_run["input_classification"],
        "signal_as_of": decision,
        "run_ids": [],
        "model_ids": list(pair_run.get("model_ids", [])),
        "pair_run_id": pair_run["run_id"],
        "rows": rows,
        "diagnostics": [{
            "factor": "pair_weights",
            "proposals": len(proposals),
            "instruments": len(rows),
            "confidence": {"state": "UNKNOWN", "value": None,
                           "method": "NO_PROBABILITY_CALIBRATION"},
        }],
        "scoring_state": "AVAILABLE" if usable else "UNKNOWN",
        "usable_for_portfolio": usable,
        "trading_approved": False,
        "economically_validated": False,
        "metrics": {"eligible_models": 0, "instruments": len(rows), "trading_approved": False},
        "method": "PAIR_PROPOSAL_CENTERED_WEIGHT_SCORE_V1",
        "library_versions": {"numpy": np.__version__, "pandas": pd.__version__},
        "notes": [
            "Pair scores sum latest AVAILABLE proposal weights per instrument, then center.",
            "They inherit Engle–Granger / Kalman research caveats from the pair run.",
            "No broker order or trading approval is established.",
        ],
    }


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
        ics = [float(row["ic"]) for row in observations]
        if len(ics) >= 20:
            mid = len(ics) // 2
            ic_drift = drift_report(ics[:mid], ics[mid:], bins=5)
        else:
            ic_drift = {
                "psi": None, "standardized_mean_shift": None,
                "reference_count": min(len(ics), len(ics) // 2),
                "current_count": max(0, len(ics) - len(ics) // 2),
                "bins": 0, "binning": "reference_quantiles", "pseudocount": .5,
                "state": "UNKNOWN",
            }
        limit = request.get("max_ic_drift_psi")
        if (limit is not None and ic_drift.get("state") == "AVAILABLE"
                and ic_drift.get("psi") is not None and float(ic_drift["psi"]) > float(limit)):
            weight = 0.
            ic_drift = {**ic_drift, "weight_zeroed": True, "max_ic_drift_psi": float(limit)}
        else:
            ic_drift = {**ic_drift, "weight_zeroed": False,
                        "max_ic_drift_psi": None if limit is None else float(limit)}
        selected = [r for r in run["rows"] if r["timestamp"] == decision]
        if any(r.get("fit_state", "CONVERGED") != "CONVERGED" for r in selected):
            weight = 0.
        latest.append(selected)
        diagnostics.append({"run_id": run["run_id"], "model": run["model"],
            "prediction_kind": run["prediction_kind"], "ic_state": "AVAILABLE" if sufficient else "UNKNOWN",
            "mean_oos_rank_ic": mean, "observations": n, "dates": observations,
            "selected_label_evidence_hash": _hash(evidence),
            "two_instrument_dates": sum(r["instruments"] == 2 for r in observations),
            "ic_drift": ic_drift,
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
                  "Nonpositive or insufficient IC gets zero ensemble weight; no automatic sign reversal.",
                  "ic_drift is a descriptive PSI on the prior OOS IC path; max_ic_drift_psi may zero a model weight."]}


class SignalWorkspace:
    def __init__(self, base_dir: Path):
        self.workspace = TerminalWorkspace(base_dir)
        self.base_dir = self.workspace.base_dir

    def _path(self, name):
        return self.workspace._path("research-signals/" + name)

    @staticmethod
    def _contract(document, state=State.AVAILABLE):
        schema = document.get("schema_version", IMPLEMENTATION)
        if schema == FACTOR_SCHEMA:
            title = "Unapproved cross-sectional factor research ranks"
            evidence = "HASH_VERIFIED_CS_FACTOR_RESEARCH"
        elif schema == PAIRS_SCHEMA:
            title = "Unapproved statistical-arbitrage pair weight scores"
            evidence = "HASH_VERIFIED_PAIR_WEIGHT_RESEARCH"
        else:
            title = "Unapproved IC-aware research ensemble"
            evidence = "HASH_VERIFIED_PIT_OOS_RESEARCH"
        return Contract.admit(state, {**document, "title": title}
            if state in {State.AVAILABLE, State.STALE} else None, source=document["source"],
            classification="MODEL_ESTIMATE", as_of=document["as_of"], scope="workspace.research-signals",
            source_identity=document["dataset_id"], model_identity=document["signal_id"],
            model_version=schema, evidence_state=evidence, readiness="RESEARCH_ONLY")

    def _persist(self, body: dict[str, Any]) -> Contract:
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

    def run(self, payload: dict[str, Any]) -> Contract:
        if isinstance(payload, dict) and payload.get("operation") == "from-dataset":
            body = {key: value for key, value in payload.items() if key != "operation"}
            return self.run_from_dataset(body)
        if isinstance(payload, dict) and payload.get("operation") == "from-pairs":
            body = {key: value for key, value in payload.items() if key != "operation"}
            return self.run_from_pairs(body)
        fields = _fields(payload, {"run_ids", "now", "lookback", "min_ic_observations",
                                   "max_ic_drift_psi", "operation"},
                         {"run_ids", "now"})
        ids = fields["run_ids"]
        if not isinstance(ids, list) or not 1 <= len(ids) <= 8 or len(set(ids)) != len(ids):
            raise ValueError("one to eight distinct saved run identities required")
        request = {"run_ids": sorted(sha256(i) for i in ids), "now": aware_timestamp(fields["now"]).isoformat(),
            "lookback": integer(fields.get("lookback", 20), 2, 200, "lookback"),
            "min_ic_observations": integer(fields.get("min_ic_observations", 5), 2, 200, "min_ic_observations")}
        if "max_ic_drift_psi" in fields:
            limit = float(fields["max_ic_drift_psi"])
            if not np.isfinite(limit) or limit <= 0:
                raise ValueError("max_ic_drift_psi must be a positive finite decimal")
            request["max_ic_drift_psi"] = limit
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
        return self._persist(score_runs(runs, dataset, request))

    def run_from_dataset(self, payload: dict[str, Any]) -> Contract:
        """Cross-sectional momentum / mean-reversion ranks from a saved return panel."""
        fields = _fields(payload, {"dataset_id", "now", "mode", "momentum_lookback", "reversion_lookback",
                                   "momentum_weight", "reversion_weight"},
                         {"dataset_id", "now"})
        mode = fields.get("mode", "blend")
        if mode not in {"momentum", "mean_reversion", "blend", "market_neutral_momentum"}:
            raise ValueError("mode must be momentum, mean_reversion, blend, or market_neutral_momentum")
        request = {
            "dataset_id": sha256(fields["dataset_id"]),
            "now": aware_timestamp(fields["now"]).isoformat(),
            "mode": mode,
            "momentum_lookback": integer(fields.get("momentum_lookback", 20), 2, 252, "momentum_lookback"),
            "reversion_lookback": integer(fields.get("reversion_lookback", 5), 2, 60, "reversion_lookback"),
            "momentum_weight": float(fields.get("momentum_weight", 0.5)),
            "reversion_weight": float(fields.get("reversion_weight", 0.5)),
        }
        try:
            dataset = PersistedDatasetWorkspace(self.base_dir).load(request["dataset_id"])
        except FileNotFoundError:
            return Contract.admit(State.UNAVAILABLE, None, source="persisted-research-signals",
                                  classification="UNAVAILABLE", scope="workspace.research-signals",
                                  error=public_error("DATA_UNAVAILABLE"))
        return self._persist(score_factor_dataset(dataset, request))

    def run_from_pairs(self, payload: dict[str, Any]) -> Contract:
        """Project saved pair-target weights into unapproved research scores."""
        fields = _fields(payload, {"pair_run_id", "now"}, {"pair_run_id", "now"})
        request = {
            "pair_run_id": sha256(fields["pair_run_id"]),
            "now": aware_timestamp(fields["now"]).isoformat(),
        }
        try:
            from gig.service.pairs_workspace import PairsResearchWorkspace
            pair_run = PairsResearchWorkspace(self.base_dir).load(request["pair_run_id"])
        except FileNotFoundError:
            return Contract.admit(State.UNAVAILABLE, None, source="persisted-research-signals",
                                  classification="UNAVAILABLE", scope="workspace.research-signals",
                                  error=public_error("DATA_UNAVAILABLE"))
        return self._persist(score_pair_run(pair_run, request))

    def load(self, signal_id: str) -> dict[str, Any]:
        identity = sha256(signal_id)
        document = loads(self._path("runs/" + identity + ".json").read_bytes())
        body = {k: v for k, v in document.items() if k != "signal_id"}
        schema = body.get("schema_version")
        if document.get("signal_id") != identity or _hash(body) != identity or schema not in SCHEMAS:
            raise ValueError("signal artifact integrity mismatch")
        with sqlite3.connect(self._path("signals.sqlite").as_uri() + "?mode=ro", uri=True) as connection:
            row = connection.execute("SELECT as_of FROM signals WHERE identity=?", (identity,)).fetchone()
        if row is None or row[0] != document["as_of"]:
            raise ValueError("signal index differs from preserved artifact")
        dataset = PersistedDatasetWorkspace(self.base_dir).load(document["dataset_id"])
        if schema == FACTOR_SCHEMA:
            if score_factor_dataset(dataset, document["request"]) != body:
                raise ValueError("signal evidence differs from deterministic replay")
            return document
        if schema == PAIRS_SCHEMA:
            from gig.service.pairs_workspace import PairsResearchWorkspace
            pair_run = PairsResearchWorkspace(self.base_dir).load(document["request"]["pair_run_id"])
            if score_pair_run(pair_run, document["request"]) != body:
                raise ValueError("signal evidence differs from deterministic replay")
            return document
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
