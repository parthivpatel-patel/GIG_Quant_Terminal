"""Research-only buy / hold / sell labels from a saved IC-weighted signal.

The screen does not approve a trade. Labels are derived from the pinned
ensemble scores with caller thresholds. A previous screen may be compared for
entry and exit research events. Broker connectivity stays off.
"""
from __future__ import annotations

import hashlib
import sqlite3
from datetime import timedelta
from pathlib import Path
from typing import Any

from gig.artifacts import write_immutable
from gig.research.model_pipeline import sha256
from gig.service.contracts import Contract, State
from gig.service.options_workspace import OptionsWorkspace
from gig.service.serialization import dumps, loads, public_error
from gig.service.signal_workspace import SignalWorkspace
from gig.service.workspace import TerminalWorkspace, _fields
from gig.timestamps import aware_timestamp

IMPLEMENTATION = "gig.research-screen.v1"
MONITOR = "gig.research-screen.monitor.v1"
OPTIONS_LINK = "gig.research-screen.options-link.v1"
OPTIONS_SCAN = "gig.research-screen.options-scan.v1"
OPTIONS_STRUCTURES = "gig.research-screen.options-structures.v1"
FI_ENRICH = "gig.research-screen.fi-enrich.v1"
FI_FINDINGS = "gig.research-screen.fi-findings.v1"
_FI_FINDINGS_POLICIES = frozenset({"annotate", "gate-hold", "weight-adjust"})
_FI_FINDINGS_OPS = frozenset({"investigation", "watch-check", "entity-graph"})
STANCES = ("RESEARCH_BUY", "RESEARCH_HOLD", "RESEARCH_SELL")
EVENTS = ("ENTER_LONG", "EXIT_LONG", "ENTER_SHORT", "EXIT_SHORT", "CONTINUE", "FLIP", "ENTER_HOLD")
OPTION_CANDIDATES = ("RESEARCH_CALL_CANDIDATE", "RESEARCH_PUT_CANDIDATE", "RESEARCH_NO_OPTION_MATCH",
                     "RESEARCH_HOLD_NO_OPTION")


def match_mentions(rows: list[dict[str, Any]], entity_mentions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Exact string match of FI entity mentions onto screen instrument identities.

    Mentions keep their FI link_state. A match is lexical only; it is not an
    issuer authority link or a trading signal. Exchange-qualified mentions such
    as ``NYSE:AAA`` may resolve through the local ticker suffix of ``US:AAA``.
    """
    if not isinstance(entity_mentions, list):
        raise ValueError("entity_mentions must be a list")
    instruments = {row["instrument_id"] for row in rows}
    aliases: dict[str, str] = {}
    for instrument in instruments:
        aliases[instrument] = instrument
        aliases[instrument.upper()] = instrument
        if ":" in instrument:
            suffix = instrument.split(":", 1)[1]
            aliases[suffix] = instrument
            aliases[suffix.upper()] = instrument
    matched: list[dict[str, Any]] = []
    for mention in entity_mentions:
        if not isinstance(mention, dict) or not isinstance(mention.get("mention"), str):
            raise ValueError("entity mention requires a mention string")
        candidates = [mention["mention"].strip()]
        if isinstance(mention.get("symbol_mention"), str) and mention["symbol_mention"].strip():
            candidates.append(mention["symbol_mention"].strip())
        text = mention["mention"].strip()
        if ":" in text:
            candidates.append(text.split(":", 1)[1].strip())
        instrument = None
        for candidate in candidates:
            instrument = aliases.get(candidate) or aliases.get(candidate.upper())
            if instrument is not None:
                break
        if instrument is None:
            continue
        row = next(item for item in rows if item["instrument_id"] == instrument)
        matched.append({
            "instrument_id": instrument,
            "stance": row["stance"],
            "research_weight": float(row["research_weight"]),
            "mention": text,
            "link_state": mention.get("link_state", "UNKNOWN_NO_ISSUER_AUTHORITY"),
            "trading_approved": False,
        })
    return matched


def _hash(value: Any) -> str:
    return hashlib.sha256(dumps(value).encode()).hexdigest()


def _number(value: Any, name: str, *, low: float, high: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    number = float(value)
    if not low <= number <= high or number != number:
        raise ValueError(f"{name} is outside its allowed range")
    return number


def label_rows(rows: list[dict[str, Any]], *, buy_threshold: float, sell_threshold: float,
               ttl_seconds: int | None = None, signal_as_of: str | None = None,
               now: str | None = None) -> list[dict[str, Any]]:
    if buy_threshold <= sell_threshold:
        raise ValueError("buy_threshold must exceed sell_threshold")
    labeled = []
    for row in rows:
        score = float(row["prediction"])
        if score >= buy_threshold:
            stance = "RESEARCH_BUY"
        elif score <= sell_threshold:
            stance = "RESEARCH_SELL"
        else:
            stance = "RESEARCH_HOLD"
        labeled.append({
            "instrument_id": row["instrument_id"],
            "timestamp": row["timestamp"],
            "prediction": score,
            "prediction_kind": row["prediction_kind"],
            "stance": stance,
            "trading_approved": False,
            "confidence": {"state": "UNKNOWN", "value": None},
            "uncertainty": {"state": "NOT_CALIBRATED", "value": None},
            "opportunity_rank": None,
            "expires_at": None,
            "expiration_state": "UNKNOWN",
        })
    actionable = [row for row in labeled if row["stance"] != "RESEARCH_HOLD"]
    total = sum(abs(row["prediction"]) for row in actionable)
    peak = max((abs(row["prediction"]) for row in actionable), default=0.0)
    for row in labeled:
        if row["stance"] == "RESEARCH_HOLD" or total <= 0:
            row["research_weight"] = 0.0
        else:
            signed = abs(row["prediction"]) / total
            row["research_weight"] = signed if row["stance"] == "RESEARCH_BUY" else -signed
        if peak > 0 and row["stance"] != "RESEARCH_HOLD":
            row["confidence"] = {
                "state": "UNCALIBRATED_RELATIVE_ABS_SCORE",
                "value": abs(row["prediction"]) / peak,
            }
    ranked = sorted(actionable, key=lambda item: (-abs(item["research_weight"]), item["instrument_id"]))
    for index, row in enumerate(ranked, start=1):
        row["opportunity_rank"] = index
    if ttl_seconds is not None:
        if signal_as_of is None:
            raise ValueError("signal_as_of is required when ttl_seconds is set")
        if ttl_seconds < 1:
            raise ValueError("ttl_seconds must be a positive integer")
        origin = aware_timestamp(signal_as_of)
        expires = origin + timedelta(seconds=int(ttl_seconds))
        clock = aware_timestamp(now) if now is not None else origin
        expires_at = expires.isoformat()
        state = "EXPIRED" if clock > expires else "ACTIVE"
        for row in labeled:
            row["expires_at"] = expires_at
            row["expiration_state"] = state if row["stance"] != "RESEARCH_HOLD" else "NOT_APPLICABLE"
    return labeled


def option_candidate(rows: list[dict[str, Any]], underlying_id: str) -> dict[str, Any]:
    if not isinstance(underlying_id, str) or not underlying_id or len(underlying_id) > 64:
        raise ValueError("underlying identity must be a short string")
    match = next((row for row in rows if row.get("instrument_id") == underlying_id), None)
    if match is None:
        return {"option_candidate": "RESEARCH_NO_OPTION_MATCH", "equity_stance": None, "research_weight": 0.0}
    stance = match["stance"]
    if stance == "RESEARCH_BUY":
        return {"option_candidate": "RESEARCH_CALL_CANDIDATE", "equity_stance": stance,
                "research_weight": float(match["research_weight"])}
    if stance == "RESEARCH_SELL":
        return {"option_candidate": "RESEARCH_PUT_CANDIDATE", "equity_stance": stance,
                "research_weight": float(match["research_weight"])}
    return {"option_candidate": "RESEARCH_HOLD_NO_OPTION", "equity_stance": stance, "research_weight": 0.0}


def timing_events(current: list[dict[str, Any]], previous: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    if previous is None:
        return [{"instrument_id": row["instrument_id"], "event": "ENTER_LONG" if row["stance"] == "RESEARCH_BUY"
                 else "ENTER_SHORT" if row["stance"] == "RESEARCH_SELL" else "ENTER_HOLD",
                 "from_stance": None, "to_stance": row["stance"]} for row in current]
    prior = {row["instrument_id"]: row["stance"] for row in previous}
    events = []
    for row in current:
        before = prior.get(row["instrument_id"])
        after = row["stance"]
        if before is None:
            event = "ENTER_LONG" if after == "RESEARCH_BUY" else "ENTER_SHORT" if after == "RESEARCH_SELL" else "ENTER_HOLD"
        elif before == after:
            event = "CONTINUE"
        elif before == "RESEARCH_HOLD" and after == "RESEARCH_BUY":
            event = "ENTER_LONG"
        elif before == "RESEARCH_HOLD" and after == "RESEARCH_SELL":
            event = "ENTER_SHORT"
        elif before == "RESEARCH_BUY" and after == "RESEARCH_HOLD":
            event = "EXIT_LONG"
        elif before == "RESEARCH_SELL" and after == "RESEARCH_HOLD":
            event = "EXIT_SHORT"
        else:
            event = "FLIP"
        events.append({"instrument_id": row["instrument_id"], "event": event,
                       "from_stance": before, "to_stance": after})
    return events


def build_screen(signal: dict[str, Any], request: dict[str, Any],
                 previous: dict[str, Any] | None) -> dict[str, Any]:
    if not signal.get("usable_for_portfolio") or not signal.get("rows"):
        raise ValueError("signal is not usable for a research screen")
    rows = label_rows(
        signal["rows"], buy_threshold=request["buy_threshold"],
        sell_threshold=request["sell_threshold"],
        ttl_seconds=request.get("ttl_seconds"),
        signal_as_of=signal["signal_as_of"],
        now=request.get("now"),
    )
    prior_rows = previous["rows"] if previous is not None else None
    events = timing_events(rows, prior_rows)
    buys = sum(row["stance"] == "RESEARCH_BUY" for row in rows)
    sells = sum(row["stance"] == "RESEARCH_SELL" for row in rows)
    holds = sum(row["stance"] == "RESEARCH_HOLD" for row in rows)
    expired = any(row.get("expiration_state") == "EXPIRED" for row in rows)
    ranked = [row for row in rows if row.get("opportunity_rank") is not None]
    ranked.sort(key=lambda item: item["opportunity_rank"])
    return {
        "schema_version": IMPLEMENTATION,
        "request": request,
        "as_of": signal["as_of"],
        "signal_as_of": signal["signal_as_of"],
        "dataset_id": signal["dataset_id"],
        "signal_id": signal["signal_id"],
        "source": signal["source"],
        "input_classification": signal["input_classification"],
        "previous_screen_id": request.get("previous_screen_id"),
        "rows": rows,
        "opportunity_ranking": [
            {"instrument_id": row["instrument_id"], "opportunity_rank": row["opportunity_rank"],
             "stance": row["stance"], "research_weight": row["research_weight"],
             "confidence": row["confidence"], "expiration_state": row["expiration_state"],
             "expires_at": row["expires_at"], "trading_approved": False}
            for row in ranked
        ],
        "timing": events,
        "metrics": {
            "instruments": len(rows),
            "research_buys": buys,
            "research_sells": sells,
            "research_holds": holds,
            "ranked_opportunities": len(ranked),
            "signal_expired": expired,
            "timing_entries": sum(e["event"].startswith("ENTER_") for e in events),
            "timing_exits": sum(e["event"].startswith("EXIT_") for e in events),
            "timing_flips": sum(e["event"] == "FLIP" for e in events),
            "trading_approved": False,
        },
        "usable_for_portfolio": (
            signal["usable_for_portfolio"] and not expired
            and any(row["stance"] != "RESEARCH_HOLD" for row in rows)
        ),
        "trading_approved": False,
        "economically_validated": False,
        "method": "CENTERED_RANK_THRESHOLD_STANCE_V1",
        "notes": [
            "RESEARCH_BUY / RESEARCH_SELL / RESEARCH_HOLD are threshold labels on an uncalibrated ensemble score.",
            "research_weight is a relative research share among actionable names, not an order size.",
            "opportunity_rank orders actionable names by absolute research_weight; confidence is uncalibrated relative abs score.",
            "ttl_seconds expires actionable stances after signal_as_of; expired screens are not usable_for_portfolio.",
            "Timing events compare two saved screens only; they are not live market monitors.",
            "No broker order, allocation approval or funded certification is established.",
        ],
    }


class ResearchScreenWorkspace:
    def __init__(self, base_dir: Path):
        self.workspace = TerminalWorkspace(base_dir)
        self.base_dir = self.workspace.base_dir
        self.signals = SignalWorkspace(base_dir)

    def _path(self, name: str) -> Path:
        return self.workspace._path("research-screens/" + name)

    def _persist_artifact(self, directory: str, identity_key: str, document: dict[str, Any]) -> dict[str, Any]:
        """Write a hash-addressed sidecar artifact next to research screens."""
        identity = document[identity_key]
        folder = self._path(directory)
        folder.mkdir(parents=True, exist_ok=True)
        write_immutable(folder / (identity + ".json"), dumps(document).encode())
        return document

    def _load_artifact(self, directory: str, identity_key: str, artifact_id: str,
                       schema: str) -> dict[str, Any]:
        identity = sha256(artifact_id)
        path = self._path(directory) / (identity + ".json")
        if not path.is_file():
            raise FileNotFoundError(f"research screen {identity_key} unavailable")
        document = loads(path.read_bytes())
        body = {key: value for key, value in document.items() if key != identity_key}
        if (document.get(identity_key) != identity or _hash(body) != identity
                or document.get("schema_version") != schema):
            raise ValueError(f"{identity_key} artifact integrity mismatch")
        return document

    def discover(self, payload: dict[str, Any]) -> Contract:
        """Score saved evidence, then label research stances in one step.

        Accepts exactly one of ``run_ids``, ``dataset_id``, or ``pair_run_id``.
        """
        fields = _fields(payload, {"run_ids", "now", "lookback", "min_ic_observations",
                                   "buy_threshold", "sell_threshold", "previous_screen_id",
                                   "ttl_seconds", "dataset_id", "mode", "momentum_lookback",
                                   "reversion_lookback", "momentum_weight", "reversion_weight",
                                   "pair_run_id"},
                         {"now"})
        sources = [key for key in ("run_ids", "dataset_id", "pair_run_id") if key in fields]
        if len(sources) != 1:
            raise ValueError("discover requires exactly one of run_ids, dataset_id, or pair_run_id")
        if "dataset_id" in fields:
            signal_payload = {key: fields[key] for key in fields
                              if key in {"dataset_id", "now", "mode", "momentum_lookback",
                                         "reversion_lookback", "momentum_weight", "reversion_weight"}}
            signal_contract = self.signals.run_from_dataset(signal_payload)
            discovery_op = "from-dataset"
        elif "pair_run_id" in fields:
            signal_contract = self.signals.run_from_pairs({
                "pair_run_id": fields["pair_run_id"], "now": fields["now"],
            })
            discovery_op = "from-pairs"
        else:
            signal_payload = {"run_ids": fields["run_ids"], "now": fields["now"]}
            for key in ("lookback", "min_ic_observations"):
                if key in fields:
                    signal_payload[key] = fields[key]
            signal_contract = self.signals.run(signal_payload)
            discovery_op = "from-runs"
        signal = signal_contract.to_dict()
        if signal["state"] != State.AVAILABLE.value or not isinstance(signal.get("data"), dict):
            return signal_contract
        discovery: dict[str, Any] = {
            "operation": discovery_op, "scoring_state": signal["data"]["scoring_state"],
        }
        if discovery_op == "from-runs":
            discovery["run_ids"] = list(signal["data"]["run_ids"])
        elif discovery_op == "from-pairs":
            discovery["pair_run_id"] = signal["data"]["pair_run_id"]
        else:
            discovery["dataset_id"] = signal["data"]["dataset_id"]
            discovery["mode"] = signal["data"]["request"]["mode"]
        screen_payload: dict[str, Any] = {
            "signal_id": signal["data"]["signal_id"],
            "discovery": discovery,
            "now": fields["now"],
        }
        for key in ("buy_threshold", "sell_threshold", "previous_screen_id", "ttl_seconds"):
            if key in fields:
                screen_payload[key] = fields[key]
        return self.run(screen_payload)

    @staticmethod
    def _contract(document: dict[str, Any], *, state: State = State.AVAILABLE) -> Contract:
        return Contract.admit(
            state,
            {**document, "title": "Unapproved research buy/hold/sell screen"}
            if state in {State.AVAILABLE, State.STALE} else None,
            source=document["source"],
            classification="MODEL_ESTIMATE",
            as_of=document["as_of"],
            scope="workspace.research-screen",
            source_identity=document["dataset_id"],
            model_identity=document["screen_id"],
            model_version=IMPLEMENTATION,
            evidence_state="HASH_VERIFIED_RESEARCH_SCREEN",
            readiness="RESEARCH_ONLY",
        )

    def run(self, payload: dict[str, Any]) -> Contract:
        fields = _fields(payload, {"signal_id", "buy_threshold", "sell_threshold", "previous_screen_id",
                                   "discovery", "ttl_seconds", "now"},
                         {"signal_id"})
        request = {
            "signal_id": sha256(fields["signal_id"]),
            "buy_threshold": _number(fields.get("buy_threshold", 0.25), "buy_threshold", low=-1.0, high=1.0),
            "sell_threshold": _number(fields.get("sell_threshold", -0.25), "sell_threshold", low=-1.0, high=1.0),
        }
        if "previous_screen_id" in fields:
            request["previous_screen_id"] = sha256(fields["previous_screen_id"])
        if "ttl_seconds" in fields:
            ttl = int(fields["ttl_seconds"])
            if ttl < 1:
                raise ValueError("ttl_seconds must be a positive integer")
            request["ttl_seconds"] = ttl
        if "now" in fields:
            request["now"] = aware_timestamp(fields["now"]).isoformat()
        discovery = fields.get("discovery")
        if discovery is not None:
            if not isinstance(discovery, dict) or discovery.get("operation") not in {
                    "from-runs", "from-dataset", "from-pairs"}:
                raise ValueError("discovery must record a from-runs, from-dataset, or from-pairs operation")
            if discovery["operation"] == "from-runs":
                if not isinstance(discovery.get("run_ids"), list) or not discovery["run_ids"]:
                    raise ValueError("discovery requires the ensemble run identities")
            elif discovery["operation"] == "from-pairs":
                if not isinstance(discovery.get("pair_run_id"), str) or not discovery["pair_run_id"]:
                    raise ValueError("discovery requires the pair run identity")
            elif not isinstance(discovery.get("dataset_id"), str) or not discovery["dataset_id"]:
                raise ValueError("discovery requires the dataset identity")
        try:
            signal = self.signals.load(str(request["signal_id"]))
            previous = self.load(str(request["previous_screen_id"])) if "previous_screen_id" in request else None
        except FileNotFoundError:
            return Contract.admit(State.UNAVAILABLE, None, source="persisted-research-screen",
                                  classification="UNAVAILABLE", scope="workspace.research-screen",
                                  error=public_error("DATA_UNAVAILABLE"))
        if previous is not None:
            if previous["dataset_id"] != signal["dataset_id"]:
                raise ValueError("previous screen must share the signal dataset")
            if aware_timestamp(previous["signal_as_of"]) > aware_timestamp(signal["signal_as_of"]):
                raise ValueError("previous screen cannot follow a later signal decision")
            if aware_timestamp(previous["as_of"]) >= aware_timestamp(signal["as_of"]):
                raise ValueError("previous screen must precede the current evaluation time")
        body = build_screen(signal, request, previous)
        if discovery is not None:
            if discovery["operation"] == "from-dataset":
                body["discovery"] = {
                    "operation": "from-dataset", "signal_id": signal["signal_id"],
                    "dataset_id": discovery["dataset_id"], "mode": discovery.get("mode"),
                    "scoring_state": discovery.get("scoring_state"),
                }
            elif discovery["operation"] == "from-pairs":
                body["discovery"] = {
                    "operation": "from-pairs", "signal_id": signal["signal_id"],
                    "pair_run_id": discovery["pair_run_id"],
                    "scoring_state": discovery.get("scoring_state"),
                }
            else:
                body["discovery"] = {
                    "operation": "from-runs", "signal_id": signal["signal_id"],
                    "run_ids": list(discovery["run_ids"]),
                    "scoring_state": discovery.get("scoring_state"),
                }
        identity = _hash(body)
        document = {**body, "screen_id": identity}
        result = self._contract(document)
        self._path("runs").mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self._path("screens.sqlite")) as connection:
            connection.execute("PRAGMA synchronous=FULL")
            connection.execute("CREATE TABLE IF NOT EXISTS screens(identity TEXT PRIMARY KEY,as_of TEXT NOT NULL)")
            count = connection.execute("SELECT COUNT(*) FROM screens").fetchone()[0]
            old = connection.execute("SELECT as_of FROM screens WHERE identity=?", (identity,)).fetchone()
            if old is not None and old[0] != document["as_of"]:
                raise ValueError("screen index conflicts with immutable evidence")
            if old is None and count >= 1000:
                raise ValueError("screen registry exceeds retention bound")
            write_immutable(self._path("runs/" + identity + ".json"), dumps(document).encode())
            connection.execute("INSERT OR IGNORE INTO screens VALUES(?,?)", (identity, document["as_of"]))
        return result

    def load(self, screen_id: str) -> dict[str, Any]:
        identity = sha256(screen_id)
        path = self._path("runs/" + identity + ".json")
        if not path.is_file():
            raise FileNotFoundError("research screen unavailable")
        document = loads(path.read_bytes())
        body = {k: v for k, v in document.items() if k != "screen_id"}
        if document.get("screen_id") != identity or _hash(body) != identity or body.get("schema_version") != IMPLEMENTATION:
            raise ValueError("screen artifact integrity mismatch")
        with sqlite3.connect(self._path("screens.sqlite").as_uri() + "?mode=ro", uri=True) as connection:
            row = connection.execute("SELECT as_of FROM screens WHERE identity=?", (identity,)).fetchone()
        if row is None or row[0] != document["as_of"]:
            raise ValueError("screen index differs from preserved artifact")
        signal = self.signals.load(document["signal_id"])
        previous = self.load(document["previous_screen_id"]) if document.get("previous_screen_id") else None
        rebuilt = build_screen(signal, document["request"], previous)
        if document.get("discovery") is not None:
            rebuilt["discovery"] = document["discovery"]
        if rebuilt != body:
            raise ValueError("screen evidence differs from deterministic replay")
        return document

    def replay(self, payload: dict[str, Any]) -> Contract:
        _fields(payload, {"screen_id"}, {"screen_id"})
        try:
            document = self.load(payload["screen_id"])
        except FileNotFoundError:
            return Contract.admit(State.UNAVAILABLE, None, source="persisted-research-screen",
                                  classification="UNAVAILABLE", scope="workspace.research-screen",
                                  error=public_error("DATA_UNAVAILABLE"))
        return self._contract(document)

    def monitor(self, payload: dict[str, Any]) -> Contract:
        """Chain screens across successive saved signals for research-only monitoring."""
        fields = _fields(payload, {"signal_ids", "buy_threshold", "sell_threshold", "ttl_seconds", "now"},
                         {"signal_ids"})
        ids = fields["signal_ids"]
        if not isinstance(ids, list) or not 2 <= len(ids) <= 32 or len(set(ids)) != len(ids):
            raise ValueError("monitor requires two to thirty-two distinct signal identities")
        screens: list[dict[str, Any]] = []
        previous_id: str | None = None
        for signal_id in ids:
            request: dict[str, Any] = {"signal_id": signal_id}
            for key in ("buy_threshold", "sell_threshold", "ttl_seconds", "now"):
                if key in fields:
                    request[key] = fields[key]
            if previous_id is not None:
                request["previous_screen_id"] = previous_id
            contract = self.run(request)
            result = contract.to_dict()
            if result["state"] != State.AVAILABLE.value or not isinstance(result.get("data"), dict):
                return contract
            screens.append(result["data"])
            previous_id = result["data"]["screen_id"]
        timeline = [{"screen_id": screen["screen_id"], "signal_id": screen["signal_id"],
                     "signal_as_of": screen["signal_as_of"],
                     "research_buys": screen["metrics"]["research_buys"],
                     "research_sells": screen["metrics"]["research_sells"],
                     "research_holds": screen["metrics"]["research_holds"],
                     "timing_entries": screen["metrics"]["timing_entries"],
                     "timing_exits": screen["metrics"]["timing_exits"],
                     "timing_flips": screen["metrics"]["timing_flips"]} for screen in screens]
        body = {
            "schema_version": MONITOR,
            "as_of": screens[-1]["as_of"],
            "dataset_id": screens[-1]["dataset_id"],
            "source": screens[-1]["source"],
            "input_classification": screens[-1]["input_classification"],
            "signal_ids": [sha256(i) for i in ids],
            "screen_ids": [screen["screen_id"] for screen in screens],
            "timeline": timeline,
            "latest_screen_id": screens[-1]["screen_id"],
            "screen_id": screens[-1]["screen_id"],
            "rows": screens[-1]["rows"],
            "timing": screens[-1]["timing"],
            "metrics": {
                "screens": len(screens),
                "timing_entries": sum(item["timing_entries"] for item in timeline),
                "timing_exits": sum(item["timing_exits"] for item in timeline),
                "timing_flips": sum(item["timing_flips"] for item in timeline),
                "trading_approved": False,
            },
            "trading_approved": False,
            "economically_validated": False,
            "usable_for_portfolio": screens[-1]["usable_for_portfolio"],
            "notes": [
                "Monitor chains saved signals in caller order; it is not a live market subscription.",
                "Each step is a persisted research screen. No broker order or trading approval is established.",
            ],
        }
        identity = _hash(body)
        document = {**body, "monitor_id": identity}
        self._persist_artifact("monitors", "monitor_id", document)
        return self._contract(document)

    def load_monitor(self, monitor_id: str) -> dict[str, Any]:
        return self._load_artifact("monitors", "monitor_id", monitor_id, MONITOR)

    def replay_monitor(self, payload: dict[str, Any]) -> Contract:
        _fields(payload, {"monitor_id"}, {"monitor_id"})
        try:
            document = self.load_monitor(payload["monitor_id"])
        except FileNotFoundError:
            return Contract.admit(State.UNAVAILABLE, None, source="persisted-research-screen",
                                  classification="UNAVAILABLE", scope="workspace.research-screen",
                                  error=public_error("DATA_UNAVAILABLE"))
        return self._contract(document)

    def link_options(self, payload: dict[str, Any]) -> Contract:
        """Attach research option candidates to an equity screen. Not an options order."""
        fields = _fields(payload, {"screen_id", "options_run_id"}, {"screen_id", "options_run_id"})
        try:
            screen = self.load(fields["screen_id"])
            options = OptionsWorkspace(self.base_dir).load(fields["options_run_id"])
        except FileNotFoundError:
            return Contract.admit(State.UNAVAILABLE, None, source="persisted-research-screen",
                                  classification="UNAVAILABLE", scope="workspace.research-screen.options",
                                  error=public_error("DATA_UNAVAILABLE"))
        underlying = options["underlying_id"]
        linked = option_candidate(screen["rows"], underlying)
        quotes = options.get("metrics", {}).get("admitted_quotes", 0)
        body = {
            "schema_version": OPTIONS_LINK,
            "as_of": screen["as_of"],
            "screen_id": screen["screen_id"],
            "options_run_id": options["run_id"],
            "dataset_id": screen["dataset_id"],
            "signal_id": screen["signal_id"],
            "source": screen["source"],
            "input_classification": screen["input_classification"],
            "underlying_id": underlying,
            "equity_stance": linked["equity_stance"],
            "option_candidate": linked["option_candidate"],
            "research_weight": linked["research_weight"],
            "admitted_option_quotes": quotes,
            "options_trading_approved": False,
            "trading_approved": False,
            "economically_validated": False,
            "usable_for_portfolio": False,
            "notes": [
                "Option candidates follow the equity research stance only; they are not priced entry or exit orders.",
                "A CALL candidate means the equity screen is RESEARCH_BUY; a PUT candidate means RESEARCH_SELL.",
                "No broker connectivity, OPRA entitlement or trading approval is established.",
            ],
            "metrics": {"trading_approved": False, "option_candidate": linked["option_candidate"],
                        "admitted_option_quotes": quotes},
        }
        identity = _hash(body)
        document = {**body, "link_id": identity}
        self._persist_artifact("options-links", "link_id", document)
        return Contract.admit(
            State.AVAILABLE,
            {**document, "title": "Unapproved equity/options research link"},
            source=document["source"],
            classification="MODEL_ESTIMATE",
            as_of=document["as_of"],
            scope="workspace.research-screen.options",
            source_identity=document["dataset_id"],
            model_identity=identity,
            model_version=OPTIONS_LINK,
            evidence_state="HASH_VERIFIED_RESEARCH_SCREEN_OPTIONS_LINK",
            readiness="RESEARCH_ONLY",
        )

    def load_options_link(self, link_id: str) -> dict[str, Any]:
        return self._load_artifact("options-links", "link_id", link_id, OPTIONS_LINK)

    def replay_options_link(self, payload: dict[str, Any]) -> Contract:
        _fields(payload, {"link_id"}, {"link_id"})
        try:
            document = self.load_options_link(payload["link_id"])
        except FileNotFoundError:
            return Contract.admit(State.UNAVAILABLE, None, source="persisted-research-screen",
                                  classification="UNAVAILABLE",
                                  scope="workspace.research-screen.options",
                                  error=public_error("DATA_UNAVAILABLE"))
        return Contract.admit(
            State.AVAILABLE,
            {**document, "title": "Unapproved equity/options research link"},
            source=document["source"],
            classification="MODEL_ESTIMATE",
            as_of=document["as_of"],
            scope="workspace.research-screen.options",
            source_identity=document["dataset_id"],
            model_identity=document["link_id"],
            model_version=OPTIONS_LINK,
            evidence_state="HASH_VERIFIED_RESEARCH_SCREEN_OPTIONS_LINK",
            readiness="RESEARCH_ONLY",
        )

    def scan_options(self, payload: dict[str, Any]) -> Contract:
        """Scan multiple saved options runs against one equity research screen."""
        fields = _fields(payload, {"screen_id", "options_run_ids"}, {"screen_id", "options_run_ids"})
        ids = fields["options_run_ids"]
        if not isinstance(ids, list) or not 1 <= len(ids) <= 32 or len(set(ids)) != len(ids):
            raise ValueError("options scan requires one to thirty-two distinct options run identities")
        try:
            screen = self.load(fields["screen_id"])
        except FileNotFoundError:
            return Contract.admit(State.UNAVAILABLE, None, source="persisted-research-screen",
                                  classification="UNAVAILABLE", scope="workspace.research-screen.options",
                                  error=public_error("DATA_UNAVAILABLE"))
        workspace = OptionsWorkspace(self.base_dir)
        rows: list[dict[str, Any]] = []
        for options_run_id in ids:
            try:
                options = workspace.load(options_run_id)
            except FileNotFoundError:
                return Contract.admit(State.UNAVAILABLE, None, source="persisted-research-screen",
                                      classification="UNAVAILABLE", scope="workspace.research-screen.options",
                                      error=public_error("DATA_UNAVAILABLE"))
            linked = option_candidate(screen["rows"], options["underlying_id"])
            term = options.get("term_structure") or []
            skew = options.get("skew") or []
            vts_rows = (options.get("vts") or {}).get("rows") or []
            atm_iv = term[0].get("implied_volatility") if term else None
            skew_secant = next((item.get("value") for item in skew if item.get("value") is not None), None)
            term_slope = next((item.get("value") for item in vts_rows if item.get("value") is not None), None)
            parts = [abs(float(value)) for value in (skew_secant, term_slope) if value is not None]
            opportunity = (sum(parts) / len(parts)) if parts else None
            surface_state = (options.get("surface") or {}).get("state")
            rows.append({
                "options_run_id": options["run_id"],
                "underlying_id": options["underlying_id"],
                "equity_stance": linked["equity_stance"],
                "option_candidate": linked["option_candidate"],
                "research_weight": linked["research_weight"],
                "admitted_option_quotes": options.get("metrics", {}).get("admitted_quotes", 0),
                "atm_implied_volatility": atm_iv,
                "skew_secant": skew_secant,
                "term_structure_slope": term_slope,
                "iv_opportunity_score": opportunity,
                "iv_opportunity_rank": None,
                "surface_state": surface_state,
                "confidence": {
                    "state": "UNCALIBRATED_RELATIVE_ABS_SCORE",
                    "value": opportunity,
                    "method": "MEAN_ABS_SKEW_AND_TERM_SLOPE",
                },
                "trading_approved": False,
            })
        scored = sorted(
            (row for row in rows if row["iv_opportunity_score"] is not None),
            key=lambda row: (-float(row["iv_opportunity_score"]), row["underlying_id"]),
        )
        for index, row in enumerate(scored, start=1):
            row["iv_opportunity_rank"] = index
        actionable = [row for row in rows if row["option_candidate"] in
                      {"RESEARCH_CALL_CANDIDATE", "RESEARCH_PUT_CANDIDATE"}]
        ranked_actionable = sorted(
            actionable,
            key=lambda row: (
                row["iv_opportunity_rank"] is None,
                row["iv_opportunity_rank"] if row["iv_opportunity_rank"] is not None else 10**9,
                row["underlying_id"],
            ),
        )
        body = {
            "schema_version": OPTIONS_SCAN,
            "as_of": screen["as_of"],
            "screen_id": screen["screen_id"],
            "dataset_id": screen["dataset_id"],
            "signal_id": screen["signal_id"],
            "source": screen["source"],
            "input_classification": screen["input_classification"],
            "options_run_ids": [sha256(i) for i in ids],
            "rows": rows,
            "actionable": ranked_actionable,
            "options_trading_approved": False,
            "trading_approved": False,
            "economically_validated": False,
            "usable_for_portfolio": False,
            "notes": [
                "Options scan maps each saved options run onto the equity research screen by underlying identity.",
                "IV opportunity scores use absolute skew secant and adjacent-expiry ATM IV slope; uncalibrated only.",
                "Candidates are research labels only; no priced order, OPRA entitlement, or broker path is created.",
            ],
            "metrics": {
                "scanned": len(rows),
                "actionable": len(actionable),
                "ranked_iv_opportunities": len(scored),
                "calls": sum(1 for row in actionable if row["option_candidate"] == "RESEARCH_CALL_CANDIDATE"),
                "puts": sum(1 for row in actionable if row["option_candidate"] == "RESEARCH_PUT_CANDIDATE"),
                "trading_approved": False,
            },
        }
        identity = _hash(body)
        document = {**body, "scan_id": identity}
        self._persist_artifact("options-scans", "scan_id", document)
        return Contract.admit(
            State.AVAILABLE,
            {**document, "title": "Unapproved equity/options research opportunity scan"},
            source=document["source"],
            classification="MODEL_ESTIMATE",
            as_of=document["as_of"],
            scope="workspace.research-screen.options",
            source_identity=document["dataset_id"],
            model_identity=identity,
            model_version=OPTIONS_SCAN,
            evidence_state="HASH_VERIFIED_RESEARCH_SCREEN_OPTIONS_SCAN",
            readiness="RESEARCH_ONLY",
        )

    def load_options_scan(self, scan_id: str) -> dict[str, Any]:
        return self._load_artifact("options-scans", "scan_id", scan_id, OPTIONS_SCAN)

    def replay_options_scan(self, payload: dict[str, Any]) -> Contract:
        _fields(payload, {"scan_id"}, {"scan_id"})
        try:
            document = self.load_options_scan(payload["scan_id"])
        except FileNotFoundError:
            return Contract.admit(State.UNAVAILABLE, None, source="persisted-research-screen",
                                  classification="UNAVAILABLE",
                                  scope="workspace.research-screen.options",
                                  error=public_error("DATA_UNAVAILABLE"))
        return Contract.admit(
            State.AVAILABLE,
            {**document, "title": "Unapproved equity/options research opportunity scan"},
            source=document["source"],
            classification="MODEL_ESTIMATE",
            as_of=document["as_of"],
            scope="workspace.research-screen.options",
            source_identity=document["dataset_id"],
            model_identity=document["scan_id"],
            model_version=OPTIONS_SCAN,
            evidence_state="HASH_VERIFIED_RESEARCH_SCREEN_OPTIONS_SCAN",
            readiness="RESEARCH_ONLY",
        )

    def discover_option_structures(self, payload: dict[str, Any]) -> Contract:
        """Discover same-expiry verticals on equity-linked options opportunities.

        BUY screens prefer debit call verticals; SELL screens prefer debit put
        verticals. Profiles are expiry-only research diagnostics, not orders.
        """
        fields = _fields(payload, {"screen_id", "options_run_ids", "width_strikes"},
                         {"screen_id", "options_run_ids"})
        ids = fields["options_run_ids"]
        if not isinstance(ids, list) or not 1 <= len(ids) <= 16 or len(set(ids)) != len(ids):
            raise ValueError("structure discovery requires one to sixteen distinct options run identities")
        width = int(fields.get("width_strikes", 1))
        if not 1 <= width <= 5:
            raise ValueError("width_strikes must lie between 1 and 5")
        try:
            screen = self.load(fields["screen_id"])
        except FileNotFoundError:
            return Contract.admit(State.UNAVAILABLE, None, source="persisted-research-screen",
                                  classification="UNAVAILABLE", scope="workspace.research-screen.options",
                                  error=public_error("DATA_UNAVAILABLE"))
        from gig.options.exotics import strategy as option_strategy
        workspace = OptionsWorkspace(self.base_dir)
        structures: list[dict[str, Any]] = []
        for options_run_id in ids:
            try:
                options = workspace.load(options_run_id)
            except FileNotFoundError:
                return Contract.admit(State.UNAVAILABLE, None, source="persisted-research-screen",
                                      classification="UNAVAILABLE", scope="workspace.research-screen.options",
                                      error=public_error("DATA_UNAVAILABLE"))
            linked = option_candidate(screen["rows"], options["underlying_id"])
            if linked["option_candidate"] not in {"RESEARCH_CALL_CANDIDATE", "RESEARCH_PUT_CANDIDATE"}:
                continue
            right = "CALL" if linked["option_candidate"] == "RESEARCH_CALL_CANDIDATE" else "PUT"
            admitted = [
                row for row in options.get("rows", [])
                if row.get("right") == right and row.get("state") == "AVAILABLE"
                and row.get("midpoint") is not None and row.get("strike") is not None
                and row.get("expiry") is not None
            ]
            by_expiry: dict[str, list[dict[str, Any]]] = {}
            for row in admitted:
                by_expiry.setdefault(row["expiry"], []).append(row)
            for expiry, quotes in sorted(by_expiry.items()):
                ordered = sorted(quotes, key=lambda item: float(item["strike"]))
                if len(ordered) <= width:
                    continue
                for index in range(len(ordered) - width):
                    low, high = ordered[index], ordered[index + width]
                    low_strike = float(low["strike"])
                    high_strike = float(high["strike"])
                    low_mid = float(low["midpoint"])
                    high_mid = float(high["midpoint"])
                    if right == "CALL":
                        long_strike, short_strike = low_strike, high_strike
                        long_mid, short_mid = low_mid, high_mid
                        structure = "DEBIT_CALL_VERTICAL"
                        kind = "CALL"
                    else:
                        long_strike, short_strike = high_strike, low_strike
                        long_mid, short_mid = high_mid, low_mid
                        structure = "DEBIT_PUT_VERTICAL"
                        kind = "PUT"
                    legs = [
                        {"kind": kind, "strike": long_strike, "quantity": 1, "premium": long_mid},
                        {"kind": kind, "strike": short_strike, "quantity": -1, "premium": short_mid},
                    ]
                    multiplier = low.get("multiplier")
                    profile = option_strategy(
                        legs, multiplier=float(multiplier) if multiplier is not None else 1.0)
                    structures.append({
                        "options_run_id": options["run_id"],
                        "underlying_id": options["underlying_id"],
                        "equity_stance": linked["equity_stance"],
                        "option_candidate": linked["option_candidate"],
                        "structure": structure,
                        "expiry": expiry,
                        "long_strike": long_strike,
                        "short_strike": short_strike,
                        "width": high_strike - low_strike,
                        "net_premium": profile["net_premium"],
                        "premium_direction": profile["premium_direction"],
                        "max_profit": profile["max_profit"],
                        "max_loss": profile["max_loss"],
                        "breakevens": profile["breakevens"],
                        "trading_approved": False,
                        "options_trading_approved": False,
                    })
        ranked = sorted(
            structures,
            key=lambda row: (
                -(float(row["max_profit"]) if row["max_profit"] is not None else -1e18),
                row["underlying_id"], row["expiry"], row["long_strike"],
            ),
        )
        for index, row in enumerate(ranked, start=1):
            row["structure_rank"] = index
        body = {
            "schema_version": OPTIONS_STRUCTURES,
            "as_of": screen["as_of"],
            "screen_id": screen["screen_id"],
            "dataset_id": screen["dataset_id"],
            "signal_id": screen["signal_id"],
            "source": screen["source"],
            "input_classification": screen["input_classification"],
            "options_run_ids": [sha256(i) for i in ids],
            "request": {"width_strikes": width},
            "rows": ranked,
            "options_trading_approved": False,
            "trading_approved": False,
            "economically_validated": False,
            "usable_for_portfolio": False,
            "notes": [
                "Structures are same-expiry debit verticals linked to equity research stances.",
                "P&L profiles are exact expiry payoffs on midpoint premiums; no financing, fees, or margin.",
                "No OPRA entitlement, broker path, or trading approval is established.",
            ],
            "metrics": {
                "structures": len(ranked),
                "call_verticals": sum(1 for row in ranked if row["structure"] == "DEBIT_CALL_VERTICAL"),
                "put_verticals": sum(1 for row in ranked if row["structure"] == "DEBIT_PUT_VERTICAL"),
                "trading_approved": False,
            },
        }
        identity = _hash(body)
        document = {**body, "structures_id": identity}
        self._persist_artifact("options-structures", "structures_id", document)
        return Contract.admit(
            State.AVAILABLE,
            {**document, "title": "Unapproved multi-leg options structure discovery"},
            source=document["source"],
            classification="MODEL_ESTIMATE",
            as_of=document["as_of"],
            scope="workspace.research-screen.options",
            source_identity=document["dataset_id"],
            model_identity=identity,
            model_version=OPTIONS_STRUCTURES,
            evidence_state="HASH_VERIFIED_RESEARCH_OPTIONS_STRUCTURES",
            readiness="RESEARCH_ONLY",
        )

    def load_option_structures(self, structures_id: str) -> dict[str, Any]:
        return self._load_artifact("options-structures", "structures_id", structures_id,
                                   OPTIONS_STRUCTURES)

    def replay_option_structures(self, payload: dict[str, Any]) -> Contract:
        _fields(payload, {"structures_id"}, {"structures_id"})
        try:
            document = self.load_option_structures(payload["structures_id"])
        except FileNotFoundError:
            return Contract.admit(State.UNAVAILABLE, None, source="persisted-research-screen",
                                  classification="UNAVAILABLE",
                                  scope="workspace.research-screen.options",
                                  error=public_error("DATA_UNAVAILABLE"))
        return Contract.admit(
            State.AVAILABLE,
            {**document, "title": "Unapproved multi-leg options structure discovery"},
            source=document["source"],
            classification="MODEL_ESTIMATE",
            as_of=document["as_of"],
            scope="workspace.research-screen.options",
            source_identity=document["dataset_id"],
            model_identity=document["structures_id"],
            model_version=OPTIONS_STRUCTURES,
            evidence_state="HASH_VERIFIED_RESEARCH_OPTIONS_STRUCTURES",
            readiness="RESEARCH_ONLY",
        )

    def enrich_mentions(self, payload: dict[str, Any]) -> Contract:
        """Attach Financial Intelligence entity mentions that lexically match screen names."""
        fields = _fields(payload, {"screen_id", "document_id", "nlp_profile"}, {"screen_id", "document_id"})
        profile = fields.get("nlp_profile", "v1")
        try:
            screen = self.load(fields["screen_id"])
            from gig.service.financial_workspace import FinancialIntelligenceWorkspace
            corpus = FinancialIntelligenceWorkspace(self.base_dir).corpus
            admitted = corpus.document(fields["document_id"])
        except FileNotFoundError:
            return Contract.admit(State.UNAVAILABLE, None, source="persisted-research-screen",
                                  classification="UNAVAILABLE", scope="workspace.research-screen.fi",
                                  error=public_error("DATA_UNAVAILABLE"))
        from gig.financial_intelligence import nlp
        extracted = nlp.extract(admitted["passages"], profile=profile)
        matches = match_mentions(screen["rows"], extracted.get("entity_mentions", []))
        body = {
            "schema_version": FI_ENRICH,
            "as_of": screen["as_of"],
            "screen_id": screen["screen_id"],
            "document_id": admitted["metadata"]["document_id"],
            "dataset_id": screen["dataset_id"],
            "signal_id": screen["signal_id"],
            "source": screen["source"],
            "input_classification": screen["input_classification"],
            "nlp_profile": profile,
            "matched_mentions": matches,
            "entity_mention_count": len(extracted.get("entity_mentions", [])),
            "matched_count": len(matches),
            "trading_approved": False,
            "economically_validated": False,
            "usable_for_portfolio": False,
            "notes": [
                "Matches are exact lexical overlaps between FI entity mentions and screen instrument identities.",
                "link_state stays UNKNOWN_NO_ISSUER_AUTHORITY unless the FI extract already set another state.",
                "Enrichment does not change the screen hash, stance, or trading approval.",
            ],
            "metrics": {"matched_count": len(matches),
                        "entity_mention_count": len(extracted.get("entity_mentions", [])),
                        "trading_approved": False},
        }
        identity = _hash(body)
        document_out = {**body, "enrichment_id": identity}
        self._persist_artifact("fi-enrichments", "enrichment_id", document_out)
        return Contract.admit(
            State.AVAILABLE,
            {**document_out, "title": "Unapproved FI mention enrichment of a research screen"},
            source=document_out["source"],
            classification="MODEL_ESTIMATE",
            as_of=document_out["as_of"],
            scope="workspace.research-screen.fi",
            source_identity=document_out["dataset_id"],
            model_identity=identity,
            model_version=FI_ENRICH,
            evidence_state="HASH_VERIFIED_RESEARCH_SCREEN_FI_ENRICHMENT",
            readiness="RESEARCH_ONLY",
        )

    def load_enrichment(self, enrichment_id: str) -> dict[str, Any]:
        return self._load_artifact("fi-enrichments", "enrichment_id", enrichment_id, FI_ENRICH)

    def replay_enrichment(self, payload: dict[str, Any]) -> Contract:
        _fields(payload, {"enrichment_id"}, {"enrichment_id"})
        try:
            document = self.load_enrichment(payload["enrichment_id"])
        except FileNotFoundError:
            return Contract.admit(State.UNAVAILABLE, None, source="persisted-research-screen",
                                  classification="UNAVAILABLE",
                                  scope="workspace.research-screen.fi",
                                  error=public_error("DATA_UNAVAILABLE"))
        return Contract.admit(
            State.AVAILABLE,
            {**document, "title": "Unapproved FI mention enrichment of a research screen"},
            source=document["source"],
            classification="MODEL_ESTIMATE",
            as_of=document["as_of"],
            scope="workspace.research-screen.fi",
            source_identity=document["dataset_id"],
            model_identity=document["enrichment_id"],
            model_version=FI_ENRICH,
            evidence_state="HASH_VERIFIED_RESEARCH_SCREEN_FI_ENRICHMENT",
            readiness="RESEARCH_ONLY",
        )

    def annotate_fi_findings(self, payload: dict[str, Any]) -> Contract:
        """Annotate a research screen from a saved FI investigation, watch-check, or entity-graph.

        Policies may annotate lineage only, gate matched names to RESEARCH_HOLD on
        structural evidence, or multiplicatively demote weights from mention counts.
        The original screen is never rewritten. Director binding stays NOT_BOUND.
        """
        fields = _fields(payload, {"screen_id", "investigation_id", "watch_check_id",
                                   "entity_graph_id", "nlp_profile", "policy",
                                   "weight_demote"},
                         {"screen_id"})
        sources = [key for key in ("investigation_id", "watch_check_id", "entity_graph_id")
                   if key in fields]
        if len(sources) != 1:
            raise ValueError(
                "annotate-fi-findings requires exactly one of investigation_id, "
                "watch_check_id, or entity_graph_id")
        policy = fields.get("policy", "annotate")
        if policy not in _FI_FINDINGS_POLICIES:
            raise ValueError("policy must be annotate, gate-hold, or weight-adjust")
        profile = fields.get("nlp_profile", "v1")
        if profile not in {"v1", "v2"}:
            raise ValueError("nlp_profile must be v1 or v2")
        demote = float(fields.get("weight_demote", 0.85))
        if not 0.1 <= demote <= 1.0:
            raise ValueError("weight_demote must lie in [0.1, 1.0]")
        try:
            screen = self.load(fields["screen_id"])
            from gig.financial_intelligence import nlp
            from gig.financial_intelligence import research as fi_research
            from gig.service.financial_workspace import FinancialIntelligenceWorkspace
            corpus = FinancialIntelligenceWorkspace(self.base_dir).corpus
            record_id = fields[sources[0]]
            record = fi_research.load(corpus, record_id)
        except FileNotFoundError:
            return Contract.admit(State.UNAVAILABLE, None, source="persisted-research-screen",
                                  classification="UNAVAILABLE", scope="workspace.research-screen.fi",
                                  error=public_error("DATA_UNAVAILABLE"))
        operation = record.get("operation")
        expected = {"investigation_id": "investigation", "watch_check_id": "watch-check",
                    "entity_graph_id": "entity-graph"}[sources[0]]
        if operation != expected or operation not in _FI_FINDINGS_OPS:
            raise ValueError(f"{sources[0]} must reference a saved {expected} record")

        entity_mentions: list[dict[str, Any]] = []
        evidence_docs: list[str] = []
        structural_gates: dict[str, str] = {}
        alert_changed = False

        if operation == "entity-graph":
            for node in record.get("nodes") or []:
                if not isinstance(node, dict) or not isinstance(node.get("mention"), str):
                    continue
                entity_mentions.append({
                    "mention": node["mention"],
                    "link_state": node.get("link_state", "UNKNOWN_NO_ISSUER_AUTHORITY"),
                    "passage_mentions": node.get("passage_mentions", 0),
                    "document_ids": list(node.get("document_ids") or []),
                })
            evidence_docs = list(record.get("document_ids") or [])
        elif operation == "investigation":
            evidence_docs = sorted({
                reviewed["document_id"]
                for reviewed in (record.get("documents_reviewed") or [])
                if isinstance(reviewed, dict) and isinstance(reviewed.get("document_id"), str)
            })
            for document_id in evidence_docs:
                try:
                    admitted = corpus.document(document_id)
                except FileNotFoundError:
                    continue
                extracted = nlp.extract(admitted["passages"], profile=profile)
                for mention in extracted.get("entity_mentions", []):
                    if isinstance(mention, dict):
                        entity_mentions.append(mention)
            inconsistent = {
                reviewed["document_id"]
                for reviewed in (record.get("documents_reviewed") or [])
                if isinstance(reviewed, dict) and reviewed.get("inconsistent_checks")
            }
            unanswered = list(record.get("unanswered_questions") or [])
            if unanswered:
                structural_gates["UNANSWERED_QUESTIONS"] = f"{len(unanswered)} unanswered"
            if inconsistent:
                structural_gates["INCONSISTENT_STATEMENTS"] = (
                    f"{len(inconsistent)} reviewed documents with INCONSISTENT checks")
        else:  # watch-check
            alert_changed = record.get("alert_state") == "CHANGED"
            evidence_docs = list(record.get("new_document_ids") or [])
            for document_id in evidence_docs:
                try:
                    admitted = corpus.document(document_id)
                except FileNotFoundError:
                    continue
                extracted = nlp.extract(admitted["passages"], profile=profile)
                for mention in extracted.get("entity_mentions", []):
                    if isinstance(mention, dict):
                        entity_mentions.append(mention)
            if alert_changed and evidence_docs:
                structural_gates["WATCH_ALERT_CHANGED"] = "watch-check alert_state=CHANGED"

        matches = match_mentions(screen["rows"], entity_mentions)
        mention_counts: dict[str, int] = {}
        evidence_by_name: dict[str, list[dict[str, Any]]] = {}
        for match in matches:
            name = match["instrument_id"]
            mention_counts[name] = mention_counts.get(name, 0) + 1
            evidence_by_name.setdefault(name, []).append({
                "mention": match["mention"],
                "link_state": match["link_state"],
                "evidence_record_id": record["record_id"],
                "evidence_operation": operation,
                "trading_approved": False,
            })

        rows_out: list[dict[str, Any]] = []
        gated = 0
        adjusted = 0
        for row in screen["rows"]:
            name = row["instrument_id"]
            base_stance = row["stance"]
            base_weight = float(row["research_weight"])
            evidence = evidence_by_name.get(name, [])
            stance_override = None
            adjusted_weight = None
            gate_reason = None
            if evidence and policy == "gate-hold" and structural_gates:
                stance_override = "RESEARCH_HOLD"
                gate_reason = sorted(structural_gates)[0]
                gated += 1
            if evidence and policy == "weight-adjust":
                hits = mention_counts.get(name, 0)
                factor = demote ** min(hits, 8)
                adjusted_weight = float(base_weight) * float(factor)
                adjusted += 1
            rows_out.append({
                "instrument_id": name,
                "base_stance": base_stance,
                "base_research_weight": base_weight,
                "evidence": evidence,
                "mention_count": mention_counts.get(name, 0),
                "stance_override": stance_override,
                "adjusted_weight": adjusted_weight,
                "gate_reason": gate_reason,
                "link_state": (evidence[0]["link_state"] if evidence
                               else "UNKNOWN_NO_ISSUER_AUTHORITY"),
                "trading_approved": False,
            })

        body = {
            "schema_version": FI_FINDINGS,
            "as_of": screen["as_of"],
            "screen_id": screen["screen_id"],
            "dataset_id": screen["dataset_id"],
            "signal_id": screen["signal_id"],
            "evidence_kind": operation,
            "evidence_record_id": record["record_id"],
            "evidence_document_ids": evidence_docs,
            "source": screen["source"],
            "input_classification": screen["input_classification"],
            "evidence_classification": record.get("classification", "UNKNOWN"),
            "nlp_profile": profile,
            "policy": policy,
            "request": {
                "policy": policy,
                "weight_demote": demote if policy == "weight-adjust" else None,
                "source_field": sources[0],
            },
            "structural_gates": structural_gates,
            "rows": rows_out,
            "director_binding_state": "NOT_BOUND_HUMAN_APPROVED_OBJECTIVE_REQUIRED",
            "broker_connected": False,
            "trading_approved": False,
            "economically_validated": False,
            "usable_for_portfolio": False,
            "classification": "MODEL_ESTIMATE",
            "metrics": {
                "names": len(rows_out),
                "matched": sum(1 for row in rows_out if row["mention_count"] > 0),
                "gated": gated,
                "adjusted": adjusted,
                "entity_mention_count": len(entity_mentions),
                "trading_approved": False,
            },
            "notes": [
                "FI findings annotate a research screen from saved investigate, watch-check, or entity-graph records.",
                "Matches are lexical only; issuer identity and trading approval stay unestablished.",
                "gate-hold requires structural evidence (unanswered questions, inconsistent statements, or CHANGED watch).",
                "weight-adjust demotes research_weight by weight_demote^min(hits,8); weights stay uncalibrated.",
                "The original screen artifact is not rewritten.",
                "director_binding_state remains NOT_BOUND_HUMAN_APPROVED_OBJECTIVE_REQUIRED.",
            ],
        }
        identity = _hash(body)
        document = {**body, "findings_id": identity}
        self._path("fi-findings").mkdir(parents=True, exist_ok=True)
        write_immutable(self._path("fi-findings/" + identity + ".json"), dumps(document).encode())
        return Contract.admit(
            State.AVAILABLE,
            {**document, "title": "Unapproved FI findings annotation of a research screen"},
            source=document["source"],
            classification="MODEL_ESTIMATE",
            as_of=document["as_of"],
            scope="workspace.research-screen.fi",
            source_identity=document["dataset_id"],
            model_identity=identity,
            model_version=FI_FINDINGS,
            evidence_state="HASH_VERIFIED_RESEARCH_SCREEN_FI_FINDINGS",
            readiness="RESEARCH_ONLY",
        )

    def load_fi_findings(self, findings_id: str) -> dict[str, Any]:
        identity = sha256(findings_id)
        path = self._path("fi-findings/" + identity + ".json")
        if not path.is_file():
            raise FileNotFoundError("research screen FI findings unavailable")
        document = loads(path.read_bytes())
        body = {k: v for k, v in document.items() if k != "findings_id"}
        if (document.get("findings_id") != identity or _hash(body) != identity
                or body.get("schema_version") != FI_FINDINGS):
            raise ValueError("FI findings artifact integrity mismatch")
        return document

    def replay_fi_findings(self, payload: dict[str, Any]) -> Contract:
        _fields(payload, {"findings_id"}, {"findings_id"})
        try:
            document = self.load_fi_findings(payload["findings_id"])
        except FileNotFoundError:
            return Contract.admit(State.UNAVAILABLE, None, source="persisted-research-screen",
                                  classification="UNAVAILABLE", scope="workspace.research-screen.fi",
                                  error=public_error("DATA_UNAVAILABLE"))
        return Contract.admit(
            State.AVAILABLE,
            {**document, "title": "Unapproved FI findings annotation of a research screen"},
            source=document["source"],
            classification="MODEL_ESTIMATE",
            as_of=document["as_of"],
            scope="workspace.research-screen.fi",
            source_identity=document["dataset_id"],
            model_identity=document["findings_id"],
            model_version=FI_FINDINGS,
            evidence_state="HASH_VERIFIED_RESEARCH_SCREEN_FI_FINDINGS",
            readiness="RESEARCH_ONLY",
        )

