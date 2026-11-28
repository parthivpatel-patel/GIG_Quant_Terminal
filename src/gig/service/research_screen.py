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
OPTIONS_LINK = "gig.research-screen.options-link.v1"
OPTIONS_SCAN = "gig.research-screen.options-scan.v1"
FI_ENRICH = "gig.research-screen.fi-enrich.v1"
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

    def discover(self, payload: dict[str, Any]) -> Contract:
        """Fit an ensemble from saved model runs, then label research stances in one step."""
        fields = _fields(payload, {"run_ids", "now", "lookback", "min_ic_observations",
                                   "buy_threshold", "sell_threshold", "previous_screen_id",
                                   "ttl_seconds"},
                         {"run_ids", "now"})
        signal_payload: dict[str, Any] = {"run_ids": fields["run_ids"], "now": fields["now"]}
        for key in ("lookback", "min_ic_observations"):
            if key in fields:
                signal_payload[key] = fields[key]
        signal_contract = self.signals.run(signal_payload)
        signal = signal_contract.to_dict()
        if signal["state"] != State.AVAILABLE.value or not isinstance(signal.get("data"), dict):
            return signal_contract
        screen_payload: dict[str, Any] = {
            "signal_id": signal["data"]["signal_id"],
            "discovery": {"operation": "from-runs", "run_ids": list(signal["data"]["run_ids"]),
                          "scoring_state": signal["data"]["scoring_state"]},
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
            if not isinstance(discovery, dict) or discovery.get("operation") != "from-runs":
                raise ValueError("discovery must record a from-runs operation")
            if not isinstance(discovery.get("run_ids"), list) or not discovery["run_ids"]:
                raise ValueError("discovery requires the ensemble run identities")
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
            body["discovery"] = {"operation": "from-runs", "signal_id": signal["signal_id"],
                                 "run_ids": list(discovery["run_ids"]),
                                 "scoring_state": discovery.get("scoring_state")}
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
            "schema_version": IMPLEMENTATION + ".monitor",
            "as_of": screens[-1]["as_of"],
            "dataset_id": screens[-1]["dataset_id"],
            "source": screens[-1]["source"],
            "input_classification": screens[-1]["input_classification"],
            "signal_ids": [sha256(i) for i in ids],
            "screen_ids": [screen["screen_id"] for screen in screens],
            "timeline": timeline,
            "latest_screen_id": screens[-1]["screen_id"],
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
        document = {**body, "monitor_id": identity, "screen_id": screens[-1]["screen_id"]}
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
            rows.append({
                "options_run_id": options["run_id"],
                "underlying_id": options["underlying_id"],
                "equity_stance": linked["equity_stance"],
                "option_candidate": linked["option_candidate"],
                "research_weight": linked["research_weight"],
                "admitted_option_quotes": options.get("metrics", {}).get("admitted_quotes", 0),
                "trading_approved": False,
            })
        actionable = [row for row in rows if row["option_candidate"] in
                      {"RESEARCH_CALL_CANDIDATE", "RESEARCH_PUT_CANDIDATE"}]
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
            "actionable": actionable,
            "options_trading_approved": False,
            "trading_approved": False,
            "economically_validated": False,
            "usable_for_portfolio": False,
            "notes": [
                "Options scan maps each saved options run onto the equity research screen by underlying identity.",
                "Candidates are research labels only; no priced order, OPRA entitlement, or broker path is created.",
            ],
            "metrics": {
                "scanned": len(rows),
                "actionable": len(actionable),
                "calls": sum(1 for row in actionable if row["option_candidate"] == "RESEARCH_CALL_CANDIDATE"),
                "puts": sum(1 for row in actionable if row["option_candidate"] == "RESEARCH_PUT_CANDIDATE"),
                "trading_approved": False,
            },
        }
        identity = _hash(body)
        document = {**body, "scan_id": identity}
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
