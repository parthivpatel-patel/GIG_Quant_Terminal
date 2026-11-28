"""Research-only buy / hold / sell labels from a saved IC-weighted signal.

The screen does not approve a trade. Labels are derived from the pinned
ensemble scores with caller thresholds. A previous screen may be compared for
entry and exit research events. Broker connectivity stays off.
"""
from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path
from typing import Any

from gig.artifacts import write_immutable
from gig.research.model_pipeline import sha256
from gig.service.contracts import Contract, State
from gig.service.serialization import dumps, loads, public_error
from gig.service.signal_workspace import SignalWorkspace
from gig.service.workspace import TerminalWorkspace, _fields
from gig.timestamps import aware_timestamp

IMPLEMENTATION = "gig.research-screen.v1"
STANCES = ("RESEARCH_BUY", "RESEARCH_HOLD", "RESEARCH_SELL")
EVENTS = ("ENTER_LONG", "EXIT_LONG", "ENTER_SHORT", "EXIT_SHORT", "CONTINUE", "FLIP", "ENTER_HOLD")


def _hash(value: Any) -> str:
    return hashlib.sha256(dumps(value).encode()).hexdigest()


def _number(value: Any, name: str, *, low: float, high: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    number = float(value)
    if not low <= number <= high or number != number:
        raise ValueError(f"{name} is outside its allowed range")
    return number


def label_rows(rows: list[dict[str, Any]], *, buy_threshold: float, sell_threshold: float) -> list[dict[str, Any]]:
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
        })
    actionable = [row for row in labeled if row["stance"] != "RESEARCH_HOLD"]
    total = sum(abs(row["prediction"]) for row in actionable)
    for row in labeled:
        if row["stance"] == "RESEARCH_HOLD" or total <= 0:
            row["research_weight"] = 0.0
        else:
            signed = abs(row["prediction"]) / total
            row["research_weight"] = signed if row["stance"] == "RESEARCH_BUY" else -signed
    return labeled


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
    rows = label_rows(signal["rows"], buy_threshold=request["buy_threshold"],
                      sell_threshold=request["sell_threshold"])
    prior_rows = previous["rows"] if previous is not None else None
    events = timing_events(rows, prior_rows)
    buys = sum(row["stance"] == "RESEARCH_BUY" for row in rows)
    sells = sum(row["stance"] == "RESEARCH_SELL" for row in rows)
    holds = sum(row["stance"] == "RESEARCH_HOLD" for row in rows)
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
        "timing": events,
        "metrics": {
            "instruments": len(rows),
            "research_buys": buys,
            "research_sells": sells,
            "research_holds": holds,
            "timing_entries": sum(e["event"].startswith("ENTER_") for e in events),
            "timing_exits": sum(e["event"].startswith("EXIT_") for e in events),
            "timing_flips": sum(e["event"] == "FLIP" for e in events),
            "trading_approved": False,
        },
        "usable_for_portfolio": signal["usable_for_portfolio"] and any(row["stance"] != "RESEARCH_HOLD" for row in rows),
        "trading_approved": False,
        "economically_validated": False,
        "method": "CENTERED_RANK_THRESHOLD_STANCE_V1",
        "notes": [
            "RESEARCH_BUY / RESEARCH_SELL / RESEARCH_HOLD are threshold labels on an uncalibrated ensemble score.",
            "research_weight is a relative research share among actionable names, not an order size.",
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
        fields = _fields(payload, {"signal_id", "buy_threshold", "sell_threshold", "previous_screen_id"},
                         {"signal_id"})
        request = {
            "signal_id": sha256(fields["signal_id"]),
            "buy_threshold": _number(fields.get("buy_threshold", 0.25), "buy_threshold", low=-1.0, high=1.0),
            "sell_threshold": _number(fields.get("sell_threshold", -0.25), "sell_threshold", low=-1.0, high=1.0),
        }
        if "previous_screen_id" in fields:
            request["previous_screen_id"] = sha256(fields["previous_screen_id"])
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
            if aware_timestamp(previous["signal_as_of"]) >= aware_timestamp(signal["signal_as_of"]):
                raise ValueError("previous screen must precede the current signal decision")
        body = build_screen(signal, request, previous)
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
        if build_screen(signal, document["request"], previous) != body:
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
