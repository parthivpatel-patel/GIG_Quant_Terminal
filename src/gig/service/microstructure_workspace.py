"""Persisted original-decision quote/print research; no inferred executions."""
from __future__ import annotations

import hashlib
import math
import sqlite3
from dataclasses import asdict
from functools import lru_cache
from importlib.metadata import version
from pathlib import Path
from typing import Any

from gig.artifacts import write_immutable
from gig.data.validated import CLASSIFICATIONS
from gig.microstructure import (
    Quote,
    Trade,
    kyle_lambda,
    lee_ready,
    match_quotes,
    order_flow_imbalance,
    vpin,
)
from gig.service.contracts import Contract, State
from gig.service.serialization import dumps, loads, public_error
from gig.service.workspace import TerminalWorkspace, _age, _fields, _text
from gig.timestamps import aware_timestamp

VERSION = "persisted-original-decision-microstructure-v1"
HEADERS = {"source", "classification", "feed", "instrument_id", "currency"}
EVENT = {"event_id", "kind", "event_at", "acquired_at", "knowledge_at", "sequence"}
QUOTE = {"bid", "ask", "bid_size", "ask_size"}
TRADE = {"price", "size"}
RUN = {"source", "classification", "feed", "instrument_id", "start", "end", "as_of",
       "max_quote_age_ns", "lag_ns", "bucket_volume", "window_buckets", "participation_rate", "proposed_quantity"}


def _hash(value):
    return hashlib.sha256(dumps(value).encode()).hexdigest()


def _identity(value):
    if not isinstance(value, str) or len(value) != 64 or any(x not in "0123456789abcdef" for x in value):
        raise ValueError("canonical microstructure evidence identity required")
    return value


def _number(value, *, positive=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError("finite event number required")
    if not (0 < value <= 1e12 if positive else 0 <= value <= 1e12):
        raise ValueError("event numeric bounds exceeded")
    return float(value)


def _integer(value, upper):
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= upper:
        raise ValueError("bounded nonnegative exact integer required")
    return value


@lru_cache(maxsize=4096)
def _clock(value: str):
    return aware_timestamp(value)


def _key(row):
    return _clock(row["event_at"]).value, row["sequence"]


def _known(row):
    return _clock(row["knowledge_at"])


def _first(rows):
    groups = {}
    for row in sorted(rows, key=lambda r: (_known(r), r["event_hash"])):
        groups.setdefault(row["event_id"], row)
    return sorted(groups.values(), key=_key)


def _latest(rows, cutoff):
    groups = {}
    for row in sorted(rows, key=lambda r: (_known(r), r["event_hash"])):
        if _known(row) <= cutoff:
            groups[row["event_id"]] = row
    return sorted(groups.values(), key=_key)


def _quote(row):
    return Quote(_key(row)[0], row["instrument_id"], row["bid"], row["ask"], row["bid_size"], row["ask_size"],
                 row["sequence"], row["source"], row["classification"])


def _trade(row):
    return Trade(_key(row)[0], row["instrument_id"], row["price"], row["size"], row["sequence"],
                 row["source"], row["classification"])


class MicrostructureWorkspace:
    def __init__(self, base_dir: Path):
        self.workspace = TerminalWorkspace(base_dir)

    def _path(self, name):
        return self.workspace._path("microstructure-research/" + name)

    @staticmethod
    def _header(document):
        for key in ("source", "feed", "instrument_id"):
            _text(document[key], bound=128)
        if (not isinstance(document["classification"], str) or document["classification"] not in CLASSIFICATIONS
                or document["currency"] != "USD"):
            raise ValueError("explicit observed input classification and USD units required")

    @staticmethod
    def _admit_event(body):
        _fields(body, HEADERS | EVENT | QUOTE | TRADE, HEADERS | EVENT)
        value = loads(dumps(body))
        MicrostructureWorkspace._header(value)
        if value["kind"] not in ("QUOTE", "TRADE"):
            raise ValueError("only explicit quote or print observations admitted")
        fields = QUOTE if value["kind"] == "QUOTE" else TRADE
        _fields(value, HEADERS | EVENT | fields, HEADERS | EVENT | fields)
        _text(value["event_id"], bound=128)
        value["sequence"] = _integer(value["sequence"], 2**53-1)
        for key in ("event_at", "acquired_at", "knowledge_at"):
            value[key] = aware_timestamp(value[key]).isoformat()
        if not aware_timestamp(value["event_at"]) <= aware_timestamp(value["acquired_at"]) <= _known(value):
            raise ValueError("event must precede acquisition and revision knowledge")
        for key in fields:
            value[key] = _number(value[key], positive=key not in ("bid_size", "ask_size"))
        if value["kind"] == "QUOTE" and value["bid"] > value["ask"]:
            raise ValueError("crossed event quote rejected")
        return value

    def ingest(self, payload: dict[str, Any]) -> Contract:
        _fields(payload, HEADERS | {"events"}, HEADERS | {"events"})
        header = {k: payload[k] for k in HEADERS}
        self._header(header)
        if not isinstance(payload["events"], list) or not 1 <= len(payload["events"]) <= 1000:
            raise ValueError("one to 1000 explicit event observations required")
        events = []
        for row in payload["events"]:
            if not isinstance(row, dict) or set(row) & HEADERS:
                raise ValueError("event rows cannot override admitted source headers")
            body = self._admit_event({**header, **row})
            events.append({**body, "event_hash": _hash(body)})
        events.sort(key=lambda r: (_key(r), _known(r), r["event_id"]))
        if len({(r["event_id"], r["knowledge_at"]) for r in events}) != len(events):
            raise ValueError("duplicate event revision in admission batch")
        snapshot = {**header, "event_hashes": [r["event_hash"] for r in events]}
        identity = _hash(snapshot)
        contract = Contract.admit(State.AVAILABLE, {"title": "Admitted quote and print evidence", **snapshot,
            "snapshot_id": identity, "metrics": {"observations": len(events), "vendor_verified": False}},
            source=header["source"], classification=header["classification"],
            as_of=max(_known(r) for r in events).isoformat(), source_identity=identity,
            scope="workspace.microstructure.evidence", evidence_state="CALLER_SUPPLIED_NOT_VENDOR_VERIFIED")
        index = self._path("events.sqlite")
        index.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(index) as connection:
            connection.execute("PRAGMA synchronous=FULL")
            connection.execute("BEGIN IMMEDIATE")
            connection.execute("CREATE TABLE IF NOT EXISTS events(source TEXT,instrument TEXT,event_id TEXT,known TEXT,digest TEXT,PRIMARY KEY(source,instrument,event_id,known))")
            connection.execute("CREATE TABLE IF NOT EXISTS identities(source TEXT,instrument TEXT,event_id TEXT,kind TEXT,event_at TEXT,sequence INTEGER,terms TEXT,PRIMARY KEY(source,instrument,event_id),UNIQUE(source,instrument,event_at,sequence))")
            additions = 0
            for row in events:
                terms = dumps({key: row[key] for key in HEADERS | {"event_id", "kind", "event_at", "sequence"}})
                old = connection.execute("SELECT terms FROM identities WHERE source=? AND instrument=? AND event_id=?",
                    (row["source"], row["instrument_id"], row["event_id"])).fetchone()
                if old and old[0] != terms:
                    raise ValueError("event identity cannot change observation or source terms")
                collision = connection.execute("SELECT event_id FROM identities WHERE source=? AND instrument=? AND event_at=? AND sequence=?",
                    (row["source"], row["instrument_id"], row["event_at"], row["sequence"])).fetchone()
                if collision and collision[0] != row["event_id"]:
                    raise ValueError("event ordering identity is ambiguous")
                prior = connection.execute("SELECT digest FROM events WHERE source=? AND instrument=? AND event_id=? AND known=?",
                    (row["source"], row["instrument_id"], row["event_id"], row["knowledge_at"])).fetchone()
                if prior and prior[0] != row["event_hash"]:
                    raise ValueError("conflicting immutable event revision")
                additions += prior is None
                connection.execute("INSERT OR IGNORE INTO identities VALUES(?,?,?,?,?,?,?)",
                    (row["source"], row["instrument_id"], row["event_id"], row["kind"], row["event_at"], row["sequence"], terms))
            if connection.execute("SELECT COUNT(*) FROM events").fetchone()[0]+additions > 1000:
                raise ValueError("event revision retention bound exceeded")
            for row in events:
                write_immutable(self._path("events/" + row["event_hash"] + ".json"), dumps(row).encode())
                connection.execute("INSERT OR IGNORE INTO events VALUES(?,?,?,?,?)",
                    (row["source"], row["instrument_id"], row["event_id"], row["knowledge_at"], row["event_hash"]))
            write_immutable(self._path("snapshots/" + identity + ".json"), dumps(snapshot).encode())
        return contract

    def _event(self, identity):
        identity = _identity(identity)
        value = loads(self._path("events/" + identity + ".json").read_bytes())
        body = {k: v for k, v in value.items() if k != "event_hash"}
        if value.get("event_hash") != identity or _hash(body) != identity or self._admit_event(body) != body:
            raise ValueError("microstructure event evidence integrity failure")
        index = self._path("events.sqlite")
        with sqlite3.connect(index.as_uri()+"?mode=ro", uri=True) as connection:
            row = connection.execute("SELECT digest FROM events WHERE source=? AND instrument=? AND event_id=? AND known=?",
                (body["source"], body["instrument_id"], body["event_id"], body["knowledge_at"])).fetchone()
        if row != (identity,):
            raise ValueError("event index differs from immutable evidence")
        return value

    @staticmethod
    def _request(payload):
        _fields(payload, RUN, RUN)
        request = loads(dumps(payload))
        MicrostructureWorkspace._header({**request, "currency": "USD"})
        for key in ("start", "end", "as_of"):
            request[key] = aware_timestamp(request[key]).isoformat()
        if not aware_timestamp(request["start"]) <= aware_timestamp(request["end"]) <= aware_timestamp(request["as_of"]):
            raise ValueError("ordered historical research interval and knowledge cutoff required")
        for key in ("max_quote_age_ns", "lag_ns"):
            request[key] = _integer(request[key], 86_400_000_000_000)
        request["window_buckets"] = _integer(request["window_buckets"], 1000)
        if request["window_buckets"] == 0:
            raise ValueError("positive VPIN window required")
        for key in ("bucket_volume", "participation_rate", "proposed_quantity"):
            request[key] = _number(request[key], positive=True)
        if request["participation_rate"] > 1:
            raise ValueError("participation fraction cannot exceed one")
        return request

    def _select(self, request):
        path = self._path("events.sqlite")
        if not path.is_file():
            return []
        with sqlite3.connect(path.as_uri()+"?mode=ro", uri=True) as connection:
            if not connection.execute("SELECT 1 FROM sqlite_master WHERE name='events'").fetchone():
                return []
            rows = connection.execute("SELECT known,digest FROM events WHERE source=? AND instrument=? LIMIT 1001",
                (request["source"], request["instrument_id"])).fetchall()
        if len(rows) > 1000:
            raise ValueError("event read bound exceeded")
        result = []
        for known, identity in rows:
            if aware_timestamp(known) <= aware_timestamp(request["as_of"]):
                event = self._event(identity)
                if all(event[k] == request[k] for k in ("source", "classification", "feed", "instrument_id")) and _key(event)[0] <= aware_timestamp(request["end"]).value:
                    result.append(event)
        return sorted(result, key=lambda r: (_key(r), _known(r), r["event_id"]))

    @staticmethod
    def _compute(request, evidence):
        start, end = aware_timestamp(request["start"]), aware_timestamp(request["end"])
        quotes = [row for row in evidence if row["kind"] == "QUOTE"]
        trades = [row for row in evidence if row["kind"] == "TRADE"]
        original_quotes = [r for r in _first(quotes) if start <= aware_timestamp(r["event_at"]) <= end]
        original_trades = [r for r in _first(trades) if start <= aware_timestamp(r["event_at"]) <= end]
        rows, ofi_rows = [], []
        for row in original_trades:
            known = _known(row)
            eligible_quotes = [q for q in _latest(quotes, known) if _key(q) <= _key(row)]
            match = match_quotes([_trade(row)], [_quote(q) for q in eligible_quotes], lag_ns=request["lag_ns"],
                max_age_ns=request["max_quote_age_ns"])[0]
            past = [t for t in _latest(trades, known) if _key(t) < _key(row)]
            prior_matches = match_quotes([_trade(t) for t in past], [], max_age_ns=0)
            classified = lee_ready([*prior_matches, match])[-1]
            quote_row = next((q for q in eligible_quotes if match.quote is not None and _key(q) == (match.quote.timestamp_ns, match.quote.sequence)), None)
            rows.append({"event_id": row["event_id"], "event_at": row["event_at"], "decision_at": row["knowledge_at"],
                "sequence": row["sequence"], "price": row["price"], "size": row["size"],
                "estimated_sign": classified.sign, "classification_method": classified.method,
                "sign_state": "AVAILABLE" if classified.sign else "UNKNOWN", "quote_match": match.reason,
                "quote_age_ns": match.quote_age_ns, "quote_event_hash": quote_row["event_hash"] if quote_row else None,
                "event_hash": row["event_hash"], "trading_approved": False})
        for row in original_quotes:
            prior = [q for q in _latest(quotes, _known(row)) if _key(q) < _key(row)]
            value = float(order_flow_imbalance([_quote(prior[-1]), _quote(row)])[-1]) if prior else None
            ofi_rows.append({"event_id": row["event_id"], "event_at": row["event_at"], "decision_at": row["knowledge_at"],
                "sequence": row["sequence"], "best_level_ofi": value, "state": "AVAILABLE" if prior else "UNAVAILABLE",
                "event_hash": row["event_hash"], "previous_event_hash": prior[-1]["event_hash"] if prior else None})
        intervals, unknown_intervals = [], 0
        for left, right in zip(original_quotes, original_quotes[1:], strict=False):
            inside = [(trade, result) for trade, result in zip(original_trades, rows, strict=True) if _key(left) < _key(trade) <= _key(right)]
            complete = (_known(left) <= _known(right) and all(_known(t) <= _known(right) and r["estimated_sign"] for t, r in inside))
            unknown_intervals += not complete
            intervals.append({"start": left["event_at"], "end": right["event_at"], "known_at": right["knowledge_at"],
                "price_change": _quote(right).midpoint-_quote(left).midpoint,
                "signed_volume": math.fsum(r["size"]*r["estimated_sign"] for _, r in inside) if complete else None,
                "state": "AVAILABLE" if complete else "UNKNOWN"})
        if unknown_intervals:
            kyle = {"state": "UNKNOWN", "coefficient": None, "intercept": None, "r_squared": None, "sample_count": len(intervals)}
        else:
            kyle = asdict(kyle_lambda([r["price_change"] for r in intervals], [r["signed_volume"] for r in intervals]))
        sizes, signs = [r["size"] for r in rows], [r["estimated_sign"] for r in rows]
        estimate = asdict(vpin(sizes, signs, bucket_volume=request["bucket_volume"], window_buckets=request["window_buckets"], max_buckets=1000))
        estimate["kernel_state"] = estimate["state"]
        if estimate["excluded_unknown_volume"]:
            estimate["state"] = "UNKNOWN"
        volume = math.fsum(sizes)
        capacity = volume*request["participation_rate"]
        return {"schema_version": VERSION, "request": request, "source": request["source"], "feed": request["feed"],
            "implementation_provenance": {"numpy": version("numpy"), "kernel": "gig.microstructure.analytics", "version": VERSION},
            "instrument_id": request["instrument_id"], "input_classification": request["classification"],
            "as_of": request["as_of"], "observed_at": max((r["event_at"] for r in [*original_quotes, *original_trades]), key=aware_timestamp),
            "event_hashes": sorted(r["event_hash"] for r in evidence), "rows": rows, "ofi_rows": ofi_rows,
            "kyle_lambda": {**kyle, "unit": "USD price change per signed underlying-volume unit", "intervals": intervals,
                "unknown_intervals": unknown_intervals, "method": "OLS_ON_ALL_CONTIGUOUS_QUOTE_INTERVALS_NO_UNKNOWN_INTERVAL_DROPPING"},
            "vpin": {**estimate, "method": "LEE_READY_EVENT_SIGN_EQUAL_VOLUME_BUCKET_NOT_BULK_VOLUME_VPIN"},
            "capacity": {"state": "AVAILABLE" if volume else "UNAVAILABLE", "observed_print_volume": volume,
                "participation_rate": request["participation_rate"], "modeled_quantity_budget": capacity if volume else None,
                "proposed_quantity": request["proposed_quantity"], "within_observed_volume_budget": request["proposed_quantity"] <= capacity if volume else None,
                "executable_capacity": {"state": "UNKNOWN", "value": None},
                "scope": "DECLARED_PARTICIPATION_OF_ADMITTED_PRINT_SAMPLE_NOT_DISPLAYED_OR_EXECUTABLE_LIQUIDITY"},
            "execution_costs": {"state": "UNAVAILABLE", "value": None, "reason": "NO_OWN_EXECUTION_SIDE_AND_ARRIVAL_BENCHMARK"},
            "market_impact": {"state": kyle["state"], "linear_price_change": kyle["coefficient"]*request["proposed_quantity"] if kyle["coefficient"] is not None else None,
                "scope": "DESCRIPTIVE_KYLE_LINEAR_ASSUMPTION_NOT_CAUSAL_EXECUTION_COST"},
            "metrics": {"quotes": len(original_quotes), "prints": len(rows), "unknown_signs": sum(s == 0 for s in signs),
                "best_level_ofi_sum": {"value": math.fsum(r["best_level_ofi"] for r in ofi_rows if r["best_level_ofi"] is not None)
                    if any(r["best_level_ofi"] is not None for r in ofi_rows) else None,
                    "state": "AVAILABLE" if any(r["best_level_ofi"] is not None for r in ofi_rows) else "UNAVAILABLE",
                    "unit": "admitted best displayed-size changes"},
                "observed_print_volume": volume, "trading_approved": False},
            "provenance": {"revision_policy": "FROZEN_ORIGINAL_DECISION_TRADE_AND_QUOTE_FEATURES",
                "matching": "QUOTE_EVENT_TIME_SEQUENCE_AND_KNOWLEDGE_MUST_PRECEDE_TRADE_DECISION",
                "clock_authority": "CALLER_DECLARED_NOT_VENDOR_VERIFIED", "print_coverage": "ADMITTED_SAMPLE_ONLY_NOT_CONSOLIDATED_COMPLETE_TAPE",
                "ofi": "CONT_STYLE_BEST_DISPLAYED_QUEUE_CHANGES_NOT_EXECUTED_ORDER_FLOW", "tick_history": "KNOWN_PRIOR_PRINTS_ONLY"},
            "trading_approved": False, "economically_validated": False,
            "notes": ["Best-level displayed OFI is not traded volume or executable liquidity.",
                "Trade signs are estimates; unknown signs and unknown Kyle intervals are not imputed as zero evidence.",
                "Historical features preserve original decisions; subsequent corrections are archived and can inform later knowledge clocks.",
                "Source/feed classifications establish no provider authenticity, IEX/SIP/OPRA coverage or entitlement."]}

    @staticmethod
    def _contract(document, state=State.AVAILABLE):
        return Contract.admit(state, {"title": "Persisted quote and print microstructure", **document}
            if state in {State.AVAILABLE, State.STALE} else None, source=document["source"], classification="MODEL_ESTIMATE",
            as_of=document["as_of"], scope="workspace.microstructure.research", instrument_identity=document["instrument_id"],
            model_identity=document["report_id"], model_version=VERSION, source_identity=_hash(document["event_hashes"]),
            evidence_state="HASH_VERIFIED_ORIGINAL_DECISION_EVENT_RESEARCH", readiness="RESEARCH_ONLY")

    @staticmethod
    def _missing():
        return Contract.admit(State.UNAVAILABLE, None, source="persisted-microstructure", classification="UNAVAILABLE",
            scope="workspace.microstructure", error=public_error("DATA_UNAVAILABLE"))

    def run(self, payload: dict[str, Any]) -> Contract:
        request = self._request(payload)
        evidence = self._select(request)
        if not evidence or not any(aware_timestamp(request["start"]) <= aware_timestamp(r["event_at"]) for r in evidence):
            return self._missing()
        body = self._compute(request, evidence)
        identity = _hash(body)
        document = {**body, "report_id": identity}
        contract = self._contract(document)
        write_immutable(self._path("runs/" + identity + ".json"), dumps(document).encode())
        with sqlite3.connect(self._path("events.sqlite")) as connection:
            connection.execute("PRAGMA synchronous=FULL")
            connection.execute("CREATE TABLE IF NOT EXISTS reports(report_id TEXT PRIMARY KEY,as_of TEXT NOT NULL)")
            old = connection.execute("SELECT as_of FROM reports WHERE report_id=?", (identity,)).fetchone()
            if old is not None and old[0] != document["as_of"]:
                raise ValueError("microstructure report index conflict")
            if old is None and connection.execute("SELECT COUNT(*) FROM reports").fetchone()[0] >= 1000:
                raise ValueError("microstructure report retention exceeded")
            connection.execute("INSERT OR IGNORE INTO reports VALUES(?,?)", (identity, document["as_of"]))
        return contract

    def load(self, report_id: str) -> dict[str, Any]:
        identity = _identity(report_id)
        encoded = self._path("runs/" + identity + ".json").read_bytes()
        document = loads(encoded)
        body = {k: v for k, v in document.items() if k != "report_id"}
        if document.get("report_id") != identity or _hash(body) != identity or body.get("schema_version") != VERSION or dumps(document).encode() != encoded:
            raise ValueError("microstructure report integrity mismatch")
        with sqlite3.connect(self._path("events.sqlite").as_uri()+"?mode=ro", uri=True) as connection:
            entry = connection.execute("SELECT as_of FROM reports WHERE report_id=?", (identity,)).fetchone()
        if entry != (document["as_of"],):
            raise ValueError("microstructure report index differs from immutable evidence")
        events = [self._event(value) for value in document["event_hashes"]]
        if dumps(self._compute(document["request"], events)) != dumps(body):
            raise ValueError("microstructure deterministic replay differs from saved evidence")
        return document

    def replay(self, payload):
        _fields(payload, {"report_id"}, {"report_id"})
        try:
            return self._contract(self.load(payload["report_id"]))
        except FileNotFoundError:
            return self._missing()

    def section(self, section, as_of, max_age_seconds):
        if section not in {"markets", "stat-arb", "research", "execution"}:
            return None
        cutoff, maximum = _age(as_of, max_age_seconds)
        path = self._path("events.sqlite")
        if not path.is_file():
            return None
        with sqlite3.connect(path.as_uri()+"?mode=ro", uri=True) as connection:
            if not connection.execute("SELECT 1 FROM sqlite_master WHERE name='reports'").fetchone():
                return None
            rows = connection.execute("SELECT report_id,as_of FROM reports ORDER BY rowid LIMIT 1001").fetchall()
        if len(rows) > 1000:
            raise ValueError("microstructure report read bound exceeded")
        visible = [(aware_timestamp(stamp), i, identity) for i, (identity, stamp) in enumerate(rows) if aware_timestamp(stamp) <= cutoff]
        if not visible:
            return None
        document = self.load(max(visible)[2])
        state = State.UNKNOWN if maximum is None else State.STALE if (cutoff-aware_timestamp(document["observed_at"])).value/1e9 > maximum else State.AVAILABLE
        return self._contract(document, state)
