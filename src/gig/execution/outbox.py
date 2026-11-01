"""Durable offline transport intents and replayable simulated execution evidence.

A claim commits before transport. A crash or exception leaves an uncertain
intent which is never resent until an explicit authoritative-absence query and
explicit retry. This protocol never constructs a network or brokerage client.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any, Protocol

from gig.execution.oms import TERMINAL, OfflineOMS, OrderError, _decimal, _exact, _text, _time
from gig.timestamps import aware_timestamp

MAX_EVENTS = 1000


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, allow_nan=False, separators=(",", ":"))


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def event_value(payload: dict[str, Any]) -> dict[str, Any]:
    """Strict immutable protocol values; source occurrence time is never inferred."""
    required = {"event_id", "client_id", "kind", "occurred_at"}
    if not isinstance(payload, dict) or not required <= payload.keys():
        raise OrderError("missing event identity or occurrence timestamp")
    kind = payload["kind"]
    if kind not in {"ACK", "FILL", "CANCELLED", "REJECTED", "EXPIRED"}:
        raise OrderError("unsupported simulated event kind")
    allowed = required | ({"quantity", "price", "fee"} if kind == "FILL" else set())
    if set(payload) - allowed or (kind == "FILL" and not {"quantity", "price"} <= payload.keys()):
        raise OrderError("event fields do not match event kind")
    stamp = aware_timestamp(payload["occurred_at"])
    if stamp.nanosecond:
        raise OrderError("execution event precision is limited to microseconds")
    result = {
        "event_id": _text(payload["event_id"], "event_id"),
        "client_id": _text(payload["client_id"], "client_id"),
        "kind": kind,
        "occurred_at": stamp.isoformat(),
    }
    if kind == "FILL":
        result.update(
            quantity=str(_decimal(payload["quantity"])),
            price=str(_decimal(payload["price"])),
            fee=str(_decimal(payload.get("fee", "0"), positive=False)),
        )
    return result


class OfflineExecutionAdapter(Protocol):
    """Only a deliberately offline implementation may be supplied in this cycle."""

    offline: bool

    def submit(self, intent: dict[str, Any], *, now: datetime) -> dict[str, Any]: ...
    def cancel(self, client_id: str, *, now: datetime) -> dict[str, Any]: ...
    def lookup(self, client_id: str) -> dict[str, Any]: ...
    def positions(self) -> dict[str, str]: ...


class ExecutionOutbox:
    def __init__(self, oms: OfflineOMS, *, fault_hook: Callable[[str], None] | None = None):
        self.oms = oms
        self.hook = fault_hook or (lambda _: None)
        self.db = oms.connection
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS transport_outbox (
                identity TEXT PRIMARY KEY, client_id TEXT NOT NULL REFERENCES orders(client_id),
                kind TEXT NOT NULL, intent_hash TEXT NOT NULL, payload TEXT NOT NULL,
                state TEXT NOT NULL, attempts INTEGER NOT NULL DEFAULT 0,
                admitted_at TEXT NOT NULL, updated_at TEXT NOT NULL,
                UNIQUE(client_id,kind)
            );
            CREATE TABLE IF NOT EXISTS transport_events (
                sequence INTEGER PRIMARY KEY AUTOINCREMENT, event_id TEXT UNIQUE NOT NULL,
                client_id TEXT NOT NULL, kind TEXT NOT NULL, occurred_at TEXT NOT NULL,
                received_at TEXT NOT NULL, payload TEXT NOT NULL, digest TEXT NOT NULL,
                disposition TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS transport_queries (
                sequence INTEGER PRIMARY KEY AUTOINCREMENT, client_id TEXT NOT NULL,
                queried_at TEXT NOT NULL, state TEXT NOT NULL, response_hash TEXT NOT NULL
            );
        """)

    def _adapter(self, adapter: OfflineExecutionAdapter) -> None:
        if getattr(adapter, "offline", None) is not True:
            raise OrderError("only explicit offline adapters are supported")

    def intents(self) -> list[dict[str, Any]]:
        return [
            dict(row)
            for row in self.db.execute(
                "SELECT * FROM transport_outbox ORDER BY admitted_at,identity"
            )
        ]

    @_exact
    def admit_order(
        self,
        client_id: str,
        symbol: str,
        side: str,
        quantity: str | int | Decimal,
        *,
        reference_price: str | int | Decimal,
        price_asof: datetime,
        max_notional: str | int | Decimal,
        max_position_quantity: str | int | Decimal,
        limit_price: str | int | Decimal | None = None,
        now: datetime | None = None,
    ) -> dict:
        at = _time(now or datetime.now(UTC))
        payload = {
            "client_id": _text(client_id, "client_id"),
            "symbol": _text(symbol, "symbol"),
            "side": side,
            "quantity": str(_decimal(quantity)),
            "limit_price": None if limit_price is None else str(_decimal(limit_price)),
            "risk": {
                "reference_price": str(_decimal(reference_price)),
                "price_asof": _time(price_asof).isoformat(),
                "max_notional": str(_decimal(max_notional)),
                "max_position_quantity": str(_decimal(max_position_quantity)),
            },
        }
        digest = _digest(payload)
        with self.oms._transaction():
            prior = self.db.execute(
                "SELECT * FROM transport_outbox WHERE client_id=? AND kind='SUBMIT'", (client_id,)
            ).fetchone()
            if prior:
                if prior["intent_hash"] != digest:
                    raise OrderError("client_id conflicts with durable transport intent")
                return self.oms.order(client_id)
            self.oms.create(client_id, symbol, side, quantity, limit_price=limit_price, now=at)
            self.oms.approve(
                client_id,
                reference_price=reference_price,
                price_asof=price_asof,
                max_notional=max_notional,
                max_position_quantity=max_position_quantity,
                now=at,
            )
            self.oms.reserve_submission(client_id, now=at)
            self.db.execute(
                "INSERT INTO transport_outbox VALUES(?,?,?, ?,?,'READY',0,?,?)",
                (
                    _digest([client_id, "SUBMIT"]),
                    client_id,
                    "SUBMIT",
                    digest,
                    _canonical(payload),
                    at.isoformat(),
                    at.isoformat(),
                ),
            )
            self.hook("after_intent_write")
        return self.oms.order(client_id)

    def request_cancel(self, client_id: str, *, now: datetime) -> dict:
        at = _time(now)
        with self.oms._transaction():
            order = self.oms.order(client_id)
            if order["state"] in TERMINAL:
                return {
                    "state": order["state"],
                    "cancel_queued": False,
                    "classification": "SIMULATED",
                }
            if order["state"] not in {"SUBMITTING", "UNKNOWN", "ACKNOWLEDGED", "PARTIALLY_FILLED"}:
                raise OrderError("cancel requires a reserved or working intent")
            submit = self.db.execute(
                "SELECT state FROM transport_outbox WHERE client_id=? AND kind='SUBMIT'",
                (client_id,),
            ).fetchone()
            if submit is None:
                raise OrderError("cancel requires a transport-managed intent")
            if submit[0] == "READY":
                self.oms.terminate(client_id, "CANCELLED", now=at)
                self.db.execute(
                    "UPDATE transport_outbox SET state='RESOLVED',updated_at=? WHERE client_id=? AND kind='SUBMIT'",
                    (at.isoformat(), client_id),
                )
                return {"state": "CANCELLED", "cancel_queued": False, "classification": "SIMULATED"}
            payload = {"client_id": client_id, "kind": "CANCEL"}
            self.db.execute(
                "INSERT OR IGNORE INTO transport_outbox VALUES(?,?,?, ?,?,'READY',0,?,?)",
                (
                    _digest([client_id, "CANCEL"]),
                    client_id,
                    "CANCEL",
                    _digest(payload),
                    _canonical(payload),
                    at.isoformat(),
                    at.isoformat(),
                ),
            )
        return {"state": "CANCEL_PENDING", "cancel_queued": True, "classification": "SIMULATED"}

    def dispatch(
        self, adapter: OfflineExecutionAdapter, *, now: datetime, limit: int = 100
    ) -> dict:
        self._adapter(adapter)
        at = _time(now)
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 100:
            raise OrderError("dispatch limit must lie in 1..100")
        sent = 0
        unknown = 0
        event_errors = 0
        for row in self.db.execute(
            "SELECT * FROM transport_outbox WHERE state='READY' ORDER BY admitted_at,identity LIMIT ?",
            (limit,),
        ).fetchall():
            if row["kind"] == "SUBMIT" and self.oms.kill_switch:
                continue
            with self.oms._transaction():
                current = self.db.execute(
                    "SELECT state FROM transport_outbox WHERE identity=?", (row["identity"],)
                ).fetchone()
                if current[0] != "READY":
                    continue
                if row["kind"] == "SUBMIT":
                    self.oms._check_submission(self.oms.order(row["client_id"]), at)
                self.db.execute(
                    "UPDATE transport_outbox SET state='SENDING',attempts=attempts+1,updated_at=? WHERE identity=?",
                    (at.isoformat(), row["identity"]),
                )
            self.hook("after_claim")
            try:
                response = (
                    adapter.submit(json.loads(row["payload"]), now=at)
                    if row["kind"] == "SUBMIT"
                    else adapter.cancel(row["client_id"], now=at)
                )
                sent += 1
                self.hook("after_transport")
                outcome = self.ingest_event(response, received_at=at)
                event_errors += outcome["state"] == "ERROR"
            except Exception:
                with self.oms._transaction():
                    self.db.execute(
                        "UPDATE transport_outbox SET state='UNKNOWN',updated_at=? WHERE identity=?",
                        (at.isoformat(), row["identity"]),
                    )
                    if row["kind"] == "SUBMIT":
                        self.oms.submission_unknown(row["client_id"], now=at)
                unknown += 1
        return {
            "state": "ERROR" if event_errors else "UNKNOWN" if unknown else "AVAILABLE",
            "sent": sent,
            "ambiguous": unknown,
            "event_errors": event_errors,
            "broker_connected": False,
            "classification": "SIMULATED",
        }

    def ingest_event(self, payload: dict[str, Any], *, received_at: datetime) -> dict:
        event = event_value(payload)
        at = _time(received_at)
        occurred = datetime.fromisoformat(event["occurred_at"])
        if occurred > at:
            raise OrderError("event occurrence cannot exceed receipt time")
        encoded, digest = _canonical(event), _digest(event)
        with self.oms._transaction():
            old = self.db.execute(
                "SELECT digest,disposition FROM transport_events WHERE event_id=?",
                (event["event_id"],),
            ).fetchone()
            if old:
                if old["digest"] != digest:
                    raise OrderError("event_id conflicts with preserved event")
                return {
                    "state": "ERROR"
                    if old["disposition"] in {"QUARANTINED", "APPLIED_RISK_BREACH"}
                    else "AVAILABLE",
                    "disposition": "DUPLICATE",
                    "classification": "SIMULATED",
                }
            self.db.execute(
                "INSERT INTO transport_events(event_id,client_id,kind,occurred_at,received_at,payload,digest,disposition) VALUES(?,?,?,?,?,?,?,'RECEIVED')",
                (
                    event["event_id"],
                    event["client_id"],
                    event["kind"],
                    event["occurred_at"],
                    at.isoformat(),
                    encoded,
                    digest,
                ),
            )
            try:
                with self.oms._transaction():
                    disposition = self._apply_event(event, at)
            except OrderError:
                disposition = "QUARANTINED"
                self.oms.set_kill_switch(True, now=at)
            self.db.execute(
                "UPDATE transport_events SET disposition=? WHERE event_id=?",
                (disposition, event["event_id"]),
            )
            self.hook("before_event_commit")
        return {
            "state": "ERROR"
            if disposition in {"QUARANTINED", "APPLIED_RISK_BREACH"}
            else "AVAILABLE",
            "disposition": disposition,
            "classification": "SIMULATED",
            "broker_connected": False,
        }

    @_exact
    def _apply_event(self, event: dict[str, Any], at: datetime) -> str:
        client = event["client_id"]
        order = self.oms.order(client)
        occurred = datetime.fromisoformat(event["occurred_at"])
        if occurred < datetime.fromisoformat(order["created_at"]):
            raise OrderError("event precedes its intent")
        at = max(at, datetime.fromisoformat(order["updated_at"]))
        kind = event["kind"]
        risk_breach = False
        if kind == "ACK":
            if order["state"] not in TERMINAL:
                self.oms.acknowledge(client, now=at)
        elif kind == "FILL":
            terminal = order["state"] in {"CANCELLED", "EXPIRED"}
            if terminal:
                cutoff = self.db.execute(
                    "SELECT occurred_at FROM transport_events WHERE client_id=? AND kind IN ('CANCELLED','EXPIRED') AND disposition='APPLIED' ORDER BY sequence DESC LIMIT 1",
                    (client,),
                ).fetchone()
                if cutoff is None or occurred > datetime.fromisoformat(cutoff[0]):
                    raise OrderError("fill contradicts terminal occurrence")
            elif order["state"] in {"SUBMITTING", "UNKNOWN"}:
                self.oms.acknowledge(client, now=at)
            self.oms.fill(
                client,
                event["event_id"],
                event["quantity"],
                event["price"],
                fee=event["fee"],
                now=at,
                occurred_at=occurred,
                reconciled_terminal=terminal,
            )
            position = Decimal(self.oms.positions().get(order["symbol"], "0"))
            if abs(position) > Decimal(order["risk"]["max_position_quantity"]):
                self.oms.set_kill_switch(True, now=at)
                risk_breach = True
        elif order["state"] not in TERMINAL:
            self.oms.terminate(client, kind, now=at)
        else:
            return "IGNORED_TERMINAL"
        current = self.oms.order(client)
        self.db.execute(
            "UPDATE transport_outbox SET state=?,updated_at=? WHERE client_id=? AND kind='SUBMIT'",
            (
                "RESOLVED" if current["state"] in TERMINAL else "ACKNOWLEDGED",
                at.isoformat(),
                client,
            ),
        )
        if kind in {"CANCELLED", "REJECTED", "EXPIRED", "FILL"} and current["state"] in TERMINAL:
            self.db.execute(
                "UPDATE transport_outbox SET state='RESOLVED',updated_at=? WHERE client_id=? AND kind='CANCEL'",
                (at.isoformat(), client),
            )
        return "APPLIED_RISK_BREACH" if risk_breach else "APPLIED"

    def reconcile(self, adapter: OfflineExecutionAdapter, *, now: datetime) -> dict:
        self._adapter(adapter)
        at = _time(now)
        queries = []
        event_errors = False
        clients = [
            r[0]
            for r in self.db.execute(
                "SELECT DISTINCT client_id FROM transport_outbox WHERE state NOT IN ('RESOLVED','ABSENT_PROVEN') ORDER BY client_id"
            )
        ]
        for client in clients:
            try:
                response = adapter.lookup(client)
            except Exception:
                response = {"state": "UNKNOWN"}
            state = response.get("state")
            if state not in {"FOUND", "ABSENT", "UNKNOWN"}:
                state, response = "UNKNOWN", {"state": "UNKNOWN"}
            with self.oms._transaction():
                self.db.execute(
                    "INSERT INTO transport_queries(client_id,queried_at,state,response_hash) VALUES(?,?,?,?)",
                    (client, at.isoformat(), state, _digest(response)),
                )
                if state == "ABSENT":
                    self.db.execute(
                        "UPDATE transport_outbox SET state='ABSENT_PROVEN',updated_at=? WHERE client_id=? AND kind='SUBMIT' AND state IN ('SENDING','UNKNOWN')",
                        (at.isoformat(), client),
                    )
                elif state == "FOUND":
                    existing = self.db.execute(
                        "SELECT intent_hash FROM transport_outbox WHERE client_id=? AND kind='SUBMIT'",
                        (client,),
                    ).fetchone()
                    if existing is None or existing[0] != response.get("intent_hash"):
                        self.oms.set_kill_switch(True, now=at)
                        state = "CONFLICT"
                    else:
                        events = response.get("events")
                        if not isinstance(events, list) or len(events) > MAX_EVENTS:
                            raise OrderError("reconciliation events exceed bounds")
                        for event in events:
                            outcome = self.ingest_event(event, received_at=at)
                            event_errors = event_errors or outcome["state"] == "ERROR"
            queries.append({"client_id": client, "state": state})
        try:
            positions = self.oms.reconcile(adapter.positions())
        except Exception:
            positions = {"state": "UNKNOWN", "matched": False, "differences": {}}
        known_error = (
            event_errors
            or positions.get("state") == "ERROR"
            or any(q["state"] == "CONFLICT" for q in queries)
            or self.db.execute(
                "SELECT COUNT(*) FROM transport_events WHERE disposition='QUARANTINED'"
            ).fetchone()[0]
            > 0
        )
        return {
            "state": "ERROR"
            if known_error
            else "AVAILABLE"
            if positions.get("matched") and all(q["state"] in {"FOUND", "ABSENT"} for q in queries)
            else "UNKNOWN",
            "queries": queries,
            "positions": positions,
            "classification": "SIMULATED_COMPARISON",
            "broker_connected": False,
            "automatic_retries": 0,
        }

    def retry_absent(self, client_id: str, *, now: datetime) -> dict:
        at = _time(now)
        with self.oms._transaction():
            row = self.db.execute(
                "SELECT state FROM transport_outbox WHERE client_id=? AND kind='SUBMIT'",
                (client_id,),
            ).fetchone()
            if row is None or row[0] != "ABSENT_PROVEN":
                raise OrderError("retry requires explicit authoritative absence reconciliation")
            order = self.oms.order(client_id)
            if order["state"] not in {"SUBMITTING", "UNKNOWN"}:
                raise OrderError("retry cannot replace known acknowledgement or execution")
            self.db.execute("UPDATE orders SET state='APPROVED' WHERE client_id=?", (client_id,))
            self.oms.reserve_submission(client_id, now=at)
            self.db.execute(
                "UPDATE transport_outbox SET state='READY',updated_at=? WHERE client_id=? AND kind='SUBMIT'",
                (at.isoformat(), client_id),
            )
        return self.oms.order(client_id)

    def recover(self) -> dict:
        pending = [row for row in self.intents() if row["state"] in {"SENDING", "UNKNOWN"}]
        quarantined = self.db.execute(
            "SELECT COUNT(*) FROM transport_events WHERE disposition='QUARANTINED'"
        ).fetchone()[0]
        return {
            "pending_ambiguities": pending,
            "requires_reconciliation": bool(pending or quarantined),
            "quarantined_events": quarantined,
            "auto_resubmitted": 0,
            "positions": self.oms.positions(),
            "kill_switch": self.oms.kill_switch,
            "classification": "SIMULATED",
            "broker_connected": False,
        }

    def replay_events(self) -> list[dict]:
        return [
            {
                **json.loads(row["payload"]),
                "received_at": row["received_at"],
                "disposition": row["disposition"],
                "classification": "SIMULATED",
            }
            for row in self.db.execute(
                "SELECT * FROM transport_events ORDER BY sequence LIMIT ?", (MAX_EVENTS,)
            )
        ]
