"""A durable local simulated counterparty and strict Alpaca-shaped fixtures.

No SDK, socket, credentials, account, or real submission is implemented here.
A separate SQLite file represents the simulated external system for crash tests.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

from gig.execution.oms import OrderError, _decimal, _exact, _text, _time
from gig.execution.outbox import _canonical, _digest, event_value


class OfflinePaperAdapter:
    offline = True

    def __init__(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path, isolation_level=None, timeout=10)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS remote_orders (
                client_id TEXT PRIMARY KEY,intent_hash TEXT NOT NULL,payload TEXT NOT NULL,state TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS remote_events (
                sequence INTEGER PRIMARY KEY AUTOINCREMENT,event_id TEXT UNIQUE NOT NULL,
                client_id TEXT NOT NULL,payload TEXT NOT NULL
            );
        """)
        self.disconnected = False
        self.fault: str | None = None
        self.reject_next = False
        self.queries_unknown = False
        self.submit_calls = 0

    def close(self) -> None:
        self.db.close()

    def _event(self, client_id: str, kind: str, now: datetime, **values) -> dict:
        index = self.db.execute(
            "SELECT COUNT(*) FROM remote_events WHERE client_id=?", (client_id,)
        ).fetchone()[0]
        event = event_value(
            {
                "event_id": "offline-" + _digest([client_id, kind, index]),
                "client_id": client_id,
                "kind": kind,
                "occurred_at": _time(now).isoformat(),
                **values,
            }
        )
        self.db.execute(
            "INSERT INTO remote_events(event_id,client_id,payload) VALUES(?,?,?)",
            (event["event_id"], client_id, _canonical(event)),
        )
        return event

    def _ready(self) -> None:
        if self.disconnected:
            raise ConnectionError("offline simulated disconnection")

    def submit(self, intent: dict[str, Any], *, now: datetime) -> dict:
        self._ready()
        self.submit_calls += 1
        if self.fault == "before_accept":
            raise TimeoutError("simulated timeout before acceptance")
        client = _text(intent["client_id"], "client_id")
        digest = _digest(intent)
        self.db.execute("BEGIN IMMEDIATE")
        try:
            prior = self.db.execute(
                "SELECT intent_hash FROM remote_orders WHERE client_id=?", (client,)
            ).fetchone()
            if prior:
                if prior[0] != digest:
                    raise OrderError("offline remote client identity conflict")
                event = json.loads(
                    self.db.execute(
                        "SELECT payload FROM remote_events WHERE client_id=? ORDER BY sequence LIMIT 1",
                        (client,),
                    ).fetchone()[0]
                )
            else:
                kind = "REJECTED" if self.reject_next else "ACK"
                self.reject_next = False
                self.db.execute(
                    "INSERT INTO remote_orders VALUES(?,?,?,?)",
                    (client, digest, _canonical(intent), kind),
                )
                event = self._event(client, kind, now)
            self.db.commit()
        except BaseException:
            self.db.rollback()
            raise
        if self.fault == "after_accept":
            raise TimeoutError("simulated timeout after durable acceptance")
        return event

    def lookup(self, client_id: str) -> dict:
        self._ready()
        if self.queries_unknown:
            return {"state": "UNKNOWN"}
        row = self.db.execute(
            "SELECT * FROM remote_orders WHERE client_id=?", (client_id,)
        ).fetchone()
        if row is None:
            return {"state": "ABSENT", "classification": "SIMULATED_AUTHORITATIVE_ABSENCE"}
        events = [
            json.loads(row[0])
            for row in self.db.execute(
                "SELECT payload FROM remote_events WHERE client_id=? ORDER BY sequence",
                (client_id,),
            )
        ]
        if len(events) > 1000:
            raise OrderError("offline event page exceeds declared bound")
        return {
            "state": "FOUND",
            "intent_hash": row["intent_hash"],
            "events": events,
            "classification": "SIMULATED",
        }

    def cancel(self, client_id: str, *, now: datetime) -> dict:
        self._ready()
        self.db.execute("BEGIN IMMEDIATE")
        try:
            row = self.db.execute(
                "SELECT state FROM remote_orders WHERE client_id=?", (client_id,)
            ).fetchone()
            if row is None:
                raise OrderError("offline cancel target is absent")
            if row[0] in {"CANCELLED", "REJECTED", "FILLED", "EXPIRED"}:
                event = json.loads(
                    self.db.execute(
                        "SELECT payload FROM remote_events WHERE client_id=? ORDER BY sequence DESC LIMIT 1",
                        (client_id,),
                    ).fetchone()[0]
                )
            else:
                event = self._event(client_id, "CANCELLED", now)
                self.db.execute(
                    "UPDATE remote_orders SET state='CANCELLED' WHERE client_id=?", (client_id,)
                )
            self.db.commit()
            return event
        except BaseException:
            self.db.rollback()
            raise

    @_exact
    def fill(
        self,
        client_id: str,
        quantity: str | int | Decimal,
        price: str | int | Decimal,
        *,
        now: datetime,
        fee: str | int | Decimal = "0",
    ) -> dict:
        """Explicit test-fixture execution; never triggered merely by submission."""
        self._ready()
        qty, px, cost = _decimal(quantity), _decimal(price), _decimal(fee, positive=False)
        self.db.execute("BEGIN IMMEDIATE")
        try:
            row = self.db.execute(
                "SELECT * FROM remote_orders WHERE client_id=?", (client_id,)
            ).fetchone()
            if row is None or row["state"] not in {"ACK", "PARTIALLY_FILLED"}:
                raise OrderError("offline fill requires a simulated working order")
            total = (
                sum(
                    (
                        Decimal(json.loads(r[0])["quantity"])
                        for r in self.db.execute(
                            "SELECT payload FROM remote_events WHERE client_id=?", (client_id,)
                        )
                        if json.loads(r[0])["kind"] == "FILL"
                    ),
                    Decimal(0),
                )
                + qty
            )
            intended = Decimal(json.loads(row["payload"])["quantity"])
            if total > intended:
                raise OrderError("offline fill exceeds remaining quantity")
            event = self._event(
                client_id, "FILL", now, quantity=str(qty), price=str(px), fee=str(cost)
            )
            state = "FILLED" if total == intended else "PARTIALLY_FILLED"
            self.db.execute(
                "UPDATE remote_orders SET state=? WHERE client_id=?", (state, client_id)
            )
            self.db.commit()
            return event
        except BaseException:
            self.db.rollback()
            raise

    @_exact
    def positions(self) -> dict[str, str]:
        self._ready()
        positions: dict[str, Decimal] = {}
        for row in self.db.execute(
            "SELECT remote_events.payload,remote_orders.payload FROM remote_events JOIN remote_orders USING(client_id) ORDER BY remote_events.sequence"
        ):
            event, intent = json.loads(row[0]), json.loads(row[1])
            if event["kind"] == "FILL":
                symbol = intent["symbol"]
                positions[symbol] = positions.get(symbol, Decimal(0)) + Decimal(
                    event["quantity"]
                ) * (1 if intent["side"] == "buy" else -1)
        return {key: str(value) for key, value in sorted(positions.items())}


def parse_alpaca_fixture(payload: dict[str, Any], *, expected: dict[str, Any]) -> dict:
    """Map a caller-supplied paper event fixture after checking intent identity.

    No provider authenticity is implied. `qty` is an incremental execution
    quantity, never the order's requested or cumulative filled quantity.
    """
    if not isinstance(payload, dict) or set(payload) - {
        "event",
        "event_id",
        "timestamp",
        "order",
        "qty",
        "price",
        "fee",
    }:
        raise OrderError("unsupported Alpaca fixture fields")
    order = payload.get("order")
    if not isinstance(order, dict):
        raise OrderError("Alpaca fixture requires order identity")
    for key, expected_key in (
        ("client_order_id", "client_id"),
        ("symbol", "symbol"),
        ("side", "side"),
    ):
        if order.get(key) != expected[expected_key]:
            raise OrderError("Alpaca fixture differs from admitted intent")
    requested_quantity = order.get("qty")
    if requested_quantity is None:
        raise OrderError("Alpaca fixture requires requested quantity")
    if _decimal(requested_quantity) != _decimal(expected["quantity"]):
        raise OrderError("Alpaca fixture requested quantity differs from intent")
    names = {
        "new": "ACK",
        "accepted": "ACK",
        "partial_fill": "FILL",
        "fill": "FILL",
        "canceled": "CANCELLED",
        "rejected": "REJECTED",
        "expired": "EXPIRED",
    }
    if payload.get("event") not in names:
        raise OrderError("unsupported Alpaca paper lifecycle fixture")
    event = {
        "event_id": payload["event_id"],
        "client_id": expected["client_id"],
        "kind": names[payload["event"]],
        "occurred_at": payload["timestamp"],
    }
    if event["kind"] == "FILL":
        if "qty" not in payload or "price" not in payload:
            raise OrderError("fill fixture needs incremental execution quantity and price")
        event.update(quantity=payload["qty"], price=payload["price"], fee=payload.get("fee", "0"))
    return event_value(event)
