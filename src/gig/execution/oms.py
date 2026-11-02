"""Crash-safe offline order ledger. This module never constructs a broker client.

Quantities, prices, fees, and positions are exact decimal strings in SQLite.
All state transitions and fill/position updates commit together. A recovered
working order requires reconciliation; recovery never resubmits an order.
This is an offline simulator, not broker-observed execution evidence.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from decimal import Context, Decimal, Inexact, InvalidOperation, localcontext
from functools import wraps
from pathlib import Path


class OrderError(ValueError):
    """An invalid request or forbidden state transition."""


TERMINAL = frozenset({"FILLED", "CANCELLED", "REJECTED", "EXPIRED"})
WORKING = frozenset({"ACKNOWLEDGED", "PARTIALLY_FILLED"})


def _exact(function):
    """Keep monetary operations independent of the process decimal context."""

    @wraps(function)
    def invoke(*args, **kwargs):
        context = Context(prec=100)
        context.traps[Inexact] = True
        with localcontext(context):
            return function(*args, **kwargs)

    return invoke


def _decimal(value: str | int | Decimal, *, positive: bool = True) -> Decimal:
    if isinstance(value, (float, bool)):
        raise OrderError("use an exact decimal string, integer, or Decimal")
    if (isinstance(value, str) and len(value) > 128) or (
        isinstance(value, int) and value.bit_length() > 256
    ):
        raise OrderError("decimal exceeds admission size bound")
    try:
        number = Decimal(value)
    except (InvalidOperation, TypeError, ValueError):
        raise OrderError("invalid decimal") from None
    if not number.is_finite() or (number <= 0 if positive else number < 0):
        raise OrderError(
            "decimal must be finite and positive" if positive else "decimal must be nonnegative"
        )
    decimal_tuple = number.as_tuple()
    exponent = decimal_tuple.exponent
    if not isinstance(exponent, int):
        raise OrderError("decimal must have a finite numeric exponent")
    if len(decimal_tuple.digits) > 24 or abs(exponent) > 12:
        raise OrderError("decimal exceeds supported precision")
    return number


def _time(value: datetime) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise OrderError("timezone-aware timestamp required")
    if getattr(value, "nanosecond", 0):
        raise OrderError("offline execution timestamps require microsecond precision")
    return value.astimezone(UTC)


def _text(value: str, name: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or len(value) > 128
        or any(ord(c) < 32 or 0xD800 <= ord(c) <= 0xDFFF for c in value)
    ):
        raise OrderError(f"invalid {name}")
    return value


class OfflineOMS:
    """SQLite OMS with explicit risk approval and deterministic simulation fills.

    Scope is one local strategy account, initially flat. Broker reconciliation
    compares external positions but never imports or overwrites them silently.
    Callers must close connections (or use the context manager) at shutdown.
    """

    def __init__(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path, timeout=10, isolation_level=None)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA foreign_keys=ON")
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA synchronous=FULL")
        self.connection.executescript("""
            CREATE TABLE IF NOT EXISTS orders (
                client_id TEXT PRIMARY KEY, symbol TEXT NOT NULL, side TEXT NOT NULL,
                quantity TEXT NOT NULL, limit_price TEXT, state TEXT NOT NULL,
                filled TEXT NOT NULL DEFAULT '0', created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL, risk_json TEXT
            );
            CREATE TABLE IF NOT EXISTS executions (
                execution_id TEXT PRIMARY KEY, client_id TEXT NOT NULL REFERENCES orders(client_id),
                quantity TEXT NOT NULL, price TEXT NOT NULL, fee TEXT NOT NULL,
                timestamp TEXT NOT NULL, classification TEXT NOT NULL DEFAULT 'SIMULATED'
            );
            CREATE TABLE IF NOT EXISTS positions (symbol TEXT PRIMARY KEY, quantity TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS events (
                sequence INTEGER PRIMARY KEY AUTOINCREMENT, client_id TEXT,
                state TEXT NOT NULL, timestamp TEXT NOT NULL, detail TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS controls (name TEXT PRIMARY KEY, value TEXT NOT NULL);
            INSERT OR IGNORE INTO controls VALUES ('kill_switch','false');
            CREATE TABLE IF NOT EXISTS simulation_runs (
                run_id TEXT PRIMARY KEY, intent_sha256 TEXT NOT NULL,
                state TEXT NOT NULL, result_json TEXT
            );
        """)

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> OfflineOMS:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    @contextmanager
    def _transaction(self) -> Iterator[None]:
        nested = self.connection.in_transaction
        counter = getattr(self, "_savepoint_counter", 0) + 1
        self._savepoint_counter = counter
        savepoint = f"oms_{counter}"
        self.connection.execute(f"SAVEPOINT {savepoint}" if nested else "BEGIN IMMEDIATE")
        try:
            yield
        except BaseException:
            if nested:
                self.connection.execute(f"ROLLBACK TO SAVEPOINT {savepoint}")
                self.connection.execute(f"RELEASE SAVEPOINT {savepoint}")
            else:
                self.connection.rollback()
            raise
        else:
            if nested:
                self.connection.execute(f"RELEASE SAVEPOINT {savepoint}")
            else:
                self.connection.commit()

    def _event(self, client_id: str | None, state: str, at: datetime, detail: str) -> None:
        self.connection.execute(
            "INSERT INTO events(client_id,state,timestamp,detail) VALUES(?,?,?,?)",
            (client_id, state, at.isoformat(), detail),
        )

    def order(self, client_id: str) -> dict:
        row = self.connection.execute(
            "SELECT * FROM orders WHERE client_id=?", (client_id,)
        ).fetchone()
        if row is None:
            raise OrderError("unknown order")
        result = dict(row)
        result["classification"] = "SIMULATED"
        risk_json = result.pop("risk_json")
        result["risk"] = json.loads(risk_json) if risk_json else None
        return result

    @_exact
    def create(
        self,
        client_id: str,
        symbol: str,
        side: str,
        quantity: str | int | Decimal,
        *,
        limit_price: str | int | Decimal | None = None,
        now: datetime | None = None,
    ) -> dict:
        """Idempotent by client_id; conflicting retries fail without mutation."""
        _text(client_id, "client_id")
        _text(symbol, "symbol")
        if side not in {"buy", "sell"}:
            raise OrderError("side must be buy or sell")
        qty = _decimal(quantity)
        limit = _decimal(limit_price) if limit_price is not None else None
        at = _time(now or datetime.now(UTC))
        with self._transaction():
            row = self.connection.execute(
                "SELECT * FROM orders WHERE client_id=?", (client_id,)
            ).fetchone()
            if row:
                if (
                    row["symbol"] != symbol
                    or row["side"] != side
                    or Decimal(row["quantity"]) != qty
                    or (Decimal(row["limit_price"]) if row["limit_price"] else None) != limit
                ):
                    raise OrderError("client_id conflicts with existing intent")
            else:
                self.connection.execute(
                    "INSERT INTO orders(client_id,symbol,side,quantity,limit_price,state,created_at,updated_at) VALUES(?,?,?,?,?,'CREATED',?,?)",
                    (
                        client_id,
                        symbol,
                        side,
                        str(qty),
                        str(limit) if limit else None,
                        at.isoformat(),
                        at.isoformat(),
                    ),
                )
                self._event(client_id, "CREATED", at, "offline intent admitted")
        return self.order(client_id)

    @_exact
    def approve(
        self,
        client_id: str,
        *,
        reference_price: str | int | Decimal,
        price_asof: datetime,
        max_notional: str | int | Decimal,
        max_position_quantity: str | int | Decimal,
        max_age_seconds: int = 60,
        now: datetime | None = None,
    ) -> dict:
        """Admit bounded offline risk evidence; this grants no broker authority."""
        at, asof = _time(now or datetime.now(UTC)), _time(price_asof)
        if (
            isinstance(max_age_seconds, bool)
            or not isinstance(max_age_seconds, int)
            or not 0 < max_age_seconds <= 86400
        ):
            raise OrderError("invalid freshness bound")
        price, notional, cap = (
            _decimal(reference_price),
            _decimal(max_notional),
            _decimal(max_position_quantity),
        )
        if not timedelta(0) <= at - asof <= timedelta(seconds=max_age_seconds):
            raise OrderError("reference price is stale or in the future")
        risk = {
            "reference_price": str(price),
            "price_asof": asof.isoformat(),
            "max_notional": str(notional),
            "max_position_quantity": str(cap),
            "max_age_seconds": max_age_seconds,
            "classification": "SIMULATED",
        }
        with self._transaction():
            order = self.order(client_id)
            if order["state"] != "CREATED":
                raise OrderError("risk approval requires CREATED state")
            if at < datetime.fromisoformat(order["updated_at"]):
                raise OrderError("risk approval timestamp precedes order state")
            qty = Decimal(order["quantity"])
            # A buy limit above reference must fit the notional cap too.
            bound = max(price, Decimal(order["limit_price"])) if order["limit_price"] else price
            if qty * bound > notional or qty > cap:
                raise OrderError("order exceeds risk bounds")
            self.connection.execute(
                "UPDATE orders SET state='APPROVED',risk_json=?,updated_at=? WHERE client_id=?",
                (json.dumps(risk, sort_keys=True), at.isoformat(), client_id),
            )
            self._event(client_id, "APPROVED", at, "offline risk bounds accepted")
        return self.order(client_id)

    @_exact
    def _check_submission(self, order: dict, at: datetime) -> None:
        if self.kill_switch:
            raise OrderError("kill switch is engaged")
        risk = order["risk"]
        age = at - datetime.fromisoformat(risk["price_asof"])
        if not timedelta(0) <= age <= timedelta(seconds=risk["max_age_seconds"]):
            raise OrderError("risk price is stale or in the future")
        positions = self.positions()
        # Bound either fill ordering. Opposing orders never offset their
        # risk reservations; an exit from a position is still permitted.
        pending = self.connection.execute(
            "SELECT side,quantity,filled FROM orders WHERE symbol=? AND state IN ('ACKNOWLEDGED','PARTIALLY_FILLED','SUBMITTING','UNKNOWN') AND client_id != ?",
            (order["symbol"], order["client_id"]),
        ).fetchall()
        buys = sum(
            (
                Decimal(row["quantity"]) - Decimal(row["filled"])
                for row in pending
                if row["side"] == "buy"
            ),
            Decimal(0),
        )
        sells = sum(
            (
                Decimal(row["quantity"]) - Decimal(row["filled"])
                for row in pending
                if row["side"] == "sell"
            ),
            Decimal(0),
        )
        position = Decimal(positions.get(order["symbol"], "0"))
        buys += Decimal(order["quantity"]) if order["side"] == "buy" else 0
        sells += Decimal(order["quantity"]) if order["side"] == "sell" else 0
        if max(abs(position + buys), abs(position - sells)) > Decimal(
            risk["max_position_quantity"]
        ):
            raise OrderError("position plus working orders exceeds quantity cap")

    @_exact
    def reserve_submission(self, client_id: str, *, now: datetime) -> dict:
        """Reserve risk before transport, without inventing an acknowledgement."""
        at = _time(now)
        with self._transaction():
            order = self.order(client_id)
            if order["state"] != "APPROVED":
                raise OrderError("reservation requires APPROVED state")
            if at < datetime.fromisoformat(order["updated_at"]):
                raise OrderError("reservation timestamp precedes order state")
            self._check_submission(order, at)
            self.connection.execute(
                "UPDATE orders SET state='SUBMITTING',updated_at=? WHERE client_id=?",
                (at.isoformat(), client_id),
            )
            self._event(
                client_id, "SUBMITTING", at, "offline outbox risk reservation; no acknowledgement"
            )
        return self.order(client_id)

    def submission_unknown(self, client_id: str, *, now: datetime) -> None:
        at = _time(now)
        with self._transaction():
            order = self.order(client_id)
            if order["state"] in {"SUBMITTING", "UNKNOWN"}:
                if at < datetime.fromisoformat(order["updated_at"]):
                    raise OrderError("unknown timestamp precedes order state")
                self.connection.execute(
                    "UPDATE orders SET state='UNKNOWN',updated_at=? WHERE client_id=?",
                    (at.isoformat(), client_id),
                )
                self._event(
                    client_id, "UNKNOWN", at, "submission outcome requires explicit reconciliation"
                )

    @_exact
    def acknowledge(self, client_id: str, *, now: datetime | None = None) -> dict:
        """Local simulation submission; atomic approval/freshness/exposure check."""
        at = _time(now or datetime.now(UTC))
        with self._transaction():
            order = self.order(client_id)
            if order["state"] in WORKING | {"FILLED"}:
                return order
            if order["state"] not in {"APPROVED", "SUBMITTING", "UNKNOWN"}:
                raise OrderError("submission requires approved or reserved state")
            if at < datetime.fromisoformat(order["updated_at"]):
                raise OrderError("submission timestamp precedes order state")
            if order["state"] == "APPROVED":
                self._check_submission(order, at)
            self.connection.execute(
                "UPDATE orders SET state='ACKNOWLEDGED',updated_at=? WHERE client_id=?",
                (at.isoformat(), client_id),
            )
            self._event(client_id, "ACKNOWLEDGED", at, "local simulator acknowledgement")
        return self.order(client_id)

    @_exact
    def fill(
        self,
        client_id: str,
        execution_id: str,
        quantity: str | int | Decimal,
        price: str | int | Decimal,
        *,
        fee: str | int | Decimal = "0",
        now: datetime | None = None,
        occurred_at: datetime | None = None,
        reconciled_terminal: bool = False,
    ) -> dict:
        """Record one simulated execution atomically; duplicates never double count."""
        _text(execution_id, "execution_id")
        qty, px, cost = _decimal(quantity), _decimal(price), _decimal(fee, positive=False)
        at = _time(now or datetime.now(UTC))
        execution_at = _time(occurred_at) if occurred_at is not None else at
        if execution_at > at:
            raise OrderError("execution occurrence cannot be in the future")
        with self._transaction():
            old = self.connection.execute(
                "SELECT * FROM executions WHERE execution_id=?", (execution_id,)
            ).fetchone()
            if old:
                if (
                    old["client_id"] != client_id
                    or Decimal(old["quantity"]) != qty
                    or Decimal(old["price"]) != px
                    or Decimal(old["fee"]) != cost
                    or datetime.fromisoformat(old["timestamp"]) != execution_at
                ):
                    raise OrderError("execution_id conflicts with existing execution")
                return self.order(client_id)
            order = self.order(client_id)
            prior_terminal = order["state"] if order["state"] in {"CANCELLED", "EXPIRED"} else None
            if order["state"] not in WORKING and not (reconciled_terminal and prior_terminal):
                raise OrderError("fill requires a working order")
            if execution_at < datetime.fromisoformat(order["created_at"]):
                raise OrderError("execution occurrence precedes order creation")
            if at < datetime.fromisoformat(order["updated_at"]):
                raise OrderError("execution timestamp precedes order state")
            filled, total = Decimal(order["filled"]) + qty, Decimal(order["quantity"])
            if filled > total:
                raise OrderError("execution exceeds remaining quantity")
            limit = Decimal(order["limit_price"]) if order["limit_price"] else None
            if limit and (
                (order["side"] == "buy" and px > limit) or (order["side"] == "sell" and px < limit)
            ):
                raise OrderError("execution violates limit price")
            executed = sum(
                (
                    Decimal(row["quantity"]) * Decimal(row["price"]) + Decimal(row["fee"])
                    for row in self.connection.execute(
                        "SELECT quantity,price,fee FROM executions WHERE client_id=?", (client_id,)
                    )
                ),
                Decimal(0),
            )
            if executed + qty * px + cost > Decimal(order["risk"]["max_notional"]):
                raise OrderError("execution exceeds approved notional")
            self.connection.execute(
                "INSERT INTO executions(execution_id,client_id,quantity,price,fee,timestamp) VALUES(?,?,?,?,?,?)",
                (execution_id, client_id, str(qty), str(px), str(cost), execution_at.isoformat()),
            )
            prior = self.connection.execute(
                "SELECT quantity FROM positions WHERE symbol=?", (order["symbol"],)
            ).fetchone()
            position = Decimal(prior[0]) if prior else Decimal(0)
            position += qty if order["side"] == "buy" else -qty
            self.connection.execute(
                "INSERT INTO positions VALUES(?,?) ON CONFLICT(symbol) DO UPDATE SET quantity=excluded.quantity",
                (order["symbol"], str(position)),
            )
            state = "FILLED" if filled == total else (prior_terminal or "PARTIALLY_FILLED")
            self.connection.execute(
                "UPDATE orders SET state=?,filled=?,updated_at=? WHERE client_id=?",
                (state, str(filled), at.isoformat(), client_id),
            )
            self._event(client_id, state, at, f"simulated execution {execution_id}")
        return self.order(client_id)

    def terminate(self, client_id: str, state: str, *, now: datetime | None = None) -> dict:
        if state not in {"CANCELLED", "REJECTED", "EXPIRED"}:
            raise OrderError("invalid terminal state")
        at = _time(now or datetime.now(UTC))
        with self._transaction():
            order = self.order(client_id)
            if order["state"] == state:
                return order
            if order["state"] in TERMINAL:
                raise OrderError("terminal order cannot transition")
            if at < datetime.fromisoformat(order["updated_at"]):
                raise OrderError("transition timestamp precedes order state")
            self.connection.execute(
                "UPDATE orders SET state=?,updated_at=? WHERE client_id=?",
                (state, at.isoformat(), client_id),
            )
            self._event(client_id, state, at, "local simulator terminal transition")
        return self.order(client_id)

    @property
    def kill_switch(self) -> bool:
        return (
            self.connection.execute(
                "SELECT value FROM controls WHERE name='kill_switch'"
            ).fetchone()[0]
            == "true"
        )

    def set_kill_switch(self, engaged: bool, *, now: datetime | None = None) -> None:
        if not isinstance(engaged, bool):
            raise OrderError("kill switch requires a boolean")
        at = _time(now or datetime.now(UTC))
        with self._transaction():
            self.connection.execute(
                "UPDATE controls SET value=? WHERE name='kill_switch'",
                ("true" if engaged else "false",),
            )
            self._event(None, "KILL_SWITCH", at, "engaged" if engaged else "released")

    def positions(self) -> dict[str, str]:
        return {
            row["symbol"]: row["quantity"]
            for row in self.connection.execute("SELECT * FROM positions ORDER BY symbol")
        }

    @_exact
    def reconcile(self, observed: dict[str, str | int | Decimal]) -> dict:
        """Compare quantities without mutating local evidence or submitting orders."""
        actual = {}
        for symbol, quantity in observed.items():
            _text(symbol, "symbol")
            # Position quantities are signed; exact decimal validation applies.
            sign = -1 if str(quantity).startswith("-") else 1
            if isinstance(quantity, (bool, float)):
                raise OrderError("positions require exact quantities")
            try:
                parsed = Decimal(quantity)
            except (InvalidOperation, TypeError, ValueError):
                raise OrderError("invalid observed position") from None
            actual[symbol] = sign * _decimal(abs(parsed), positive=False)
        local = {symbol: Decimal(qty) for symbol, qty in self.positions().items()}
        differences = {
            symbol: {
                "local": str(local.get(symbol, Decimal(0))),
                "observed": str(actual.get(symbol, Decimal(0))),
            }
            for symbol in sorted(local.keys() | actual.keys())
            if local.get(symbol, Decimal(0)) != actual.get(symbol, Decimal(0))
        }
        return {
            "state": "ERROR" if differences else "AVAILABLE",
            "matched": not differences,
            "differences": differences,
            "classification": "SIMULATED_COMPARISON",
            "broker_connected": False,
        }

    def recover(self) -> dict:
        rows = self.connection.execute(
            "SELECT client_id FROM orders WHERE state IN ('ACKNOWLEDGED','PARTIALLY_FILLED','SUBMITTING','UNKNOWN') ORDER BY client_id"
        ).fetchall()
        pending_runs = [
            row[0]
            for row in self.connection.execute(
                "SELECT run_id FROM simulation_runs WHERE state='RUNNING' ORDER BY run_id"
            )
        ]
        return {
            "working_orders": [self.order(row[0]) for row in rows],
            "positions": self.positions(),
            "requires_reconciliation": bool(rows or pending_runs),
            "auto_resubmitted": 0,
            "incomplete_simulations": pending_runs,
            "kill_switch": self.kill_switch,
            "classification": "SIMULATED",
        }

    def events(self) -> list[dict]:
        return [
            dict(row) for row in self.connection.execute("SELECT * FROM events ORDER BY sequence")
        ]
