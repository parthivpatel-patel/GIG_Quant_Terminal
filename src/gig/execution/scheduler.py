"""Explicit-clock, crash-atomic offline TWAP admission into the existing outbox.

Advancement reserves due child intents. It never dispatches, acknowledges or
fills them. The parent deadline expires unscheduled children; missed deadlines
do not cause automatic catch-up execution.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from datetime import datetime
from decimal import Decimal
from typing import Any

from gig.execution.oms import OfflineOMS, OrderError, _decimal, _exact, _text
from gig.execution.outbox import ExecutionOutbox
from gig.execution.twap import plan_twap
from gig.timestamps import aware_timestamp

MAX_CHILDREN = 100


def _stamp(value: Any) -> datetime:
    stamp = aware_timestamp(value)
    if stamp.nanosecond:
        raise OrderError("scheduler timestamps require microsecond precision")
    return stamp.to_pydatetime()


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


class DurableTWAPScheduler:
    """One strategy's bounded scheduler in the OMS transaction domain."""

    def __init__(
        self,
        oms: OfflineOMS,
        outbox: ExecutionOutbox,
        *,
        fault_hook: Callable[[str], None] | None = None,
        create: bool = True,
    ) -> None:
        if outbox.oms is not oms:
            raise OrderError("scheduler and outbox must share one OMS transaction domain")
        self.oms, self.outbox, self.db = oms, outbox, oms.connection
        self.hook = fault_hook or (lambda _: None)
        self.enabled = bool(self.db.execute(
            "SELECT 1 FROM sqlite_master WHERE name='twap_schedules'"
        ).fetchone())
        if not create:
            return
        # Individual DDL statements preserve a caller's enclosing transaction.
        for statement in (
            "CREATE TABLE IF NOT EXISTS twap_schedules (client_id TEXT PRIMARY KEY, intent_hash TEXT NOT NULL, payload TEXT NOT NULL, created_at TEXT NOT NULL, state TEXT NOT NULL)",
            "CREATE TABLE IF NOT EXISTS twap_children (client_id TEXT PRIMARY KEY, parent_id TEXT NOT NULL REFERENCES twap_schedules(client_id), ordinal INTEGER NOT NULL, due_at TEXT NOT NULL, quantity TEXT NOT NULL, state TEXT NOT NULL, admitted_at TEXT, UNIQUE(parent_id,ordinal))",
            "CREATE TABLE IF NOT EXISTS twap_clock (singleton INTEGER PRIMARY KEY CHECK(singleton=1), timestamp TEXT NOT NULL)",
            "CREATE TABLE IF NOT EXISTS twap_advances (timestamp TEXT PRIMARY KEY, request_hash TEXT NOT NULL, result_json TEXT NOT NULL, result_hash TEXT NOT NULL)",
            "CREATE TABLE IF NOT EXISTS twap_cancellations (client_id TEXT PRIMARY KEY REFERENCES twap_schedules(client_id), result_json TEXT NOT NULL, result_hash TEXT NOT NULL)",
        ):
            self.db.execute(statement)
        self.enabled = True

    @property
    def clock(self) -> str | None:
        if not self.enabled:
            return None
        row = self.db.execute("SELECT timestamp FROM twap_clock WHERE singleton=1").fetchone()
        return None if row is None else _stamp(row[0]).isoformat()

    def ensure_not_before(self, now: datetime) -> None:
        at = _stamp(now)
        if self.clock is not None and at < _stamp(self.clock):
            raise OrderError("operation precedes durable simulation clock")
        # Existing OMS events may have advanced simulated evidence since the
        # last explicit scheduler operation. Every operation in this workspace,
        # including explicit prepare/dispatch/fill, must respect that evidence.
        for row in self.db.execute("SELECT timestamp FROM events"):
            if at < _stamp(row[0]):
                raise OrderError("simulation clock precedes existing execution evidence")

    def _check_clock(self, at: datetime) -> None:
        self.ensure_not_before(at)

    def _set_clock(self, at: datetime) -> None:
        self.db.execute(
            "INSERT INTO twap_clock VALUES(1,?) ON CONFLICT(singleton) DO UPDATE SET timestamp=excluded.timestamp",
            (at.isoformat(),),
        )

    @staticmethod
    def _intent(values: dict[str, Any]) -> dict[str, Any]:
        required = {"client_id", "symbol", "side", "quantity", "start", "end", "slices",
                    "max_notional", "max_position_quantity"}
        if not isinstance(values, dict) or not required <= values.keys() or set(values) - (
            required | {"lot_size", "limit_price"}
        ):
            raise OrderError("unsupported or missing schedule fields")
        client = _text(values["client_id"], "client_id")
        symbol = _text(values["symbol"], "symbol")
        if not isinstance(values["side"], str) or values["side"] not in {"buy", "sell"}:
            raise OrderError("unsupported scheduled side")
        slices = values["slices"]
        if isinstance(slices, bool) or not isinstance(slices, int) or not 1 <= slices <= MAX_CHILDREN:
            raise OrderError("schedule supports one to 100 slices")
        quantity = _decimal(values["quantity"])
        cap = _decimal(values["max_position_quantity"])
        if quantity > cap:
            raise OrderError("parent quantity exceeds its position quantity cap")
        start, end = _stamp(values["start"]), _stamp(values["end"])
        result = {
            "client_id": client, "symbol": symbol, "side": values["side"],
            "quantity": str(quantity), "start": start.isoformat(), "end": end.isoformat(),
            "slices": slices, "lot_size": str(_decimal(values.get("lot_size", "1"))),
            "max_notional": str(_decimal(values["max_notional"])),
            "max_position_quantity": str(cap),
            "limit_price": None if values.get("limit_price") is None else str(_decimal(values["limit_price"])),
            "deadline_policy": "EXPIRE_UNADMITTED_AT_END",
        }
        plan_twap(client, str(quantity), start, end, slices, lot_size=result["lot_size"])
        return result

    @staticmethod
    def _children(intent: dict[str, Any]):
        return plan_twap(
            intent["client_id"], intent["quantity"], _stamp(intent["start"]),
            _stamp(intent["end"]), intent["slices"], lot_size=intent["lot_size"],
        )

    def _load(self, client_id: str) -> tuple[dict[str, Any], Any, list[Any]]:
        row = self.db.execute("SELECT * FROM twap_schedules WHERE client_id=?", (client_id,)).fetchone()
        if row is None:
            raise OrderError("unknown durable schedule")
        try:
            intent = json.loads(row["payload"])
            if _digest(intent) != row["intent_hash"] or intent["client_id"] != client_id:
                raise OrderError("schedule intent integrity failure")
            fields = {key: value for key, value in intent.items() if key != "deadline_policy"}
            if self._intent(fields) != intent:
                raise OrderError("invalid durable schedule intent")
            if row["state"] not in {"ACTIVE", "CANCELLED", "COMPLETE"}:
                raise OrderError("invalid durable schedule state")
            _stamp(row["created_at"])
            children = self.db.execute(
                "SELECT * FROM twap_children WHERE parent_id=? ORDER BY ordinal", (client_id,)
            ).fetchall()
            plan = self._children(intent)
            if len(children) != len(plan):
                raise OrderError("schedule child count integrity failure")
            for i, (child, expected) in enumerate(zip(children, plan, strict=True)):
                if (child["client_id"], child["ordinal"], child["due_at"], child["quantity"]) != (
                    expected.client_id, i + 1, expected.timestamp.isoformat(), str(expected.quantity)
                ) or child["state"] not in {"PENDING", "ADMITTED", "CANCELLED", "EXPIRED"}:
                    raise OrderError("schedule child integrity failure")
                if (child["state"] == "ADMITTED") != (child["admitted_at"] is not None):
                    raise OrderError("schedule admission timestamp integrity failure")
                if child["state"] == "ADMITTED":
                    order = self.oms.order(child["client_id"])
                    if (order["symbol"], order["side"], order["quantity"], order["created_at"], order["limit_price"]) != (
                        intent["symbol"], intent["side"], child["quantity"], child["admitted_at"], intent["limit_price"]
                    ):
                        raise OrderError("scheduled order differs from parent intent")
            if row["state"] in {"COMPLETE", "CANCELLED"} and any(c["state"] == "PENDING" for c in children):
                raise OrderError("terminal schedule contains pending children")
            return intent, row, children
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            raise OrderError("durable schedule integrity failure") from None

    def schedules(self) -> list[dict[str, Any]]:
        if not self.enabled:
            return []
        ids = [r[0] for r in self.db.execute("SELECT client_id FROM twap_schedules ORDER BY created_at,client_id")]
        if len(ids) > MAX_CHILDREN:
            raise OrderError("schedule archive exceeds workspace bound")
        result = []
        for client in ids:
            intent, row, children = self._load(client)
            result.append({
                **intent, "created_at": row["created_at"], "state": row["state"],
                "intent_sha256": row["intent_hash"], "classification": "SIMULATED_PLAN",
                "children": [
                    {"client_id": c["client_id"], "due_at": c["due_at"], "quantity": c["quantity"],
                     "state": c["state"], "admitted_at": c["admitted_at"],
                     "order_state": self.oms.order(c["client_id"])["state"] if c["state"] == "ADMITTED" else None}
                    for c in children
                ],
            })
        return result

    def schedule(self, values: dict[str, Any], *, now: datetime) -> dict[str, Any]:
        intent, at = self._intent(values), _stamp(now)
        with self.oms._transaction():
            self._check_clock(at)
            prior = self.db.execute(
                "SELECT intent_hash FROM twap_schedules WHERE client_id=?", (intent["client_id"],)
            ).fetchone()
            if prior:
                if prior[0] != _digest(intent):
                    raise OrderError("parent client_id conflicts with durable schedule")
                self._load(intent["client_id"])
                return {"state": "AVAILABLE", "client_id": intent["client_id"], "replayed": True,
                        "intent_sha256": prior[0], "simulation_clock": self.clock}
            if _stamp(intent["start"]) < at:
                raise OrderError("schedule start precedes admission clock")
            plan = self._children(intent)
            count = self.db.execute("SELECT COUNT(*) FROM twap_children").fetchone()[0]
            if count + len(plan) > MAX_CHILDREN:
                raise OrderError("workspace retains at most 100 scheduled children")
            for child in plan:
                if self.db.execute("SELECT 1 FROM orders WHERE client_id=?", (child.client_id,)).fetchone():
                    raise OrderError("scheduled child conflicts with existing order identity")
            self.db.execute("INSERT INTO twap_schedules VALUES(?,?,?,?, 'ACTIVE')",
                            (intent["client_id"], _digest(intent), _canonical(intent), at.isoformat()))
            for i, child in enumerate(plan):
                self.db.execute("INSERT INTO twap_children VALUES(?,?,?,?,?,'PENDING',NULL)",
                                (child.client_id, intent["client_id"], i + 1,
                                 child.timestamp.isoformat(), str(child.quantity)))
            self._set_clock(at)
            self.oms._event(None, "SCHEDULED", at, "offline TWAP parent admitted; no orders submitted")
        return {"state": "AVAILABLE", "client_id": intent["client_id"], "replayed": False,
                "intent_sha256": _digest(intent), "simulation_clock": at.isoformat()}

    @staticmethod
    def _marks(values: Any) -> dict[str, dict[str, str]]:
        if not isinstance(values, dict) or len(values) > MAX_CHILDREN:
            raise OrderError("advance requires a bounded explicit marks mapping")
        result = {}
        for symbol, mark in values.items():
            _text(symbol, "symbol")
            if not isinstance(mark, dict) or set(mark) != {"reference_price", "price_asof"}:
                raise OrderError("each advance mark requires price and source timestamp")
            result[symbol] = {"reference_price": str(_decimal(mark["reference_price"])),
                              "price_asof": _stamp(mark["price_asof"]).isoformat()}
        return result

    @_exact
    def _parent_risk(self, intent: dict[str, Any], children: list[Any], mark: dict[str, str]) -> None:
        filled = Decimal(0)
        notional = Decimal(0)
        for child in children:
            for execution in self.db.execute(
                "SELECT quantity,price FROM executions WHERE client_id=?", (child["client_id"],)
            ):
                quantity, price = Decimal(execution[0]), Decimal(execution[1])
                filled += quantity
                notional += quantity * price
        remaining = Decimal(intent["quantity"]) - filled
        price = Decimal(mark["reference_price"])
        if intent["limit_price"] is not None:
            price = max(price, Decimal(intent["limit_price"]))
        if remaining < 0 or notional + remaining * price > Decimal(intent["max_notional"]):
            raise OrderError("parent executed plus remaining notional exceeds cap")

    def advance(self, marks: dict[str, Any], *, now: datetime) -> dict[str, Any]:
        at = _stamp(now)
        quotes = self._marks(marks)
        request_hash = _digest({"now": at.isoformat(), "marks": quotes})
        with self.oms._transaction():
            self._check_clock(at)
            previous = self.db.execute("SELECT request_hash,result_json,result_hash FROM twap_advances WHERE timestamp=?",
                                       (at.isoformat(),)).fetchone()
            if previous:
                if previous[0] != request_hash:
                    raise OrderError("clock timestamp conflicts with preserved advance intent")
                result = json.loads(previous[1])
                if _digest(result) != previous[2]:
                    raise OrderError("clock advancement result integrity failure")
                return result
            self.schedules()  # Validate all parent/child evidence before mutation.
            ids = [r[0] for r in self.db.execute("SELECT client_id FROM twap_schedules WHERE state='ACTIVE' ORDER BY client_id")]
            due: list[tuple[dict[str, Any], Any]] = []
            expired: list[str] = []
            all_symbols = set()
            for client in ids:
                intent, _, children = self._load(client)
                all_symbols.add(intent["symbol"])
                pending = [child for child in children if child["state"] == "PENDING"]
                if at >= _stamp(intent["end"]):
                    expired.extend(child["client_id"] for child in pending)
                else:
                    selected = [child for child in pending if _stamp(child["due_at"]) <= at]
                    if selected:
                        if intent["symbol"] not in quotes:
                            raise OrderError("due scheduled children require fresh explicit marks")
                        self._parent_risk(intent, children, quotes[intent["symbol"]])
                        due.extend((intent, child) for child in selected)
            if set(quotes) - all_symbols:
                raise OrderError("advance mark has no active scheduled instrument")
            if due and self.oms.kill_switch:
                raise OrderError("kill switch blocks scheduled child admission")
            count = self.db.execute("SELECT COUNT(*) FROM orders").fetchone()[0]
            if count + len(due) > MAX_CHILDREN:
                raise OrderError("workspace retains at most 100 orders")
            for child_id in expired:
                self.db.execute("UPDATE twap_children SET state='EXPIRED' WHERE client_id=?", (child_id,))
            admitted = []
            for intent, child in due:
                mark = quotes[intent["symbol"]]
                if self.db.execute("SELECT 1 FROM orders WHERE client_id=?", (child["client_id"],)).fetchone():
                    raise OrderError("pending schedule child conflicts with existing order")
                self.hook("before_child_admission")
                self.outbox.admit_order(
                    child["client_id"], intent["symbol"], intent["side"], child["quantity"],
                    reference_price=mark["reference_price"], price_asof=_stamp(mark["price_asof"]),
                    max_notional=intent["max_notional"], max_position_quantity=intent["max_position_quantity"],
                    limit_price=intent["limit_price"], now=at,
                )
                self.hook("after_child_admission")
                self.db.execute("UPDATE twap_children SET state='ADMITTED',admitted_at=? WHERE client_id=?",
                                (at.isoformat(), child["client_id"]))
                admitted.append(child["client_id"])
            self.db.execute("UPDATE twap_schedules SET state='COMPLETE' WHERE state='ACTIVE' AND NOT EXISTS (SELECT 1 FROM twap_children WHERE parent_id=twap_schedules.client_id AND state='PENDING')")
            self._set_clock(at)
            result = {"state": "AVAILABLE", "simulation_clock": at.isoformat(),
                      "admitted_children": admitted, "expired_children": expired,
                      "automatically_dispatched": 0, "classification": "SIMULATED"}
            self.db.execute("INSERT INTO twap_advances VALUES(?,?,?,?)",
                            (at.isoformat(), request_hash, _canonical(result), _digest(result)))
            self.oms._event(None, "CLOCK_ADVANCED", at, "explicit offline clock; no transport or fills")
            self.hook("before_advance_commit")
        return result

    def cancel(self, client_id: str, *, now: datetime) -> dict[str, Any]:
        at, client_id = _stamp(now), _text(client_id, "client_id")
        with self.oms._transaction():
            self._check_clock(at)
            self._load(client_id)
            previous = self.db.execute("SELECT result_json,result_hash FROM twap_cancellations WHERE client_id=?",
                                       (client_id,)).fetchone()
            if previous:
                result = json.loads(previous[0])
                if _digest(result) != previous[1]:
                    raise OrderError("schedule cancellation result integrity failure")
                return result
            count = self.db.execute("SELECT COUNT(*) FROM twap_children WHERE parent_id=? AND state='PENDING'",
                                    (client_id,)).fetchone()[0]
            self.db.execute("UPDATE twap_children SET state='CANCELLED' WHERE parent_id=? AND state='PENDING'",
                            (client_id,))
            self.db.execute("UPDATE twap_schedules SET state='CANCELLED' WHERE client_id=?", (client_id,))
            self._set_clock(at)
            self.oms._event(None, "SCHEDULE_CANCELLED", at, "pending children cancelled; existing orders unchanged")
            result = {"state": "AVAILABLE", "client_id": client_id, "pending_children_cancelled": count,
                      "existing_orders_cancelled": 0, "simulation_clock": at.isoformat()}
            self.db.execute("INSERT INTO twap_cancellations VALUES(?,?,?)",
                            (client_id, _canonical(result), _digest(result)))
        return result
