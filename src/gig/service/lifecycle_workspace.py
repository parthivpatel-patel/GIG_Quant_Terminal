"""Immutable equity lifecycle evidence and same-OMS cash/position transitions."""
from __future__ import annotations

import hashlib
from datetime import timedelta
from decimal import Decimal
from pathlib import Path

from gig.artifacts import write_immutable
from gig.execution.oms import TERMINAL, OfflineOMS, _decimal, _exact, _text
from gig.portfolio.lifecycle import dividend_entitlement
from gig.service.execution_workspace import _stamp
from gig.service.funded_workspace import FundedWorkspace
from gig.service.serialization import dumps, loads
from gig.service.workspace import _fields
from gig.timestamps import aware_timestamp


def _hash(value):
    return hashlib.sha256(dumps(value).encode()).hexdigest()


def lifecycle_schema(oms):
    for sql in (
        "CREATE TABLE IF NOT EXISTS funded_lifecycle_policy(singleton INTEGER PRIMARY KEY CHECK(singleton=1),wire TEXT NOT NULL)",
        "CREATE TABLE IF NOT EXISTS funded_lifecycle_actions(digest TEXT PRIMARY KEY,action_id TEXT NOT NULL,revision INTEGER NOT NULL,wire TEXT NOT NULL,UNIQUE(action_id,revision))",
        "CREATE TABLE IF NOT EXISTS funded_lifecycle_events(event_id TEXT PRIMARY KEY,wire TEXT NOT NULL)",
        "CREATE TABLE IF NOT EXISTS funded_lifecycle_applied(action_id TEXT PRIMARY KEY,digest TEXT NOT NULL,entitlement TEXT NOT NULL,total_cash TEXT NOT NULL)",
        "CREATE TABLE IF NOT EXISTS funded_dividend_accruals(action_id TEXT PRIMARY KEY,digest TEXT NOT NULL,entitlement TEXT NOT NULL,total_income TEXT NOT NULL)",
        "CREATE TABLE IF NOT EXISTS funded_trade_settlements(execution_id TEXT PRIMARY KEY REFERENCES executions(execution_id),settlement_at TEXT NOT NULL)",
    ):
        oms.connection.execute(sql)


@_exact
def lifecycle_snapshot(oms):
    if not oms.connection.execute("SELECT 1 FROM sqlite_master WHERE name='funded_lifecycle_policy'").fetchone():
        return None
    row = oms.connection.execute("SELECT wire FROM funded_lifecycle_policy WHERE singleton=1").fetchone()
    policy = loads(row[0]) if row else {"settlement_policy": "IMMEDIATE"}
    event_rows = list(oms.connection.execute("SELECT event_id,wire FROM funded_lifecycle_events ORDER BY rowid LIMIT 1001"))
    action_rows = list(oms.connection.execute("SELECT digest,action_id,revision,wire FROM funded_lifecycle_actions ORDER BY action_id,revision LIMIT 1001"))
    events = [loads(row[1]) for row in event_rows]
    actions = [loads(row[3]) for row in action_rows]
    applied = {row[0]: row[1] for row in oms.connection.execute("SELECT action_id,digest FROM funded_lifecycle_applied")}
    trades = [{"execution_id": row[0], "settlement_at": row[1]} for row in oms.connection.execute("SELECT execution_id,settlement_at FROM funded_trade_settlements ORDER BY execution_id LIMIT 10001")]
    if len(events) > 1000 or len(actions) > 1000 or len(trades) > 10000:
        raise ValueError("lifecycle replay bounds exceeded")
    directory = Path(oms.connection.execute("PRAGMA database_list").fetchone()[2]).parent
    def verify(relative, value):
        path = directory/relative
        if any(p.is_symlink() for p in (path, *path.parents)) or not path.is_file() or path.read_bytes() != dumps(value).encode():
            raise ValueError("lifecycle record differs from immutable evidence")
    for row, action in zip(action_rows, actions, strict=True):
        if row[0] != _hash(action) or row[1] != action["action_id"] or row[2] != action["revision"]:
            raise ValueError("corporate action index differs from immutable identity")
        verify("lifecycle-actions/"+row[0]+".json", action)
    for row, event in zip(event_rows, events, strict=True):
        if row[0] != event["event_id"] or row[0] != _hash([event["action_id"], event["action_hash"]]+(["ACCRUAL"] if event["kind"] == "DIVIDEND_ACCRUAL" else [])):
            raise ValueError("lifecycle event index differs from immutable identity")
        verify("lifecycle-events/"+row[0]+".json", event)
    expected_paid = {}
    for event in events:
        if event["kind"] == "SPLIT":
            expected_paid[event["action_id"]] = (event["action_hash"], Decimal(0), Decimal(0))
        elif event["kind"] == "CASH_DIVIDEND" or (event["kind"] == "DIVIDEND_ACCRUAL"
                and (Decimal(event["cash_amount"]) or event.get("status") == "CANCELLED")):
            previous = expected_paid.get(event["action_id"], (None, Decimal(0), Decimal(0)))
            expected_paid[event["action_id"]] = (event["action_hash"], Decimal(event["entitlement_quantity"]), previous[2]+Decimal(event["cash_amount"]))
    indexed_paid = {row[0]: (row[1], Decimal(row[2]), Decimal(row[3]))
                    for row in oms.connection.execute("SELECT action_id,digest,entitlement,total_cash FROM funded_lifecycle_applied")}
    if indexed_paid != expected_paid:
        raise ValueError("lifecycle paid index differs from immutable events")
    if oms.connection.execute("SELECT 1 FROM sqlite_master WHERE name='funded_dividend_accruals'").fetchone():
        indexed = {row[0]: tuple(row[1:]) for row in oms.connection.execute("SELECT action_id,digest,entitlement,total_income FROM funded_dividend_accruals")}
        expected = {e["action_id"]: (e["action_hash"], e["entitlement_quantity"], e["total_income"]) for e in events if e["kind"] == "DIVIDEND_ACCRUAL"}
        if indexed != expected:
            raise ValueError("dividend accrual index differs from immutable events")
    for row in trades:
        path = directory/"trade-settlements"/(_hash(row["execution_id"])+".json")
        if any(p.is_symlink() for p in (path, *path.parents)) or not path.is_file() or path.read_bytes() != dumps(row).encode():
            raise ValueError("dated trade settlement differs from immutable evidence")
    latest = {}
    for action in actions:
        latest[action["action_id"]] = action
    pending = [row for row in latest.values() if applied.get(row["action_id"]) != _hash(row)]
    accrual_rows = []
    sources = {_hash(action): action for action in actions}
    for action_id, latest_source in latest.items():
        if latest_source["kind"] != "CASH_DIVIDEND":
            continue
        recognized = next((e for e in reversed(events) if e["kind"] == "DIVIDEND_ACCRUAL" and e["action_id"] == action_id), None)
        if recognized is None:
            continue
        source = sources[recognized["action_hash"]]
        paid_row = oms.connection.execute("SELECT total_cash FROM funded_lifecycle_applied WHERE action_id=?", (action_id,)).fetchone()
        paid_total = Decimal(paid_row[0]) if paid_row else Decimal(0)
        accrual_rows.append({"action_id": action_id, "revision": source["revision"], "action_hash": recognized["action_hash"],
            "instrument_id": source["instrument_id"], "source": source["source"], "known_at": source["known_at"],
            "effective_at": source["effective_at"], "record_at": source["record_at"], "pay_at": source["pay_at"],
            "entitlement_quantity": recognized["entitlement_quantity"], "recognized_income": recognized["total_income"],
            "paid_cash": str(paid_total), "outstanding_receivable": str(Decimal(recognized["total_income"])-paid_total),
            "status": source.get("status", "ACTIVE"), "classification": "SIMULATED"})
    return {"dividend_receivables": accrual_rows, "lifecycle_policy": policy, "lifecycle_events": events, "trade_settlements": trades,
            "pending_corporate_actions": pending, "corporate_action_sources": actions,
            "cash_settlement_evidence_state": "MODELED_SCHEDULED_DATES_NOT_BROKER_CONFIRMED"}


def validate_trade_settlement(oms, event, terms, provided, *, replay=False):
    state = lifecycle_snapshot(oms)
    policy = state["lifecycle_policy"]["settlement_policy"] if state else "IMMEDIATE"
    if event["kind"] != "FILL" or terms["kind"] != "EQUITY":
        if provided is not None:
            raise ValueError("dated settlement is supported only for equity fills")
        return None
    if state and not replay:
        for entry in state["lifecycle_events"]:
            if entry["kind"] == "SPLIT" and entry["instrument_id"] == terms["instrument_id"] and _stamp(event["occurred_at"]) <= _stamp(entry["applied_at"]):
                raise ValueError("new execution occurrence must follow applied split knowledge")
    if policy == "IMMEDIATE":
        if provided is not None and _stamp(provided) != _stamp(event["occurred_at"]):
            raise ValueError("immediate policy cannot silently become dated settlement")
        return None
    if provided is None:
        raise ValueError("explicit equity settlement_at required by account policy")
    observed, due = _stamp(event["occurred_at"]), _stamp(provided)
    if not observed <= due <= observed+timedelta(days=10):
        raise ValueError("bounded settlement date cannot precede trade occurrence")
    if policy == "XNYS_T_PLUS_ONE":
        from gig.data.calendars import ExchangeCalendar
        local_date = aware_timestamp(observed).tz_convert("America/New_York").date()
        calendar = ExchangeCalendar("XNYS", coverage_start=f"{local_date.year}-01-01", coverage_end=f"{local_date.year+1}-12-31")
        rows = calendar.schedule(local_date.isoformat(), (local_date+timedelta(days=10)).isoformat())["rows"]
        index = next((n for n, row in enumerate(rows) if _stamp(row["open"]) <= observed <= _stamp(row["close"])), None)
        if index is None or index+1 >= len(rows) or due != _stamp(rows[index+1]["close"]):
            raise ValueError("XNYS T+1 requires a regular-session trade and next scheduled session close")
    return {"execution_id": event["event_id"], "settlement_at": due.isoformat()}


@_exact
def recognize_dividends(book, oms, at, *, only=None, strict=False):
    """Same-transaction recognition; frozen entitlement never follows later fills."""
    if not oms.connection.in_transaction:
        raise ValueError("dividend recognition requires the existing OMS transaction")
    state = lifecycle_snapshot(oms)
    if not state:
        return
    lifecycle_schema(oms)
    latest = {}
    for action in state["corporate_action_sources"]:
        latest[action["action_id"]] = action
    if only is not None and not any(_hash(a) == only for a in latest.values()):
        raise ValueError("accrual requires latest admitted action revision")
    for action in latest.values():
        digest = _hash(action)
        if action["kind"] != "CASH_DIVIDEND" or (only is not None and digest != only):
            continue
        if _stamp(action["known_at"]) > at or _stamp(action["effective_at"]) > at:
            if strict:
                raise ValueError("dividend must be known and ex-date reached before recognition")
            continue
        prior = oms.connection.execute("SELECT digest,entitlement,total_income FROM funded_dividend_accruals WHERE action_id=?", (action["action_id"],)).fetchone()
        if prior is not None and prior[0] == digest:
            continue
        paid = oms.connection.execute("SELECT digest,entitlement,total_cash FROM funded_lifecycle_applied WHERE action_id=?", (action["action_id"],)).fetchone()
        if prior is None and paid is not None and paid[0] == digest:
            continue  # Legacy paid income must never be recognized twice.
        fills = book._fills(oms)
        try:
            if any(oms.order(row[0])["state"] not in TERMINAL and _stamp(oms.order(row[0])["created_at"]) < _stamp(action["effective_at"])
                   for row in oms.connection.execute("SELECT client_id FROM orders WHERE symbol=?", (action["instrument_id"],))):
                raise ValueError("pre-ex working orders prevent final entitlement recognition")
            if prior is not None or paid is not None:
                quantity = Decimal((prior or paid)[1])
            else:
                dates = {r["execution_id"]: _stamp(r["settlement_at"]) for r in state["trade_settlements"]}
                if at < _stamp(action["record_at"]) and any(f["symbol"] == action["instrument_id"] and _stamp(f["timestamp"]) < _stamp(action["effective_at"]) and at < dates.get(f["execution_id"], _stamp(f["timestamp"])) <= _stamp(action["record_at"]) for f in fills):
                    raise ValueError("record-date ownership depends on unsettled future evidence")
                for row in oms.connection.execute("SELECT occurred_at,received_at,payload FROM transport_events WHERE kind='FILL'"):
                    event = loads(row[2])
                    matching = next((f for f in fills if f["execution_id"] == event["event_id"]), None)
                    if matching and matching["symbol"] == action["instrument_id"] and _stamp(row[0]) < _stamp(action["effective_at"]) and _stamp(row[1]) > _stamp(action["record_at"]):
                        raise ValueError("late execution knowledge makes original entitlement unavailable")
                quantity = dividend_entitlement(fills, state["trade_settlements"], state["lifecycle_events"], instrument_id=action["instrument_id"], ex_at=action["effective_at"], record_at=action["record_at"])
        except ValueError:
            if strict:
                raise
            continue  # Snapshot retains an explicit unmodeled obligation.
        total = quantity*Decimal(action["amount_per_share"])
        old_total = Decimal(prior[2]) if prior else Decimal(paid[2]) if paid else Decimal(0)
        delta = total-old_total
        paid_total = Decimal(paid[2]) if paid else Decimal(0)
        cash_recovery = min(total-paid_total, Decimal(0))
        if cash_recovery and Decimal(book._snapshot(oms, at, recognize=False)["account"]["available_cash"])+cash_recovery < 0:
            if strict:
                raise ValueError("dividend correction requires unsupported financing")
            continue  # Preserve admitted source, unresolved obligation and fail-closed funding.
        event = {"event_id": _hash([action["action_id"], digest, "ACCRUAL"]), "kind": "DIVIDEND_ACCRUAL", "instrument_id": action["instrument_id"],
                 "action_id": action["action_id"], "action_hash": digest, "known_at": action["known_at"], "effective_at": action["effective_at"],
                 "record_at": action["record_at"], "pay_at": action["pay_at"], "applied_at": at.isoformat(), "cash_amount": str(cash_recovery),
                 "income_amount": str(delta), "receivable_amount": str(delta-cash_recovery), "entitlement_quantity": str(quantity),
                 "amount_per_share": action["amount_per_share"], "total_income": str(total), "status": action.get("status", "ACTIVE"),
                 "record_evidence": "KNOWN_TRADE_AND_SCHEDULED_SETTLEMENT_EVIDENCE; NOT_EXTERNAL_CONFIRMATION"}
        if len(state["lifecycle_events"]) >= 1000:
            raise ValueError("lifecycle replay bounds exceeded")
        write_immutable(book._path("lifecycle-events/"+event["event_id"]+".json"), dumps(event).encode())
        oms.connection.execute("INSERT INTO funded_lifecycle_events VALUES(?,?)", (event["event_id"], dumps(event)))
        if cash_recovery or action.get("status") == "CANCELLED":
            oms.connection.execute("INSERT INTO funded_lifecycle_applied VALUES(?,?,?,?) ON CONFLICT(action_id) DO UPDATE SET digest=excluded.digest,entitlement=excluded.entitlement,total_cash=excluded.total_cash", (action["action_id"], digest, str(quantity), str(paid_total+cash_recovery)))
        oms.connection.execute("INSERT INTO funded_dividend_accruals VALUES(?,?,?,?) ON CONFLICT(action_id) DO UPDATE SET digest=excluded.digest,entitlement=excluded.entitlement,total_income=excluded.total_income", (action["action_id"], digest, str(quantity), str(total)))


class FundedLifecycleWorkspace:
    def __init__(self, base_dir: Path, *, fault_hook=None):
        self.book = FundedWorkspace(base_dir)
        self.hook = fault_hook or (lambda _: None)

    @staticmethod
    def _action(value):
        base = {"action_id", "revision", "instrument_id", "source", "kind", "known_at", "effective_at", "record_at", "pay_at"}
        if not isinstance(value, dict) or value.get("kind") not in {"SPLIT", "CASH_DIVIDEND"}:
            raise ValueError("supported explicit equity action required")
        keys = {"ratio_new", "ratio_old"} if value["kind"] == "SPLIT" else {"amount_per_share"}
        optional = {"status"} if value["kind"] == "CASH_DIVIDEND" else set()
        _fields(value, base|keys|optional, base|keys)
        action = loads(dumps(value))
        for name in ("action_id", "instrument_id", "source"):
            _text(action[name], name)
        if isinstance(action["revision"], bool) or not isinstance(action["revision"], int) or not 1 <= action["revision"] <= 1000:
            raise ValueError("bounded positive action revision required")
        for name in ("known_at", "effective_at", "record_at", "pay_at"):
            action[name] = _stamp(action[name]).isoformat()
        if not _stamp(action["effective_at"]) <= _stamp(action["record_at"]) <= _stamp(action["pay_at"]):
            raise ValueError("ex/effective, record and payment clocks must increase")
        if action["kind"] == "SPLIT":
            if action["record_at"] != action["effective_at"] or action["pay_at"] != action["effective_at"]:
                raise ValueError("split record/payment clocks equal the declared effective instant")
            for key in keys:
                if isinstance(action[key], bool) or not isinstance(action[key], int) or not 1 <= action[key] <= 1000:
                    raise ValueError("bounded positive integer split ratio required")
        else:
            action["amount_per_share"] = str(_decimal(action["amount_per_share"], positive=False))
            if action.get("status", "ACTIVE") not in {"ACTIVE", "CANCELLED"} or (action.get("status") == "CANCELLED" and Decimal(action["amount_per_share"]) != 0):
                raise ValueError("explicit dividend cancellation requires zero per-share amount")
        return action

    @_exact
    def run(self, payload):
        allowed = {"configure": {"settlement_policy"}, "admit": {"action"}, "apply": {"action_hash"}, "accrue": {"action_hash"}, "state": set(), "replay": {"snapshot_id"}}
        if not isinstance(payload, dict) or payload.get("operation") not in allowed:
            raise ValueError("supported lifecycle operation required")
        op = payload["operation"]
        _fields(payload, {"operation", "now"}|allowed[op], {"operation", "now"}|allowed[op])
        request = loads(dumps(payload))
        at = _stamp(request["now"])
        if op == "replay":
            return self.book.run({"operation": "replay", "now": at.isoformat(), "snapshot_id": request["snapshot_id"]})
        path = self.book._path("oms.sqlite")
        if not path.is_file():
            return self.book._missing()
        with OfflineOMS(path) as oms, oms._transaction():
            self.book._schema(oms)
            lifecycle_schema(oms)
            clock = self.book._config(oms, "clock")
            if clock is not None and _stamp(clock) > at:
                raise ValueError("lifecycle clock cannot move backward")
            action_hash = None
            if op == "configure":
                policy = request["settlement_policy"]
                if policy not in {"IMMEDIATE", "EXPLICIT_EQUITY_DATES", "XNYS_T_PLUS_ONE"}:
                    raise ValueError("unsupported settlement policy")
                old = oms.connection.execute("SELECT wire FROM funded_lifecycle_policy WHERE singleton=1").fetchone()
                wire = dumps({"settlement_policy": policy})
                if old is not None and old[0] != wire and oms.connection.execute("SELECT COUNT(*) FROM orders").fetchone()[0]:
                    raise ValueError("settlement policy cannot change after orders exist")
                if old is None and oms.connection.execute("SELECT COUNT(*) FROM orders").fetchone()[0]:
                    raise ValueError("dated policy requires an initially empty account")
                oms.connection.execute("INSERT INTO funded_lifecycle_policy VALUES(1,?) ON CONFLICT(singleton) DO UPDATE SET wire=excluded.wire", (wire,))
            elif op == "admit":
                action = self._action(request["action"])
                if _stamp(action["known_at"]) > at:
                    raise ValueError("future corporate-action knowledge cannot be admitted as current")
                terms = self.book.instruments(oms).get(action["instrument_id"])
                if terms is None or terms["kind"] != "EQUITY":
                    raise ValueError("corporate actions require a registered equity")
                action_hash = _hash(action)
                prior = oms.connection.execute("SELECT digest,wire FROM funded_lifecycle_actions WHERE action_id=? ORDER BY revision DESC LIMIT 1", (action["action_id"],)).fetchone()
                exact = oms.connection.execute("SELECT digest FROM funded_lifecycle_actions WHERE action_id=? AND revision=?", (action["action_id"], action["revision"])).fetchone()
                if exact is not None and exact[0] != action_hash:
                    raise ValueError("conflicting action revision identity")
                if exact is None and prior is not None:
                    old = loads(prior[1])
                    fixed = ("instrument_id", "source", "kind", "effective_at", "record_at", "pay_at")
                    if action["revision"] != old["revision"]+1 or _stamp(action["known_at"]) <= _stamp(old["known_at"]) or any(action[k] != old[k] for k in fixed):
                        raise ValueError("revision must preserve terms and advance revision/knowledge time")
                if prior is None and action["revision"] != 1:
                    raise ValueError("original action revision must be one")
                if exact is None and oms.connection.execute("SELECT COUNT(*) FROM funded_lifecycle_actions").fetchone()[0] >= 1000:
                    raise ValueError("corporate action retention bound exceeded")
                write_immutable(self.book._path("lifecycle-actions/"+action_hash+".json"), dumps(action).encode())
                oms.connection.execute("INSERT OR IGNORE INTO funded_lifecycle_actions VALUES(?,?,?,?)", (action_hash, action["action_id"], action["revision"], dumps(action)))
            elif op in {"apply", "accrue"}:
                action_hash = request["action_hash"]
                if not isinstance(action_hash, str) or len(action_hash) != 64 or any(c not in "0123456789abcdef" for c in action_hash):
                    raise ValueError("canonical action identity required")
                found = oms.connection.execute("SELECT wire FROM funded_lifecycle_actions WHERE digest=?", (action_hash,)).fetchone()
                if found is None:
                    return self.book._missing()
                action = loads(found[0])
                if _hash(action) != action_hash or self.book._path("lifecycle-actions/"+action_hash+".json").read_bytes() != dumps(action).encode():
                    raise ValueError("corporate action evidence differs from immutable source")
                if op == "accrue":
                    recognize_dividends(self.book, oms, at, only=action_hash, strict=True)
                else:
                    self._apply(oms, action, action_hash, at)
            self.hook("after_lifecycle_transition")
            from gig.service.funded_risk_workspace import reconcile_funded_risk
            reconcile_funded_risk(self.book.base_dir, oms, at)
            document = self.book._snapshot(oms, at)
            document["lifecycle_operation"] = op
            if action_hash is not None:
                document["action_hash"] = action_hash
            return self.book._publish(oms, document)

    def _apply(self, oms, action, digest, at):
        recognize_dividends(self.book, oms, at, only=digest, strict=True) if action["kind"] == "CASH_DIVIDEND" else None
        latest = oms.connection.execute("SELECT digest FROM funded_lifecycle_actions WHERE action_id=? ORDER BY revision DESC LIMIT 1", (action["action_id"],)).fetchone()[0]
        if latest != digest or _stamp(action["known_at"]) > at or _stamp(action["pay_at"]) > at:
            raise ValueError("action must be the latest known revision and already effective/payable")
        applied = oms.connection.execute("SELECT digest,entitlement,total_cash FROM funded_lifecycle_applied WHERE action_id=?", (action["action_id"],)).fetchone()
        if applied is not None and applied[0] == digest:
            return
        symbol = action["instrument_id"]
        orders = [oms.order(row[0]) for row in oms.connection.execute("SELECT client_id FROM orders WHERE symbol=?", (symbol,))]
        if any(order["state"] not in TERMINAL for order in orders):
            raise ValueError("corporate action requires affected working orders terminated")
        state = lifecycle_snapshot(oms)
        fills = self.book._fills(oms)
        record = _stamp(action["record_at"])
        for row in oms.connection.execute("SELECT occurred_at,received_at,payload FROM transport_events WHERE kind='FILL'"):
            event = loads(row[2])
            match = next((f for f in fills if f["execution_id"] == event["event_id"]), None)
            if match and match["symbol"] == symbol and _stamp(row[0]) < _stamp(action["effective_at"]) and _stamp(row[1]) > record:
                raise ValueError("late execution knowledge makes original entitlement unavailable")
        event = {"event_id": _hash([action["action_id"], digest]), "kind": action["kind"], "instrument_id": symbol,
                 "action_id": action["action_id"], "action_hash": digest, "known_at": action["known_at"],
                 "effective_at": action["effective_at"], "applied_at": at.isoformat()}
        entitlement = total_cash = Decimal(0)
        if action["kind"] == "SPLIT":
            if any(e["kind"] == "DIVIDEND_ACCRUAL" and e["instrument_id"] == symbol
                   and _stamp(e["effective_at"]) >= _stamp(action["effective_at"]) for e in state["lifecycle_events"]):
                raise ValueError("late split correction crosses a frozen dividend entitlement")
            if any(f["symbol"] == symbol and _stamp(f["timestamp"]) >= _stamp(action["effective_at"]) for f in fills):
                raise ValueError("late split or revised split after intervening trades is unsupported")
            dates = {r["execution_id"]: _stamp(r["settlement_at"]) for r in state["trade_settlements"]}
            if any(f["symbol"] == symbol and dates.get(f["execution_id"], _stamp(f["timestamp"])) > _stamp(action["effective_at"]) for f in fills):
                raise ValueError("split cannot transform unsettled equity delivery")
            if any(e["kind"] == "SPLIT" and e["instrument_id"] == symbol and e["action_id"] != action["action_id"] and _stamp(e["applied_at"]) >= _stamp(action["effective_at"]) for e in state["lifecycle_events"]):
                raise ValueError("split revision crosses another action")
            old = loads(oms.connection.execute("SELECT wire FROM funded_lifecycle_actions WHERE digest=?", (applied[0],)).fetchone()[0]) if applied else None
            event.update(ratio_new=action["ratio_new"]*(old["ratio_old"] if old else 1), ratio_old=action["ratio_old"]*(old["ratio_new"] if old else 1))
            marks = self.book._config(oms, "marks", {"marks": {}, "max_age_seconds": 60})
            marks["marks"].pop(symbol, None)
            oms.connection.execute("INSERT INTO funded_config VALUES('marks',?) ON CONFLICT(name) DO UPDATE SET wire=excluded.wire", (dumps(marks),))
        else:
            accrued = oms.connection.execute("SELECT entitlement FROM funded_dividend_accruals WHERE action_id=?", (action["action_id"],)).fetchone()
            entitlement = Decimal(accrued[0]) if accrued else Decimal(applied[1]) if applied else dividend_entitlement(fills, state["trade_settlements"], state["lifecycle_events"],
                instrument_id=symbol, ex_at=action["effective_at"], record_at=action["record_at"])
            total_cash = entitlement*Decimal(action["amount_per_share"])
            amount = total_cash-(Decimal(applied[2]) if applied else Decimal(0))
            if Decimal(self.book._snapshot(oms, at)["account"]["available_cash"])+amount < 0:
                raise ValueError("corporate-action revision would require unsupported financing")
            event.update(cash_amount=str(amount), entitlement_quantity=str(entitlement), amount_per_share=action["amount_per_share"],
                         income_amount="0", receivable_amount=str(-amount), status=action.get("status", "ACTIVE"))
        write_immutable(self.book._path("lifecycle-events/"+event["event_id"]+".json"), dumps(event).encode())
        oms.connection.execute("INSERT INTO funded_lifecycle_events VALUES(?,?)", (event["event_id"], dumps(event)))
        oms.connection.execute("INSERT INTO funded_lifecycle_applied VALUES(?,?,?,?) ON CONFLICT(action_id) DO UPDATE SET digest=excluded.digest,entitlement=excluded.entitlement,total_cash=excluded.total_cash", (action["action_id"], digest, str(entitlement), str(total_cash)))

    def section(self, section, as_of, max_age_seconds):
        return self.book.section(section, as_of, max_age_seconds)
