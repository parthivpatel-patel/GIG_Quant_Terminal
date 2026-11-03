"""Isolated supported-scope funded simulation using the existing OMS and outbox."""
from __future__ import annotations

import hashlib
import sqlite3
from decimal import Decimal
from pathlib import Path
from typing import Any

from gig.artifacts import write_immutable
from gig.execution.offline_adapter import OfflinePaperAdapter
from gig.execution.oms import TERMINAL, WORKING, OfflineOMS, _decimal, _exact, _text
from gig.execution.outbox import ExecutionOutbox, event_value
from gig.execution.scheduler import DurableTWAPScheduler
from gig.portfolio.funded import funded_account
from gig.service.contracts import Contract, State
from gig.service.execution_workspace import _stamp
from gig.service.serialization import dumps, loads, public_error
from gig.service.workspace import TerminalWorkspace, _age, _fields
from gig.timestamps import aware_timestamp

VERSION = "funded-simulation-v1"


def _hash(value):
    return hashlib.sha256(dumps(value).encode()).hexdigest()


class FundedWorkspace:
    def __init__(self, base_dir: Path, *, fault_hook=None):
        self.workspace = TerminalWorkspace(base_dir)
        self.base_dir = self.workspace.base_dir
        self.hook = fault_hook or (lambda _: None)

    def _path(self, name):
        return self.workspace._path("funded-account/"+name)

    @staticmethod
    def _schema(oms):
        for statement in (
            "CREATE TABLE IF NOT EXISTS funded_config(name TEXT PRIMARY KEY,wire TEXT NOT NULL)",
            "CREATE TABLE IF NOT EXISTS funded_instruments(instrument_id TEXT PRIMARY KEY,wire TEXT NOT NULL)",
            "CREATE TABLE IF NOT EXISTS funded_orders(client_id TEXT PRIMARY KEY REFERENCES orders(client_id),wire TEXT NOT NULL)",
            "CREATE TABLE IF NOT EXISTS funded_settlements(instrument_id TEXT PRIMARY KEY,settlement_id TEXT UNIQUE NOT NULL,wire TEXT NOT NULL)",
            "CREATE TABLE IF NOT EXISTS funded_snapshots(snapshot_id TEXT PRIMARY KEY,as_of TEXT NOT NULL)",
            "CREATE TABLE IF NOT EXISTS funded_requests(identity TEXT PRIMARY KEY,snapshot_id TEXT NOT NULL)",
        ):
            oms.connection.execute(statement)

    @staticmethod
    def _config(oms, name, default=None):
        row = oms.connection.execute("SELECT wire FROM funded_config WHERE name=?", (name,)).fetchone()
        return loads(row[0]) if row else default

    @staticmethod
    def instruments(oms):
        return {row[0]: loads(row[1]) for row in oms.connection.execute("SELECT instrument_id,wire FROM funded_instruments ORDER BY instrument_id")}

    @staticmethod
    def _fills(oms):
        return [dict(row) for row in oms.connection.execute(
            "SELECT e.execution_id,e.client_id,o.symbol,o.side,e.quantity,e.price,e.fee,e.timestamp FROM executions e JOIN orders o USING(client_id) ORDER BY e.rowid")]

    @_exact
    def _snapshot(self, oms, at, marks=None, max_age_seconds=None):
        initial = self._config(oms, "initial_cash")
        if initial is None:
            raise ValueError("funded account is not initialized")
        registry = self.instruments(oms)
        stored_marks = self._config(oms, "marks", {"marks": {}, "max_age_seconds": 60})
        marks = stored_marks["marks"] if marks is None else marks
        maximum = stored_marks["max_age_seconds"] if max_age_seconds is None else max_age_seconds
        fills = self._fills(oms)
        orders = []
        for row in oms.connection.execute("SELECT client_id FROM orders ORDER BY client_id"):
            order = oms.order(row[0])
            saved = oms.connection.execute("SELECT wire FROM funded_orders WHERE client_id=?", (row[0],)).fetchone()
            if saved is None:
                raise ValueError("unmanaged OMS order in isolated funded account")
            orders.append({**order, **loads(saved[0])})
        settlements = [loads(row[0]) for row in oms.connection.execute("SELECT wire FROM funded_settlements ORDER BY settlement_id")]
        from gig.service.lifecycle_workspace import lifecycle_snapshot
        lifecycle = lifecycle_snapshot(oms)
        if len(fills) > 10000 or len(orders) > 100 or len(registry) > 64:
            raise ValueError("funded replay size bound exceeded")
        if any(fill["symbol"] not in registry for fill in fills):
            raise ValueError("unregistered funded execution instrument")
        if lifecycle and lifecycle["lifecycle_policy"]["settlement_policy"] != "IMMEDIATE":
            identities = {row["execution_id"] for row in lifecycle["trade_settlements"]}
            if any(fill["symbol"] in registry and registry[fill["symbol"]]["kind"] == "EQUITY" and fill["execution_id"] not in identities for fill in fills):
                raise ValueError("dated account has equity executions without settlement evidence")
        account = funded_account(initial, registry, fills, orders, settlements, marks, at.isoformat(), maximum,
            lifecycle_events=lifecycle["lifecycle_events"] if lifecycle else None,
            trade_settlements=lifecycle["trade_settlements"] if lifecycle else None)
        if lifecycle:
            due = [row for row in lifecycle["pending_corporate_actions"] if aware_timestamp(row["pay_at"]) <= aware_timestamp(at)]
            account["pending_corporate_actions"] = [row["action_id"] for row in due]
            if due:
                account["funding_state"] = account["reconciliation_state"] = "UNKNOWN"
            accruals = [row["action_id"] for row in lifecycle["pending_corporate_actions"]
                if row["kind"] == "CASH_DIVIDEND" and aware_timestamp(row["effective_at"]) <= aware_timestamp(at)]
            account["unmodeled_dividend_accruals"] = accruals
            account["dividend_accrual_state"] = "UNKNOWN" if accruals else "NOT_REQUIRED"
            if accruals or due:
                # Cash remains spendable only under the existing funding rules;
                # an unbooked entitlement cannot be treated as complete NAV.
                account["nav"] = account["total_pnl"] = None
                account["reconciliation_state"] = "UNKNOWN"
        actual = {key: Decimal(value) for key, value in oms.positions().items() if Decimal(value)}
        expected = {key: Decimal(value) for key, value in account["raw_oms_positions"].items() if Decimal(value)}
        if actual != expected:
            raise ValueError("funded accounting and OMS positions disagree")
        body = {"schema_version": VERSION, "as_of": at.isoformat(), "account": account,
            "account_state": account["reconciliation_state"],
            "rows": account["rows"], "instruments": registry, "marks": marks, "orders": orders,
            "settlements": settlements, "executions": fills,
            "source_identity": _hash({"initial_cash": initial, "instruments": registry, "fills": fills, "settlements": settlements}),
            "metrics": {"cash": account["cash"], "nav": account["nav"], "reserved_cash": account["reserved_cash"],
                "available_cash": account["available_cash"], "cash_secured_put_collateral": account["cash_secured_put_collateral"],
                "funding_state": account["funding_state"], "mark_state": account["mark_state"], "kill_switch": oms.kill_switch},
            "advanced_risk_state": "UNKNOWN_UNLESS_SEPARATELY_CONFIGURED_AND_VERIFIED",
            "classification": "SIMULATED", "trading_approved": False, "broker_connected": False,
            "scope": "ISOLATED_OFFLINE_ACCOUNT_NOT_LEGACY_OMS_OR_BROKER_POSITIONS",
            "mark_evidence_state": "CALLER_SUPPLIED_NOT_VENDOR_VERIFIED"}
        if lifecycle:
            body.update(lifecycle)
            body["source_identity"] = _hash({"initial_cash": initial, "instruments": registry, "fills": fills,
                "settlements": settlements, **lifecycle})
        return body

    @staticmethod
    def _contract(document, state=None):
        if state is None:
            state = State.ERROR if document["account"]["funding_state"] == "ERROR" else State(document.get("operation_state", "AVAILABLE"))
        return Contract.admit(state, {"title": "Supported-scope funded offline account", **document}
            if state in {State.AVAILABLE, State.STALE} else None,
            source="funded-offline-account", classification="SYNTHETIC", as_of=document["as_of"],
            scope="workspace.funded-account", source_identity=document["snapshot_id"], model_version=VERSION,
            evidence_state="SIMULATED_OMS_AND_MULTIPLIER_AWARE_CASH_LEDGER",
            readiness="RESEARCH_ONLY", error=public_error("INVALID_REQUEST") if state == State.ERROR else None)

    @staticmethod
    def _missing():
        return Contract.admit(State.UNAVAILABLE, None, source="funded-offline-account", classification="UNAVAILABLE",
            scope="workspace.funded-account", error=public_error("DATA_UNAVAILABLE"))

    def load(self, snapshot_id):
        if not isinstance(snapshot_id, str) or len(snapshot_id) != 64 or any(c not in "0123456789abcdef" for c in snapshot_id):
            raise ValueError("canonical funded snapshot identity required")
        path = self._path("oms.sqlite")
        if not path.is_file():
            raise FileNotFoundError("funded account unavailable")
        wire = self._path("snapshots/"+snapshot_id+".json").read_bytes()
        if hashlib.sha256(wire).hexdigest() != snapshot_id:
            raise ValueError("funded snapshot integrity mismatch")
        document = loads(wire)
        if document.get("schema_version") != VERSION or dumps(document).encode() != wire:
            raise ValueError("unsupported funded checkpoint")
        with sqlite3.connect(path.as_uri()+"?mode=ro", uri=True) as connection:
            row = connection.execute("SELECT as_of FROM funded_snapshots WHERE snapshot_id=?", (snapshot_id,)).fetchone()
        if row is None or row[0] != document["as_of"]:
            raise ValueError("funded snapshot index mismatch")
        return {**document, "snapshot_id": snapshot_id}

    @staticmethod
    def _instrument(value):
        base = {"instrument_id", "kind", "currency", "multiplier"}
        option = {"underlying_id", "expiry", "strike", "right", "style", "settlement"}
        _fields(value, base|option, base)
        item = loads(dumps(value))
        _text(item["instrument_id"], "instrument_id")
        if item["currency"] != "USD" or isinstance(item["multiplier"], bool) or not isinstance(item["multiplier"], int) or not 1 <= item["multiplier"] <= 1000000:
            raise ValueError("explicit USD currency and positive integer multiplier required")
        if item["kind"] == "EQUITY":
            if item["instrument_id"].upper().startswith(("ECB:EXR:", "FX:", "REFERENCE:")):
                raise ValueError("reference FX observations are not funded equity instruments")
            if set(item) != base or item["multiplier"] != 1:
                raise ValueError("equities require unit multiplier and no option terms")
        elif item["kind"] == "OPTION":
            if set(item) != base|option or item["right"] not in {"CALL", "PUT"} or item["style"] != "EUROPEAN" or item["settlement"] != "CASH":
                raise ValueError("only European cash-settled CALL/PUT contracts supported")
            _text(item["underlying_id"], "underlying_id")
            item["expiry"] = _stamp(item["expiry"]).isoformat()
            item["strike"] = str(_decimal(item["strike"]))
        else:
            raise ValueError("unsupported funded asset kind")
        return item

    def _publish(self, oms, document, request_id=None):
        identity = _hash(document)
        result = self._contract({**document, "snapshot_id": identity})
        write_immutable(self._path("snapshots/"+identity+".json"), dumps(document).encode())
        if oms.connection.execute("SELECT COUNT(*) FROM funded_snapshots").fetchone()[0] >= 1000 and not oms.connection.execute("SELECT 1 FROM funded_snapshots WHERE snapshot_id=?", (identity,)).fetchone():
            raise ValueError("funded snapshot retention bound exceeded")
        oms.connection.execute("INSERT OR IGNORE INTO funded_snapshots VALUES(?,?)", (identity, document["as_of"]))
        if request_id:
            oms.connection.execute("INSERT INTO funded_requests VALUES(?,?)", (request_id, identity))
        oms.connection.execute("INSERT INTO funded_config VALUES('clock',?) ON CONFLICT(name) DO UPDATE SET wire=excluded.wire", (dumps(document["as_of"]),))
        self.hook("before_snapshot_commit")
        return result

    @_exact
    def run(self, payload: dict[str, Any]):
        if not isinstance(payload, dict) or not {"operation", "now"} <= payload.keys():
            raise ValueError("funded operation and explicit now required")
        request = loads(dumps(payload))
        at = _stamp(request["now"])
        request["now"] = at.isoformat()
        operation = request["operation"]
        allowed = {"initialize": {"initial_cash"}, "register": {"instruments"},
            "reserve": {"client_id", "instrument_id", "side", "quantity", "limit_price", "price_asof", "fee_budget"},
            "dispatch": set(), "event": {"event"}, "settle": {"settlement_id", "instrument_id", "spot", "observed_at", "available_at", "fee"},
            "mark": {"marks", "max_age_seconds"}, "state": set(), "replay": {"snapshot_id"}, "kill": {"engaged"}}
        allowed["cancel"] = {"client_id"}
        if not isinstance(operation, str) or operation not in allowed:
            raise ValueError("unsupported funded operation")
        fields = {"operation", "now"}|allowed[operation]
        if operation == "event":
            fields = fields | {"settlement_at"}
        _fields(request, fields, fields-({"settlement_at"} if operation == "event" else set()))
        if operation == "replay":
            try:
                document = self.load(request["snapshot_id"])
                return self._contract(document) if aware_timestamp(document["as_of"]) <= aware_timestamp(at) else self._missing()
            except FileNotFoundError:
                return self._missing()
        path = self._path("oms.sqlite")
        if not path.is_file() and operation != "initialize":
            return self._missing()
        # Initial precision and fixed-field admission precede every file/DDL mutation.
        if operation == "initialize":
            request["initial_cash"] = str(_decimal(request["initial_cash"]))
        request_id = _hash(request)
        # Reads, transport advancement, mark refreshes and control changes must
        # observe the current book even when their clock/payload is repeated.
        # Immutable order/event/lifecycle requests retain original receipts.
        remember_request = operation not in {"state", "dispatch", "mark", "kill"}
        with OfflineOMS(path) as oms:
            outbox = ExecutionOutbox(oms)
            self._schema(oms)
            prior = oms.connection.execute("SELECT snapshot_id FROM funded_requests WHERE identity=?", (request_id,)).fetchone()
            if prior and remember_request:
                return self._contract(self.load(prior[0]))
            clock = self._config(oms, "clock")
            if clock is not None and aware_timestamp(clock) > aware_timestamp(at):
                raise ValueError("funded simulation clock cannot move backward")
            operation_result = {"state": "AVAILABLE"}
            if operation == "dispatch":
                from gig.service.funded_risk_workspace import reconcile_funded_risk
                with oms._transaction():
                    if self._snapshot(oms, at)["account"]["funding_state"] != "AVAILABLE":
                        raise ValueError("unresolved funding or lifecycle obligations block dispatch")
                    reconcile_funded_risk(self.base_dir, oms, at)
                DurableTWAPScheduler(oms, outbox, create=False).ensure_not_before(at)
                with_adapter = OfflinePaperAdapter(self._path("counterparty.sqlite"))
                try:
                    operation_result = outbox.dispatch(with_adapter, now=at)
                finally:
                    with_adapter.close()
            with oms._transaction():
                # Another process may have completed this request after the
                # optimistic receipt read. Recheck under the writer lock.
                if remember_request:
                    prior = oms.connection.execute("SELECT snapshot_id FROM funded_requests WHERE identity=?", (request_id,)).fetchone()
                    if prior:
                        return self._contract(self.load(prior[0]))
                clock = self._config(oms, "clock")
                if clock is not None and aware_timestamp(clock) > aware_timestamp(at):
                    raise ValueError("funded simulation clock cannot move backward")
                DurableTWAPScheduler(oms, outbox, create=False).ensure_not_before(at)
                if operation == "initialize":
                    previous = self._config(oms, "initial_cash")
                    if previous is not None and previous != request["initial_cash"]:
                        raise ValueError("initial funded capital cannot change")
                    if previous is None and oms.connection.execute("SELECT COUNT(*) FROM executions").fetchone()[0]:
                        raise ValueError("funded initialization requires an empty isolated OMS")
                    oms.connection.execute("INSERT OR IGNORE INTO funded_config VALUES('initial_cash',?)", (dumps(request["initial_cash"]),))
                else:
                    self._snapshot(oms, at)
                if operation == "register":
                    values = request["instruments"]
                    if not isinstance(values, list) or not 1 <= len(values) <= 64:
                        raise ValueError("one to 64 explicit instruments required")
                    for value in values:
                        item = self._instrument(value)
                        if item["kind"] == "OPTION" and _stamp(item["expiry"]) <= at:
                            raise ValueError("cannot register an expired contract")
                        old = oms.connection.execute("SELECT wire FROM funded_instruments WHERE instrument_id=?", (item["instrument_id"],)).fetchone()
                        if old and old[0] != dumps(item):
                            raise ValueError("instrument identity cannot change terms")
                        oms.connection.execute("INSERT OR IGNORE INTO funded_instruments VALUES(?,?)", (item["instrument_id"], dumps(item)))
                elif operation == "reserve":
                    self._reserve(oms, outbox, request, at)
                elif operation == "event":
                    operation_result = self._event(oms, outbox, request["event"], at, settlement_at=request.get("settlement_at"))
                elif operation == "cancel":
                    outbox.request_cancel(request["client_id"], now=at)
                elif operation == "settle":
                    self._settle(oms, request, at)
                elif operation == "mark":
                    maximum = request["max_age_seconds"]
                    if isinstance(maximum, bool) or not isinstance(maximum, (float, int)) or not 0 <= maximum <= 86400:
                        raise ValueError("finite bounded mark freshness required")
                    registry = self.instruments(oms)
                    if not isinstance(request["marks"], dict) or len(request["marks"]) > 64:
                        raise ValueError("bounded funded marks required")
                    for symbol, mark in request["marks"].items():
                        if symbol not in registry:
                            raise ValueError("mark instrument is unregistered")
                        _fields(mark, {"price", "as_of"}, {"price", "as_of"})
                        mark["price"] = str(_decimal(mark["price"], positive=registry[symbol]["kind"] == "EQUITY"))
                        mark["as_of"] = aware_timestamp(mark["as_of"]).isoformat()
                        if aware_timestamp(mark["as_of"]) > aware_timestamp(at):
                            raise ValueError("mark cannot lie after observation cutoff")
                    oms.connection.execute("INSERT INTO funded_config VALUES('marks',?) ON CONFLICT(name) DO UPDATE SET wire=excluded.wire",
                        (dumps({"marks": request["marks"], "max_age_seconds": maximum}),))
                elif operation == "kill":
                    oms.set_kill_switch(request["engaged"], now=at)
                if operation in {"mark", "settle", "kill"}:
                    from gig.service.funded_risk_workspace import reconcile_funded_risk
                    reconcile_funded_risk(self.base_dir, oms, at)
                document = self._snapshot(oms, at)
                document["operation_state"] = operation_result["state"]
                document["operation_result"] = operation_result
                if document["account"]["funding_state"] == "ERROR":
                    oms.set_kill_switch(True, now=at)
                    document = self._snapshot(oms, at)
                    document["operation_state"] = "ERROR"
                return self._publish(oms, document, request_id if remember_request else None)

    def _reserve(self, oms, outbox, request, at):
        registry = self.instruments(oms)
        symbol = request["instrument_id"]
        if symbol not in registry:
            raise ValueError("reserve requires a registered instrument")
        terms = registry[symbol]
        quantity, price = _decimal(request["quantity"]), _decimal(request["limit_price"])
        _decimal(request["fee_budget"], positive=False)
        if terms["kind"] == "OPTION" and (quantity != quantity.to_integral_value() or _stamp(terms["expiry"]) <= at):
            raise ValueError("whole live option contracts required")
        if self._snapshot(oms, at)["account"]["pending_expiries"]:
            raise ValueError("expired unsettled obligations block new orders")
        _stamp(request["price_asof"])
        # A configured advanced policy is an additional guard, never a substitute
        # for multiplier-aware funding and existing OMS/outbox approval.
        from gig.service.funded_risk_workspace import guard_funded_order
        guard_funded_order(self.base_dir, oms, request, at)
        intent = {"fee_budget": str(_decimal(request["fee_budget"], positive=False)), "funded_limit_price": str(price)}
        old = oms.connection.execute("SELECT wire FROM funded_orders WHERE client_id=?", (request["client_id"],)).fetchone()
        if old and old[0] != dumps(intent):
            raise ValueError("funded order reservation conflicts with original")
        outbox.admit_order(request["client_id"], symbol, request["side"], str(quantity), limit_price=str(price),
            reference_price=str(price), price_asof=_stamp(request["price_asof"]),
            max_notional=str(quantity*price*terms["multiplier"]+Decimal(intent["fee_budget"])),
            max_position_quantity="1000000", now=at)
        oms.connection.execute("INSERT OR IGNORE INTO funded_orders VALUES(?,?)", (request["client_id"], dumps(intent)))
        account = self._snapshot(oms, at)["account"]
        if account["funding_state"] != "AVAILABLE":
            raise ValueError("cash, fees and conservative collateral cannot fund all working orders")
        self.hook("after_funded_order")

    def _event(self, oms, outbox, payload, at, *, settlement_at=None):
        event = event_value(payload)
        old = oms.connection.execute("SELECT digest FROM transport_events WHERE event_id=?", (event["event_id"],)).fetchone()
        from gig.service.lifecycle_workspace import validate_trade_settlement
        terms = self.instruments(oms)[oms.order(event["client_id"])["symbol"]]
        dated = validate_trade_settlement(oms, event, terms, settlement_at, replay=old is not None)
        if dated is not None:
            prior_date = oms.connection.execute("SELECT settlement_at FROM funded_trade_settlements WHERE execution_id=?", (dated["execution_id"],)).fetchone()
            if prior_date is not None and prior_date[0] != dated["settlement_at"]:
                raise ValueError("execution settlement date conflicts with original evidence")
        if old is None:
            sent = oms.connection.execute("SELECT state FROM transport_outbox WHERE client_id=? AND kind='SUBMIT'", (event["client_id"],)).fetchone()
            if sent is None or sent[0] == "READY":
                raise ValueError("counterparty event requires actual offline dispatch first")
        if old is None and event["kind"] == "FILL":
            order = oms.order(event["client_id"])
            if order["state"] not in WORKING:
                raise ValueError("funded fill fixture requires a dispatched acknowledged order")
            terms = self.instruments(oms)[order["symbol"]]
            if terms["kind"] == "OPTION" and (Decimal(event["quantity"]) != Decimal(event["quantity"]).to_integral_value()
                    or _stamp(event["occurred_at"]) >= _stamp(terms["expiry"])):
                raise ValueError("option fill requires whole contracts before expiry")
            if oms.connection.execute("SELECT 1 FROM funded_settlements WHERE instrument_id=?", (order["symbol"],)).fetchone():
                raise ValueError("late fill after cash settlement requires unsupported lifecycle revision")
            budget = loads(oms.connection.execute("SELECT wire FROM funded_orders WHERE client_id=?", (event["client_id"],)).fetchone()[0])["fee_budget"]
            paid = sum((Decimal(row[0]) for row in oms.connection.execute("SELECT fee FROM executions WHERE client_id=?", (event["client_id"],))), Decimal(0))
            if paid+Decimal(event["fee"]) > Decimal(budget):
                raise ValueError("fill fee exceeds reserved fee cash")
        result = outbox.ingest_event(event, received_at=at)
        if dated is not None and oms.connection.execute("SELECT 1 FROM executions WHERE execution_id=?", (dated["execution_id"],)).fetchone():
            write_immutable(self._path("trade-settlements/"+_hash(dated["execution_id"])+".json"), dumps(dated).encode())
            oms.connection.execute("INSERT OR IGNORE INTO funded_trade_settlements VALUES(?,?)", (dated["execution_id"], dated["settlement_at"]))
        self.hook("after_funded_event")
        if result["state"] == "ERROR":
            return result
        from gig.service.funded_risk_workspace import reconcile_funded_risk
        reconcile_funded_risk(self.base_dir, oms, at)
        return result

    def _settle(self, oms, request, at):
        registry = self.instruments(oms)
        terms = registry.get(request["instrument_id"])
        if terms is None or terms["kind"] != "OPTION":
            raise ValueError("cash settlement requires an option identity")
        observed, available = _stamp(request["observed_at"]), _stamp(request["available_at"])
        if observed != _stamp(terms["expiry"]) or not observed <= available <= at:
            raise ValueError("settlement requires original expiry spot and its explicit knowledge time")
        spot = _decimal(request["spot"], positive=False)
        _decimal(request["fee"], positive=False)
        _text(request["settlement_id"], "settlement_id")
        old = oms.connection.execute("SELECT wire FROM funded_settlements WHERE instrument_id=?", (request["instrument_id"],)).fetchone()
        if old:
            previous = loads(old[0])
            expected = {"settlement_id": request["settlement_id"], "instrument_id": request["instrument_id"],
                "spot": str(spot), "fee": str(Decimal(request["fee"])), "observed_at": observed.isoformat(), "available_at": available.isoformat()}
            if any(previous[key] != expected[key] for key in expected):
                raise ValueError("settlement conflicts with immutable lifecycle evidence")
            return
        orders = [oms.order(row[0]) for row in oms.connection.execute("SELECT client_id FROM orders WHERE symbol=?", (request["instrument_id"],))]
        if any(row["state"] not in TERMINAL for row in orders):
            raise ValueError("cash settlement requires all option working orders terminated")
        quantity = Decimal(self._snapshot(oms, at)["account"]["effective_positions"].get(request["instrument_id"], "0"))
        strike = Decimal(terms["strike"])
        payoff = max(Decimal(0), spot-strike if terms["right"] == "CALL" else strike-spot)
        cash = Decimal(self._snapshot(oms, at)["account"]["cash"])
        if cash+quantity*payoff*terms["multiplier"]-Decimal(request["fee"]) < 0:
            raise ValueError("cash settlement fee would require unsupported borrowed cash")
        item = {"settlement_id": request["settlement_id"], "instrument_id": request["instrument_id"],
            "position_quantity": str(quantity), "spot": str(spot), "payoff": str(payoff), "fee": str(Decimal(request["fee"])),
            "observed_at": observed.isoformat(), "available_at": available.isoformat(),
            "assumption": "AUTOMATIC_INTRINSIC_CASH_EXERCISE_LONG_EUROPEAN_OPTIONS_ONLY"}
        oms.connection.execute("INSERT INTO funded_settlements VALUES(?,?,?)", (request["instrument_id"], request["settlement_id"], dumps(item)))

    def section(self, section, as_of, max_age_seconds):
        if section not in {"portfolio", "risk", "options", "execution", "research"}:
            return None
        cutoff, max_age_seconds = _age(as_of, max_age_seconds)
        path = self._path("oms.sqlite")
        if not path.is_file():
            return None
        with sqlite3.connect(path.as_uri()+"?mode=ro", uri=True) as connection:
            if not connection.execute("SELECT 1 FROM sqlite_master WHERE name='funded_snapshots'").fetchone():
                return None
            rows = connection.execute("SELECT snapshot_id,as_of FROM funded_snapshots ORDER BY rowid LIMIT 1001").fetchall()
        if len(rows) > 1000:
            raise ValueError("funded history read bound exceeded")
        eligible = [(aware_timestamp(stamp), n, identity) for n, (identity, stamp) in enumerate(rows) if aware_timestamp(stamp) <= cutoff]
        if not eligible:
            return None
        stamp, _, identity = max(eligible)
        state = State.UNKNOWN if max_age_seconds is None else State.STALE if (cutoff-stamp).total_seconds() > max_age_seconds else State.AVAILABLE
        document = self.load(identity)
        if document["account"]["funding_state"] == "ERROR" or document.get("operation_state") == "ERROR":
            state = State.ERROR
        elif document["account"]["reconciliation_state"] != "AVAILABLE":
            state = State.STALE if document["account"]["mark_state"] == "STALE" else State.UNKNOWN
        return self._contract(document, state)
