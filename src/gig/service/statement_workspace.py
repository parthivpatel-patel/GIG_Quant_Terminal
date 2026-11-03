"""Independent supplied statement fixtures versus immutable offline-account books.

Content hashes establish integrity, never supplier authenticity. A completed
report can contain discrepancies or unknown coverage; it is not broker access or
external account certification and never mutates orders or kill controls.
"""
from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path
from typing import Any

from gig.artifacts import write_immutable
from gig.portfolio.ledger import signed_decimal
from gig.service.contracts import Contract, State
from gig.service.funded_workspace import FundedWorkspace
from gig.service.serialization import dumps, loads, public_error
from gig.service.workspace import TerminalWorkspace, _age, _fields, _text
from gig.timestamps import aware_timestamp

VERSION = "independent-offline-statement-v1"
BALANCES = {"cash", "nav", "realized_gross_pnl", "unrealized_pnl", "fees", "reserved_cash", "available_cash"}
LIFECYCLE_BALANCES = {"settled_cash", "unsettled_sale_receivables", "unsettled_buy_payables", "dividend_income", "dividend_receivable", "unsettled_cash"}
POSITION = {"instrument_id", "kind", "currency", "multiplier", "quantity", "cost_basis", "market_value", "unrealized_pnl"}
OPTION = {"underlying_id", "expiry", "strike", "right", "style", "settlement"}
ORDER = {"client_id", "instrument_id", "side", "quantity", "filled", "limit_price", "state", "acknowledged", "created_at", "updated_at"}
FILL = {"execution_id", "client_id", "instrument_id", "side", "quantity", "price", "fee", "occurred_at"}
SETTLEMENT = {"settlement_id", "instrument_id", "position_quantity", "spot", "payoff", "fee", "observed_at", "available_at"}
RESERVATION = {"client_id", "instrument_id", "remaining_quantity", "buy_cash", "short_put_collateral", "fee_cash", "state"}
LIFECYCLE = {"event_id", "kind", "instrument_id", "action_id", "applied_at", "effective_at", "known_at", "cash_amount", "entitlement_quantity"}
NUMERIC = BALANCES|LIFECYCLE_BALANCES|{"quantity", "filled", "limit_price", "cost_basis", "market_value", "position_quantity",
    "price", "fee", "spot", "payoff", "remaining_quantity", "buy_cash", "short_put_collateral", "fee_cash",
    "cash_amount", "entitlement_quantity", "amount_per_share", "ratio_new", "ratio_old", "multiplier"}


def _hash(value):
    return hashlib.sha256(dumps(value).encode()).hexdigest()


def _identity(value):
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError("canonical statement/report identity required")
    return value


def _amount(value):
    return None if value is None else str(signed_decimal(value))


def _stamp(value):
    return aware_timestamp(value).isoformat()


def _rows(value, maximum):
    if not isinstance(value, list) or len(value) > maximum:
        raise ValueError("statement record count exceeds supported bound")
    return value


def _normalize_record(row, fields, numbers=(), stamps=(), nullable=()):
    _fields(row, fields, fields)
    for key in fields-set(numbers)-set(stamps)-set(nullable)-{"acknowledged", "multiplier", "ratio_new", "ratio_old"}:
        _text(row[key], bound=256)
    for key in numbers:
        row[key] = _amount(row[key])
    for key in stamps:
        row[key] = _stamp(row[key])
    for key in nullable:
        row[key] = _amount(row[key])
    return row


class StatementWorkspace:
    def __init__(self, base_dir: Path, *, fault_hook=None):
        self.base_dir = Path(base_dir).absolute()
        self.workspace = TerminalWorkspace(base_dir)
        self.hook = fault_hook or (lambda _: None)

    def _path(self, name):
        return self.workspace._path("independent-statements/"+name)

    @staticmethod
    def _admit(payload):
        required = {"source", "statement_id", "observed_at", "available_at", "currency", "balances", "positions", "orders", "fills", "settlements", "reservations"}
        optional = {"lifecycle_events", "trade_settlements", "settled_trade_positions"}
        _fields(payload, required|optional, required)
        value = loads(dumps(payload))
        for key in ("source", "statement_id"):
            _text(value[key], bound=256)
        if value["currency"] != "USD":
            raise ValueError("statement comparison supports USD only")
        for key in ("observed_at", "available_at"):
            value[key] = _stamp(value[key])
        if aware_timestamp(value["observed_at"]) > aware_timestamp(value["available_at"]):
            raise ValueError("statement acquisition cannot precede observation")
        _fields(value["balances"], BALANCES|LIFECYCLE_BALANCES, BALANCES)
        value["balances"] = {key: _amount(amount) for key, amount in value["balances"].items()}
        for row in _rows(value["positions"], 128):
            if not isinstance(row, dict):
                raise ValueError("statement position must be an object")
            fields = POSITION|OPTION if row.get("kind") == "OPTION" else POSITION
            _normalize_record(row, fields, ("quantity", "cost_basis", "market_value", "unrealized_pnl"))
            if row["currency"] != "USD" or row["kind"] not in {"EQUITY", "OPTION"}:
                raise ValueError("explicit supported statement asset kind and currency required")
            if isinstance(row["multiplier"], bool) or not isinstance(row["multiplier"], int) or not 1 <= row["multiplier"] <= 1_000_000:
                raise ValueError("positive bounded integer multiplier required")
            if row["kind"] == "OPTION":
                row["expiry"], row["strike"] = _stamp(row["expiry"]), _amount(row["strike"])
        for row in _rows(value["orders"], 200):
            _normalize_record(row, ORDER, ("quantity", "filled", "limit_price"), ("created_at", "updated_at"))
            if not isinstance(row["acknowledged"], bool) or row["side"] not in {"buy", "sell"}:
                raise ValueError("explicit acknowledgement and order side required")
        for row in _rows(value["fills"], 2000):
            _normalize_record(row, FILL, ("quantity", "price", "fee"), ("occurred_at",))
            if row["side"] not in {"buy", "sell"}:
                raise ValueError("explicit fill side required")
        for row in _rows(value["settlements"], 128):
            _normalize_record(row, SETTLEMENT, ("position_quantity", "spot", "payoff", "fee"), ("observed_at", "available_at"))
        for row in _rows(value["reservations"], 200):
            _normalize_record(row, RESERVATION, ("remaining_quantity", "buy_cash", "short_put_collateral", "fee_cash"))
        for row in _rows(value.get("lifecycle_events", []), 1000):
            if not isinstance(row, dict) or row.get("kind") not in {"SPLIT", "CASH_DIVIDEND"}:
                raise ValueError("supported explicit lifecycle statement records required")
            additional = {"amount_per_share"} if row["kind"] == "CASH_DIVIDEND" else {"ratio_new", "ratio_old"}
            _normalize_record(row, LIFECYCLE|additional, ("cash_amount", "entitlement_quantity"), ("applied_at", "effective_at", "known_at"))
            if row["kind"] == "CASH_DIVIDEND":
                row["amount_per_share"] = _amount(row["amount_per_share"])
            else:
                for key in ("ratio_new", "ratio_old"):
                    if isinstance(row[key], bool) or not isinstance(row[key], int) or not 1 <= row[key] <= 1_000_000:
                        raise ValueError("positive bounded split ratio required")
        for row in _rows(value.get("trade_settlements", []), 2000):
            _normalize_record(row, {"execution_id", "settlement_at"}, stamps=("settlement_at",))
        if "settled_trade_positions" in value:
            positions = value["settled_trade_positions"]
            if not isinstance(positions, dict) or len(positions) > 128:
                raise ValueError("bounded settled position mapping required")
            for symbol, amount in list(positions.items()):
                _text(symbol, bound=256)
                positions[symbol] = _amount(amount)
        return value

    @staticmethod
    def _missing():
        return Contract.admit(State.UNAVAILABLE, None, source="independent-offline-statement",
            classification="UNAVAILABLE", scope="workspace.statement-reconciliation", error=public_error("DATA_UNAVAILABLE"))

    def ingest(self, payload: dict[str, Any]) -> Contract:
        statement = self._admit(payload)
        body = {"schema_version": VERSION, **statement}
        identity = _hash(body)
        result = Contract.admit(State.AVAILABLE, {"title": "Admitted supplied offline statement fixture", "statement_hash": identity,
            "statement_id": statement["statement_id"], "source": statement["source"], "observed_at": statement["observed_at"],
            "available_at": statement["available_at"], "fixture_only": True, "vendor_authenticity": "UNKNOWN",
            "external_reconciled": False}, source=statement["source"], classification="SYNTHETIC", as_of=statement["available_at"],
            source_identity=identity, scope="workspace.statement-admission", evidence_state="SUPPLIED_OFFLINE_STATEMENT_FIXTURE_NOT_BROKER_OBSERVED")
        path = self._path("statements.sqlite")
        path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(path) as db:
            db.execute("PRAGMA synchronous=FULL")
            db.execute("BEGIN IMMEDIATE")
            db.execute("CREATE TABLE IF NOT EXISTS statements(source TEXT,statement_id TEXT,available_at TEXT,digest TEXT,PRIMARY KEY(source,statement_id,available_at))")
            key = (statement["source"], statement["statement_id"], statement["available_at"])
            old = db.execute("SELECT digest FROM statements WHERE source=? AND statement_id=? AND available_at=?", key).fetchone()
            if old is not None and old[0] != identity:
                raise ValueError("statement revision conflicts at identical knowledge time")
            if old is None and db.execute("SELECT COUNT(*) FROM statements").fetchone()[0] >= 1000:
                raise ValueError("statement retention bound exceeded")
            write_immutable(self._path("sources/"+identity+".json"), dumps(body).encode())
            db.execute("INSERT OR IGNORE INTO statements VALUES(?,?,?,?)", (*key, identity))
        return result

    def _statement(self, identity):
        identity = _identity(identity)
        wire = self._path("sources/"+identity+".json").read_bytes()
        body = loads(wire)
        if _hash(body) != identity or dumps(body).encode() != wire or body.get("schema_version") != VERSION:
            raise ValueError("statement source integrity failure")
        value = {k: v for k, v in body.items() if k != "schema_version"}
        if self._admit(value) != value:
            raise ValueError("statement source is not canonical")
        with sqlite3.connect(self._path("statements.sqlite").as_uri()+"?mode=ro", uri=True) as db:
            entry = db.execute("SELECT digest FROM statements WHERE source=? AND statement_id=? AND available_at=?",
                (body["source"], body["statement_id"], body["available_at"])).fetchone()
        if entry != (identity,):
            raise ValueError("statement publication index differs from source")
        return body

    @staticmethod
    def _compare(statement, snapshot, request):
        findings = []
        def finding(code, path, expected=None, observed=None, severity="DISCREPANCY"):
            findings.append({"code": code, "path": path, "expected": expected, "observed": observed, "severity": severity})
        def compare(path, expected, observed):
            if expected is None or observed is None:
                finding("VALUE_UNKNOWN", path, expected, observed, "UNKNOWN")
            elif isinstance(expected, bool) or isinstance(observed, bool):
                if type(expected) is not type(observed) or expected != observed:
                    finding("VALUE_MISMATCH", path, expected, observed)
            else:
                equal = signed_decimal(expected) == signed_decimal(observed) if path.rsplit(".", 1)[-1] in NUMERIC else expected == observed
                if not equal:
                    finding("VALUE_MISMATCH", path, expected, observed)
        def records(label, expected, observed, key, fields):
            wanted = {row[key]: row for row in expected}
            supplied = {}
            for row in observed:
                identity = row[key]
                if identity in supplied:
                    finding("DUPLICATE_STATEMENT_RECORD", label+"."+identity)
                supplied.setdefault(identity, row)
            for identity in sorted(wanted.keys() | supplied.keys()):
                if identity not in wanted:
                    finding("UNEXPECTED_STATEMENT_RECORD", label+"."+identity, None, supplied[identity])
                elif identity not in supplied:
                    finding("MISSING_STATEMENT_RECORD", label+"."+identity, wanted[identity], None)
                else:
                    for field in sorted(fields):
                        compare(label+"."+identity+"."+field, wanted[identity].get(field), supplied[identity].get(field))
        now = aware_timestamp(request["now"])
        timing_ok = True
        for name, value in (("statement.available_at", statement["available_at"]), ("engine.as_of", snapshot["as_of"])):
            if aware_timestamp(value) > now:
                timing_ok = False
                finding("EVIDENCE_AFTER_CUTOFF", name, request["now"], value, "UNKNOWN")
        if aware_timestamp(statement["observed_at"]) != aware_timestamp(snapshot["as_of"]):
            timing_ok = False
            finding("OBSERVATION_CLOCK_MISMATCH", "statement.observed_at", snapshot["as_of"], statement["observed_at"], "UNKNOWN")
        account = snapshot["account"]
        if (account["mark_state"] != "AVAILABLE" or account["funding_state"] != "AVAILABLE"
                or account.get("reconciliation_state", "AVAILABLE") != "AVAILABLE" or account["pending_expiries"]):
            finding("ENGINE_FINANCIAL_STATE_INCOMPLETE", "engine.account", severity="UNKNOWN")
        for key in sorted(BALANCES):
            compare("balances."+key, account[key], statement["balances"][key])
        expected_positions = [{k: row[k] for k in POSITION | (OPTION if row["kind"] == "OPTION" else set())}
                              for row in account["rows"] if signed_decimal(row["quantity"])]
        position_fields = POSITION | OPTION
        # Equity rows do not have option terms; compare those separately to
        # avoid turning a correctly absent option field into unknown equity risk.
        records("positions", expected_positions, statement["positions"], "instrument_id", POSITION)
        for supplied in statement["positions"]:
            expected = next((row for row in expected_positions if row["instrument_id"] == supplied["instrument_id"]), None)
            if supplied["kind"] == "OPTION" and expected is not None:
                for key in sorted(position_fields-POSITION):
                    compare("positions."+supplied["instrument_id"]+"."+key, expected.get(key), supplied.get(key))
        expected_orders = []
        for order in snapshot["orders"]:
            state = order["state"]
            ack = True if state in {"ACKNOWLEDGED", "PARTIALLY_FILLED", "FILLED"} else False if state in {"CREATED", "APPROVED"} else None
            expected_orders.append({"client_id": order["client_id"], "instrument_id": order["symbol"],
                "side": order["side"], "quantity": order["quantity"], "filled": order["filled"], "limit_price": order["limit_price"],
                "state": state, "acknowledged": ack, "created_at": order["created_at"], "updated_at": order["updated_at"]})
        records("orders", expected_orders, statement["orders"], "client_id", ORDER)
        expected_fills = [{"execution_id": row["execution_id"], "client_id": row["client_id"], "instrument_id": row["symbol"],
            "side": row["side"], "quantity": row["quantity"], "price": row["price"], "fee": row["fee"], "occurred_at": row["timestamp"]}
            for row in snapshot["executions"]]
        records("fills", expected_fills, statement["fills"], "execution_id", FILL)
        records("settlements", snapshot["settlements"], statement["settlements"], "settlement_id", SETTLEMENT)
        records("reservations", account["reservations"], statement["reservations"], "client_id", RESERVATION)
        for fill in statement["fills"]:
            if aware_timestamp(fill["occurred_at"]) > aware_timestamp(statement["observed_at"]):
                timing_ok = False
                finding("STATEMENT_EVENT_AFTER_OBSERVATION", "fills."+fill["execution_id"], severity="UNKNOWN")
        lifecycle = snapshot.get("lifecycle_events", [])
        policy = snapshot.get("lifecycle_policy", {}).get("settlement_policy", "IMMEDIATE")
        coverage = {"orders": True, "fills": True, "positions": True, "cash_and_pnl": True, "reservations": True,
                    "acknowledgements": all(r["acknowledged"] is not None for r in expected_orders),
                    "option_settlements": True, "equity_settlements": policy == "IMMEDIATE", "corporate_actions": not lifecycle}
        if lifecycle:
            if "lifecycle_events" not in statement:
                finding("LIFECYCLE_COVERAGE_MISSING", "lifecycle_events", severity="UNKNOWN")
            else:
                coverage["corporate_actions"] = True
                for kind in ("SPLIT", "CASH_DIVIDEND"):
                    fields = LIFECYCLE | ({"ratio_new", "ratio_old"} if kind == "SPLIT" else {"amount_per_share"})
                    records("lifecycle_events."+kind, [r for r in lifecycle if r["kind"] == kind],
                            [r for r in statement["lifecycle_events"] if r["kind"] == kind], "event_id", fields)
                if any(r["kind"] == "CASH_DIVIDEND" for r in lifecycle):
                    compare("balances.dividend_income", account.get("dividend_income"), statement["balances"].get("dividend_income"))
        elif statement.get("lifecycle_events"):
            finding("UNEXPECTED_STATEMENT_RECORD", "lifecycle_events", [], statement["lifecycle_events"])
        if snapshot.get("pending_corporate_actions"):
            # The supported statement schema covers applied actions, not future
            # entitlement accrual or unapplied revisions. Empty event lists do
            # not establish that these known obligations have been reconciled.
            coverage["corporate_actions"] = False
            finding("PENDING_CORPORATE_ACTION_COVERAGE_UNKNOWN", "pending_corporate_actions", severity="UNKNOWN")
        if policy != "IMMEDIATE":
            for key in ("settled_cash", "unsettled_sale_receivables", "unsettled_buy_payables"):
                compare("balances."+key, account.get(key), statement["balances"].get(key))
            if "trade_settlements" not in statement or "settled_trade_positions" not in statement:
                finding("TRADE_SETTLEMENT_COVERAGE_MISSING", "trade_settlements", severity="UNKNOWN")
            else:
                coverage["equity_settlements"] = True
                records("trade_settlements", snapshot.get("trade_settlements", []), statement["trade_settlements"], "execution_id", {"execution_id", "settlement_at"})
                expected = account.get("settled_trade_positions")
                if not isinstance(expected, dict):
                    finding("SETTLED_POSITION_COVERAGE_UNKNOWN", "settled_trade_positions", severity="UNKNOWN")
                else:
                    records("settled_trade_positions", [{"instrument_id": k, "quantity": v} for k, v in expected.items()],
                        [{"instrument_id": k, "quantity": v} for k, v in statement["settled_trade_positions"].items()], "instrument_id", {"quantity"})
        elif statement.get("trade_settlements"):
            records("trade_settlements", snapshot.get("trade_settlements", []), statement["trade_settlements"], "execution_id", {"execution_id", "settlement_at"})
        discrepancy_count = sum(row["severity"] == "DISCREPANCY" for row in findings)
        unknown_count = sum(row["severity"] == "UNKNOWN" for row in findings)
        state = "UNKNOWN" if not timing_ok else "DISCREPANCY" if discrepancy_count else "UNKNOWN" if unknown_count else "MATCH"
        return {"schema_version": VERSION, "request": request, "as_of": request["now"], "source": statement["source"],
            "statement_id": statement["statement_id"], "statement_hash": request["statement_hash"], "snapshot_id": request["snapshot_id"],
            "engine_source_identity": snapshot["source_identity"], "engine_as_of": snapshot["as_of"], "statement_observed_at": statement["observed_at"],
            "reconciliation_state": state, "fixture_match": state == "MATCH", "external_reconciled": False,
            "fixture_only": True, "supplier_authenticity": "UNKNOWN", "coverage": coverage, "rows": findings,
            "source_independence": "CALLER_SUPPLIED_NOT_AUTHENTICATED",
            "engine_financial_state": {"mark_state": account["mark_state"], "funding_state": account["funding_state"], "pending_expiries": account["pending_expiries"]},
            "metrics": {"discrepancies": discrepancy_count, "unknowns": unknown_count, "fixture_match": state == "MATCH"},
            "rounding_policy": "EXACT_DECIMAL_NO_TOLERANCE_OR_IMPLICIT_ROUNDING",
            "comparison_scope": "IMMUTABLE_OFFLINE_STATEMENT_AND_ENGINE_SNAPSHOT_NOT_CURRENT_ACCOUNT_APPROVAL",
            "acknowledgement_scope": "ACK_OR_FILL_PROTOCOL_STATE; AMBIGUOUS_TERMINAL_ACK_HISTORY_REMAINS_UNKNOWN",
            "fully_funded_certified": False, "trading_approved": False,
            "notes": ["Statement values are supplied separately; supplier authenticity and source independence are unverified.",
                "Hashes establish content integrity, not a broker or supplier connection.",
                "An AVAILABLE report is not successful reconciliation when reconciliation_state is DISCREPANCY or UNKNOWN.",
                "This comparison does not release a kill switch, import balances, create orders or overwrite positions."]}

    @staticmethod
    def _contract(document, state=State.AVAILABLE):
        return Contract.admit(state, {"title": "Independent supplied statement comparison", **document} if state in {State.AVAILABLE, State.STALE} else None,
            source=document["source"], classification="SYNTHETIC", as_of=document["as_of"], scope="workspace.statement-reconciliation",
            source_identity=document["statement_hash"], model_identity=document["report_id"], model_version=VERSION,
            evidence_state="INDEPENDENTLY_SUPPLIED_OFFLINE_FIXTURE_NOT_BROKER_OBSERVED", readiness="RESEARCH_ONLY")

    def run(self, payload: dict[str, Any]) -> Contract:
        _fields(payload, {"statement_hash", "snapshot_id", "now"}, {"statement_hash", "snapshot_id", "now"})
        request = {"statement_hash": _identity(payload["statement_hash"]), "snapshot_id": _identity(payload["snapshot_id"]), "now": _stamp(payload["now"])}
        try:
            statement = self._statement(request["statement_hash"])
            snapshot = FundedWorkspace(self.base_dir).load(request["snapshot_id"])
        except FileNotFoundError:
            return self._missing()
        if (aware_timestamp(statement["available_at"]) > aware_timestamp(request["now"])
                or aware_timestamp(snapshot["as_of"]) > aware_timestamp(request["now"])):
            return Contract.admit(State.UNKNOWN, None, source="independent-offline-statement", classification="UNKNOWN",
                as_of=request["now"], scope="workspace.statement-reconciliation",
                evidence_state="PARENT_EVIDENCE_NOT_KNOWN_BY_REQUESTED_CUTOFF")
        body = self._compare(statement, snapshot, request)
        identity = _hash(body)
        document = {**body, "report_id": identity}
        result = self._contract(document)
        path = self._path("reports.sqlite")
        path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(path) as db:
            db.execute("PRAGMA synchronous=FULL")
            db.execute("BEGIN IMMEDIATE")
            db.execute("CREATE TABLE IF NOT EXISTS reports(report_id TEXT PRIMARY KEY,as_of TEXT NOT NULL)")
            old = db.execute("SELECT as_of FROM reports WHERE report_id=?", (identity,)).fetchone()
            if old is not None and old[0] != body["as_of"]:
                raise ValueError("statement report publication conflict")
            if old is None and db.execute("SELECT COUNT(*) FROM reports").fetchone()[0] >= 1000:
                raise ValueError("statement report retention bound exceeded")
            write_immutable(self._path("reports/"+identity+".json"), dumps(body).encode())
            self.hook("report_written")
            db.execute("INSERT OR IGNORE INTO reports VALUES(?,?)", (identity, body["as_of"]))
            self.hook("report_indexed")
        return result

    def load(self, report_id):
        identity = _identity(report_id)
        wire = self._path("reports/"+identity+".json").read_bytes()
        body = loads(wire)
        if _hash(body) != identity or body.get("schema_version") != VERSION or dumps(body).encode() != wire:
            raise ValueError("statement report integrity failure")
        with sqlite3.connect(self._path("reports.sqlite").as_uri()+"?mode=ro", uri=True) as db:
            entry = db.execute("SELECT as_of FROM reports WHERE report_id=?", (identity,)).fetchone()
        if entry != (body["as_of"],):
            raise ValueError("statement report index differs from immutable evidence")
        statement = self._statement(body["statement_hash"])
        snapshot = FundedWorkspace(self.base_dir).load(body["snapshot_id"])
        if self._compare(statement, snapshot, body["request"]) != body:
            raise ValueError("statement deterministic replay differs")
        return {**body, "report_id": identity}

    def replay(self, payload):
        _fields(payload, {"report_id"}, {"report_id"})
        try:
            return self._contract(self.load(payload["report_id"]))
        except FileNotFoundError:
            return self._missing()

    def section(self, section, as_of, max_age_seconds):
        if section not in {"portfolio", "risk", "execution", "operations"}:
            return None
        cutoff, maximum = _age(as_of, max_age_seconds)
        path = self._path("reports.sqlite")
        if not path.is_file():
            return None
        with sqlite3.connect(path.as_uri()+"?mode=ro", uri=True) as db:
            if not db.execute("SELECT 1 FROM sqlite_master WHERE name='reports'").fetchone():
                return None
            entries = db.execute("SELECT report_id,as_of FROM reports ORDER BY rowid LIMIT 1001").fetchall()
        if len(entries) > 1000:
            raise ValueError("statement report read bound exceeded")
        visible = [(aware_timestamp(at), n, identity) for n, (identity, at) in enumerate(entries) if aware_timestamp(at) <= cutoff]
        if not visible:
            return None
        at, _, identity = max(visible)
        document = self.load(identity)
        age = (cutoff.value-aware_timestamp(document["statement_observed_at"]).value)/1e9
        state = State.UNKNOWN if maximum is None else State.STALE if age > maximum else State.AVAILABLE
        return self._contract(document, state)
