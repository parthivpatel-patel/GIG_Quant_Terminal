"""Independent literal statement values, discrepancy detection and exact replay."""
from __future__ import annotations

import copy
import hashlib
import sqlite3
import tempfile
import unittest
from datetime import UTC, datetime, timedelta
from pathlib import Path

from gig.service.contracts import EnvelopeSchema
from gig.service.funded_workspace import FundedWorkspace
from gig.service.statement_workspace import StatementWorkspace

NOW = datetime(2026, 11, 3, 15, 0, 0, 123456, tzinfo=UTC)
AT = NOW.isoformat()
EXPIRY = (NOW+timedelta(days=1)).isoformat()


def supplied_statement(**changes):
    """Hand-calculated supplier fixture; never assembled from an engine snapshot."""
    return {"source": "authored-independent-supplier", "statement_id": "equity-reference-1", "observed_at": AT,
        "available_at": AT, "currency": "USD", "balances": {"cash": "899.8", "nav": "999.8", "realized_gross_pnl": "0",
            "unrealized_pnl": "0", "fees": "0.2", "reserved_cash": "0", "available_cash": "899.8"},
        "positions": [{"instrument_id": "SYN101", "kind": "EQUITY", "currency": "USD", "multiplier": 1,
            "quantity": "10", "cost_basis": "100", "market_value": "100", "unrealized_pnl": "0"}],
        "orders": [{"client_id": "stock", "instrument_id": "SYN101", "side": "buy", "quantity": "10", "filled": "10",
            "limit_price": "10", "state": "FILLED", "acknowledged": True, "created_at": AT, "updated_at": AT}],
        "fills": [{"execution_id": "stock-fill", "client_id": "stock", "instrument_id": "SYN101", "side": "buy",
            "quantity": "10", "price": "10", "fee": "0.2", "occurred_at": AT}], "settlements": [], "reservations": [], **changes}


def funded_reference(base, *, partial=False, mark=True):
    book = FundedWorkspace(base)
    book.run({"operation": "initialize", "now": AT, "initial_cash": "1000"})
    book.run({"operation": "register", "now": AT, "instruments": [
        {"instrument_id": "SYN101", "kind": "EQUITY", "currency": "USD", "multiplier": 1}]})
    book.run({"operation": "reserve", "now": AT, "client_id": "stock", "instrument_id": "SYN101", "side": "buy",
        "quantity": "10", "limit_price": "10", "price_asof": AT, "fee_budget": "0.2"})
    book.run({"operation": "dispatch", "now": AT})
    result = book.run({"operation": "event", "now": AT, "event": {"event_id": "stock-fill", "client_id": "stock",
        "kind": "FILL", "occurred_at": AT, "quantity": "5" if partial else "10", "price": "10", "fee": "0.1" if partial else "0.2"}})
    if mark:
        result = book.run({"operation": "mark", "now": AT, "marks": {"SYN101": {"price": "10", "as_of": AT}}, "max_age_seconds": 60})
    return book, result.to_dict()["data"]["snapshot_id"]


class StatementReconciliationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.base = Path(self.temp.name)/"state"
        self.book, self.snapshot = funded_reference(self.base)
        self.workspace = StatementWorkspace(self.base)

    def tearDown(self):
        self.temp.cleanup()

    def request(self, statement=None, *, now=AT, workspace=None):
        workspace = workspace or self.workspace
        receipt = workspace.ingest(statement or supplied_statement()).to_dict()
        return {"statement_hash": receipt["data"]["statement_hash"], "snapshot_id": self.snapshot, "now": now}

    def oms_hash(self):
        path = self.base/"funded-account/oms.sqlite"
        with sqlite3.connect(path.as_uri()+"?mode=ro", uri=True) as db:
            return hashlib.sha256("\n".join(db.iterdump()).encode()).hexdigest()

    def test_literal_independent_cash_positions_orders_and_fills_match_without_vendor_claim(self):
        before = self.oms_hash()
        result = self.workspace.run(self.request()).to_dict()
        EnvelopeSchema.model_validate(result)
        self.assertEqual(result["state"], "AVAILABLE")
        self.assertEqual(result["data"]["reconciliation_state"], "MATCH")
        self.assertTrue(result["data"]["fixture_match"])
        self.assertEqual(result["data"]["rows"], [])
        self.assertFalse(result["data"]["external_reconciled"])
        self.assertFalse(result["data"]["fully_funded_certified"])
        self.assertEqual(result["provenance"]["classification"], "SYNTHETIC")
        self.assertEqual(self.oms_hash(), before)

    def test_discrepancies_duplicate_missing_and_unexpected_events_are_not_success(self):
        fixture = supplied_statement()
        fixture["balances"]["cash"] = "899.81"
        fixture["positions"][0]["quantity"] = "11"
        fixture["orders"][0]["acknowledged"] = False
        fixture["fills"].append(copy.deepcopy(fixture["fills"][0]))
        fixture["fills"].append({**fixture["fills"][0], "execution_id": "unexpected"})
        fixture["reservations"] = [{"client_id": "ghost", "instrument_id": "SYN101", "remaining_quantity": "1",
            "buy_cash": "10", "short_put_collateral": "0", "fee_cash": "0", "state": "ACKNOWLEDGED"}]
        data = self.workspace.run(self.request(fixture)).to_dict()["data"]
        self.assertEqual(data["reconciliation_state"], "DISCREPANCY")
        self.assertFalse(data["fixture_match"])
        codes = {r["code"] for r in data["rows"]}
        self.assertIn("DUPLICATE_STATEMENT_RECORD", codes)
        self.assertIn("UNEXPECTED_STATEMENT_RECORD", codes)
        self.assertIn("VALUE_MISMATCH", codes)
        missing = supplied_statement(fills=[])
        missing["available_at"] = (NOW+timedelta(seconds=1)).isoformat()
        data = self.workspace.run(self.request(missing, now=missing["available_at"])).to_dict()["data"]
        self.assertIn("MISSING_STATEMENT_RECORD", {r["code"] for r in data["rows"]})

    def test_exact_decimal_comparison_and_strict_instrument_identity(self):
        fixture = supplied_statement()
        fixture["balances"]["cash"] = "899.800000000000"
        fixture["positions"][0]["quantity"] = "10.000"
        self.assertEqual(self.workspace.run(self.request(fixture)).to_dict()["data"]["reconciliation_state"], "MATCH")
        fixture = supplied_statement(statement_id="wrong-symbol")
        fixture["orders"][0]["instrument_id"] = "1.0"
        data = self.workspace.run(self.request(fixture)).to_dict()["data"]
        self.assertEqual(data["reconciliation_state"], "DISCREPANCY")
        with self.assertRaises(ValueError):
            self.workspace.ingest(supplied_statement(balances={**supplied_statement()["balances"], "cash": 899.8}))

    def test_unknown_financial_metrics_and_clock_mismatch_remain_unknown(self):
        unknown = supplied_statement()
        unknown["balances"]["nav"] = None
        self.assertEqual(self.workspace.run(self.request(unknown)).to_dict()["data"]["reconciliation_state"], "UNKNOWN")
        wrong_clock = supplied_statement(statement_id="wrong-clock", observed_at="2026-11-03T15:00:00.123455999Z")
        data = self.workspace.run(self.request(wrong_clock)).to_dict()["data"]
        self.assertEqual(data["reconciliation_state"], "UNKNOWN")
        self.assertFalse(data["fixture_match"])
        self.assertIn("OBSERVATION_CLOCK_MISMATCH", {r["code"] for r in data["rows"]})

    def test_future_knowledge_does_not_leak_statement_data_into_earlier_decision(self):
        future = supplied_statement(available_at="2026-11-03T15:00:00.123456001Z")
        request = self.request(future)
        result = self.workspace.run(request).to_dict()
        self.assertEqual(result["state"], "UNKNOWN")
        self.assertIsNone(result["data"])
        self.assertFalse((self.base/"independent-statements/reports.sqlite").exists())
        result = self.workspace.run({**request, "now": future["available_at"]}).to_dict()
        self.assertEqual(result["data"]["reconciliation_state"], "MATCH")

    def test_partial_fill_fee_and_cash_reservations_reference(self):
        base = self.base/"partial"
        _, snapshot = funded_reference(base, partial=True)
        workspace = StatementWorkspace(base)
        fixture = supplied_statement()
        fixture["balances"].update(cash="949.9", nav="999.9", fees="0.1", reserved_cash="50.1", available_cash="899.8")
        fixture["positions"][0].update(quantity="5", cost_basis="50", market_value="50")
        fixture["orders"][0].update(filled="5", state="PARTIALLY_FILLED")
        fixture["fills"][0].update(quantity="5", fee="0.1")
        fixture["reservations"] = [{"client_id": "stock", "instrument_id": "SYN101", "remaining_quantity": "5",
            "buy_cash": "50", "short_put_collateral": "0", "fee_cash": "0.1", "state": "PARTIALLY_FILLED"}]
        request = self.request(fixture, workspace=workspace)
        request["snapshot_id"] = snapshot
        self.assertEqual(workspace.run(request).to_dict()["data"]["reconciliation_state"], "MATCH")

    def test_missing_engine_marks_never_pass_reconciliation_even_with_matching_cash(self):
        base = self.base/"unmarked"
        _, snapshot = funded_reference(base, mark=False)
        workspace = StatementWorkspace(base)
        request = self.request(workspace=workspace)
        request["snapshot_id"] = snapshot
        result = workspace.run(request).to_dict()["data"]
        self.assertEqual(result["reconciliation_state"], "UNKNOWN")
        self.assertIn("ENGINE_FINANCIAL_STATE_INCOMPLETE", {r["code"] for r in result["rows"]})

    def test_option_multiplier_premium_and_cash_expiry_reference(self):
        base = self.base/"option"
        book = FundedWorkspace(base)
        book.run({"operation": "initialize", "now": AT, "initial_cash": "1000"})
        terms = {"instrument_id": "OPT:SYN101:C100", "kind": "OPTION", "currency": "USD", "multiplier": 100,
            "underlying_id": "SYN101", "expiry": EXPIRY, "strike": "100", "right": "CALL", "style": "EUROPEAN", "settlement": "CASH"}
        book.run({"operation": "register", "now": AT, "instruments": [terms]})
        book.run({"operation": "reserve", "now": AT, "client_id": "option", "instrument_id": terms["instrument_id"],
            "side": "buy", "quantity": "1", "limit_price": "2.5", "price_asof": AT, "fee_budget": "0.2"})
        book.run({"operation": "dispatch", "now": AT})
        book.run({"operation": "event", "now": AT, "event": {"event_id": "option-fill", "client_id": "option", "kind": "FILL",
            "occurred_at": AT, "quantity": "1", "price": "2.5", "fee": "0.2"}})
        snapshot = book.run({"operation": "mark", "now": AT, "marks": {terms["instrument_id"]: {"price": "2.5", "as_of": AT}},
            "max_age_seconds": 60}).to_dict()["data"]["snapshot_id"]
        fixture = supplied_statement()
        fixture["balances"].update(cash="749.8", available_cash="749.8")
        fixture["positions"] = [{**terms, "quantity": "1", "cost_basis": "250", "market_value": "250", "unrealized_pnl": "0"}]
        fixture["orders"] = [{"client_id": "option", "instrument_id": terms["instrument_id"], "side": "buy", "quantity": "1",
            "filled": "1", "limit_price": "2.5", "state": "FILLED", "acknowledged": True, "created_at": AT, "updated_at": AT}]
        fixture["fills"] = [{"execution_id": "option-fill", "client_id": "option", "instrument_id": terms["instrument_id"],
            "side": "buy", "quantity": "1", "price": "2.5", "fee": "0.2", "occurred_at": AT}]
        workspace = StatementWorkspace(base)
        request = self.request(fixture, workspace=workspace)
        request["snapshot_id"] = snapshot
        self.assertEqual(workspace.run(request).to_dict()["data"]["reconciliation_state"], "MATCH")
        snapshot = book.run({"operation": "settle", "now": EXPIRY, "settlement_id": "cash-expiry", "instrument_id": terms["instrument_id"],
            "spot": "110", "observed_at": EXPIRY, "available_at": EXPIRY, "fee": "0.1"}).to_dict()["data"]["snapshot_id"]
        fixture["available_at"] = fixture["observed_at"] = EXPIRY
        fixture["positions"] = []
        fixture["balances"].update(cash="1749.7", nav="1749.7", available_cash="1749.7", realized_gross_pnl="750", fees="0.3")
        fixture["settlements"] = [{"settlement_id": "cash-expiry", "instrument_id": terms["instrument_id"], "position_quantity": "1",
            "spot": "110", "payoff": "10", "fee": "0.1", "observed_at": EXPIRY, "available_at": EXPIRY}]
        request = self.request(fixture, now=EXPIRY, workspace=workspace)
        request["snapshot_id"] = snapshot
        self.assertEqual(workspace.run(request).to_dict()["data"]["reconciliation_state"], "MATCH")

    def test_reopen_replay_detachment_and_later_statement_revision_preserve_earlier_result(self):
        request = self.request()
        result = self.workspace.run(request).to_dict()
        identity = result["data"]["report_id"]
        reopened = StatementWorkspace(self.base)
        self.assertEqual(reopened.replay({"report_id": identity}).to_dict(), result)
        loaded = reopened.load(identity)
        loaded["reconciliation_state"] = "DISCREPANCY"
        self.assertEqual(reopened.load(identity)["reconciliation_state"], "MATCH")
        revised = supplied_statement(available_at="2026-11-03T15:00:00.123456001Z")
        revised["balances"]["fees"] = "99"
        later = self.request(revised, now=revised["available_at"])
        self.assertEqual(reopened.run(later).to_dict()["data"]["reconciliation_state"], "DISCREPANCY")
        self.assertEqual(reopened.replay({"report_id": identity}).to_dict(), result)
        self.assertEqual(reopened.run(request).to_dict(), result)

    def test_lifecycle_coverage_is_required_for_enabled_obligations(self):
        statement = supplied_statement()
        snapshot = self.book.load(self.snapshot)
        snapshot["lifecycle_policy"] = {"settlement_policy": "EXPLICIT_EQUITY_DATES"}
        snapshot["account"].update(settled_cash="899.8", unsettled_sale_receivables="0", unsettled_buy_payables="0",
                                   settled_trade_positions={"SYN101": "10"})
        snapshot["trade_settlements"] = [{"execution_id": "stock-fill", "settlement_at": AT}]
        request = {"statement_hash": "0"*64, "snapshot_id": self.snapshot, "now": AT}
        unknown = StatementWorkspace._compare(statement, snapshot, request)
        self.assertEqual(unknown["reconciliation_state"], "UNKNOWN")
        self.assertFalse(unknown["coverage"]["equity_settlements"])
        statement["balances"].update(settled_cash="899.8", unsettled_sale_receivables="0", unsettled_buy_payables="0")
        statement["settled_trade_positions"] = {"SYN101": "10"}
        statement["trade_settlements"] = [{"execution_id": "stock-fill", "settlement_at": AT}]
        self.assertEqual(StatementWorkspace._compare(statement, snapshot, request)["reconciliation_state"], "MATCH")
        snapshot["lifecycle_events"] = [{"event_id": "div", "kind": "CASH_DIVIDEND", "instrument_id": "SYN101",
            "action_id": "issuer-1", "applied_at": AT, "effective_at": AT, "known_at": AT,
            "cash_amount": "1", "entitlement_quantity": "10", "amount_per_share": "0.1"}]
        self.assertEqual(StatementWorkspace._compare(statement, snapshot, request)["reconciliation_state"], "UNKNOWN")

    def test_known_unapplied_actions_cannot_be_silently_covered_by_empty_lifecycle(self):
        statement = supplied_statement()
        snapshot = self.book.load(self.snapshot)
        snapshot["pending_corporate_actions"] = [{"action_id": "future-dividend", "kind": "CASH_DIVIDEND",
            "instrument_id": "SYN101", "pay_at": EXPIRY, "amount_per_share": "0.1"}]
        request = {"statement_hash": "0"*64, "snapshot_id": self.snapshot, "now": AT}
        result = StatementWorkspace._compare(statement, snapshot, request)
        self.assertEqual(result["reconciliation_state"], "UNKNOWN")
        self.assertFalse(result["coverage"]["corporate_actions"])
        self.assertFalse(result["fixture_match"])
        self.assertIn("PENDING_CORPORATE_ACTION_COVERAGE_UNKNOWN", {r["code"] for r in result["rows"]})

    def test_unreconciled_engine_cannot_match_an_independent_statement(self):
        snapshot = self.book.load(self.snapshot)
        snapshot["account"]["reconciliation_state"] = "UNKNOWN"
        request = {"statement_hash": "0"*64, "snapshot_id": self.snapshot, "now": AT}
        result = StatementWorkspace._compare(supplied_statement(), snapshot, request)
        self.assertEqual(result["reconciliation_state"], "UNKNOWN")
        self.assertIn("ENGINE_FINANCIAL_STATE_INCOMPLETE", {r["code"] for r in result["rows"]})

    def test_fault_injection_preserves_unindexed_artifact_without_admitting_report(self):
        request = self.request()
        before = self.oms_hash()
        for phase in ("report_written", "report_indexed"):
            def fail(name, target=phase):
                if name == target:
                    raise RuntimeError("injected interruption")
            with self.subTest(phase=phase), self.assertRaises(RuntimeError):
                StatementWorkspace(self.base, fault_hook=fail).run(request)
            with sqlite3.connect(self.base/"independent-statements/reports.sqlite") as db:
                if db.execute("SELECT 1 FROM sqlite_master WHERE name='reports'").fetchone():
                    self.assertEqual(db.execute("SELECT COUNT(*) FROM reports").fetchone()[0], 0)
            self.assertIsNone(self.workspace.section("risk", AT, 60))
        self.assertEqual(self.oms_hash(), before)
        self.assertEqual(self.workspace.run(request).to_dict()["data"]["reconciliation_state"], "MATCH")

    def test_source_index_tampering_and_conflicting_revisions_fail_closed(self):
        request = self.request()
        result = self.workspace.run(request).to_dict()
        conflict = supplied_statement()
        conflict["balances"]["cash"] = "0"
        with self.assertRaisesRegex(ValueError, "conflict"):
            self.workspace.ingest(conflict)
        with sqlite3.connect(self.base/"independent-statements/reports.sqlite") as db:
            db.execute("UPDATE reports SET as_of='2020-01-01T00:00:00Z'")
        with self.assertRaisesRegex(ValueError, "index"):
            self.workspace.load(result["data"]["report_id"])
        source = self.base/"independent-statements/sources"/(request["statement_hash"]+".json")
        source.write_text("{}")
        with self.assertRaisesRegex(ValueError, "integrity"):
            self.workspace.run(request)

    def test_empty_reads_missing_ids_freshness_and_path_bounds(self):
        workspace = StatementWorkspace(self.base/"missing")
        self.assertIsNone(workspace.section("operations", AT, 60))
        self.assertEqual(workspace.replay({"report_id": "0"*64}).to_dict()["state"], "UNAVAILABLE")
        self.assertFalse((self.base/"missing").exists())
        result = self.workspace.run(self.request()).to_dict()
        self.assertEqual(self.workspace.section("risk", AT, None).to_dict()["state"], "UNKNOWN")
        self.assertEqual(self.workspace.section("risk", (NOW+timedelta(seconds=61)).isoformat(), 60).to_dict()["state"], "STALE")
        self.assertIsNone(self.workspace.section("options", AT, 60))
        with self.assertRaises(ValueError):
            self.workspace.load("../outside")
        self.assertEqual(self.workspace.replay({"report_id": result["data"]["report_id"]}).to_dict(), result)
