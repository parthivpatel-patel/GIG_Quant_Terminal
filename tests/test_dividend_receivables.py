"""Literal cash/receivable references, corrections, atomicity and supplier coverage."""
from __future__ import annotations

import copy
import json
import os
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from decimal import Decimal, localcontext
from pathlib import Path
from threading import Barrier

from gig.service.funded_workspace import FundedWorkspace
from gig.service.lifecycle_workspace import FundedLifecycleWorkspace, _hash
from gig.service.statement_workspace import StatementWorkspace
from test_funded_accounting import NOW, reserve_request, seed_funded
from test_statement_reconciliation import NOW as SUPPLIER_NOW
from test_statement_reconciliation import funded_reference, supplied_statement

EX = NOW+timedelta(days=1)
PAY = NOW+timedelta(days=2)


def action(*, revision=1, amount="1", known=NOW, status=None, symbol="A", ex=EX, pay=PAY, record=None):
    value = {"action_id": "authored-dividend", "revision": revision, "instrument_id": symbol,
             "source": "AUTHORED_INDEPENDENT_DIVIDEND", "kind": "CASH_DIVIDEND", "known_at": known.isoformat(),
             "effective_at": ex.isoformat(), "record_at": (record or ex).isoformat(), "pay_at": pay.isoformat(),
             "amount_per_share": amount}
    if status is not None:
        value["status"] = status
    return value


class DividendReceivableTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.base = Path(self.temp.name)/"state"
        self.book = seed_funded(self.base)
        self.lifecycle = FundedLifecycleWorkspace(self.base)
        self.lifecycle.run({"operation": "configure", "now": NOW.isoformat(), "settlement_policy": "IMMEDIATE"})
        self.fill("stock", "10", "10", NOW, fee="0.2")
        self.mark("10", NOW)
        self.identity = self.admit(action(), NOW)

    def tearDown(self):
        self.temp.cleanup()

    def fill(self, client, quantity, price, at, *, fee="0", occurred=None):
        self.book.run(reserve_request(client, "A", now=at, quantity=quantity, limit_price=price, fee_budget=fee))
        self.book.run({"operation": "dispatch", "now": at.isoformat()})
        payload = {"operation": "event", "now": at.isoformat(), "event": {"event_id": "fill-"+client,
            "client_id": client, "kind": "FILL", "occurred_at": (occurred or at).isoformat(),
            "quantity": quantity, "price": price, "fee": fee}}
        return self.book.run(payload).to_dict()["data"]

    def admit(self, value, at):
        return self.lifecycle.run({"operation": "admit", "now": at.isoformat(), "action": value}).to_dict()["data"]["action_hash"]

    def mark(self, price, at):
        return self.book.run({"operation": "mark", "now": at.isoformat(),
            "marks": {"A": {"price": price, "as_of": at.isoformat()}}, "max_age_seconds": 60}).to_dict()["data"]

    def apply(self, identity, at, operation="apply"):
        return self.lifecycle.run({"operation": operation, "now": at.isoformat(), "action_hash": identity}).to_dict()["data"]

    def counts(self):
        with sqlite3.connect(self.base/"funded-account/oms.sqlite") as db:
            return tuple(db.execute("SELECT COUNT(*) FROM "+table).fetchone()[0] for table in
                         ("funded_lifecycle_events", "funded_dividend_accruals", "funded_lifecycle_applied"))

    def test_ex_date_income_and_pay_date_conversion_conserve_nav_without_funding_future_cash(self):
        before = self.book.run({"operation": "state", "now": NOW.isoformat()}).to_dict()["data"]
        ex = self.mark("9", EX)
        account = ex["account"]
        self.assertEqual((account["cash"], account["dividend_receivable"], account["dividend_income"], account["nav"], account["total_pnl"]),
                         ("899.8", "10", "10", "999.8", "-0.2"))
        self.assertEqual(account["available_cash"], "899.8")
        self.assertFalse(account["fully_funded_certified"])
        with self.assertRaises(ValueError):
            self.book.run(reserve_request("future-income", "A", now=EX, quantity="100", limit_price="9", fee_budget="0"))
        self.mark("9", PAY)
        paid = self.apply(self.identity, PAY)
        self.assertEqual((paid["account"]["cash"], paid["account"]["dividend_receivable"], paid["account"]["dividend_income"], paid["account"]["nav"]),
                         ("909.8", "0", "10", "999.8"))
        self.assertEqual(self.apply(self.identity, PAY)["account"], paid["account"])
        self.assertEqual(self.counts(), (2, 1, 1))
        replay = FundedWorkspace(self.base).run({"operation": "replay", "now": PAY.isoformat(), "snapshot_id": ex["snapshot_id"]}).to_dict()["data"]
        self.assertEqual(replay, ex)
        self.assertEqual(self.book.load(before["snapshot_id"])["account"]["dividend_receivable"], "0")

    def test_revision_and_cancellation_before_payment_are_only_compensating_income_deltas(self):
        original = self.mark("9", EX)
        revised_at = EX+timedelta(seconds=1)
        second = self.admit(action(revision=2, amount="1.5", known=revised_at), revised_at)
        revised = self.apply(second, revised_at, "accrue")
        self.assertEqual((revised["account"]["cash"], revised["account"]["dividend_receivable"], revised["account"]["dividend_income"], revised["account"]["nav"]),
                         ("899.8", "15.0", "15.0", "1004.8"))
        self.assertNotEqual(original["source_identity"], revised["source_identity"])
        again = self.book.run({"operation": "state", "now": revised_at.isoformat()}).to_dict()["data"]
        self.assertEqual(again["source_identity"], revised["source_identity"])
        cancel_at = revised_at+timedelta(seconds=1)
        cancelled = self.admit(action(revision=3, amount="0", known=cancel_at, status="CANCELLED"), cancel_at)
        final = self.apply(cancelled, PAY)
        self.assertEqual((Decimal(final["account"]["cash"]), Decimal(final["account"]["dividend_receivable"]), Decimal(final["account"]["dividend_income"])),
                         (Decimal("899.8"), Decimal(0), Decimal(0)))
        self.assertEqual(self.book.load(original["snapshot_id"])["account"]["dividend_income"], "10")

    def test_paid_cancellation_claws_back_cash_once_without_negative_receivables(self):
        self.mark("9", EX)
        self.mark("9", PAY)
        self.apply(self.identity, PAY)
        at = PAY+timedelta(seconds=1)
        identity = self.admit(action(revision=2, amount="0", known=at, status="CANCELLED"), at)
        result = self.apply(identity, at)
        self.assertEqual((result["account"]["cash"], result["account"]["dividend_receivable"], result["account"]["dividend_income"]), ("899.8", "0", "0"))
        self.assertEqual(self.apply(identity, at)["account"], result["account"])

    def test_unfundable_paid_correction_retains_source_and_unknown_obligation_without_borrowing(self):
        self.mark("9", EX)
        self.mark("10", PAY)
        self.apply(self.identity, PAY)
        at = PAY+timedelta(seconds=1)
        self.fill("spent", "90", "10", at)
        cancel_at = at+timedelta(seconds=1)
        identity = self.admit(action(revision=2, amount="0", known=cancel_at, status="CANCELLED"), cancel_at)
        current = self.book.run({"operation": "state", "now": cancel_at.isoformat()}).to_dict()["data"]
        self.assertEqual(current["account"]["cash"], "9.8")
        self.assertEqual(current["account"]["funding_state"], "UNKNOWN")
        self.assertIsNone(current["account"]["nav"])
        self.assertEqual(current["dividend_receivables"][0]["revision"], 1)
        self.assertEqual(current["dividend_receivables"][0]["known_at"], NOW.isoformat())
        before = self.counts()
        with self.assertRaisesRegex(ValueError, "financing"):
            self.apply(identity, cancel_at, "accrue")
        self.assertEqual(self.counts(), before)

    def test_record_date_future_settlement_does_not_supply_early_entitlement(self):
        base = self.base/"dated"
        book = seed_funded(base)
        lifecycle = FundedLifecycleWorkspace(base)
        lifecycle.run({"operation": "configure", "now": NOW.isoformat(), "settlement_policy": "EXPLICIT_EQUITY_DATES"})
        book.run(reserve_request("dated", "A", quantity="10", limit_price="10", fee_budget="0"))
        book.run({"operation": "dispatch", "now": NOW.isoformat()})
        record = EX+timedelta(hours=12)
        book.run({"operation": "event", "now": NOW.isoformat(), "settlement_at": record.isoformat(), "event": {
            "event_id": "dated-fill", "client_id": "dated", "kind": "FILL", "occurred_at": NOW.isoformat(), "quantity": "10", "price": "10", "fee": "0"}})
        identity = lifecycle.run({"operation": "admit", "now": NOW.isoformat(), "action": action(record=record)}).to_dict()["data"]["action_hash"]
        early = book.run({"operation": "state", "now": EX.isoformat()}).to_dict()["data"]
        self.assertEqual(early["account"]["dividend_receivable"], "0")
        self.assertEqual(early["account"]["dividend_accrual_state"], "UNKNOWN")
        with self.assertRaisesRegex(ValueError, "future evidence"):
            lifecycle.run({"operation": "accrue", "now": EX.isoformat(), "action_hash": identity})
        reached = lifecycle.run({"operation": "accrue", "now": record.isoformat(), "action_hash": identity}).to_dict()["data"]
        self.assertEqual(reached["account"]["dividend_receivable"], "10")

    def test_late_pre_ex_fill_cannot_rewrite_frozen_entitlement(self):
        self.mark("9", EX)
        at = EX+timedelta(seconds=1)
        with self.assertRaisesRegex(ValueError, "frozen dividend"):
            self.fill("late", "1", "9", at, occurred=NOW)
        self.assertEqual(self.book.run({"operation": "state", "now": at.isoformat()}).to_dict()["data"]["account"]["dividend_income"], "10")

    def test_duplicate_concurrent_recognition_and_payment_have_single_economic_effect(self):
        barrier = Barrier(2)
        def accrue(_):
            barrier.wait(timeout=10)
            return FundedLifecycleWorkspace(self.base).run({"operation": "accrue", "now": EX.isoformat(), "action_hash": self.identity}).to_dict()["data"]
        with ThreadPoolExecutor(max_workers=2) as executor:
            result = list(executor.map(accrue, range(2)))
        self.assertEqual([r["account"]["dividend_receivable"] for r in result], ["10", "10"])
        self.assertEqual(self.counts(), (1, 1, 0))

    def test_accrual_and_payment_failure_rollback_and_process_restart(self):
        def fail(_):
            raise RuntimeError("injected dividend checkpoint failure")
        with self.assertRaises(RuntimeError):
            FundedLifecycleWorkspace(self.base, fault_hook=fail).run({"operation": "accrue", "now": EX.isoformat(), "action_hash": self.identity})
        self.assertEqual(self.counts(), (0, 0, 0))
        payload = {"operation": "accrue", "now": EX.isoformat(), "action_hash": self.identity}
        code = ("import os,sys,json;from pathlib import Path;from gig.service.lifecycle_workspace import FundedLifecycleWorkspace;"
                "FundedLifecycleWorkspace(Path(sys.argv[1]),fault_hook=lambda _:os._exit(67)).run(json.loads(sys.argv[2]))")
        result = subprocess.run([sys.executable, "-c", code, str(self.base), json.dumps(payload)], env=os.environ.copy(), capture_output=True, timeout=30)
        self.assertEqual(result.returncode, 67, result.stderr.decode())
        self.assertEqual(self.counts(), (0, 0, 0))
        self.apply(self.identity, EX, "accrue")
        with self.assertRaises(RuntimeError):
            FundedLifecycleWorkspace(self.base, fault_hook=fail).run({"operation": "apply", "now": PAY.isoformat(), "action_hash": self.identity})
        self.assertEqual(self.counts(), (1, 1, 0))
        self.assertEqual(self.apply(self.identity, PAY)["account"]["cash"], "909.8")

    def test_exact_precision_and_invalid_clock_reject_before_mutation(self):
        with self.assertRaises(ValueError):
            self.lifecycle.run({"operation": "accrue", "now": "2026-11-03T15:00:00.000000001Z", "action_hash": self.identity})
        self.assertEqual(self.counts(), (0, 0, 0))
        with localcontext() as context:
            context.prec = 6
            result = self.mark("9", EX)
        self.assertEqual(result["account"]["nav"], "999.8")
        invalid = action(revision=2, amount="1", known=EX, status="CANCELLED")
        with self.assertRaises(ValueError):
            self.admit(invalid, EX)

    def test_accrual_and_paid_index_tampering_fail_closed(self):
        self.mark("9", EX)
        self.apply(self.identity, PAY)
        with sqlite3.connect(self.base/"funded-account/oms.sqlite") as db:
            db.execute("UPDATE funded_lifecycle_applied SET total_cash='999'")
        with self.assertRaisesRegex(ValueError, "paid index"):
            self.book.run({"operation": "state", "now": PAY.isoformat()})

    def test_archive_labels_cannot_be_promoted_to_funded_equity(self):
        with self.assertRaises(ValueError):
            self.book.run({"operation": "register", "now": NOW.isoformat(), "instruments": [
                {"instrument_id": "DJIA_MEMBER_LABEL:IBM", "kind": "EQUITY", "currency": "USD", "multiplier": 1}]})

    def test_legacy_paid_income_migrates_without_recognition_or_cash_duplication(self):
        self.mark("9", EX)
        paid = self.apply(self.identity, PAY)
        # Construct the documented pre-Sprint8 paid-event schema as a literal
        # legacy fixture. Existing snapshots remain untouched and still replay.
        from gig.service.serialization import dumps, loads
        with sqlite3.connect(self.base/"funded-account/oms.sqlite") as db:
            row = db.execute("SELECT event_id,wire FROM funded_lifecycle_events WHERE json_extract(wire,'$.kind')='CASH_DIVIDEND'").fetchone()
            legacy = loads(row[1])
            for key in ("income_amount", "receivable_amount", "status"):
                legacy.pop(key)
            db.execute("DELETE FROM funded_lifecycle_events WHERE json_extract(wire,'$.kind')='DIVIDEND_ACCRUAL'")
            db.execute("DROP TABLE funded_dividend_accruals")
            db.execute("UPDATE funded_lifecycle_events SET wire=? WHERE event_id=?", (dumps(legacy), row[0]))
            (self.base/"funded-account/lifecycle-events"/(row[0]+".json")).write_bytes(dumps(legacy).encode())
        unchanged = self.book.run({"operation": "state", "now": PAY.isoformat()}).to_dict()["data"]
        self.assertEqual((unchanged["account"]["cash"], unchanged["account"]["dividend_income"], unchanged["account"]["dividend_receivable"]), ("909.8", "10", "0"))
        at = PAY+timedelta(seconds=1)
        second = self.admit(action(revision=2, amount="1.5", known=at), at)
        revised = self.apply(second, at)
        self.assertEqual((revised["account"]["cash"], revised["account"]["dividend_income"], revised["account"]["dividend_receivable"]), ("914.8", "15.0", "0.0"))
        self.assertEqual(self.book.load(paid["snapshot_id"])["account"]["cash"], "909.8")

    def test_late_split_correction_cannot_rewrite_recognized_dividend_quantity(self):
        self.mark("9", EX)
        at = EX+timedelta(seconds=1)
        split = {"action_id": "late-split", "revision": 1, "instrument_id": "A", "source": "AUTHORED",
            "kind": "SPLIT", "known_at": at.isoformat(), "effective_at": NOW.isoformat(), "record_at": NOW.isoformat(),
            "pay_at": NOW.isoformat(), "ratio_new": 2, "ratio_old": 1}
        identity = self.admit(split, at)
        before = self.counts()
        with self.assertRaisesRegex(ValueError, "frozen dividend"):
            self.apply(identity, at)
        self.assertEqual(self.counts(), before)


class DividendStatementTests(unittest.TestCase):
    def test_literal_accrued_receivable_statement_matches_and_missing_or_wrong_amount_does_not(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)/"state"
            book, _ = funded_reference(base)
            lifecycle = FundedLifecycleWorkspace(base)
            ex, pay = SUPPLIER_NOW+timedelta(days=1), SUPPLIER_NOW+timedelta(days=2)
            source = action(symbol="SYN101", known=SUPPLIER_NOW, ex=ex, pay=pay)
            identity = lifecycle.run({"operation": "admit", "now": SUPPLIER_NOW.isoformat(), "action": source}).to_dict()["data"]["action_hash"]
            saved = book.run({"operation": "mark", "now": ex.isoformat(), "marks": {"SYN101": {"price": "9", "as_of": ex.isoformat()}}, "max_age_seconds": 60}).to_dict()["data"]
            fixture = supplied_statement()
            fixture.update(observed_at=ex.isoformat(), available_at=ex.isoformat())
            fixture["balances"].update(nav="999.8", unrealized_pnl="-10", dividend_income="10", dividend_receivable="10")
            fixture["positions"][0].update(market_value="90", unrealized_pnl="-10")
            fixture["lifecycle_events"] = [{"event_id": _hash([source["action_id"], identity, "ACCRUAL"]), "kind": "DIVIDEND_ACCRUAL",
                "instrument_id": "SYN101", "action_id": source["action_id"], "applied_at": ex.isoformat(), "effective_at": ex.isoformat(),
                "known_at": SUPPLIER_NOW.isoformat(), "record_at": ex.isoformat(), "pay_at": pay.isoformat(),
                "cash_amount": "0", "entitlement_quantity": "10", "amount_per_share": "1", "income_amount": "10",
                "receivable_amount": "10", "total_income": "10", "status": "ACTIVE"}]
            fixture["dividend_receivables"] = [{"action_id": source["action_id"], "revision": 1, "instrument_id": "SYN101",
                "source": source["source"], "known_at": SUPPLIER_NOW.isoformat(), "effective_at": ex.isoformat(), "record_at": ex.isoformat(),
                "pay_at": pay.isoformat(), "entitlement_quantity": "10", "recognized_income": "10", "paid_cash": "0",
                "outstanding_receivable": "10", "status": "ACTIVE"}]
            workspace = StatementWorkspace(base)
            def compare(value):
                admitted = workspace.ingest(value).to_dict()["data"]
                return workspace.run({"statement_hash": admitted["statement_hash"], "snapshot_id": saved["snapshot_id"], "now": ex.isoformat()}).to_dict()
            result = compare(fixture)
            self.assertEqual(result["data"]["reconciliation_state"], "MATCH")
            self.assertFalse(result["data"]["external_reconciled"])
            bad = copy.deepcopy(fixture)
            bad["statement_id"] = "wrong-receivable"
            bad["balances"]["dividend_receivable"] = "0"
            self.assertEqual(compare(bad)["data"]["reconciliation_state"], "DISCREPANCY")
            missing = copy.deepcopy(fixture)
            missing["statement_id"] = "missing-receivable"
            missing.pop("dividend_receivables")
            self.assertEqual(compare(missing)["data"]["reconciliation_state"], "UNKNOWN")
            replay = StatementWorkspace(base).replay({"report_id": result["data"]["report_id"]}).to_dict()
            self.assertEqual(result, replay)
