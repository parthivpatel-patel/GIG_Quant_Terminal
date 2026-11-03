"""Exact premium accounting and real SQLite interruption recovery, offline only."""
from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from decimal import Decimal, localcontext
from pathlib import Path
from threading import Barrier, local
from unittest.mock import patch

from gig.execution.oms import OfflineOMS
from gig.service.contracts import EnvelopeSchema
from gig.service.funded_workspace import FundedWorkspace

NOW = datetime(2026, 11, 2, 15, tzinfo=UTC)
EXPIRY = NOW + timedelta(days=1)


def funded_instruments():
    return [{"instrument_id": s, "kind": "EQUITY", "currency": "USD", "multiplier": 1} for s in ("A", "B")] + [
        {"instrument_id": s, "kind": "OPTION", "currency": "USD", "multiplier": 100,
         "underlying_id": u, "expiry": EXPIRY.isoformat(), "strike": "100", "right": right,
         "style": "EUROPEAN", "settlement": "CASH"} for s, u, right in (("AC", "A", "CALL"), ("BP", "B", "PUT"))]


def reserve_request(client="buy", instrument="AC", *, now=NOW, **changes):
    return {"operation": "reserve", "now": now.isoformat(), "client_id": client,
            "instrument_id": instrument, "side": "buy", "quantity": "2", "limit_price": "3",
            "price_asof": now.isoformat(), "fee_budget": "2", **changes}


def seed_funded(base, cash="1000"):
    workspace = FundedWorkspace(base)
    workspace.run({"operation": "initialize", "now": NOW.isoformat(), "initial_cash": cash})
    workspace.run({"operation": "register", "now": NOW.isoformat(), "instruments": funded_instruments()})
    return workspace


def funded_fill(workspace, client="buy", event_id="fill-one", quantity="1", price="2.5", fee="0.1", now=NOW):
    return workspace.run({"operation": "event", "now": now.isoformat(), "event": {
        "event_id": event_id, "client_id": client, "kind": "FILL", "occurred_at": now.isoformat(),
        "quantity": quantity, "price": price, "fee": fee}})


class FundedAccountingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.base = Path(self.temp.name)/"state"
        self.workspace = seed_funded(self.base)

    def tearDown(self):
        self.temp.cleanup()

    def decode(self, result):
        result = result.to_dict()
        EnvelopeSchema.model_validate(result)
        return result

    def state(self, now=NOW):
        return self.decode(self.workspace.run({"operation": "state", "now": now.isoformat()}))["data"]

    def prepare(self, **changes):
        self.workspace.run(reserve_request(**changes))
        return self.decode(self.workspace.run({"operation": "dispatch", "now": NOW.isoformat()}))

    def test_concurrent_duplicate_intents_recheck_receipt_inside_writer_lock(self):
        barrier, visited = Barrier(2), local()
        original = FundedWorkspace._config
        def gate(oms, name, default=None):
            if name == "clock" and not getattr(visited, "waiting_done", False):
                visited.waiting_done = True
                barrier.wait(timeout=10)
            return original(oms, name, default)
        def submit(_):
            return FundedWorkspace(self.base).run(reserve_request()).to_dict()
        with patch.object(FundedWorkspace, "_config", new=staticmethod(gate)), ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(submit, range(2)))
        self.assertEqual(results[0], results[1])
        with OfflineOMS(self.base/"funded-account/oms.sqlite") as oms:
            self.assertEqual(oms.connection.execute("SELECT COUNT(*) FROM orders").fetchone()[0], 1)

    def mark(self, marks, now=NOW, maximum=60):
        return self.decode(self.workspace.run({"operation": "mark", "now": now.isoformat(),
            "marks": {key: {"price": value, "as_of": now.isoformat()} for key, value in marks.items()},
            "max_age_seconds": maximum}))

    def test_premium_fees_partial_reservations_and_independent_cash_reference(self):
        reserved = self.decode(self.workspace.run(reserve_request()))["data"]["account"]
        self.assertEqual(Decimal(reserved["cash"]), 1000)
        self.assertEqual(Decimal(reserved["reserved_cash"]), 602)
        self.assertEqual(Decimal(reserved["available_cash"]), 398)
        ack = self.decode(self.workspace.run({"operation": "dispatch", "now": NOW.isoformat()}))
        self.assertEqual(ack["data"]["account"]["effective_positions"], {})
        filled = self.decode(funded_fill(self.workspace))["data"]
        self.assertEqual(Decimal(filled["account"]["cash"]), Decimal("749.9"))
        self.assertEqual(Decimal(filled["account"]["reserved_cash"]), Decimal("301.9"))
        self.assertIsNone(filled["account"]["nav"])
        self.assertEqual(filled["account_state"], "UNKNOWN")
        self.assertEqual(self.workspace.section("portfolio", NOW, 60).to_dict()["state"], "UNKNOWN")
        marked = self.mark({"AC": "3"})["data"]["account"]
        self.assertEqual(Decimal(marked["nav"]), Decimal("1049.9"))
        self.assertEqual(Decimal(marked["total_pnl"]), Decimal("49.9"))
        self.assertFalse(marked["fully_funded_certified"])

    def test_fifo_sale_and_long_cash_exercise_conserve_premium_and_realized_pnl(self):
        self.prepare()
        funded_fill(self.workspace)
        funded_fill(self.workspace, event_id="fill-two", price="2", fee="0.2")
        self.workspace.run(reserve_request("sale", side="sell", quantity="1", limit_price="4", fee_budget="1"))
        self.workspace.run({"operation": "dispatch", "now": NOW.isoformat()})
        funded_fill(self.workspace, client="sale", event_id="sale-one", price="4")
        account = self.mark({"AC": "3"})["data"]["account"]
        self.assertEqual(Decimal(account["cash"]), Decimal("949.6"))
        self.assertEqual(Decimal(account["nav"]), Decimal("1249.6"))
        self.assertEqual(Decimal(account["realized_gross_pnl"]), 150)
        settlement = {"operation": "settle", "now": EXPIRY.isoformat(), "settlement_id": "expiry-ac",
            "instrument_id": "AC", "spot": "110", "observed_at": EXPIRY.isoformat(),
            "available_at": EXPIRY.isoformat(), "fee": "0.1"}
        result = self.decode(self.workspace.run(settlement))
        account = result["data"]["account"]
        self.assertEqual(Decimal(account["cash"]), Decimal("1949.5"))
        self.assertEqual(Decimal(account["realized_gross_pnl"]), 950)
        self.assertEqual(Decimal(account["nav"]), Decimal(account["initial_cash"])+Decimal(account["total_pnl"]))
        self.assertEqual(account["effective_positions"]["AC"], "0")
        self.assertEqual(account["raw_oms_positions"]["AC"], "1")
        self.assertEqual(result, self.decode(FundedWorkspace(self.base).run(settlement)))

    def test_multi_underlying_equity_and_contract_pnl_conservation(self):
        workspace = seed_funded(Path(self.temp.name)/"multi", "10000")
        for client, symbol, quantity, price in (("a", "A", "2", "100"), ("b", "B", "1", "50"), ("option", "AC", "1", "2.5")):
            workspace.run(reserve_request(client, symbol, quantity=quantity, limit_price=price))
            workspace.run({"operation": "dispatch", "now": NOW.isoformat()})
            funded_fill(workspace, client=client, event_id="fill-"+client, quantity=quantity, price=price)
        result = workspace.run({"operation": "mark", "now": NOW.isoformat(), "max_age_seconds": 60,
            "marks": {key: {"price": price, "as_of": NOW.isoformat()} for key, price in {"A": "110", "B": "45", "AC": "3"}.items()}}).to_dict()["data"]["account"]
        self.assertEqual(Decimal(result["cash"]), Decimal("9499.7"))
        self.assertEqual(Decimal(result["nav"]), Decimal("10064.7"))
        self.assertEqual(Decimal(result["unrealized_pnl"]), 65)
        self.assertEqual(result["external_counterparty_reconciliation"], "UNKNOWN")

    def test_no_short_put_call_equity_or_unfunded_concurrent_orders(self):
        for instrument in ("A", "AC", "BP"):
            with self.subTest(instrument=instrument), self.assertRaises(ValueError):
                self.workspace.run(reserve_request("short-"+instrument, instrument, side="sell", quantity="1"))
        self.workspace.run(reserve_request())
        with self.assertRaises(ValueError):
            self.workspace.run(reserve_request("second"))
        with OfflineOMS(self.base/"funded-account/oms.sqlite") as oms:
            self.assertEqual(oms.connection.execute("SELECT COUNT(*) FROM orders").fetchone()[0], 1)
        self.assertEqual(Decimal(self.state()["account"]["reserved_cash"]), 602)

    def test_unsupported_contract_terms_and_capital_reset_are_rejected(self):
        for changes in ({"style": "AMERICAN"}, {"settlement": "PHYSICAL"}, {"multiplier": True}, {"currency": "EUR"}):
            item = {**funded_instruments()[2], "instrument_id": "invalid", **changes}
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.workspace.run({"operation": "register", "now": NOW.isoformat(), "instruments": [item]})
        with self.assertRaises(ValueError):
            self.workspace.run({"operation": "initialize", "now": NOW.isoformat(), "initial_cash": "2000"})
        with self.assertRaises(ValueError):
            self.workspace.run(reserve_request(quantity="1.5"))
        self.assertEqual(self.state()["account"]["cash"], "1000")

    def test_duplicate_fill_id_is_idempotent_and_conflict_preserves_original(self):
        self.prepare()
        first = self.decode(funded_fill(self.workspace))
        self.assertEqual(first, self.decode(funded_fill(self.workspace)))
        with self.assertRaises(ValueError):
            funded_fill(self.workspace, price="2.6")
        with OfflineOMS(self.base/"funded-account/oms.sqlite") as oms:
            self.assertEqual(oms.connection.execute("SELECT COUNT(*) FROM executions").fetchone()[0], 1)
        self.assertEqual(self.state()["account"]["cash"], "749.9")

    def test_existing_oms_limit_violation_is_quarantined_and_kills_without_cash_fill(self):
        self.prepare()
        bad = self.decode(funded_fill(self.workspace, price="4"))
        self.assertEqual(bad["state"], "ERROR")
        with OfflineOMS(self.base/"funded-account/oms.sqlite") as oms:
            self.assertTrue(oms.kill_switch)
            self.assertEqual(oms.connection.execute("SELECT COUNT(*) FROM executions").fetchone()[0], 0)
        self.assertEqual(self.state()["account"]["cash"], "1000")

    def test_fees_above_reserved_cash_and_fill_before_dispatch_are_rejected(self):
        self.workspace.run(reserve_request())
        with self.assertRaises(ValueError):
            funded_fill(self.workspace)
        self.workspace.run({"operation": "dispatch", "now": NOW.isoformat()})
        with self.assertRaises(ValueError):
            funded_fill(self.workspace, fee="3")
        self.assertEqual(self.state()["account"]["cash"], "1000")

    def test_precision_and_clock_boundaries_no_silent_truncation(self):
        missing = Path(self.temp.name)/"missing"
        with self.assertRaises(ValueError):
            FundedWorkspace(missing).run({"operation": "initialize", "now": "2026-11-02T15:00:00.000000001Z", "initial_cash": "1000"})
        self.assertFalse(missing.exists())
        with self.assertRaises(ValueError):
            self.workspace.run(reserve_request(limit_price=2.5))
        self.workspace.run({"operation": "state", "now": (NOW+timedelta(seconds=1)).isoformat()})
        with self.assertRaises(ValueError):
            self.workspace.run({"operation": "register", "now": NOW.isoformat(), "instruments": [funded_instruments()[0]]})

    def test_expiry_boundary_pending_obligation_otm_expiry_and_no_late_revision(self):
        self.prepare(quantity="1")
        funded_fill(self.workspace)
        pending = self.state(EXPIRY)["account"]
        self.assertEqual(pending["pending_expiries"], ["AC"])
        self.assertEqual(pending["funding_state"], "UNKNOWN")
        self.assertIsNone(pending["nav"])
        with self.assertRaises(ValueError):
            self.workspace.run(reserve_request("after", "A", now=EXPIRY))
        settle = {"operation": "settle", "now": EXPIRY.isoformat(), "settlement_id": "expiry-otm", "instrument_id": "AC",
            "spot": "90", "observed_at": EXPIRY.isoformat(), "available_at": EXPIRY.isoformat(), "fee": "0"}
        result = self.decode(self.workspace.run(settle))["data"]["account"]
        self.assertEqual(result["effective_positions"]["AC"], "0")
        self.assertEqual(Decimal(result["nav"]), Decimal("749.9"))
        with self.assertRaises(ValueError):
            self.workspace.run({**settle, "spot": "110"})
        with self.assertRaises(ValueError):
            funded_fill(self.workspace, event_id="late", now=EXPIRY)

    def test_settlement_original_expiry_and_knowledge_required(self):
        self.prepare(quantity="1")
        funded_fill(self.workspace)
        settle = {"operation": "settle", "now": EXPIRY.isoformat(), "settlement_id": "bad", "instrument_id": "AC",
            "spot": "110", "observed_at": (EXPIRY-timedelta(seconds=1)).isoformat(), "available_at": EXPIRY.isoformat(), "fee": "0"}
        with self.assertRaises(ValueError):
            self.workspace.run(settle)
        with self.assertRaises(ValueError):
            self.workspace.run({**settle, "observed_at": EXPIRY.isoformat(), "available_at": (EXPIRY+timedelta(seconds=1)).isoformat()})
        with self.assertRaisesRegex(ValueError, "borrowed cash"):
            self.workspace.run({**settle, "observed_at": EXPIRY.isoformat(), "spot": "90", "fee": "1000"})

    def test_repeated_same_clock_state_mark_and_dispatch_observe_new_book(self):
        before = self.state()
        self.prepare(quantity="1")
        funded_fill(self.workspace)
        after = self.state()
        self.assertNotEqual(before["snapshot_id"], after["snapshot_id"])
        self.assertEqual(after["account"]["cash"], "749.9")
        self.mark({"AC": "3"})
        self.workspace.run(reserve_request("second", "A", quantity="1", limit_price="100"))
        self.workspace.run({"operation": "dispatch", "now": NOW.isoformat()})
        funded_fill(self.workspace, client="second", event_id="second-fill", quantity="1", price="100")
        marked = self.mark({"AC": "3"})["data"]
        self.assertEqual(marked["account"]["cash"], "649.8")
        self.assertEqual(marked["account_state"], "UNKNOWN")

    def test_atomic_failure_and_actual_process_death_leave_no_order_or_reservation(self):
        code = ("import os,sys,json;from pathlib import Path;from gig.service.funded_workspace import FundedWorkspace;"
            "FundedWorkspace(Path(sys.argv[1]),fault_hook=lambda stage:os._exit(67) if stage==sys.argv[3] else None).run(json.loads(sys.argv[2]))")
        for phase in ("after_funded_order", "before_snapshot_commit"):
            def fail(stage, target=phase):
                if stage == target:
                    raise RuntimeError("injected interruption")
            with self.subTest(phase=phase):
                with self.assertRaises(RuntimeError):
                    FundedWorkspace(self.base, fault_hook=fail).run(reserve_request())
                done = subprocess.run([sys.executable, "-c", code, str(self.base), json.dumps(reserve_request()), phase], env=os.environ.copy(), capture_output=True, timeout=30)
                self.assertEqual(done.returncode, 67, done.stderr.decode())
                with OfflineOMS(self.base/"funded-account/oms.sqlite") as oms:
                    self.assertEqual(oms.connection.execute("SELECT COUNT(*) FROM orders").fetchone()[0], 0)
                self.assertEqual(self.state()["account"]["reserved_cash"], "0")

    def test_fill_failure_rolls_back_cash_position_and_event_then_exact_retry(self):
        self.prepare()
        def fail(stage):
            if stage == "after_funded_event":
                raise RuntimeError("event interruption")
        with self.assertRaises(RuntimeError):
            funded_fill(FundedWorkspace(self.base, fault_hook=fail))
        self.assertEqual(self.state()["account"]["cash"], "1000")
        self.assertEqual(self.state()["account"]["effective_positions"], {})
        result = self.decode(funded_fill(self.workspace))
        identity = result["data"]["snapshot_id"]
        reopened = FundedWorkspace(self.base)
        with localcontext() as context:
            context.prec = 5
            replay = reopened.run({"operation": "replay", "now": NOW.isoformat(), "snapshot_id": identity}).to_dict()
        self.assertEqual(replay, result)
        saved = reopened.load(identity)
        saved["account"]["cash"] = "0"
        self.assertEqual(reopened.load(identity)["account"]["cash"], "749.9")

    def test_cancel_releases_only_unfilled_reservations(self):
        self.prepare()
        funded_fill(self.workspace)
        self.workspace.run({"operation": "cancel", "now": NOW.isoformat(), "client_id": "buy"})
        self.workspace.run({"operation": "dispatch", "now": NOW.isoformat()})
        account = self.state()["account"]
        self.assertEqual(account["reserved_cash"], "0")
        self.assertEqual(account["effective_positions"], {"AC": "1"})
        self.assertEqual(account["cash"], "749.9")

    def test_oms_disagreement_and_snapshot_tamper_fail_closed(self):
        self.prepare()
        result = self.decode(funded_fill(self.workspace))
        identity = result["data"]["snapshot_id"]
        with sqlite3.connect(self.base/"funded-account/oms.sqlite") as connection:
            connection.execute("UPDATE positions SET quantity='99'")
        with self.assertRaisesRegex(ValueError, "disagree"):
            self.workspace.run({"operation": "state", "now": (NOW+timedelta(seconds=1)).isoformat()})
        path = self.base/"funded-account/snapshots"/(identity+".json")
        path.write_text("{}")
        with self.assertRaisesRegex(ValueError, "integrity"):
            self.workspace.load(identity)
