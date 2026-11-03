"""Cash conservation, entitlement clocks and exact equity lifecycle recovery."""
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
from decimal import Decimal
from pathlib import Path
from threading import Barrier

from gig.service.funded_workspace import FundedWorkspace
from gig.service.lifecycle_workspace import FundedLifecycleWorkspace
from test_funded_accounting import NOW, reserve_request, seed_funded


def dividend(*, revision=1, amount="1", known=NOW):
    return {"action_id": "authored-dividend", "revision": revision, "instrument_id": "A", "source": "AUTHORED_SYNTHETIC",
        "kind": "CASH_DIVIDEND", "known_at": known.isoformat(), "effective_at": (NOW+timedelta(days=3)).isoformat(),
        "record_at": (NOW+timedelta(days=3)).isoformat(), "pay_at": (NOW+timedelta(days=4)).isoformat(), "amount_per_share": amount}


def split(*, revision=1, new=2, old=1, known=NOW):
    at = NOW+timedelta(days=2)
    return {"action_id": "authored-split", "revision": revision, "instrument_id": "A", "source": "AUTHORED_SYNTHETIC",
        "kind": "SPLIT", "known_at": known.isoformat(), "effective_at": at.isoformat(), "record_at": at.isoformat(),
        "pay_at": at.isoformat(), "ratio_new": new, "ratio_old": old}


class FundedLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.base = Path(self.temp.name)/"state"
        self.book = seed_funded(self.base, "1000")
        self.lifecycle = FundedLifecycleWorkspace(self.base)
        self.lifecycle.run({"operation": "configure", "now": NOW.isoformat(), "settlement_policy": "EXPLICIT_EQUITY_DATES"})

    def tearDown(self):
        self.temp.cleanup()

    def fill(self, client, *, side="buy", quantity="10", price="10", fee="0.2", now=NOW, due=None):
        self.book.run(reserve_request(client, "A", side=side, quantity=quantity, limit_price=price, fee_budget="1", now=now))
        self.book.run({"operation": "dispatch", "now": now.isoformat()})
        return self.book.run({"operation": "event", "now": now.isoformat(), "settlement_at": (due or now+timedelta(days=1)).isoformat(),
            "event": {"event_id": "fill-"+client, "client_id": client, "kind": "FILL", "occurred_at": now.isoformat(),
                      "quantity": quantity, "price": price, "fee": fee}}).to_dict()["data"]

    def state(self, at):
        return self.book.run({"operation": "state", "now": at.isoformat()}).to_dict()["data"]

    def admit(self, action, now=NOW):
        return self.lifecycle.run({"operation": "admit", "now": now.isoformat(), "action": action}).to_dict()["data"]["action_hash"]

    def apply(self, identity, at):
        return self.lifecycle.run({"operation": "apply", "now": at.isoformat(), "action_hash": identity}).to_dict()["data"]

    def mark(self, price, at):
        return self.book.run({"operation": "mark", "now": at.isoformat(), "marks": {"A": {"price": price, "as_of": at.isoformat()}}, "max_age_seconds": 60}).to_dict()["data"]

    def test_trade_date_vs_settled_cash_and_no_reuse_of_unsettled_sales(self):
        initial = self.fill("buy")["account"]
        self.assertEqual(Decimal(initial["cash"]), Decimal("899.8"))
        self.assertEqual(Decimal(initial["settled_cash"]), Decimal("999.8"))
        self.assertEqual(initial["unsettled_buy_payables"], "100")
        with self.assertRaises(ValueError):
            self.book.run(reserve_request("early-sale", "A", side="sell", quantity="1", limit_price="10"))
        later = NOW+timedelta(days=1)
        sold = self.fill("sell", side="sell", quantity="10", price="12", now=later)["account"]
        self.assertEqual(Decimal(sold["cash"]), Decimal("1019.6"))
        self.assertEqual(Decimal(sold["available_cash"]), Decimal("899.6"))
        self.assertEqual(Decimal(sold["settled_cash"]), Decimal("899.6"))
        self.assertEqual(sold["unsettled_sale_receivables"], "120")
        with self.assertRaises(ValueError):
            self.book.run(reserve_request("unfunded", "A", quantity="100", limit_price="10", now=later))
        final = self.state(later+timedelta(days=1))["account"]
        self.assertEqual(Decimal(final["available_cash"]), Decimal("1019.6"))
        self.assertEqual(final["unsettled_sale_receivables"], "0")

    def test_exact_split_and_cash_dividend_conserve_basis_pnl_and_raw_oms(self):
        self.fill("buy")
        first, second = self.admit(split()), self.admit(dividend())
        when = NOW+timedelta(days=2)
        changed = self.apply(first, when)
        self.assertEqual(changed["account"]["effective_positions"]["A"], "20")
        self.assertEqual(changed["account"]["raw_oms_positions"]["A"], "10")
        self.assertEqual(changed["account"]["rows"][0]["cost_basis"], "100")
        self.assertIsNone(changed["account"]["nav"])
        after = self.mark("5", when)
        self.assertEqual(Decimal(after["account"]["nav"]), Decimal("999.8"))
        paid = self.apply(second, NOW+timedelta(days=4))
        final = self.mark("5", NOW+timedelta(days=4))["account"]
        self.assertEqual(Decimal(final["cash"]), Decimal("919.8"))
        self.assertEqual(Decimal(final["nav"]), Decimal("1019.8"))
        self.assertEqual(final["dividend_income"], "20")
        self.assertEqual(Decimal(final["initial_cash"])+Decimal(final["total_pnl"]), Decimal(final["nav"]))
        again = self.apply(second, NOW+timedelta(days=4))["account"]
        self.assertEqual(again["cash"], paid["account"]["cash"])
        with sqlite3.connect(self.base/"funded-account/oms.sqlite") as db:
            self.assertEqual(db.execute("SELECT quantity FROM positions WHERE symbol='A'").fetchone()[0], "10")

    def test_post_ex_buys_do_not_receive_dividend_and_revision_freezes_entitlement(self):
        self.fill("buy")
        identity = self.admit(dividend())
        late = NOW+timedelta(days=3, hours=1)
        self.fill("after-ex", quantity="5", now=late, due=late+timedelta(hours=1))
        paid_at = NOW+timedelta(days=4)
        original = self.apply(identity, paid_at)
        self.assertEqual(original["account"]["dividend_income"], "10")
        revised_at = paid_at+timedelta(seconds=1)
        revised = dividend(revision=2, amount="1.5", known=revised_at)
        identity2 = self.admit(revised, revised_at)
        adjusted = self.apply(identity2, revised_at)
        self.assertEqual(adjusted["account"]["dividend_income"], "15.0")
        self.assertEqual(adjusted["lifecycle_events"][-1]["entitlement_quantity"], "10")
        self.assertEqual(adjusted["lifecycle_events"][-1]["cash_amount"], "5.0")
        reopened = FundedWorkspace(self.base)
        self.assertEqual(reopened.load(original["snapshot_id"])["account"]["dividend_income"], "10")

    def test_ex_date_receivable_completes_nav_without_becoming_future_cash(self):
        self.fill("buy")
        identity = self.admit(dividend())
        original = self.mark("10", NOW)
        at = NOW+timedelta(days=3)
        pending = self.mark("9", at)["account"]
        self.assertEqual(pending["mark_state"], "AVAILABLE")
        self.assertEqual(pending["funding_state"], "AVAILABLE")
        self.assertEqual(pending["reconciliation_state"], "AVAILABLE")
        self.assertEqual(pending["dividend_accrual_state"], "AVAILABLE")
        self.assertEqual(pending["unmodeled_dividend_accruals"], [])
        self.assertEqual(pending["dividend_receivable"], "10")
        self.assertEqual(pending["nav"], "999.8")
        self.assertEqual(pending["total_pnl"], "-0.2")
        self.assertEqual(pending["cash"], "899.8")
        self.assertEqual(pending["available_cash"], "899.8")
        due = self.state(NOW+timedelta(days=4))["account"]
        self.assertEqual(due["funding_state"], "AVAILABLE")
        self.assertEqual(due["dividend_receivable"], "10")
        self.apply(identity, NOW+timedelta(days=4))
        paid = self.mark("9", NOW+timedelta(days=4))["account"]
        self.assertEqual(paid["dividend_accrual_state"], "NOT_REQUIRED")
        self.assertEqual(paid["reconciliation_state"], "AVAILABLE")
        self.assertIsNotNone(paid["nav"])
        self.assertEqual(FundedWorkspace(self.base).load(original["snapshot_id"])["account"]["nav"], "999.8")

    def test_late_split_after_trade_and_fractional_delivery_reject_atomically(self):
        self.fill("buy", quantity="1")
        identity = self.admit(split(new=1, old=2))
        with self.assertRaisesRegex(ValueError, "fractional"):
            self.apply(identity, NOW+timedelta(days=2))
        with sqlite3.connect(self.base/"funded-account/oms.sqlite") as db:
            self.assertEqual(db.execute("SELECT COUNT(*) FROM funded_lifecycle_events").fetchone()[0], 0)
        # A due unresolved action blocks new orders. Independently reproduce a
        # late first admission after an already recorded post-effective trade.
        self.base = self.base/"late"
        self.book = seed_funded(self.base, "1000")
        self.lifecycle = FundedLifecycleWorkspace(self.base)
        self.lifecycle.run({"operation": "configure", "now": NOW.isoformat(), "settlement_policy": "EXPLICIT_EQUITY_DATES"})
        self.fill("buy", quantity="1")
        later = NOW+timedelta(days=2, seconds=1)
        self.fill("later", quantity="1", now=later)
        identity = self.admit(split(new=1, old=2), later)
        with self.assertRaisesRegex(ValueError, "intervening trades"):
            self.apply(identity, later)

    def test_exact_basis_representability_is_not_silently_rounded(self):
        self.fill("buy", quantity="1", price="10")
        identity = self.admit(split(new=3))
        with self.assertRaisesRegex(ValueError, "represented exactly"):
            self.apply(identity, NOW+timedelta(days=2))

    def test_split_economic_inventory_and_pending_sales_never_create_shorts(self):
        self.fill("buy")
        identity = self.admit(split())
        self.apply(identity, NOW+timedelta(days=2))
        sale_at = NOW+timedelta(days=2, seconds=1)
        account = self.fill("sale", side="sell", quantity="15", price="5", now=sale_at)["account"]
        self.assertEqual(account["raw_oms_positions"]["A"], "-5")
        self.assertEqual(account["effective_positions"]["A"], "5")
        self.assertEqual(account["settled_trade_positions"]["A"], "20")
        with self.assertRaises(ValueError):
            self.book.run(reserve_request("oversale", "A", side="sell", quantity="6", limit_price="5", now=sale_at))

    def test_policy_cannot_reinterpret_existing_account_and_dates_are_required(self):
        self.fill("buy")
        with self.assertRaises(ValueError):
            self.lifecycle.run({"operation": "configure", "now": NOW.isoformat(), "settlement_policy": "IMMEDIATE"})
        self.book.run(reserve_request("new", "A", quantity="1", limit_price="10"))
        self.book.run({"operation": "dispatch", "now": NOW.isoformat()})
        with self.assertRaisesRegex(ValueError, "settlement_at"):
            self.book.run({"operation": "event", "now": NOW.isoformat(), "event": {"event_id": "missing-date", "client_id": "new",
                "kind": "FILL", "occurred_at": NOW.isoformat(), "quantity": "1", "price": "10", "fee": "0"}})

    def test_conflicting_revision_future_knowledge_and_time_reversal_reject(self):
        self.fill("buy")
        source = dividend()
        first = self.admit(source)
        changed = copy.deepcopy(source)
        changed["amount_per_share"] = "999"
        with self.assertRaises(ValueError):
            self.admit(changed)
        changed["revision"] = 2
        changed["known_at"] = (NOW+timedelta(days=1)).isoformat()
        with self.assertRaises(ValueError):
            self.admit(changed)
        with self.assertRaises(ValueError):
            self.apply(first, NOW)

    def test_failure_injection_rolls_back_cash_positions_and_action_index(self):
        self.fill("buy")
        identity = self.admit(dividend())
        def fail(_):
            raise RuntimeError("injected lifecycle interruption")
        with self.assertRaises(RuntimeError):
            FundedLifecycleWorkspace(self.base, fault_hook=fail).run({"operation": "apply", "now": (NOW+timedelta(days=4)).isoformat(), "action_hash": identity})
        with sqlite3.connect(self.base/"funded-account/oms.sqlite") as db:
            self.assertEqual(db.execute("SELECT COUNT(*) FROM funded_lifecycle_events").fetchone()[0], 0)
        final = self.apply(identity, NOW+timedelta(days=4))
        self.assertEqual(final["account"]["dividend_income"], "10")

    def test_settlement_metadata_tampering_and_duplicate_conflicts_fail_closed(self):
        self.fill("buy")
        with sqlite3.connect(self.base/"funded-account/oms.sqlite") as db:
            db.execute("UPDATE funded_trade_settlements SET settlement_at=?", ((NOW+timedelta(days=2)).isoformat(),))
        with self.assertRaisesRegex(ValueError, "immutable"):
            self.state(NOW)

    def test_lifecycle_cash_event_and_action_tampering_fail_closed(self):
        self.fill("buy")
        identity = self.admit(dividend())
        at = NOW+timedelta(days=4)
        self.apply(identity, at)
        path = self.base/"funded-account/oms.sqlite"
        with sqlite3.connect(path) as db:
            original = db.execute("SELECT wire FROM funded_lifecycle_events").fetchone()[0]
            changed = json.loads(original)
            changed["cash_amount"] = "1000000"
            db.execute("UPDATE funded_lifecycle_events SET wire=?", (json.dumps(changed),))
        with self.assertRaisesRegex(ValueError, "immutable"):
            self.state(at)
        with sqlite3.connect(path) as db:
            db.execute("UPDATE funded_lifecycle_events SET wire=?", (original,))
            action = json.loads(db.execute("SELECT wire FROM funded_lifecycle_actions").fetchone()[0])
            action["amount_per_share"] = "1000000"
            db.execute("UPDATE funded_lifecycle_actions SET wire=?", (json.dumps(action),))
        with self.assertRaisesRegex(ValueError, "immutable"):
            self.state(at)

    def test_simultaneous_duplicate_action_has_one_cash_credit(self):
        self.fill("buy")
        identity = self.admit(dividend())
        at = NOW+timedelta(days=4)
        barrier = Barrier(2)
        def apply_once(_):
            barrier.wait(timeout=10)
            return FundedLifecycleWorkspace(self.base).run({"operation": "apply", "now": at.isoformat(),
                "action_hash": identity}).to_dict()["data"]
        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(apply_once, range(2)))
        self.assertEqual([row["account"]["dividend_income"] for row in results], ["10", "10"])
        with sqlite3.connect(self.base/"funded-account/oms.sqlite") as db:
            self.assertEqual(db.execute("SELECT COUNT(*) FROM funded_lifecycle_events").fetchone()[0], 2)
        self.assertEqual(FundedWorkspace(self.base).load(results[0]["snapshot_id"])["account"]["cash"], results[1]["account"]["cash"])

    def test_split_rejects_working_orders_and_unsettled_delivery(self):
        self.fill("buy", due=NOW+timedelta(days=3))
        self.book.run(reserve_request("working", "A", quantity="1", limit_price="10"))
        identity = self.admit(split())
        at = NOW+timedelta(days=2)
        with self.assertRaisesRegex(ValueError, "working orders"):
            self.apply(identity, at)
        with sqlite3.connect(self.base/"funded-account/oms.sqlite") as db:
            self.assertEqual(db.execute("SELECT COUNT(*) FROM funded_lifecycle_events").fetchone()[0], 0)
        # An independent account removes the working-order reason for rejection.
        self.base = self.base/"unsettled"
        self.book = seed_funded(self.base, "1000")
        self.lifecycle = FundedLifecycleWorkspace(self.base)
        self.lifecycle.run({"operation": "configure", "now": NOW.isoformat(), "settlement_policy": "EXPLICIT_EQUITY_DATES"})
        self.fill("buy", due=NOW+timedelta(days=3))
        identity = self.admit(split())
        with self.assertRaisesRegex(ValueError, "unsettled"):
            self.apply(identity, at)

    def test_process_death_rolls_back_lifecycle_and_exact_retry_recovers(self):
        self.fill("buy")
        identity = self.admit(dividend())
        payload = {"operation": "apply", "now": (NOW+timedelta(days=4)).isoformat(), "action_hash": identity}
        code = ("import os,sys,json;from pathlib import Path;from gig.service.lifecycle_workspace import FundedLifecycleWorkspace;"
                "FundedLifecycleWorkspace(Path(sys.argv[1]),fault_hook=lambda _:os._exit(67)).run(json.loads(sys.argv[2]))")
        result = subprocess.run([sys.executable, "-c", code, str(self.base), json.dumps(payload)],
                                env=os.environ.copy(), capture_output=True, timeout=30)
        self.assertEqual(result.returncode, 67, result.stderr.decode())
        with sqlite3.connect(self.base/"funded-account/oms.sqlite") as db:
            self.assertEqual(db.execute("SELECT COUNT(*) FROM funded_lifecycle_events").fetchone()[0], 0)
        final = FundedLifecycleWorkspace(self.base).run(payload).to_dict()["data"]
        self.assertEqual(final["account"]["dividend_income"], "10")

    def test_reference_fx_cannot_be_registered_as_a_funded_equity(self):
        for instrument in ("ECB:EXR:USD:PER_EUR", "FX:EURUSD", "REFERENCE:EURUSD"):
            with self.subTest(instrument=instrument), self.assertRaises(ValueError):
                self.book.run({"operation": "register", "now": NOW.isoformat(), "instruments": [
                    {"instrument_id": instrument, "kind": "EQUITY", "currency": "USD", "multiplier": 1}]})

    def test_xnys_t_plus_one_uses_next_scheduled_close_and_holiday(self):
        fresh = self.base/"calendar"
        book = seed_funded(fresh, "1000")
        lifecycle = FundedLifecycleWorkspace(fresh)
        lifecycle.run({"operation": "configure", "now": NOW.isoformat(), "settlement_policy": "XNYS_T_PLUS_ONE"})
        book.run(reserve_request("calendar-buy", "A", quantity="1", limit_price="10"))
        book.run({"operation": "dispatch", "now": NOW.isoformat()})
        payload = {"operation": "event", "now": NOW.isoformat(), "settlement_at": "2026-11-03T21:00:00Z",
            "event": {"event_id": "calendar-fill", "client_id": "calendar-buy", "kind": "FILL", "occurred_at": NOW.isoformat(), "quantity": "1", "price": "10", "fee": "0"}}
        good = book.run(payload).to_dict()
        self.assertEqual(good["data"]["trade_settlements"][0]["settlement_at"], "2026-11-03T21:00:00+00:00")
        payload["settlement_at"] = "2026-11-04T21:00:00Z"
        with self.assertRaises(ValueError):
            book.run(payload)

    def test_xnys_t_plus_one_crosses_thanksgiving_to_early_close(self):
        fresh = self.base/"holiday"
        book = seed_funded(fresh, "1000")
        lifecycle = FundedLifecycleWorkspace(fresh)
        lifecycle.run({"operation": "configure", "now": NOW.isoformat(), "settlement_policy": "XNYS_T_PLUS_ONE"})
        at = NOW.replace(day=25)
        book.run(reserve_request("holiday-buy", "A", quantity="1", limit_price="10", now=at))
        book.run({"operation": "dispatch", "now": at.isoformat()})
        payload = {"operation": "event", "now": at.isoformat(), "settlement_at": "2026-11-26T21:00:00Z",
            "event": {"event_id": "holiday-fill", "client_id": "holiday-buy", "kind": "FILL", "occurred_at": at.isoformat(), "quantity": "1", "price": "10", "fee": "0"}}
        with self.assertRaisesRegex(ValueError, "next scheduled"):
            book.run(payload)
        payload["settlement_at"] = "2026-11-27T18:00:00Z"
        good = book.run(payload).to_dict()["data"]
        self.assertEqual(good["trade_settlements"][0]["settlement_at"], "2026-11-27T18:00:00+00:00")
