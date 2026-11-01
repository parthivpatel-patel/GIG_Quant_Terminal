"""Durable offline transport faults and real ledger reconciliation references."""

from __future__ import annotations

import tempfile
import unittest
from datetime import UTC, datetime, timedelta
from pathlib import Path

from gig.execution.offline_adapter import OfflinePaperAdapter, parse_alpaca_fixture
from gig.execution.oms import OfflineOMS, OrderError
from gig.execution.outbox import ExecutionOutbox, event_value
from gig.service.execution_workspace import ExecutionWorkspace

NOW = datetime(2026, 9, 30, 14, tzinfo=UTC)


class OutboxTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.base = Path(self.temp.name)
        self.oms = OfflineOMS(self.base / "oms.sqlite")
        self.outbox = ExecutionOutbox(self.oms)
        self.adapter = OfflinePaperAdapter(self.base / "adapter.sqlite")

    def tearDown(self):
        self.adapter.close()
        self.oms.close()
        self.temp.cleanup()

    def prepare(self, client="one", quantity="10", cap="20", *, hook=None):
        box = ExecutionOutbox(self.oms, fault_hook=hook) if hook else self.outbox
        return box.admit_order(
            client,
            "SYN001",
            "buy",
            quantity,
            reference_price="100",
            price_asof=NOW,
            max_notional="10000",
            max_position_quantity=cap,
            limit_price="101",
            now=NOW,
        )

    def test_admit_is_atomic_and_conflicting_intent_cannot_replace_it(self):
        result = self.prepare()
        self.assertEqual(result["state"], "SUBMITTING")
        self.assertEqual(self.oms.positions(), {})
        self.assertEqual(self.prepare()["client_id"], "one")
        self.assertEqual(len(self.outbox.intents()), 1)
        with self.assertRaises(OrderError):
            self.prepare(quantity="11")
        self.assertEqual(self.oms.order("one")["quantity"], "10")

    def test_dispatch_exposes_quarantined_ack_as_error(self):
        self.prepare()
        original = self.adapter.submit

        def invalid_ack(intent, *, now):
            event = original(intent, now=now)
            event["occurred_at"] = (NOW - timedelta(seconds=1)).isoformat()
            return event

        self.adapter.submit = invalid_ack
        result = self.outbox.dispatch(self.adapter, now=NOW)
        self.assertEqual(result["state"], "ERROR")
        self.assertEqual(result["event_errors"], 1)
        self.assertTrue(self.oms.kill_switch)
        self.assertEqual(self.oms.positions(), {})

    def test_failed_intent_commit_leaves_no_order_or_reservation(self):
        def fail(stage):
            if stage == "after_intent_write":
                raise RuntimeError("fixture crash")

        with self.assertRaises(RuntimeError):
            self.prepare(hook=fail)
        self.assertEqual(
            self.oms.connection.execute("SELECT COUNT(*) FROM orders").fetchone()[0], 0
        )
        self.assertEqual(self.outbox.intents(), [])

    def test_ack_never_creates_a_fill_and_partial_events_are_exact(self):
        self.prepare()
        self.outbox.dispatch(self.adapter, now=NOW)
        self.assertEqual(self.oms.order("one")["state"], "ACKNOWLEDGED")
        self.assertEqual(self.oms.positions(), {})
        event = self.adapter.fill("one", "3", "100", now=NOW + timedelta(seconds=1))
        self.outbox.ingest_event(event, received_at=NOW + timedelta(seconds=2))
        self.assertEqual(self.oms.positions(), {"SYN001": "3"})
        self.assertEqual(self.oms.order("one")["state"], "PARTIALLY_FILLED")
        self.assertEqual(
            self.outbox.ingest_event(event, received_at=NOW + timedelta(seconds=3))["disposition"],
            "DUPLICATE",
        )
        self.assertEqual(self.oms.positions(), {"SYN001": "3"})
        event = self.adapter.fill("one", "7", "100", now=NOW + timedelta(seconds=4))
        self.outbox.ingest_event(event, received_at=NOW + timedelta(seconds=4))
        self.assertEqual(self.oms.order("one")["state"], "FILLED")
        self.assertTrue(
            self.outbox.reconcile(self.adapter, now=NOW + timedelta(seconds=5))["positions"][
                "matched"
            ]
        )

    def test_crash_after_claim_never_automatically_resends_after_restart(self):
        self.prepare()

        def crash(stage):
            if stage == "after_claim":
                raise RuntimeError("process lost")

        box = ExecutionOutbox(self.oms, fault_hook=crash)
        with self.assertRaises(RuntimeError):
            box.dispatch(self.adapter, now=NOW)
        self.assertEqual(self.adapter.submit_calls, 0)
        self.oms.close()
        self.oms = OfflineOMS(self.base / "oms.sqlite")
        self.outbox = ExecutionOutbox(self.oms)
        self.assertTrue(self.outbox.recover()["requires_reconciliation"])
        self.assertEqual(self.outbox.dispatch(self.adapter, now=NOW)["sent"], 0)
        self.outbox.reconcile(self.adapter, now=NOW)
        self.assertEqual(self.outbox.intents()[0]["state"], "ABSENT_PROVEN")
        self.outbox.retry_absent("one", now=NOW)
        self.assertEqual(self.outbox.dispatch(self.adapter, now=NOW)["sent"], 1)

    def test_accepted_then_timeout_reconciles_instead_of_resending(self):
        self.prepare()
        self.adapter.fault = "after_accept"
        result = self.outbox.dispatch(self.adapter, now=NOW)
        self.assertEqual(result["state"], "UNKNOWN")
        self.assertEqual(self.adapter.submit_calls, 1)
        self.adapter.fault = None
        result = self.outbox.reconcile(self.adapter, now=NOW + timedelta(seconds=1))
        self.assertEqual(result["state"], "AVAILABLE")
        self.assertEqual(self.oms.order("one")["state"], "ACKNOWLEDGED")
        self.assertEqual(self.outbox.dispatch(self.adapter, now=NOW)["sent"], 0)
        with self.assertRaises(OrderError):
            self.outbox.retry_absent("one", now=NOW)

    def test_disconnected_unknown_queries_do_not_prove_absence(self):
        self.prepare()
        self.adapter.disconnected = True
        self.outbox.dispatch(self.adapter, now=NOW)
        self.assertEqual(self.outbox.reconcile(self.adapter, now=NOW)["state"], "UNKNOWN")
        with self.assertRaises(OrderError):
            self.outbox.retry_absent("one", now=NOW)

    def test_crash_before_event_commit_preserves_remote_ack_for_reconciliation(self):
        self.prepare()

        def crash(stage):
            if stage == "before_event_commit":
                raise RuntimeError("lost local commit")

        box = ExecutionOutbox(self.oms, fault_hook=crash)
        self.assertEqual(box.dispatch(self.adapter, now=NOW)["state"], "UNKNOWN")
        self.assertEqual(box.replay_events(), [])
        self.assertEqual(self.adapter.lookup("one")["state"], "FOUND")
        self.outbox.reconcile(self.adapter, now=NOW + timedelta(seconds=1))
        self.assertEqual(self.oms.order("one")["state"], "ACKNOWLEDGED")
        self.assertEqual(self.adapter.submit_calls, 1)

    def test_quarantined_reconciliation_cannot_be_healthy_empty_success(self):
        self.prepare()
        import json

        intent = json.loads(self.outbox.intents()[0]["payload"])
        self.adapter.submit(intent, now=NOW - timedelta(seconds=1))
        result = self.outbox.reconcile(self.adapter, now=NOW)
        self.assertEqual(result["state"], "ERROR")
        self.assertTrue(self.oms.kill_switch)
        self.assertTrue(self.outbox.recover()["requires_reconciliation"])

    def test_stale_price_and_emergency_stop_block_ready_submissions(self):
        self.prepare()
        self.oms.set_kill_switch(True, now=NOW)
        self.assertEqual(self.outbox.dispatch(self.adapter, now=NOW)["sent"], 0)
        self.oms.set_kill_switch(False, now=NOW)
        with self.assertRaisesRegex(OrderError, "stale"):
            self.outbox.dispatch(self.adapter, now=NOW + timedelta(seconds=61))
        self.assertEqual(self.adapter.submit_calls, 0)

    def test_cancel_unsent_is_local_and_cancel_after_partial_retains_position(self):
        self.prepare()
        self.assertFalse(self.outbox.request_cancel("one", now=NOW)["cancel_queued"])
        self.assertEqual(self.outbox.dispatch(self.adapter, now=NOW)["sent"], 0)
        self.prepare("two")
        self.outbox.dispatch(self.adapter, now=NOW)
        event = self.adapter.fill("two", "2", "100", now=NOW + timedelta(seconds=1))
        self.outbox.ingest_event(event, received_at=NOW + timedelta(seconds=1))
        self.outbox.request_cancel("two", now=NOW + timedelta(seconds=2))
        self.outbox.dispatch(self.adapter, now=NOW + timedelta(seconds=2))
        self.assertEqual(self.oms.order("two")["state"], "CANCELLED")
        self.assertEqual(self.oms.positions(), {"SYN001": "2"})

    def test_full_fill_racing_cancel_stays_filled(self):
        self.prepare()
        self.outbox.dispatch(self.adapter, now=NOW)
        self.outbox.request_cancel("one", now=NOW + timedelta(seconds=1))
        fill = self.adapter.fill("one", "10", "100", now=NOW + timedelta(seconds=2))
        self.outbox.dispatch(self.adapter, now=NOW + timedelta(seconds=3))
        self.assertEqual(self.oms.order("one")["state"], "FILLED")
        self.assertEqual(
            self.outbox.ingest_event(fill, received_at=NOW + timedelta(seconds=4))["disposition"],
            "DUPLICATE",
        )

    def test_fill_before_ack_and_late_ack_do_not_regress_state(self):
        self.prepare()
        self.adapter.submit(__import__("json").loads(self.outbox.intents()[0]["payload"]), now=NOW)
        fill = self.adapter.fill("one", "2", "100", now=NOW + timedelta(seconds=1))
        self.outbox.ingest_event(fill, received_at=NOW + timedelta(seconds=2))
        ack = self.adapter.lookup("one")["events"][0]
        self.outbox.ingest_event(ack, received_at=NOW + timedelta(seconds=3))
        self.assertEqual(self.oms.order("one")["state"], "PARTIALLY_FILLED")

    def test_conflicting_event_and_impossible_fill_are_not_success(self):
        self.prepare()
        self.outbox.dispatch(self.adapter, now=NOW)
        fill = self.adapter.fill("one", "2", "100", now=NOW)
        self.outbox.ingest_event(fill, received_at=NOW)
        with self.assertRaises(OrderError):
            self.outbox.ingest_event({**fill, "quantity": "3"}, received_at=NOW)
        bad = {**fill, "event_id": "impossible", "quantity": "20"}
        result = self.outbox.ingest_event(bad, received_at=NOW)
        self.assertEqual(result["state"], "ERROR")
        self.assertTrue(self.oms.kill_switch)
        self.assertEqual(self.oms.positions(), {"SYN001": "2"})

    def test_delayed_pre_cancel_fill_records_reality_and_post_trade_breach(self):
        self.prepare(quantity="1", cap="1")
        self.outbox.dispatch(self.adapter, now=NOW)
        self.outbox.request_cancel("one", now=NOW + timedelta(seconds=2))
        self.outbox.dispatch(self.adapter, now=NOW + timedelta(seconds=2))
        self.prepare("two", quantity="1", cap="1")
        self.outbox.dispatch(self.adapter, now=NOW + timedelta(seconds=3))
        fill = self.adapter.fill("two", "1", "100", now=NOW + timedelta(seconds=3))
        self.outbox.ingest_event(fill, received_at=NOW + timedelta(seconds=3))
        delayed = {
            "event_id": "delayed-realistic-fixture",
            "client_id": "one",
            "kind": "FILL",
            "occurred_at": (NOW + timedelta(seconds=1)).isoformat(),
            "quantity": "1",
            "price": "100",
        }
        result = self.outbox.ingest_event(delayed, received_at=NOW + timedelta(seconds=4))
        self.assertEqual(result["state"], "ERROR")
        self.assertEqual(result["disposition"], "APPLIED_RISK_BREACH")
        self.assertEqual(self.oms.positions(), {"SYN001": "2"})
        self.assertTrue(self.oms.kill_switch)

    def test_future_and_submicrosecond_event_timestamps_fail_before_admission(self):
        self.prepare()
        event = {
            "event_id": "x",
            "client_id": "one",
            "kind": "ACK",
            "occurred_at": "2026-09-30T14:00:00.000000001Z",
        }
        with self.assertRaisesRegex(OrderError, "microseconds"):
            event_value(event)
        event["occurred_at"] = (NOW + timedelta(seconds=1)).isoformat()
        with self.assertRaises(OrderError):
            self.outbox.ingest_event(event, received_at=NOW)
        self.assertEqual(self.outbox.replay_events(), [])

    def test_offline_rejection_and_alpaca_fixture_identity(self):
        self.prepare()
        self.adapter.reject_next = True
        self.outbox.dispatch(self.adapter, now=NOW)
        self.assertEqual(self.oms.order("one")["state"], "REJECTED")
        expected = {"client_id": "one", "symbol": "SYN001", "side": "buy", "quantity": "10"}
        raw = {
            "event": "partial_fill",
            "event_id": "fixture-1",
            "timestamp": NOW.isoformat(),
            "order": {"client_order_id": "one", "symbol": "SYN001", "side": "buy", "qty": "10"},
            "qty": "2",
            "price": "100",
        }
        self.assertEqual(parse_alpaca_fixture(raw, expected=expected)["quantity"], "2")
        raw["order"]["symbol"] = "OTHER"
        with self.assertRaises(OrderError):
            parse_alpaca_fixture(raw, expected=expected)


class ExecutionWorkspaceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.base = Path(self.temp.name) / "engine"
        self.workspace = ExecutionWorkspace(self.base)

    def tearDown(self):
        self.temp.cleanup()

    def runop(self, op, **values):
        return self.workspace.run({"operation": op, "now": NOW.isoformat(), **values}).to_dict()

    def test_missing_read_and_invalid_inputs_never_create_state(self):
        self.assertEqual(
            self.workspace.section("execution", NOW, 60).to_dict()["state"], "UNAVAILABLE"
        )
        self.assertFalse(self.base.exists())
        with self.assertRaises(OrderError):
            self.runop(
                "prepare",
                client_id="one",
                symbol="SYN001",
                side="buy",
                quantity=True,
                reference_price="100",
                price_asof=NOW.isoformat(),
                max_notional="1000",
                max_position_quantity="20",
            )
        self.assertFalse(self.base.exists())
        with self.assertRaises(OrderError):
            self.runop("status", path="../escape")

    def test_prepare_dispatch_fill_restart_and_freshness(self):
        self.runop(
            "prepare",
            client_id="one",
            symbol="SYN001",
            side="buy",
            quantity="2",
            reference_price="100",
            price_asof=NOW.isoformat(),
            max_notional="1000",
            max_position_quantity="20",
        )
        ack = self.runop("dispatch")
        self.assertEqual(ack["data"]["positions"], {})
        result = self.runop("fill", client_id="one", quantity="1", price="100")
        self.assertEqual(result["data"]["positions"], {"SYN001": "1"})
        reopened = ExecutionWorkspace(self.base)
        portfolio = reopened.section("portfolio", NOW, 60).to_dict()
        self.assertEqual(portfolio["data"]["rows"][0]["quantity"], "1")
        self.assertIsNone(portfolio["data"]["metrics"]["nav"])
        self.assertEqual(
            reopened.section("execution", NOW + timedelta(seconds=61), 60).to_dict()["state"],
            "STALE",
        )
        self.assertEqual(reopened.section("execution", NOW, None).to_dict()["state"], "UNKNOWN")

    def test_twap_is_exact_plan_and_does_not_submit_or_create_state(self):
        result = self.runop(
            "twap",
            client_id="parent",
            quantity="100",
            start=NOW.isoformat(),
            end=(NOW + timedelta(seconds=300)).isoformat(),
            slices=3,
        )
        self.assertEqual(len(result["data"]["rows"]), 3)
        self.assertFalse(self.base.exists())
        self.assertFalse(result["data"]["broker_connected"])


if __name__ == "__main__":
    unittest.main()
