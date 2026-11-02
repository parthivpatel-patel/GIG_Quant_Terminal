"""Explicit simulation clocks and real SQLite crash recovery, never brokers."""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest
from datetime import UTC, datetime, timedelta
from pathlib import Path

from gig.execution.offline_adapter import OfflinePaperAdapter
from gig.execution.oms import OfflineOMS, OrderError
from gig.execution.outbox import ExecutionOutbox
from gig.execution.scheduler import DurableTWAPScheduler
from gig.service.contracts import EnvelopeSchema
from gig.service.execution_workspace import ExecutionWorkspace

NOW = datetime(2026, 11, 1, 15, tzinfo=UTC)


def intent(client="parent", **overrides):
    return {
        "client_id": client, "symbol": "SYN001", "side": "buy", "quantity": "10",
        "start": NOW.isoformat(), "end": (NOW + timedelta(seconds=120)).isoformat(),
        "slices": 3, "max_notional": "2000", "max_position_quantity": "10", **overrides,
    }


def marks(at=NOW, price="100"):
    return {"SYN001": {"reference_price": price, "price_asof": at.isoformat()}}


class SchedulerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name) / "oms.sqlite"
        self.oms = OfflineOMS(self.path)
        self.outbox = ExecutionOutbox(self.oms)
        self.scheduler = DurableTWAPScheduler(self.oms, self.outbox)
        self.adapter = OfflinePaperAdapter(Path(self.temp.name) / "counterparty.sqlite")

    def tearDown(self):
        self.adapter.close()
        self.oms.close()
        self.temp.cleanup()

    def reopen(self):
        self.oms.close()
        self.oms = OfflineOMS(self.path)
        self.outbox = ExecutionOutbox(self.oms)
        self.scheduler = DurableTWAPScheduler(self.oms, self.outbox)

    def order_count(self):
        return self.oms.connection.execute("SELECT COUNT(*) FROM orders").fetchone()[0]

    def test_exact_durable_plan_has_no_orders_and_replays_after_restart(self):
        result = self.scheduler.schedule(intent(), now=NOW)
        self.assertEqual(result["simulation_clock"], NOW.isoformat())
        self.assertEqual(self.order_count(), 0)
        original = self.scheduler.schedules()
        self.assertEqual([c["quantity"] for c in original[0]["children"]], ["4", "3", "3"])
        self.reopen()
        self.assertEqual(self.scheduler.schedules(), original)
        replay = self.scheduler.schedule(intent(), now=NOW)
        self.assertTrue(replay["replayed"])
        self.assertEqual(replay["intent_sha256"], result["intent_sha256"])
        self.assertEqual(self.oms.connection.execute("SELECT COUNT(*) FROM events").fetchone()[0], 1)
        with self.assertRaises(OrderError):
            self.scheduler.schedule(intent(quantity="9"), now=NOW)

    def test_advance_admits_only_due_intents_ack_is_not_a_fill_and_reconciles(self):
        self.scheduler.schedule(intent(), now=NOW)
        first = self.scheduler.advance(marks(), now=NOW)
        self.assertEqual(first["admitted_children"], ["parent:1"])
        self.assertEqual(first["automatically_dispatched"], 0)
        self.assertEqual(self.oms.order("parent:1")["state"], "SUBMITTING")
        self.assertEqual(self.oms.positions(), {})
        self.outbox.dispatch(self.adapter, now=NOW)
        self.assertEqual(self.oms.order("parent:1")["state"], "ACKNOWLEDGED")
        self.assertEqual(self.oms.positions(), {})
        event = self.adapter.fill("parent:1", "2", "100", now=NOW + timedelta(seconds=1))
        self.outbox.ingest_event(event, received_at=NOW + timedelta(seconds=1))
        self.assertEqual(self.oms.positions(), {"SYN001": "2"})
        self.reopen()
        at = NOW + timedelta(seconds=40)
        second = self.scheduler.advance(marks(at), now=at)
        self.assertEqual(second["admitted_children"], ["parent:2"])
        self.assertEqual(self.scheduler.advance(marks(at), now=at), second)
        self.assertEqual(self.order_count(), 2)
        self.assertEqual(self.outbox.reconcile(self.adapter, now=at)["state"], "AVAILABLE")
        self.assertEqual(self.oms.order("parent:1")["filled"], "2")
        self.assertEqual(self.oms.positions(), {"SYN001": "2"})

    def test_clock_is_monotonic_and_conflicting_same_tick_cannot_reprice(self):
        self.scheduler.schedule(intent(), now=NOW)
        self.scheduler.advance(marks(), now=NOW)
        with self.assertRaises(OrderError):
            self.scheduler.advance(marks(price="101"), now=NOW)
        at = NOW + timedelta(seconds=80)
        self.scheduler.advance(marks(at), now=at)
        with self.assertRaises(OrderError):
            self.scheduler.advance(marks(), now=NOW)
        self.assertEqual(self.order_count(), 3)
        self.assertEqual(self.scheduler.schedules()[0]["state"], "COMPLETE")

    def test_missing_future_stale_and_unknown_marks_rollback_the_whole_tick(self):
        self.scheduler.schedule(intent(), now=NOW)
        for quote, at in [({}, NOW), (marks(NOW + timedelta(seconds=1)), NOW),
                          (marks(), NOW + timedelta(seconds=80)),
                          ({"UNKNOWN": marks()["SYN001"]}, NOW)]:
            with self.subTest(quote=quote), self.assertRaises(OrderError):
                self.scheduler.advance(quote, now=at)
            self.assertEqual(self.order_count(), 0)
            self.assertEqual(self.scheduler.clock, NOW.isoformat())

    def test_deadline_expires_unadmitted_children_without_catch_up(self):
        self.scheduler.schedule(intent(), now=NOW)
        self.scheduler.advance(marks(), now=NOW)
        final = self.scheduler.advance({}, now=NOW + timedelta(seconds=120))
        self.assertEqual(final["admitted_children"], [])
        self.assertEqual(final["expired_children"], ["parent:2", "parent:3"])
        self.assertEqual(self.order_count(), 1)
        self.assertEqual(self.oms.order("parent:1")["state"], "SUBMITTING")

    def test_cancel_only_pending_children_and_replays_without_events(self):
        self.scheduler.schedule(intent(), now=NOW)
        self.scheduler.advance(marks(), now=NOW)
        first = self.scheduler.cancel("parent", now=NOW)
        self.assertEqual(first["pending_children_cancelled"], 2)
        self.assertEqual(first["existing_orders_cancelled"], 0)
        count = self.oms.connection.execute("SELECT COUNT(*) FROM events").fetchone()[0]
        self.reopen()
        self.assertEqual(self.scheduler.cancel("parent", now=NOW), first)
        self.assertEqual(self.oms.connection.execute("SELECT COUNT(*) FROM events").fetchone()[0], count)
        self.assertEqual(self.oms.order("parent:1")["state"], "SUBMITTING")
        self.assertEqual(self.scheduler.advance({}, now=NOW + timedelta(seconds=80))["admitted_children"], [])

    def test_kill_persists_and_blocks_atomic_child_admission(self):
        self.scheduler.schedule(intent(), now=NOW)
        self.oms.set_kill_switch(True, now=NOW)
        self.reopen()
        with self.assertRaises(OrderError):
            self.scheduler.advance(marks(), now=NOW)
        self.assertEqual(self.order_count(), 0)
        self.assertTrue(self.oms.kill_switch)
        self.oms.set_kill_switch(False, now=NOW)
        self.assertEqual(self.scheduler.advance(marks(), now=NOW)["admitted_children"], ["parent:1"])

    def test_parent_aggregate_notional_includes_prior_executions(self):
        self.scheduler.schedule(intent(max_notional="1000"), now=NOW)
        self.scheduler.advance(marks(), now=NOW)
        self.outbox.dispatch(self.adapter, now=NOW)
        event = self.adapter.fill("parent:1", "4", "150", now=NOW)
        self.outbox.ingest_event(event, received_at=NOW)
        at = NOW + timedelta(seconds=40)
        with self.assertRaisesRegex(OrderError, "parent executed"):
            self.scheduler.advance(marks(at), now=at)
        self.assertEqual(self.order_count(), 1)
        self.assertEqual(self.scheduler.clock, NOW.isoformat())

    def test_existing_position_reservations_prevent_extra_scheduled_risk(self):
        self.outbox.admit_order("existing", "SYN001", "buy", "9", reference_price="100",
                                price_asof=NOW, max_notional="2000", max_position_quantity="10", now=NOW)
        self.scheduler.schedule(intent(), now=NOW)
        with self.assertRaisesRegex(OrderError, "quantity cap"):
            self.scheduler.advance(marks(), now=NOW)
        self.assertEqual(self.order_count(), 1)
        self.assertEqual(self.scheduler.schedules()[0]["children"][0]["state"], "PENDING")

    def test_fault_after_later_child_rolls_back_all_children_and_clock(self):
        self.scheduler.schedule(intent(), now=NOW)
        visits = []

        def fail(stage):
            if stage == "after_child_admission":
                visits.append(stage)
                if len(visits) == 2:
                    raise RuntimeError("simulated worker loss")

        scheduler = DurableTWAPScheduler(self.oms, self.outbox, fault_hook=fail)
        at = NOW + timedelta(seconds=80)
        with self.assertRaisesRegex(RuntimeError, "worker loss"):
            scheduler.advance(marks(at), now=at)
        self.reopen()
        self.assertEqual(self.order_count(), 0)
        self.assertEqual(self.outbox.intents(), [])
        self.assertEqual(self.scheduler.clock, NOW.isoformat())
        self.assertEqual(len(self.scheduler.advance(marks(at), now=at)["admitted_children"]), 3)

    def test_abrupt_process_exit_preserves_atomicity_on_reopen(self):
        self.scheduler.schedule(intent(), now=NOW)
        code = '''import os,sys
from datetime import datetime
from gig.execution.oms import OfflineOMS
from gig.execution.outbox import ExecutionOutbox
from gig.execution.scheduler import DurableTWAPScheduler
with OfflineOMS(sys.argv[1]) as oms:
    outbox=ExecutionOutbox(oms)
    def die(stage):
        if stage=='after_child_admission': os._exit(17)
    scheduler=DurableTWAPScheduler(oms,outbox,fault_hook=die)
    now=datetime.fromisoformat('2026-11-01T15:00:00+00:00')
    scheduler.advance({'SYN001':{'reference_price':'100','price_asof':now.isoformat()}},now=now)
'''
        environment = {"PATH": os.environ["PATH"], "PYTHONPATH": os.environ.get("PYTHONPATH", "src"),
                       "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"}
        result = subprocess.run([sys.executable, "-c", code, str(self.path)], env=environment,
                                capture_output=True, timeout=20)
        self.assertEqual(result.returncode, 17, result.stderr.decode())
        self.reopen()
        self.assertEqual(self.order_count(), 0)
        self.assertEqual(self.outbox.intents(), [])
        self.assertEqual(self.scheduler.schedules()[0]["children"][0]["state"], "PENDING")
        self.assertEqual(self.scheduler.advance(marks(), now=NOW)["admitted_children"], ["parent:1"])

    def test_future_execution_evidence_prevents_backdated_clock(self):
        self.scheduler.schedule(intent(), now=NOW)
        self.oms.set_kill_switch(False, now=NOW + timedelta(seconds=10))
        with self.assertRaisesRegex(OrderError, "execution evidence"):
            self.scheduler.advance(marks(), now=NOW)

    def test_plan_and_child_tampering_reject_and_pending_order_collision_cannot_be_adopted(self):
        self.scheduler.schedule(intent(), now=NOW)
        self.oms.connection.execute("UPDATE twap_children SET quantity='9' WHERE client_id='parent:1'")
        with self.assertRaises(OrderError):
            self.scheduler.schedules()
        self.oms.connection.execute("UPDATE twap_children SET quantity='4' WHERE client_id='parent:1'")
        self.outbox.admit_order("parent:1", "SYN001", "buy", "4", reference_price="100",
                                price_asof=NOW, max_notional="2000", max_position_quantity="10", now=NOW)
        with self.assertRaisesRegex(OrderError, "conflicts"):
            self.scheduler.advance(marks(), now=NOW)
        self.assertEqual(self.scheduler.schedules()[0]["children"][0]["state"], "PENDING")

    def test_precision_and_size_are_rejected_before_any_parent(self):
        for request in (intent(slices=True), intent(slices=101), intent(quantity="10.01"),
                        intent(start="2026-11-01T15:00:00.123456789Z"), intent(quantity="11")):
            with self.subTest(request=request), self.assertRaises(ValueError):
                self.scheduler.schedule(request, now=NOW)
        self.assertEqual(self.scheduler.schedules(), [])


class SchedulingWorkspaceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.base = Path(self.temp.name) / "execution"
        self.workspace = ExecutionWorkspace(self.base)

    def tearDown(self):
        self.temp.cleanup()

    def op(self, operation, *, now=NOW, **values):
        result = self.workspace.run({"operation": operation, "now": now.isoformat(), **values}).to_dict()
        EnvelopeSchema.model_validate(result)
        return result

    def test_fixed_path_service_schedule_advance_partialfill_and_restart(self):
        first = self.op("schedule-twap", **intent())
        self.assertEqual(first["data"]["metrics"]["order_count"], 0)
        advanced = self.op("advance-clock", marks=marks())
        self.assertEqual(advanced["data"]["operation_result"]["admitted_children"], ["parent:1"])
        self.op("dispatch")
        filled = self.op("fill", client_id="parent:1", quantity="1", price="100")
        self.assertEqual(filled["data"]["positions"], {"SYN001": "1"})
        self.assertEqual(self.op("reconcile")["state"], "AVAILABLE")
        reopened = ExecutionWorkspace(self.base).section("execution", NOW, 60).to_dict()
        self.assertEqual(reopened["data"]["schedules"][0]["children"][0]["order_state"], "PARTIALLY_FILLED")
        self.assertEqual(reopened["data"]["metrics"]["simulation_clock"], NOW.isoformat())

    def test_missing_advance_and_invalid_schedule_leave_state_absent(self):
        self.assertEqual(self.op("advance-clock", marks={})["state"], "UNAVAILABLE")
        self.assertFalse(self.base.exists())
        with self.assertRaises(ValueError):
            self.op("schedule-twap", **intent(slices=101))
        self.assertFalse(self.base.exists())

    def test_workspace_other_operations_cannot_backdate_after_explicit_clock(self):
        self.op("schedule-twap", **intent())
        at = NOW + timedelta(seconds=40)
        self.op("advance-clock", now=at, marks=marks(at))
        with self.assertRaises(OrderError):
            self.op("dispatch")
        self.assertEqual(self.op("cancel-schedule", now=at, client_id="parent")["state"], "AVAILABLE")
        self.assertEqual(self.op("status", now=at)["data"]["metrics"]["order_count"], 2)

    def test_workspace_cannot_backdate_after_execution_events_advance_beyond_clock(self):
        self.op("schedule-twap", **intent())
        later = NOW + timedelta(seconds=10)
        self.op("kill-switch", now=later, engaged=True)
        with self.assertRaisesRegex(OrderError, "execution evidence"):
            self.op("prepare", client_id="backdated", symbol="SYN002", side="buy",
                    quantity="1", reference_price="100", price_asof=NOW.isoformat(),
                    max_notional="1000", max_position_quantity="10")
        self.assertEqual(self.op("status", now=later)["data"]["metrics"]["order_count"], 0)


if __name__ == "__main__":
    unittest.main()
