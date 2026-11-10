"""Cross-caller research lease fencing, atomic commits and crash recovery."""
from __future__ import annotations

import multiprocessing
import sqlite3
import threading
import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from gig.research_org.backlog import SEED
from gig.research_org.ledger import LeaseLost, ResearchLedger
from gig.research_org.scheduler import tick
from gig.research_org.service import ResearchWorker


def _process_tick(directory: str, output: object) -> None:
    result = tick(ResearchLedger(Path(directory)))
    output.put(result["tick"])


def _report() -> dict:
    return {"hypothesis_id": SEED[0]["hypothesis_id"], "verdict": "INCONCLUSIVE",
            "classification": "SYNTHETIC", "economic_claim": "NONE"}


class ResearchLeaseTests(unittest.TestCase):
    def test_invalid_lease_bounds_do_not_create_a_ledger(self) -> None:
        with TemporaryDirectory() as directory:
            ledger = ResearchLedger(Path(directory))
            for value in (0, -1, float("nan"), float("inf"), True, "30", 3600.001, 10**1000):
                with self.subTest(seconds=value), self.assertRaises(ValueError):
                    ledger.acquire_lease("owner", value)
                self.assertFalse(ledger.path.exists())

    def test_manual_tick_does_not_enable_restart_autostart(self) -> None:
        with TemporaryDirectory() as directory:
            ledger = ResearchLedger(Path(directory))
            self.assertEqual(tick(ledger)["tick"]["action"], "ran")
            self.assertEqual(ledger.scheduler_state()["desired"], "STOPPED")
            worker = ResearchWorker(ResearchLedger(Path(directory)))
            self.assertFalse(worker.recover()["alive"])
            self.assertEqual(ledger.count(), 1)

    def test_live_claim_cannot_be_reset_by_another_caller(self) -> None:
        with TemporaryDirectory() as directory:
            first, second = ResearchLedger(Path(directory)), ResearchLedger(Path(directory))
            self.assertTrue(first.acquire_lease("first", 2))
            first.recover_and_claim(SEED, lease_owner="first")
            self.assertIsNone(second.recover_and_claim(SEED))
            self.assertEqual(tick(second)["tick"]["action"], "busy")
            self.assertEqual(first.jobs()[0]["attempts"], 1)
            self.assertEqual(first.jobs()[0]["status"], "RUNNING")
            with self.assertRaises(LeaseLost):
                second.complete_job(SEED[0]["job_id"], "QUEUED", None)

    def test_separate_process_obeys_live_lease(self) -> None:
        with TemporaryDirectory() as directory:
            ledger = ResearchLedger(Path(directory))
            ledger.acquire_lease("parent", 30)
            context = multiprocessing.get_context("spawn")
            output = context.Queue()
            process = context.Process(target=_process_tick, args=(directory, output))
            process.start()
            process.join(15)
            if process.is_alive():
                process.terminate()
                process.join()
                self.fail("child research process did not finish")
            self.assertEqual(process.exitcode, 0)
            self.assertEqual(output.get(timeout=2)["action"], "busy")
            self.assertEqual(ledger.count(), 0)
            output.close()

    def test_long_runner_renews_and_concurrent_tick_does_not_duplicate(self) -> None:
        with TemporaryDirectory() as directory:
            ledger = ResearchLedger(Path(directory))
            entered, release = threading.Event(), threading.Event()
            outcomes = []

            def bounded_runner():
                entered.set()
                if not release.wait(3):
                    raise RuntimeError("test runner timeout")
                return "test-hash", _report()

            with patch.dict("gig.research_org.scheduler.RUNNERS", {"lagged_sign": bounded_runner}):
                first = threading.Thread(target=lambda: outcomes.append(tick(ledger, lease_seconds=0.09)))
                first.start()
                try:
                    self.assertTrue(entered.wait(2))
                    time.sleep(0.3)
                    self.assertEqual(tick(ResearchLedger(Path(directory)))["tick"]["action"], "busy")
                    self.assertEqual(ledger.count(), 0)
                    self.assertEqual(ledger.jobs()[0]["attempts"], 1)
                finally:
                    release.set()
                    first.join(3)
            self.assertFalse(first.is_alive())
            self.assertEqual(outcomes[0]["tick"]["action"], "ran")
            self.assertEqual(ledger.count(), 1)

    def test_shutdown_timeout_keeps_live_job_lease_until_runner_finishes(self) -> None:
        with TemporaryDirectory() as directory:
            ledger = ResearchLedger(Path(directory))
            entered, release = threading.Event(), threading.Event()

            def bounded_runner():
                entered.set()
                if not release.wait(3):
                    raise RuntimeError("test runner timeout")
                return "test-hash", _report()

            worker = ResearchWorker(ledger, lease_seconds=0.09)
            with patch.dict("gig.research_org.scheduler.RUNNERS", {"lagged_sign": bounded_runner}):
                worker.start(0.05)
                try:
                    self.assertTrue(entered.wait(2))
                    worker.shutdown(timeout=0.001)
                    time.sleep(0.2)
                    self.assertTrue(worker.snapshot()["alive"])
                    self.assertTrue(ResearchLedger(Path(directory)).lease_blocks(None))
                    self.assertEqual(tick(ResearchLedger(Path(directory)))["tick"]["action"], "busy")
                finally:
                    release.set()
                    worker.wait(3)
            self.assertEqual(ledger.count(), 1)
            self.assertFalse(worker.snapshot()["alive"])
            self.assertFalse(ledger.lease_blocks(None))

    def test_expired_and_replaced_owner_cannot_commit_or_change_job(self) -> None:
        with TemporaryDirectory() as directory:
            ledger = ResearchLedger(Path(directory))
            ledger.acquire_lease("old", 30)
            old = ledger.recover_and_claim(SEED, lease_owner="old")
            ledger.expire_lease()
            self.assertFalse(ledger.renew_lease("old", 30))
            ledger.acquire_lease("new", 30)
            current = ledger.recover_and_claim(SEED, lease_owner="new")
            self.assertEqual(current["attempts"], old["attempts"] + 1)
            with self.assertRaises(LeaseLost):
                ledger.finish_claim(old["job_id"], "stale", _report(), actor="test",
                                    lease_owner="old", attempt=old["attempts"])
            with self.assertRaises(LeaseLost):
                ledger.complete_job(old["job_id"], "FAILED", "stale", lease_owner="old", attempt=old["attempts"])
            self.assertEqual(ledger.count(), 0)
            ledger.finish_claim(current["job_id"], "current", _report(), actor="test",
                                lease_owner="new", attempt=current["attempts"])
            self.assertEqual(ledger.count(), 1)
            self.assertEqual(ledger.jobs()[0]["status"], "DONE")
            with self.assertRaises(LeaseLost):
                ledger.finish_claim(current["job_id"], "current", _report(), actor="test",
                                    lease_owner="new", attempt=current["attempts"])
            self.assertEqual(ledger.count(), 1)

    def test_waiting_for_sqlite_lock_cannot_revive_an_expired_lease(self) -> None:
        with TemporaryDirectory() as directory:
            ledger = ResearchLedger(Path(directory))
            ledger.acquire_lease("owner", 0.08)
            outcome = []
            writer = sqlite3.connect(ledger.path)
            writer.execute("BEGIN IMMEDIATE")
            thread = threading.Thread(target=lambda: outcome.append(ledger.renew_lease("owner", 0.08)))
            thread.start()
            try:
                time.sleep(0.2)
            finally:
                writer.rollback()
                writer.close()
                thread.join(3)
            self.assertFalse(thread.is_alive())
            self.assertEqual(outcome, [False])

    def test_report_and_job_completion_roll_back_together_then_recover(self) -> None:
        with TemporaryDirectory() as directory:
            ledger = ResearchLedger(Path(directory))
            ledger.acquire_lease("owner", 30)
            claim = ledger.recover_and_claim(SEED, lease_owner="owner")
            with sqlite3.connect(ledger.path) as db:
                db.execute("CREATE TRIGGER inject_failure BEFORE UPDATE OF status ON jobs "
                           "WHEN NEW.status='DONE' BEGIN SELECT RAISE(ABORT, 'injected'); END")
            with self.assertRaises(sqlite3.IntegrityError):
                ledger.finish_claim(claim["job_id"], "test", _report(), actor="test",
                                    lease_owner="owner", attempt=claim["attempts"])
            self.assertEqual(ledger.count(), 0)
            self.assertEqual(ledger.jobs()[0]["status"], "RUNNING")
            with sqlite3.connect(ledger.path) as db:
                db.execute("DROP TRIGGER inject_failure")
            ledger.release_lease("owner")
            restored = tick(ResearchLedger(Path(directory)))
            self.assertEqual(restored["tick"]["action"], "ran")
            self.assertEqual(restored["run_count"], 1)
            self.assertEqual(ledger.jobs()[0]["attempts"], 2)

    def test_expired_claims_do_not_reset_retry_budget(self) -> None:
        with TemporaryDirectory() as directory:
            ledger = ResearchLedger(Path(directory))
            for attempt in range(1, 4):
                owner = f"owner-{attempt}"
                ledger.acquire_lease(owner, 30)
                claim = ledger.recover_and_claim(SEED, lease_owner=owner)
                self.assertEqual(claim["attempts"], attempt)
                ledger.expire_lease()
            restored = tick(ResearchLedger(Path(directory)))
            self.assertNotEqual(restored["tick"]["job_id"], SEED[0]["job_id"])
            first = ledger.jobs()[0]
            self.assertEqual(first["status"], "FAILED")
            self.assertEqual(first["attempts"], 3)


if __name__ == "__main__":
    unittest.main()
