"""The worker runs the queue by itself, then survives a restart."""

from __future__ import annotations

import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from gig.research_org.backlog import SEED
from gig.research_org.ledger import ResearchLedger
from gig.research_org.service import ResearchWorker
from gig.research_org.treasury_experiment import run_treasury


class WorkerTests(unittest.TestCase):
    def test_treasury_result_is_historical_and_below_a_coin_flip(self) -> None:
        _, report = run_treasury()
        self.assertEqual(report["classification"], "HISTORICAL")
        self.assertEqual(report["verdict"], "INCONCLUSIVE")
        self.assertEqual(report["economic_claim"], "NONE")
        self.assertEqual(report["publication_time"], "UNKNOWN")
        self.assertEqual(report["hits"], 105)
        self.assertEqual(report["pairs"], 228)
        self.assertLess(report["hit_rate"], 0.5)
        self.assertEqual(report["period"]["start"], "2023-01-03")
        self.assertEqual(report["period"]["end"], "2023-12-29")
        self.assertEqual(report["orders_submitted"], 0)
        self.assertIsNone(report["sharpe_after_cost"])
        self.assertIn("publicdomain/zero/1.0", report["dataset"]["license"])

    def test_one_start_runs_three_cycles_and_does_not_invent_another(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "org"
            worker = ResearchWorker(ResearchLedger(path), max_jobs=4)
            worker.start(0.05)
            worker.wait(30)
            ledger = ResearchLedger(path)
            self.assertEqual(ledger.count(), 3)
            ids = {row["hypothesis_id"] for row in ledger.status()["runs"]}
            self.assertIn("treasury-10y-lagged-sign-2023", ids)
            self.assertEqual(ledger.scheduler_state()["desired"], "PAUSED")
            self.assertEqual(ledger.scheduler_state()["reason"], "approved backlog is empty")
            self.assertEqual(ledger.scheduler_state()["candidates"], 1)
            again = ResearchWorker(ledger, max_jobs=4)
            again.start(0.05)
            again.wait(10)
            self.assertEqual(ResearchLedger(path).count(), 3)

    def test_pause_holds_the_queue_and_a_new_process_finishes_it(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "org"
            first = ResearchWorker(ResearchLedger(path), max_jobs=4)
            first.start(0.8)
            for _ in range(100):
                if ResearchLedger(path).count() >= 1:
                    break
                time.sleep(0.02)
            first.pause()
            held = ResearchLedger(path).count()
            time.sleep(1.0)
            self.assertGreaterEqual(held, 1)
            self.assertEqual(ResearchLedger(path).count(), held)
            self.assertLess(held, 3)
            first.stop()
            second = ResearchWorker(ResearchLedger(path), max_jobs=4)
            second.start(0.05)
            second.wait(30)
            self.assertEqual(ResearchLedger(path).count(), 3)

    def test_expired_lease_is_reclaimed(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "org"
            ledger = ResearchLedger(path)
            claimed = ledger.recover_and_claim(SEED)
            self.assertEqual(claimed["job_id"], "job-lagged-sign-v1")
            self.assertTrue(ledger.acquire_lease("dead-worker", 60))
            ledger.expire_lease()
            worker = ResearchWorker(ledger, max_jobs=4)
            worker.start(0.05)
            worker.wait(30)
            self.assertGreaterEqual(ResearchLedger(path).count(), 1)
            lagged = next(row for row in ResearchLedger(path).jobs() if row["job_id"] == "job-lagged-sign-v1")
            self.assertEqual(lagged["status"], "DONE")

    def test_storage_budget_pauses_before_a_job(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "org"
            worker = ResearchWorker(ResearchLedger(path), max_db_bytes=1)
            worker.start(0.05)
            worker.wait(5)
            state = ResearchLedger(path).scheduler_state()
            self.assertEqual(state["desired"], "PAUSED")
            self.assertEqual(state["reason"], "storage budget")
            self.assertEqual(ResearchLedger(path).count(), 0)

    def test_shutdown_keeps_running_and_the_next_process_finishes(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "org"
            first = ResearchWorker(ResearchLedger(path), max_jobs=4)
            first.start(2.0)
            for _ in range(200):
                if ResearchLedger(path).count() >= 1:
                    break
                time.sleep(0.02)
            first.shutdown()
            held = ResearchLedger(path).count()
            self.assertGreaterEqual(held, 1)
            self.assertLess(held, 3)
            self.assertEqual(ResearchLedger(path).scheduler_state()["desired"], "RUNNING")
            second = ResearchWorker(ResearchLedger(path), max_jobs=4)
            second.recover()
            second.wait(30)
            self.assertEqual(ResearchLedger(path).count(), 3)
            self.assertEqual(ResearchLedger(path).scheduler_state()["desired"], "PAUSED")

    def test_three_failures_pause_instead_of_continuing(self) -> None:
        def boom():
            raise RuntimeError("experiment failed")

        with TemporaryDirectory() as directory, patch.dict(
            "gig.research_org.scheduler.RUNNERS", {"lagged_sign": boom}
        ):
            path = Path(directory) / "org"
            worker = ResearchWorker(ResearchLedger(path), max_jobs=4)
            worker.start(0.05)
            worker.wait(20)
            lagged = next(row for row in ResearchLedger(path).jobs() if row["job_id"] == "job-lagged-sign-v1")
            self.assertEqual(lagged["status"], "FAILED")
            self.assertEqual(lagged["attempts"], 3)
            self.assertEqual(ResearchLedger(path).count(), 0)
            self.assertEqual(ResearchLedger(path).scheduler_state()["reason"], "failure escalation")
            self.assertFalse(any(row["job_id"] == "job-futures-tsmom-v1" and row["status"] == "DONE"
                                  for row in ResearchLedger(path).jobs()))

    def test_one_http_start_records_three_runs(self) -> None:
        import asyncio

        import httpx

        from gig.service.app import create_app

        async def run() -> dict:
            app = create_app(state_dir=Path(directory))
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                started = await client.post("/api/v1/research-org/scheduler",
                                            json={"action": "start", "cadence_seconds": 0.05})
                assert started.status_code == 200
                body = started.json()
                for _ in range(200):
                    body = (await client.get("/api/v1/research-org/status")).json()
                    state = body["data"]["scheduler_state"]
                    if body["data"]["run_count"] >= 3 and state["desired"] == "PAUSED":
                        break
                    await asyncio.sleep(0.05)
            return body

        with TemporaryDirectory() as directory:
            body = asyncio.run(run())
        self.assertEqual(body["data"]["run_count"], 3)
        self.assertEqual(body["data"]["scheduler_state"]["reason"], "approved backlog is empty")
        ids = {row["report"]["hypothesis_id"] for row in body["data"]["runs"]}
        self.assertIn("treasury-10y-lagged-sign-2023", ids)
        self.assertEqual(body["data"]["orders_submitted"], 0)
        self.assertFalse(body["data"]["continuous_process"])


if __name__ == "__main__":
    unittest.main()
