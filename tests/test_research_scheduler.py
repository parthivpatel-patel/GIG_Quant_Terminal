"""Backlog ticks, a futures experiment, and restart recovery."""

from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from gig.research_org.__main__ import main
from gig.research_org.futures_experiment import run_futures
from gig.research_org.ledger import ResearchLedger
from gig.research_org.scheduler import cancel_job, submit_objective, tick


class SchedulerTests(unittest.TestCase):
    def test_futures_experiment_is_inconclusive_and_reproducible(self) -> None:
        digest, report = run_futures()
        again, repeat = run_futures()
        self.assertEqual(digest, again)
        self.assertEqual(report["verdict"], repeat["verdict"])
        self.assertEqual(report["verdict"], "INCONCLUSIVE")
        self.assertEqual(report["classification"], "SYNTHETIC")
        self.assertEqual(report["evidence_class"], "INCONCLUSIVE")
        self.assertEqual(report["economic_claim"], "NONE")
        self.assertFalse(report["lookahead_detected"])
        self.assertEqual(report["orders_submitted"], 0)
        self.assertFalse(report["code_rewritten"])
        self.assertGreaterEqual(report["n_obs"], 63)
        self.assertEqual(report["holdout"]["n_obs"], 63.0)
        self.assertIsNotNone(report["sharpe_after_cost"])
        self.assertIsNotNone(report["sortino"])
        self.assertIsNotNone(report["max_drawdown"])
        self.assertIsNotNone(report["ann_turnover"])
        self.assertGreater(len(report["equity_curve"]), 2)
        self.assertIn("authored-futures-excess-v1", report["dataset"]["id"])
        self.assertNotIn("POSITIVE", report["verdict"])
        self.assertLess(report["total_return"], 0)

    def test_two_ticks_persist_across_restart_and_then_idle(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "org"
            first = tick(ResearchLedger(path))
            second = tick(ResearchLedger(path))
            restored = tick(ResearchLedger(path))
        self.assertEqual(first["tick"]["job_id"], "job-lagged-sign-v1")
        self.assertEqual(first["last"]["verdict"], "INCONCLUSIVE")
        self.assertEqual(second["tick"]["job_id"], "job-futures-tsmom-v1")
        self.assertEqual(second["last"]["hypothesis_id"], "futures-tsmom-authored-excess-v1")
        self.assertEqual(second["run_count"], 2)
        self.assertEqual(restored["tick"]["action"], "idle")
        self.assertEqual(restored["run_count"], 2)
        ids = {row["hypothesis_id"] for row in restored["runs"]}
        self.assertEqual(ids, {"lagged-sign-momentum-synthetic-v1", "futures-tsmom-authored-excess-v1"})
        catalog = {row["id"]: row["status"] for row in restored["catalog"]}
        self.assertEqual(catalog["futures_tsmom"], "EXECUTED_ON_AUTHORED_SYNTHETIC")
        self.assertEqual(catalog["equity_ls"], "REGISTERED_NOT_EXECUTED")
        self.assertEqual(catalog["options_vrp"], "REGISTERED_NOT_EXECUTED")

    def test_running_job_is_recovered_after_a_crash(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "org"
            ledger = ResearchLedger(path)
            ledger.ensure_job({"seq": 0, "job_id": "job-lagged-sign-v1",
                               "hypothesis_id": "lagged-sign-momentum-synthetic-v1"})
            ledger.complete_job("job-lagged-sign-v1", "QUEUED", None)
            # Leave the job claimed, as a process would if it died mid-run.
            claimed = ledger.recover_and_claim((
                {"seq": 0, "job_id": "job-lagged-sign-v1", "hypothesis_id": "lagged-sign-momentum-synthetic-v1",
                 "runner": "lagged_sign"},
                {"seq": 1, "job_id": "job-futures-tsmom-v1", "hypothesis_id": "futures-tsmom-authored-excess-v1",
                 "runner": "futures_tsmom"},
            ))
            self.assertEqual(claimed["job_id"], "job-lagged-sign-v1")
            self.assertEqual(ledger.jobs()[0]["status"], "RUNNING")
            resumed = tick(ResearchLedger(path))
        self.assertEqual(resumed["tick"]["action"], "ran")
        self.assertEqual(resumed["last"]["hypothesis_id"], "lagged-sign-momentum-synthetic-v1")
        self.assertEqual(resumed["run_count"], 1)

    def test_cancel_removes_a_queued_job_and_a_blocked_objective_adds_none(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "org"
            ledger = ResearchLedger(path)
            self.assertTrue(cancel_job(ledger, "job-futures-tsmom-v1"))
            blocked = submit_objective(
                ledger,
                "Investigate whether options-flow anomalies predict subsequent equity volatility after realistic costs.",
            )
            ran = tick(ledger)
        self.assertEqual(blocked["objective"]["disposition"], "BLOCKED_NO_DATASET")
        self.assertEqual(blocked["run_count"], 0)
        self.assertEqual(ran["tick"]["job_id"], "job-lagged-sign-v1")
        self.assertEqual(ran["run_count"], 1)
        states = {row["job_id"]: row["status"] for row in ran["backlog"]}
        self.assertEqual(states["job-futures-tsmom-v1"], "CANCELLED")

    def test_failed_runner_stays_queued_then_fails_closed(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "org"

            def boom():
                raise RuntimeError("boom")

            from gig.research_org import scheduler
            original = scheduler.RUNNERS["lagged_sign"]
            scheduler.RUNNERS["lagged_sign"] = boom
            try:
                outcomes = [tick(ResearchLedger(path)) for _ in range(3)]
            finally:
                scheduler.RUNNERS["lagged_sign"] = original
        self.assertTrue(all(item["tick"]["action"] == "failed" for item in outcomes))
        self.assertEqual(outcomes[-1]["run_count"], 0)
        lagged = next(row for row in outcomes[-1]["backlog"] if row["job_id"] == "job-lagged-sign-v1")
        self.assertEqual(lagged["status"], "FAILED")
        self.assertEqual(lagged["attempts"], 3)

    def test_compute_command_prints_the_futures_verdict(self) -> None:
        with TemporaryDirectory() as directory:
            self.assertEqual(main(["--compute", "futures-tsmom-authored-excess-v1"]), 0)
            self.assertEqual(main(["--state-dir", str(Path(directory) / "org"), "--ticks", "1"]), 0)


if __name__ == "__main__":
    unittest.main()
