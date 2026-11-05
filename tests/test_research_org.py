"""The research organization runs one synthetic experiment and keeps the record."""

from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from gig.research_org.cycle import run_cycle, status
from gig.research_org.experiment import (
    authored_returns,
    causal_position,
    evaluate,
    lookahead_position,
    run_seeded,
)
from gig.research_org.ledger import ResearchLedger


class ResearchOrganizationTests(unittest.TestCase):
    def test_reading_creates_nothing(self) -> None:
        with TemporaryDirectory() as directory:
            ledger = ResearchLedger(Path(directory) / "org")
            self.assertEqual(status(ledger)["run_count"], 0)
            self.assertFalse(ledger.path.exists())

    def test_cycle_is_deterministic_inconclusive_and_deduplicated(self) -> None:
        with TemporaryDirectory() as directory:
            ledger = ResearchLedger(Path(directory) / "org")
            first = run_cycle(ledger)
            second = run_cycle(ledger)
        report = first["last"]
        self.assertEqual(report["verdict"], "INCONCLUSIVE")
        self.assertEqual(report["classification"], "SYNTHETIC")
        self.assertFalse(report["lookahead_detected"])
        self.assertEqual(report["orders_submitted"], 0)
        self.assertFalse(report["code_rewritten"])
        self.assertEqual(report["economic_claim"], "NONE")
        self.assertEqual(first["run_count"], 1)
        self.assertTrue(second["deduplicated"])
        self.assertEqual(second["run_count"], 1)
        self.assertEqual(second["last"], report)
        self.assertNotIn("POSITIVE", report["verdict"])

    def test_restart_reads_the_same_finding(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "org"
            run_cycle(ResearchLedger(path))
            restored = status(ResearchLedger(path))
        self.assertEqual(restored["run_count"], 1)
        self.assertEqual(restored["runs"][0]["report"]["verdict"], "INCONCLUSIVE")

    def test_same_bar_position_is_rejected(self) -> None:
        returns = authored_returns()
        report = evaluate(returns, lookahead_position(returns), classification="SYNTHETIC")
        self.assertTrue(report["lookahead_detected"])
        self.assertEqual(report["verdict"], "REJECTED_LOOKAHEAD")
        causal = evaluate(returns, causal_position(returns), classification="SYNTHETIC")
        self.assertFalse(causal["lookahead_detected"])

    def test_seeded_run_matches_the_causal_evaluation(self) -> None:
        _, report = run_seeded()
        direct = evaluate(authored_returns(), causal_position(authored_returns()), classification="SYNTHETIC")
        self.assertEqual(report["sharpe_after_cost"], direct["sharpe_after_cost"])
        self.assertTrue(np.isfinite(report["sharpe_after_cost"]))

    def test_catalog_does_not_pretend_registered_strategies_ran(self) -> None:
        with TemporaryDirectory() as directory:
            catalog = {row["id"]: row["status"] for row in status(ResearchLedger(Path(directory)))["catalog"]}
        self.assertEqual(catalog["equity_ls"], "REGISTERED_NOT_EXECUTED")
        self.assertEqual(catalog["futures_tsmom"], "REGISTERED_NOT_EXECUTED")
        self.assertEqual(catalog["options_vrp"], "REGISTERED_NOT_EXECUTED")


if __name__ == "__main__":
    unittest.main()
