"""Untrusted worker budgets and cadence must not disable resource controls."""
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from gig.research_org.ledger import ResearchLedger
from gig.research_org.service import ResearchWorker, _cadence


class ResearchLimitTests(TestCase):
    def test_invalid_integer_budgets_do_not_create_state(self) -> None:
        with TemporaryDirectory() as directory:
            ledger = ResearchLedger(Path(directory))
            for name in ("max_jobs", "max_db_bytes", "max_rss_kb"):
                for value in (True, False, 0, -1, 1.0, "1", None, float("nan"), float("inf")):
                    with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                        ResearchWorker(ledger, **{name: value})
                    self.assertFalse(ledger.path.exists())

    def test_invalid_numeric_budgets_do_not_create_state(self) -> None:
        with TemporaryDirectory() as directory:
            ledger = ResearchLedger(Path(directory))
            for name in ("lease_seconds", "max_cpu_seconds"):
                for value in (True, False, 0, -1, "1", None, float("nan"), float("inf"), -float("inf"), 10**1000):
                    with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                        ResearchWorker(ledger, **{name: value})
                    self.assertFalse(ledger.path.exists())
            self.assertIsNotNone(ResearchWorker(ledger, lease_seconds=1, max_cpu_seconds=0.1))
            with self.assertRaises(ValueError):
                ResearchWorker(ledger, lease_seconds=3600.001)

    def test_cadence_rejects_nonfinite_bools_and_out_of_range(self) -> None:
        for value in (True, False, 0, -1, 0.049, 3600.001, "1", None, float("nan"), float("inf"), 10**1000):
            with self.subTest(value=value), self.assertRaises(ValueError):
                _cadence(value)
        self.assertEqual(_cadence(0.05), 0.05)
        self.assertEqual(_cadence(3600), 3600.0)
