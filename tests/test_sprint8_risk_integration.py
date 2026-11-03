"""Accrued income affects NAV, never cash admission or transactional risk safety."""
from __future__ import annotations

import copy
import sqlite3
import tempfile
import unittest
from datetime import timedelta
from decimal import Decimal
from pathlib import Path
from unittest.mock import patch

from gig.execution.oms import OfflineOMS
from gig.service.funded_risk_workspace import FundedRiskWorkspace
from gig.service.lifecycle_workspace import FundedLifecycleWorkspace
from test_funded_accounting import NOW, funded_fill, reserve_request, seed_funded
from test_funded_lifecycle import dividend
from test_funded_risk import risk_request


class AccruedDividendRiskTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.base = Path(self.temporary.name)/"state"
        self.book = seed_funded(self.base, "1000")
        self.book.run(reserve_request("stock", "A", quantity="10", limit_price="10", fee_budget="1"))
        self.book.run({"operation": "dispatch", "now": NOW.isoformat()})
        funded_fill(self.book, "stock", "stock-fill", quantity="10", price="10", fee="0.2")
        self.lifecycle = FundedLifecycleWorkspace(self.base)
        self.lifecycle.run({"operation": "admit", "now": NOW.isoformat(), "action": dividend()})
        self.ex = NOW+timedelta(days=3)
        self.marks = {"A": {"price": "9", "as_of": self.ex.isoformat()}}
        self.snapshot = self.book.run({"operation": "mark", "now": self.ex.isoformat(), "marks": self.marks, "max_age_seconds": 60}).to_dict()["data"]
        self.risk = FundedRiskWorkspace(self.base)
        self.request = risk_request([], now=self.ex.isoformat(), marks=self.marks)

    def tearDown(self):
        self.temporary.cleanup()

    def test_receivable_nav_conservation_never_supplies_cash_risk_budget(self):
        account = self.snapshot["account"]
        self.assertEqual(Decimal(account["dividend_receivable"]), 10)
        self.assertEqual(Decimal(account["nav"]), Decimal("999.8"))
        self.assertEqual(Decimal(account["available_cash"]), Decimal("899.8"))
        passed = self.risk.run(self.request).to_dict()
        self.assertEqual(passed["state"], "AVAILABLE")
        self.assertEqual(Decimal(passed["data"]["account_nav"]), Decimal("999.8"))
        self.assertFalse(passed["data"]["fully_funded_certified"])
        limit = copy.deepcopy(self.request)
        limit["limits"]["min_available_cash"] = 900
        rejected = self.risk.run(limit).to_dict()
        self.assertEqual(rejected["state"], "ERROR")
        with OfflineOMS(self.base/"funded-account/oms.sqlite") as oms:
            self.assertTrue(oms.kill_switch)
        replay = self.risk.replay({"report_id": passed["data"]["report_id"]}).to_dict()
        self.assertEqual(replay, passed)

    def test_risk_publication_failure_rolls_back_cancellation_accrual_and_source(self):
        self.assertEqual(self.risk.run(self.request).to_dict()["state"], "AVAILABLE")
        cancellation = dividend(revision=2, amount="0", known=self.ex)
        cancellation["status"] = "CANCELLED"
        with patch.object(FundedRiskWorkspace, "_publish", side_effect=RuntimeError("injected risk publication fault")), self.assertRaises(RuntimeError):
            self.lifecycle.run({"operation": "admit", "now": self.ex.isoformat(), "action": cancellation})
        after = self.book.run({"operation": "state", "now": self.ex.isoformat()}).to_dict()["data"]
        self.assertEqual(after["account"]["dividend_receivable"], self.snapshot["account"]["dividend_receivable"])
        self.assertEqual(after["account"]["cash"], self.snapshot["account"]["cash"])
        self.assertEqual(after["account"]["nav"], self.snapshot["account"]["nav"])
        with sqlite3.connect(self.base/"funded-account/oms.sqlite") as connection:
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM funded_lifecycle_actions").fetchone()[0], 1)
