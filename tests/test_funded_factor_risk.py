"""Knowledge-time metadata, option-delta budgets and immutable NAV drawdowns."""
from __future__ import annotations

import copy
import tempfile
import unittest
from datetime import timedelta
from decimal import localcontext
from pathlib import Path

import pandas as pd

from gig.execution.oms import OfflineOMS
from gig.service.funded_risk_workspace import FundedRiskWorkspace, RiskUnavailable
from gig.service.serialization import loads
from test_funded_accounting import NOW, funded_fill, reserve_request, seed_funded
from test_funded_risk import model_run, risk_request


class FundedFactorRiskTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.base = Path(self.temp.name)/"state"
        self.book = seed_funded(self.base, "10000")
        self.ids = [model_run(self.base), model_run(self.base, "B", "BP", "PUT")]
        for client, symbol in (("a", "AC"), ("b", "BP")):
            self.book.run(reserve_request(client, symbol, quantity="1", limit_price="1", fee_budget="1"))
            self.book.run({"operation": "dispatch", "now": NOW.isoformat()})
            funded_fill(self.book, client, "fill-"+client, price="1")
        self.request = risk_request(self.ids)
        self.mark(self.request["marks"])
        self.workspace = FundedRiskWorkspace(self.base)

    def tearDown(self):
        self.temp.cleanup()

    def mark(self, marks, now=NOW):
        return self.book.run({"operation": "mark", "now": now.isoformat(), "marks": marks, "max_age_seconds": 60})

    def policy(self):
        request = copy.deepcopy(self.request)
        request.update(metadata={"A": {"sector": "Technology", "factors": {"market": 1.}},
                                 "B": {"sector": "Financials", "factors": {"market": 1.}}},
                       metadata_source_identity="authored-sector-and-beta-fixture",
                       metadata_available_at=NOW.isoformat())
        request["limits"].update(max_sector_gross_notional=100000., max_abs_factor_exposure={"market": 100000.})
        return request

    def test_old_requests_and_artifact_shape_remain_unchanged(self):
        request = FundedRiskWorkspace._request(self.request)
        self.assertFalse({"metadata", "metadata_available_at", "metadata_source_identity"} & request.keys())
        result = self.workspace.run(self.request).to_dict()
        self.assertNotIn("sector_factor_risk", result["data"])
        self.assertNotIn("drawdown_risk", result["data"])
        self.assertNotIn("risk_drawdown_samples", result["data"]["account_snapshot"])
        self.assertEqual(self.workspace.replay({"report_id": result["data"]["report_id"]}).to_dict(), result)

    def test_option_delta_factor_and_sector_reference_not_premium_or_netting(self):
        result = self.workspace.run(self.policy()).to_dict()
        self.assertEqual(result["state"], "AVAILABLE")
        data, factor = result["data"], result["data"]["sector_factor_risk"]
        expected = sum(abs(r["delta_notional"]) for r in data["rows"])
        signed = sum(r["delta_notional"] for r in data["rows"])
        self.assertAlmostEqual(factor["factor_abs_budget_usd"]["market"], expected, places=10)
        self.assertAlmostEqual(factor["current_signed_factor_exposure_usd"]["market"], signed, places=10)
        self.assertGreater(expected, 20*data["metrics"]["max_gross_notional"])
        self.assertLess(abs(signed), expected/10)
        self.assertAlmostEqual(sum(factor["sector_delta_gross_usd"].values()), expected, places=10)
        self.assertFalse(factor["calibrated_factor_model"])

    def test_sector_and_factor_breaches_save_error_and_existing_kill_control(self):
        request = self.policy()
        request["limits"]["max_abs_factor_exposure"]["market"] = 500
        request["limits"]["max_sector_gross_notional"] = 1000
        result = self.workspace.run(request).to_dict()
        self.assertEqual(result["state"], "ERROR")
        self.assertIsNone(result["data"])
        saved = self.workspace.load(result["provenance"]["model_identity"])
        self.assertIn("factor:market", saved["violations"])
        self.assertIn("sector:Technology", saved["violations"])
        with OfflineOMS(self.base/"funded-account/oms.sqlite") as oms:
            self.assertTrue(oms.kill_switch)

    def test_missing_metadata_provenance_or_loadings_is_unknown_not_zero(self):
        for field in ("metadata", "metadata_available_at", "metadata_source_identity"):
            request = self.policy()
            request.pop(field)
            with self.subTest(field=field):
                self.assertEqual(self.workspace.run(request).to_dict()["state"], "UNKNOWN")
        request = self.policy()
        request["metadata"]["A"]["factors"]["market"] = None
        self.assertEqual(self.workspace.run(request).to_dict()["state"], "UNKNOWN")
        with OfflineOMS(self.base/"funded-account/oms.sqlite") as oms:
            self.assertFalse(oms.kill_switch)
            self.assertIsNone(oms.connection.execute("SELECT wire FROM funded_risk_policy WHERE name='active'").fetchone())

    def test_metadata_nanosecond_freshness_and_future_boundaries(self):
        request = self.policy()
        request["metadata_available_at"] = (pd.Timestamp(NOW)-pd.Timedelta(seconds=60, nanoseconds=1)).isoformat()
        self.assertEqual(self.workspace.run(request).to_dict()["state"], "UNKNOWN")
        request["metadata_available_at"] = (pd.Timestamp(NOW)+pd.Timedelta(nanoseconds=1)).isoformat()
        self.assertEqual(self.workspace.run(request).to_dict()["state"], "UNKNOWN")
        request["metadata_available_at"] = (pd.Timestamp(NOW)-pd.Timedelta(seconds=60)).isoformat()
        self.assertEqual(self.workspace.run(request).to_dict()["state"], "AVAILABLE")

    def test_unknown_replacement_does_not_disable_a_previously_admitted_policy(self):
        original = self.policy()
        self.workspace.run(original)
        unknown = self.policy()
        unknown.pop("metadata_source_identity")
        unknown["limits"]["max_abs_factor_exposure"]["market"] = 0
        self.assertEqual(self.workspace.run(unknown).to_dict()["state"], "UNKNOWN")
        with OfflineOMS(self.base/"funded-account/oms.sqlite") as oms:
            policy = loads(oms.connection.execute("SELECT wire FROM funded_risk_policy WHERE name='active'").fetchone()[0])
        self.assertEqual(policy, FundedRiskWorkspace._request(original))

    def test_candidate_order_includes_all_working_option_delta_before_admission(self):
        request = self.policy()
        request["limits"]["max_abs_factor_exposure"]["market"] = 12000
        self.assertEqual(self.workspace.run(request).to_dict()["state"], "AVAILABLE")
        with self.assertRaises(ValueError):
            self.book.run(reserve_request("blocked-factor", "AC", quantity="1", limit_price="1", fee_budget="1"))
        with OfflineOMS(self.base/"funded-account/oms.sqlite") as oms:
            self.assertEqual(oms.connection.execute("SELECT COUNT(*) FROM orders").fetchone()[0], 2)
            self.assertFalse(oms.kill_switch)

    def test_post_fill_stale_metadata_preserves_fill_and_atomically_engages_kill(self):
        request = self.policy()
        request["metadata_available_at"] = (NOW-timedelta(seconds=59)).isoformat()
        self.workspace.run(request)
        self.book.run(reserve_request("later", "AC", quantity="1", limit_price="1", fee_budget="1"))
        self.book.run({"operation": "dispatch", "now": NOW.isoformat()})
        result = funded_fill(self.book, "later", "later-fill", price="1", now=NOW+timedelta(seconds=2)).to_dict()
        self.assertEqual(result["data"]["account"]["effective_positions"]["AC"], "2")
        self.assertTrue(result["data"]["metrics"]["kill_switch"])
        self.assertEqual(self.workspace.section("risk", NOW+timedelta(seconds=2), 60).to_dict()["state"], "ERROR")

    def test_drawdown_uses_observed_peak_and_frozen_sources_after_restart(self):
        high = copy.deepcopy(self.request["marks"])
        high["AC"]["price"] = "2"
        self.mark(high)
        self.mark(self.request["marks"])
        request = copy.deepcopy(self.request)
        request["limits"]["max_drawdown"] = .005
        result = self.workspace.run(request).to_dict()
        self.assertEqual(result["state"], "ERROR")
        identity = result["provenance"]["model_identity"]
        data = self.workspace.load(identity)
        drawdown = data["drawdown_risk"]
        self.assertAlmostEqual(drawdown["current_drawdown"], 9999.8/10099.8-1, places=14)
        self.assertEqual(drawdown["peak_nav"], "10099.8")
        self.assertEqual(drawdown["unobserved_peak_state"], "UNKNOWN")
        self.assertIn("max_drawdown", data["violations"])
        later_marks = copy.deepcopy(high)
        for mark in later_marks.values():
            mark["as_of"] = (NOW+timedelta(seconds=1)).isoformat()
        self.mark(later_marks, NOW+timedelta(seconds=1))
        self.assertEqual(FundedRiskWorkspace(self.base).replay({"report_id": identity}).to_dict(), result)

    def test_drawdown_does_not_depend_on_process_decimal_precision(self):
        high = copy.deepcopy(self.request["marks"])
        high["AC"]["price"] = "2.1234567"
        self.mark(high)
        low = copy.deepcopy(self.request["marks"])
        low["AC"]["price"] = "1.0001234"
        self.mark(low)
        request = copy.deepcopy(self.request)
        request["marks"] = low
        request["limits"]["max_drawdown"] = .1
        first = self.workspace.run(request).to_dict()
        with localcontext() as context:
            context.prec = 5
            self.assertEqual(self.workspace.run(request).to_dict(), first)
            self.assertEqual(self.workspace.replay({"report_id": first["data"]["report_id"]}).to_dict(), first)

    def test_factor_breach_rolls_back_index_policy_and_kill_together(self):
        request = self.policy()
        request["limits"]["max_sector_gross_notional"] = 0
        def fail(phase):
            if phase == "risk_controls_applied":
                raise RuntimeError("injected interruption")
        with self.assertRaises(RuntimeError):
            FundedRiskWorkspace(self.base, fault_hook=fail).run(request)
        self.assertIsNone(self.workspace.section("risk", NOW, 60))
        with OfflineOMS(self.base/"funded-account/oms.sqlite") as oms:
            self.assertFalse(oms.kill_switch)

    def test_unresolved_lifecycle_is_not_admitted_as_known_exposure(self):
        original = self.workspace.run(self.request).to_dict()["data"]
        snapshot = copy.deepcopy(original["account_snapshot"])
        snapshot["account"]["reconciliation_state"] = "UNKNOWN"
        with self.assertRaises(RiskUnavailable):
            self.workspace._evaluate(None, original["request"], snapshot_override=snapshot)

    def test_drawdown_with_missing_publication_history_is_unavailable_not_zero(self):
        request = copy.deepcopy(self.request)
        request["limits"]["max_drawdown"] = .1
        # Corrupt only this test-owned publication index, retaining the simulated
        # account, to prove a missing peak history cannot become a zero result.
        with OfflineOMS(self.base/"funded-account/oms.sqlite") as oms:
            oms.connection.execute("DELETE FROM funded_snapshots")
        result = self.workspace.run(request).to_dict()
        self.assertEqual(result["state"], "UNAVAILABLE")
        self.assertIsNone(result["data"])
