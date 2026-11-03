"""Actual funded positions, model budgets, admission and crash-atomic kill risk."""
from __future__ import annotations

import copy
import sqlite3
import tempfile
import unittest
from datetime import timedelta
from pathlib import Path

from gig.execution.oms import OfflineOMS
from gig.options import black_scholes
from gig.risk.analytics import historical_loss_risk
from gig.service.funded_risk_workspace import FundedRiskWorkspace
from gig.service.options_workspace import OptionsWorkspace
from test_funded_accounting import EXPIRY, NOW, funded_fill, reserve_request, seed_funded


def model_run(base, underlying="A", contract="AC", right="CALL"):
    workspace = OptionsWorkspace(base)
    mid = black_scholes(100, 100, .02, .01, .2, 1/365, right == "CALL")["price"]
    workspace.ingest({"source": "authored-risk-"+underlying, "classification": "SYNTHETIC", "feed": "AUTHORED_FIXTURE",
        "underlying_id": underlying, "currency": "USD", "event_at": NOW.isoformat(), "acquired_at": NOW.isoformat(),
        "knowledge_at": NOW.isoformat(), "spot": 100, "spot_source_identity": "authored-100", "spot_event_at": NOW.isoformat(),
        "spot_knowledge_at": NOW.isoformat(), "quotes": [{"contract_id": contract, "expiry": EXPIRY.isoformat(),
            "strike": 100, "right": right, "multiplier": 100, "style": "EUROPEAN", "bid": mid-.005, "ask": mid+.005,
            "volume": None, "open_interest": None}]})
    # Deliberately unrelated hypothetical size; risk must use the actual OMS 1.
    return workspace.run({"source": "authored-risk-"+underlying, "classification": "SYNTHETIC", "underlying_id": underlying,
        "observation_at": NOW.isoformat(), "as_of": NOW.isoformat(), "max_age_seconds": 60, "rate": .02, "dividend_yield": .01,
        "positions": [{"contract_id": contract, "quantity": 999}]}).to_dict()["data"]["run_id"]


def risk_request(ids, **changes):
    return {"now": NOW.isoformat(), "max_age_seconds": 60, "options_run_ids": ids,
        "marks": {s: {"price": "100" if s in {"A", "B"} else "1", "as_of": NOW.isoformat()} for s in ("A", "B", "AC", "BP")},
        "limits": {key: 1e9 for key in ("max_gross_notional", "max_name_notional", "max_abs_delta_notional",
            "max_abs_gamma_dollar", "max_abs_vega_per_point", "max_abs_theta_per_day", "max_scenario_loss")} | {"min_available_cash": 0},
        "scenarios": [{"name": "down", "spot_shocks": {"A": -.1, "B": -.1}, "vol_shocks": {"A": .02, "B": .02}, "elapsed_days": 0},
                      {"name": "up", "spot_shocks": {"A": .1, "B": .1}, "vol_shocks": {"A": 0, "B": 0}, "elapsed_days": 0}], **changes}


class FundedRiskTests(unittest.TestCase):
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
        self.book.run({"operation": "mark", "now": NOW.isoformat(), "marks": self.request["marks"], "max_age_seconds": 60})
        self.workspace = FundedRiskWorkspace(self.base)

    def tearDown(self):
        self.temp.cleanup()

    def oms(self):
        return OfflineOMS(self.base/"funded-account/oms.sqlite")

    def test_multi_underlying_actual_greeks_reference_not_research_quantities(self):
        result = self.workspace.run(self.request).to_dict()
        self.assertEqual(result["state"], "AVAILABLE")
        data = result["data"]
        self.assertEqual(data["account_nav"], "9999.8")
        rows = {r["instrument_id"]: r for r in data["rows"]}
        for symbol, call in (("AC", True), ("BP", False)):
            reference = black_scholes(100, 100, .02, .01, .2, 1/365, call)
            self.assertEqual(rows[symbol]["actual_quantity"], 1)
            self.assertAlmostEqual(rows[symbol]["gamma_dollar"], reference["gamma"]*100*10000, delta=1e-6)
            self.assertAlmostEqual(rows[symbol]["theta_per_day"], reference["theta"]*100, places=8)
        self.assertEqual(set(data["underlying_gross_notional"]), {"A", "B"})
        self.assertTrue(all(r["state"] == "UNKNOWN" for r in data["liquidity"]))
        self.assertEqual(data["historical_var_es"]["state"], "UNAVAILABLE")
        self.assertFalse(data["trading_approved"])
        self.assertNotEqual(rows["AC"]["model_mark_basis_difference"], 0)

    def test_replay_is_exact_after_restart_and_after_later_book_updates(self):
        result = self.workspace.run(self.request).to_dict()
        identity = result["data"]["report_id"]
        self.assertEqual(FundedRiskWorkspace(self.base).replay({"report_id": identity}).to_dict(), result)
        later = {**risk_request(self.ids), "now": (NOW+timedelta(seconds=1)).isoformat()}
        self.workspace.run(later)
        self.assertEqual(self.workspace.replay({"report_id": identity}).to_dict(), result)
        loaded = self.workspace.load(identity)
        loaded["rows"][0]["actual_quantity"] = 999
        self.assertEqual(self.workspace.load(identity)["rows"][0]["actual_quantity"], 1)

    def test_unknown_models_and_wrong_terms_cannot_turn_into_zero_risk(self):
        missing = {**self.request, "options_run_ids": self.ids[:1]}
        self.assertEqual(self.workspace.run(missing).to_dict()["state"], "UNAVAILABLE")
        self.assertIsNone(self.workspace.section("risk", NOW, 60))
        future = copy.deepcopy(self.request)
        future["now"] = (NOW-timedelta(seconds=1)).isoformat()
        for mark in future["marks"].values():
            mark["as_of"] = future["now"]
        self.assertEqual(self.workspace.run(future).to_dict()["state"], "UNAVAILABLE")

    def test_outstanding_orders_and_candidate_cash_minimum_are_enforced(self):
        request = copy.deepcopy(self.request)
        request["limits"]["min_available_cash"] = 9500
        self.workspace.run(request)
        with self.assertRaises(ValueError):
            self.book.run(reserve_request("cash-budget", "AC", quantity="4", limit_price="1", fee_budget="1"))
        with self.oms() as oms:
            self.assertEqual(oms.connection.execute("SELECT COUNT(*) FROM orders").fetchone()[0], 2)
        request["limits"]["min_available_cash"] = 0
        request["limits"]["max_name_notional"] = 150
        self.workspace.run(request)
        with self.assertRaises(ValueError):
            self.book.run(reserve_request("name-budget", "AC", quantity="1", limit_price="1", fee_budget="1"))

    def test_changed_marks_engage_kill_and_do_not_reset_incoming_accounting(self):
        self.workspace.run(self.request)
        marks = copy.deepcopy(self.request["marks"])
        marks["AC"]["price"] = "2"
        response = self.book.run({"operation": "mark", "now": NOW.isoformat(), "marks": marks, "max_age_seconds": 60}).to_dict()
        self.assertTrue(response["data"]["metrics"]["kill_switch"])
        self.assertEqual(response["data"]["account"]["effective_positions"]["AC"], "1")
        self.assertEqual(self.workspace.section("risk", NOW, 60).to_dict()["state"], "ERROR")

    def test_post_fill_stale_risk_preserves_fill_and_engages_kill_atomically(self):
        request = {**self.request, "max_age_seconds": 0}
        self.workspace.run(request)
        self.book.run(reserve_request("delayed", "AC", quantity="1", limit_price="1", fee_budget="1"))
        self.book.run({"operation": "dispatch", "now": NOW.isoformat()})
        later = NOW+timedelta(seconds=1)
        result = funded_fill(self.book, "delayed", "delayed-fill", price="1", now=later).to_dict()
        self.assertEqual(result["data"]["account"]["effective_positions"]["AC"], "2")
        self.assertTrue(result["data"]["metrics"]["kill_switch"])
        with self.oms() as oms:
            self.assertEqual(oms.connection.execute("SELECT COUNT(*) FROM executions WHERE execution_id='delayed-fill'").fetchone()[0], 1)

    def test_breach_index_and_kill_roll_back_together_on_failure(self):
        request = copy.deepcopy(self.request)
        request["limits"]["max_gross_notional"] = 0
        for phase in ("risk_indexed", "risk_controls_applied"):
            def fail(name, target=phase):
                if name == target:
                    raise RuntimeError("injected")
            with self.subTest(phase=phase), self.assertRaises(RuntimeError):
                FundedRiskWorkspace(self.base, fault_hook=fail).run(request)
            with self.oms() as oms:
                self.assertFalse(oms.kill_switch)
                if oms.connection.execute("SELECT 1 FROM sqlite_master WHERE name='funded_risk_reports'").fetchone():
                    self.assertEqual(oms.connection.execute("SELECT COUNT(*) FROM funded_risk_reports").fetchone()[0], 0)
        self.assertEqual(self.workspace.run(request).to_dict()["state"], "ERROR")
        with self.oms() as oms:
            self.assertTrue(oms.kill_switch)

    def test_requested_liquidity_limit_rejects_unknown_and_tamper_fails(self):
        request = {**self.request, "max_participation_rate": .1}
        result = self.workspace.run(request).to_dict()
        self.assertEqual(result["state"], "ERROR")
        report = result["provenance"]["model_identity"]
        with sqlite3.connect(self.base/"funded-account/oms.sqlite") as db:
            db.execute("UPDATE funded_risk_reports SET as_of='2020-01-01T00:00:00Z'")
        with self.assertRaisesRegex(ValueError, "index"):
            self.workspace.load(report)

    def test_precision_rejects_before_any_risk_admission(self):
        bad = {**self.request, "now": "2026-11-02T15:00:00.000000001Z"}
        with self.assertRaises(ValueError):
            self.workspace.run(bad)
        self.assertFalse((self.base/"funded-account/risk").exists())

    def historical_book(self, *, irregular=False):
        base = self.base/"history"
        book = seed_funded(base, "10000")
        book.run(reserve_request("stock", "A", quantity="10", limit_price="100", fee_budget="0"))
        book.run({"operation": "dispatch", "now": NOW.isoformat()})
        funded_fill(book, "stock", "stock-fill", quantity="10", price="100", fee="0")
        navs = []
        for index in range(21):
            stamp = NOW+timedelta(days=index+(1 if irregular and index >= 10 else 0))
            price = 100+index*(-1 if index % 3 == 0 else 1)
            book.run({"operation": "mark", "now": stamp.isoformat(), "max_age_seconds": 60,
                      "marks": {"A": {"price": str(price), "as_of": stamp.isoformat()}}})
            navs.append(9000+10*price)
        request = risk_request([], now=stamp.isoformat())
        request["marks"] = {"A": {"price": str(price), "as_of": stamp.isoformat()}}
        return FundedRiskWorkspace(base), request, navs

    def test_regular_funded_nav_history_has_reference_var_es_and_saved_sources(self):
        workspace, request, navs = self.historical_book()
        result = workspace.run(request).to_dict()
        self.assertEqual(result["state"], "AVAILABLE")
        risk = result["data"]["historical_var_es"]
        reference = historical_loss_risk([(b-a)/a for a, b in zip(navs, navs[1:], strict=False)], .99)
        self.assertEqual(risk["state"], "AVAILABLE")
        self.assertEqual(risk["observations"], 20)
        self.assertAlmostEqual(risk["var"], reference["var"], places=14)
        self.assertAlmostEqual(risk["es"], reference["expected_shortfall"], places=14)
        self.assertEqual(len(risk["source_snapshot_ids"]), 21)
        self.assertEqual(workspace.replay({"report_id": result["data"]["report_id"]}).to_dict(), result)

    def test_irregular_funded_nav_history_does_not_fabricate_daily_returns(self):
        workspace, request, _ = self.historical_book(irregular=True)
        risk = workspace.run(request).to_dict()["data"]["historical_var_es"]
        self.assertEqual(risk["state"], "UNAVAILABLE")
        self.assertIsNone(risk["var"])
