"""Read-only equity/option overlays preserve accounting and source-clock honesty."""
from __future__ import annotations

import copy
import sqlite3
import tempfile
import unittest
from datetime import UTC, datetime, timedelta
from decimal import Decimal, localcontext
from pathlib import Path

from gig.execution.oms import OfflineOMS
from gig.service.contracts import EnvelopeSchema
from gig.service.options_workspace import OptionsWorkspace
from gig.service.overlay_workspace import OptionsOverlayWorkspace
from gig.service.portfolio_workspace import PortfolioWorkspace
from test_options_workspace import AT, option_fixture, option_request
from test_portfolio_ledger import populate

FILL_AT = datetime(2026, 11, 2, 15, tzinfo=UTC)
NOW = datetime(2026, 11, 2, 15, 0, 2, tzinfo=UTC)
CAPS = {"max_abs_delta_notional": 100000., "max_abs_gamma_dollar": 100000.,
        "max_abs_vega_per_point": 100000., "max_abs_scenario_loss": 100000.}


def equity_request(**changes):
    return {"operation": "reconcile", "initial_cash": "10000", "now": NOW.isoformat(),
        "marks": {"SYN100": {"price": "100", "as_of": AT}}, "max_age_seconds": 60,
        "input_classification": "SYNTHETIC", "source_identity": "authored-synchronous-underlying-100",
        "limits": {}, **changes}


class OptionsOverlayTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.base = Path(self.temp.name)/"state"
        self.workspace = OptionsOverlayWorkspace(self.base)

    def tearDown(self):
        self.temp.cleanup()

    def seed(self, *, quantity="10", positions=None):
        populate(self.base, symbol="SYN100", quantity=quantity, price="100", now=FILL_AT)
        checkpoint = PortfolioWorkspace(self.base).run(equity_request()).to_dict()["data"]
        options = OptionsWorkspace(self.base)
        options.ingest(option_fixture())
        request = option_request() if positions is None else option_request(positions=positions)
        run = options.run(request).to_dict()["data"]
        return {"checkpoint_id": checkpoint["checkpoint_id"], "options_run_id": run["run_id"],
            "now": NOW.isoformat(), "max_age_seconds": 60, "limits": copy.deepcopy(CAPS)}, checkpoint, run

    def oms_snapshot(self):
        with sqlite3.connect((self.base/"execution-engine/oms.sqlite").as_uri()+"?mode=ro", uri=True) as connection:
            return "\n".join(connection.iterdump())

    def test_combined_delta_scenario_and_unfinanced_cash_reference(self):
        request, checkpoint, run = self.seed()
        before = self.oms_snapshot()
        response = self.workspace.run(request).to_dict()
        EnvelopeSchema.model_validate(response)
        self.assertEqual(response["state"], "AVAILABLE")
        data = response["data"]
        exposure = run["portfolio_exposure"]
        expected_delta = (Decimal(10)+Decimal(str(exposure["delta"])))*100
        expected_gamma = Decimal(str(exposure["gamma"]))*10000
        self.assertEqual(Decimal(data["metrics"]["delta_notional"]), expected_delta)
        self.assertEqual(Decimal(data["metrics"]["gamma_dollar"]), expected_gamma)
        self.assertEqual(Decimal(data["equity_cash"]), 8999)
        self.assertEqual(Decimal(data["equity_nav"]), 9999)
        self.assertEqual(Decimal(data["modeled_overlay_value"]), Decimal(9999)+Decimal(str(exposure["price"])))
        for row, option in zip(data["rows"], run["scenarios"]["scenarios"], strict=True):
            equity = Decimal(1000)*Decimal(str(option["spot_shock"]))
            self.assertEqual(Decimal(row["equity_pnl"]), equity)
            self.assertEqual(Decimal(row["combined_model_pnl"]), equity+Decimal(str(option["model_pnl"])))
        self.assertEqual(data["overlay_financing"]["state"], "UNKNOWN")
        self.assertIsNone(data["overlay_financing"]["financed_nav"])
        self.assertFalse(data["trading_approved"])
        self.assertEqual(data["historical_tail_risk"]["state"], "UNAVAILABLE")
        self.assertEqual(before, self.oms_snapshot())
        self.assertEqual(checkpoint["account"]["cash"], data["equity_cash"])

    def test_limits_save_error_evidence_without_kill_or_any_order_mutation(self):
        request, _, _ = self.seed()
        before = self.oms_snapshot()
        request["limits"]["max_abs_delta_notional"] = 0
        response = self.workspace.run(request).to_dict()
        self.assertEqual(response["state"], "ERROR")
        self.assertIsNone(response["data"])
        identity = response["provenance"]["model_identity"]
        saved = self.workspace.load(identity)
        self.assertIn("max_abs_delta_notional", saved["violations"])
        self.assertEqual(self.workspace.replay({"overlay_id": identity}).to_dict(), response)
        self.assertEqual(self.workspace.section("risk", request["now"], None).to_dict()["state"], "ERROR")
        self.assertEqual(before, self.oms_snapshot())
        with OfflineOMS(self.base/"execution-engine/oms.sqlite") as oms:
            self.assertFalse(oms.kill_switch)

    def test_restart_replay_decimal_context_and_later_fill_do_not_rewrite_old_overlay(self):
        request, _, _ = self.seed()
        original = self.workspace.run(request).to_dict()
        identity = original["data"]["overlay_id"]
        with localcontext() as context:
            context.prec = 5
            self.assertEqual(OptionsOverlayWorkspace(self.base).run(request).to_dict(), original)
            self.assertEqual(OptionsOverlayWorkspace(self.base).replay({"overlay_id": identity}).to_dict(), original)
        populate(self.base, client="later-fill", symbol="SYN100", quantity="1", price="100", now=NOW+timedelta(seconds=1))
        # Historical replay verifies preserved source checkpoints, never approves a changed account.
        self.assertEqual(self.workspace.replay({"overlay_id": identity}).to_dict(), original)
        new_request = {**request, "now": (NOW+timedelta(seconds=2)).isoformat()}
        self.assertEqual(self.workspace.run(new_request).to_dict()["state"], "UNAVAILABLE")

    def test_zero_explicit_option_quantity_is_known_but_absent_positions_are_unknown(self):
        request, _, _ = self.seed(positions=[{"contract_id": "SYN100-6-100-CALL", "quantity": 0}])
        response = self.workspace.run(request).to_dict()
        self.assertEqual(response["state"], "AVAILABLE")
        self.assertEqual(Decimal(response["data"]["metrics"]["delta_notional"]), 1000)
        self.assertEqual(Decimal(response["data"]["metrics"]["gamma_dollar"]), 0)
        self.assertEqual(Decimal(response["data"]["option_model_value"]), 0)
        run = OptionsWorkspace(self.base).run(option_request(positions=[])).to_dict()["data"]
        result = self.workspace.run({**request, "options_run_id": run["run_id"]}).to_dict()
        self.assertEqual(result["state"], "UNKNOWN")
        self.assertIsNone(result["data"])
        # An explicitly flat reconciled book and explicit zero option quantity
        # produce known zeros; missing position evidence above does not.
        flat_at = NOW+timedelta(seconds=1)
        populate(self.base, client="flatten", symbol="SYN100", side="sell", quantity="10", price="100", fee="0", now=flat_at)
        flat = PortfolioWorkspace(self.base).run(equity_request(now=flat_at.isoformat(), marks={})).to_dict()["data"]
        zero = self.workspace.run({**request, "now": flat_at.isoformat(), "checkpoint_id": flat["checkpoint_id"],
            "limits": dict.fromkeys(CAPS, 0)}).to_dict()
        self.assertEqual(zero["state"], "AVAILABLE")
        for key in ("delta_notional", "gamma_dollar", "vega_per_point", "worst_scenario_loss"):
            self.assertEqual(Decimal(zero["data"]["metrics"][key]), 0)

    def test_missing_stale_future_and_nanosecond_cutoffs_are_explicit(self):
        missing = {"checkpoint_id": "1"*64, "options_run_id": "2"*64, "now": NOW.isoformat(), "max_age_seconds": 60, "limits": CAPS}
        self.assertEqual(self.workspace.run(missing).to_dict()["state"], "UNAVAILABLE")
        self.assertFalse(self.base.exists())
        request, _, _ = self.seed()
        self.assertEqual(self.workspace.run({**request, "now": "2026-11-02T15:00:01.999999999Z"}).to_dict()["state"], "UNKNOWN")
        self.assertEqual(self.workspace.run({**request, "max_age_seconds": 1}).to_dict()["state"], "UNAVAILABLE")
        result = self.workspace.run(request).to_dict()
        self.assertEqual(self.workspace.section("risk", "2026-11-02T15:00:01.999999999Z", 60), None)
        self.assertEqual(self.workspace.section("risk", request["now"], None).to_dict()["state"], "UNKNOWN")
        self.assertEqual(self.workspace.section("risk", "2026-11-02T15:02:00Z", 60).to_dict()["state"], "STALE")
        self.assertEqual(self.workspace.replay({"overlay_id": result["data"]["overlay_id"]}).to_dict(), result)

    def test_older_error_mismatched_marks_and_other_underlyings_reject(self):
        request, _, _ = self.seed()
        latest = PortfolioWorkspace(self.base).run(equity_request(now=(NOW+timedelta(seconds=1)).isoformat(),
            marks={"SYN100": {"price": "101", "as_of": AT}})).to_dict()["data"]
        later = {**request, "now": (NOW+timedelta(seconds=1)).isoformat()}
        self.assertEqual(self.workspace.run(later).to_dict()["state"], "UNAVAILABLE")
        self.assertEqual(self.workspace.run({**later, "checkpoint_id": latest["checkpoint_id"]}).to_dict()["state"], "UNAVAILABLE")
        errored = PortfolioWorkspace(self.base).run(equity_request(now=(NOW+timedelta(seconds=2)).isoformat(), limits={"max_gross": 0})).to_dict()
        error_id = errored["provenance"]["source_identity"]
        self.assertEqual(self.workspace.run({**later, "now": (NOW+timedelta(seconds=2)).isoformat(), "checkpoint_id": error_id}).to_dict()["state"], "UNAVAILABLE")
        populate(self.base, client="other", symbol="OTHER", quantity="1", price="1", now=NOW+timedelta(seconds=3))
        other = PortfolioWorkspace(self.base).run(equity_request(now=(NOW+timedelta(seconds=3)).isoformat(),
            marks={"SYN100": {"price": "100", "as_of": AT}, "OTHER": {"price": "1", "as_of": AT}})).to_dict()["data"]
        self.assertEqual(self.workspace.run({**request, "now": (NOW+timedelta(seconds=3)).isoformat(), "checkpoint_id": other["checkpoint_id"]}).to_dict()["state"], "UNAVAILABLE")

    def test_source_position_revision_index_tamper_and_invalid_limits_reject(self):
        request, _, _ = self.seed()
        for changes in ({"max_age_seconds": None}, {"limits": {}},
                        {"limits": {**CAPS, "max_abs_scenario_loss": -1}},
                        {"limits": {**CAPS, "max_abs_vega_per_point": True}}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.workspace.run({**request, **changes})
        result = self.workspace.run(request).to_dict()
        identity = result["data"]["overlay_id"]
        with sqlite3.connect(self.base/"options-overlays/overlays.sqlite") as connection:
            connection.execute("UPDATE overlays SET checkpoint_id=? WHERE overlay_id=?", ("0"*64, identity))
        with self.assertRaises(ValueError):
            self.workspace.load(identity)
        with sqlite3.connect(self.base/"execution-engine/oms.sqlite") as connection:
            connection.execute("UPDATE positions SET quantity='11' WHERE symbol='SYN100'")
        self.assertEqual(self.workspace.run(request).to_dict()["state"], "UNAVAILABLE")
        with sqlite3.connect(self.base/"execution-engine/oms.sqlite") as connection:
            connection.execute("UPDATE positions SET quantity='10' WHERE symbol='SYN100'")
            connection.execute("UPDATE executions SET price='101'")
        self.assertEqual(self.workspace.run(request).to_dict()["state"], "UNAVAILABLE")

    def test_later_known_option_correction_does_not_silently_use_old_market_marks(self):
        request, _, _ = self.seed()
        original = self.workspace.run(request).to_dict()
        later = (NOW+timedelta(seconds=1)).isoformat()
        revision = option_fixture(acquired_at=later, knowledge_at=later)
        revision["quotes"][0]["bid"] += .1
        revision["quotes"][0]["ask"] += .1
        OptionsWorkspace(self.base).ingest(revision)
        self.assertEqual(self.workspace.run({**request, "now": later}).to_dict()["state"], "UNAVAILABLE")
        self.assertEqual(self.workspace.replay({"overlay_id": original["data"]["overlay_id"]}).to_dict(), original)


if __name__ == "__main__":
    unittest.main()
