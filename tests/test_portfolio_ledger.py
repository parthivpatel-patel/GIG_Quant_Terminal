"""Independent scalar accounting references and durable OMS-to-risk replay."""

from __future__ import annotations

import sqlite3
import tempfile
import unittest
from datetime import UTC, datetime, timedelta
from decimal import Decimal, localcontext
from pathlib import Path

from gig.execution.offline_adapter import OfflinePaperAdapter
from gig.execution.oms import OfflineOMS
from gig.execution.outbox import ExecutionOutbox
from gig.portfolio.ledger import canonical_fills, reconcile_account, risk_report
from gig.service.contracts import EnvelopeSchema
from gig.service.portfolio_workspace import PortfolioWorkspace

NOW = datetime(2026, 11, 2, 14, tzinfo=UTC)


def fill(identity, symbol, side, quantity, price, fee="0", seconds=0):
    return {"execution_id": identity, "client_id": identity, "symbol": symbol,
            "side": side, "quantity": quantity, "price": price, "fee": fee,
            "timestamp": (NOW + timedelta(seconds=seconds)).isoformat(), "classification": "SIMULATED"}


def request(now=NOW, **changes):
    return {"operation": "reconcile", "initial_cash": "10000", "now": now.isoformat(),
            "marks": {"A": {"price": "110", "as_of": now.isoformat()},
                      "B": {"price": "45", "as_of": now.isoformat()}},
            "max_age_seconds": 60, "limits": {"max_gross": 2., "max_name_weight": .5,
                                                "min_cash": "0", "max_drawdown": .5},
            "metadata": {"A": {"sector": "Technology", "factors": {"market": 1.2}},
                         "B": {"sector": "Industrials", "factors": {"market": .8}}},
            **changes}


def populate(base: Path, client="buy-a", symbol="A", side="buy", quantity="10", price="100", fee="1", now=NOW):
    with OfflineOMS(base / "execution-engine/oms.sqlite") as oms:
        outbox = ExecutionOutbox(oms)
        adapter = OfflinePaperAdapter(base / "execution-engine/counterparty.sqlite")
        try:
            outbox.admit_order(client, symbol, side, quantity, reference_price=price,
                               price_asof=now, max_notional="1000000", max_position_quantity="100000", now=now)
            outbox.dispatch(adapter, now=now)
            event = adapter.fill(client, quantity, price, fee=fee, now=now)
            outbox.ingest_event(event, received_at=now)
        finally:
            adapter.close()


class ExactAccountingTests(unittest.TestCase):
    def test_fifo_long_close_fee_cash_and_unrealized_reference(self):
        fills = [fill("one", "A", "buy", "10", "100", "1"),
                 fill("two", "A", "buy", "5", "120", "1", seconds=1),
                 fill("three", "A", "sell", "12", "130", "2", seconds=2)]
        result = reconcile_account(fills, "10000", {"A": "125"})
        row = result["rows"][0]
        self.assertEqual(Decimal(row["quantity"]), 3)
        self.assertEqual(Decimal(row["cost_basis"]), 360)
        self.assertEqual(Decimal(row["realized_gross_pnl"]), 320)
        self.assertEqual(Decimal(row["unrealized_pnl"]), 15)
        self.assertEqual(Decimal(result["cash"]), 9956)
        self.assertEqual(Decimal(result["nav"]), 10331)
        self.assertEqual(Decimal(result["total_pnl"]), 331)

    def test_short_cover_cross_flat_and_multi_asset_cash_reference(self):
        fills = [fill("one", "A", "sell", "5", "100", "1"),
                 fill("two", "A", "buy", "7", "80", "1", seconds=1),
                 fill("three", "B", "sell", "3", "50", "1", seconds=2)]
        result = reconcile_account(fills, "1000", {"A": "90", "B": "45"})
        rows = {row["instrument_id"]: row for row in result["rows"]}
        self.assertEqual(Decimal(result["cash"]), 1087)
        self.assertEqual(Decimal(rows["A"]["quantity"]), 2)
        self.assertEqual(Decimal(rows["A"]["realized_gross_pnl"]), 100)
        self.assertEqual(Decimal(rows["A"]["unrealized_pnl"]), 20)
        self.assertEqual(Decimal(rows["B"]["unrealized_pnl"]), 15)
        self.assertEqual(Decimal(result["nav"]), 1132)
        self.assertEqual(Decimal(result["total_pnl"]), 132)

    def test_fully_closed_position_requires_no_mark_and_does_not_infer_zero_price(self):
        result = reconcile_account([fill("one", "A", "buy", "1", "10"),
                                    fill("two", "A", "sell", "1", "12", seconds=1)], "100", {})
        self.assertIsNone(result["rows"][0]["mark"])
        self.assertEqual(Decimal(result["nav"]), 102)

    def test_partial_fill_event_replay_and_conflict(self):
        first = fill("one", "A", "buy", "0.1", "0.2", "0.001")
        self.assertEqual(len(canonical_fills([first, first])), 1)
        result = reconcile_account([first, first], "1", {"A": "0.3"})
        self.assertEqual(Decimal(result["cash"]), Decimal("0.979"))
        self.assertEqual(Decimal(result["total_pnl"]), Decimal("0.009"))
        with self.assertRaises(ValueError):
            canonical_fills([first, {**first, "price": "0.3"}])

    def test_process_decimal_precision_does_not_change_cash_or_risk(self):
        executions = [fill("one", "A", "buy", "1.23456789", "10.23456789", ".00000001")]
        account = reconcile_account(executions, "1000", {"A": "11.34567891"})
        with localcontext() as context:
            context.prec = 5
            repeated = reconcile_account(executions, "1000", {"A": "11.34567891"})
        self.assertEqual(account, repeated)

    def test_non_simulated_invalid_decimals_and_unmarked_positions_reject(self):
        first = fill("one", "A", "buy", "1", "10")
        for changed in ({"classification": "BROKER_OBSERVED"}, {"quantity": True},
                        {"price": "NaN"}, {"fee": "-1"}):
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                reconcile_account([{**first, **changed}], "100", {"A": "10"})
        with self.assertRaises(ValueError):
            reconcile_account([first], "100", {})


class RiskReferences(unittest.TestCase):
    def account(self):
        return reconcile_account([fill("one", "A", "buy", "10", "100"),
                                  fill("two", "B", "sell", "10", "50")], "10000",
                                 {"A": "100", "B": "50"})

    def risk(self, **kwargs):
        values = {"metadata": request()["metadata"], "limits": {}, "nav_history": [],
                  "now": NOW.isoformat(), "confidence": .95, "scenarios": []}
        values.update(kwargs)
        return risk_report(self.account(), **values)

    def test_gross_net_factor_and_stress_independent_reference(self):
        risk = self.risk(limits={"max_gross": .2, "max_abs_net": .1,
                                "max_sector_gross": .15, "max_factor_exposure": {"market": .1}},
                         scenarios=[{"name": "different shocks", "shocks": {"A": -.1, "B": .2}}])
        self.assertAlmostEqual(risk["gross_exposure"], .15)
        self.assertAlmostEqual(risk["net_exposure"], .05)
        self.assertAlmostEqual(risk["factor_exposure"]["market"]["value"], .08)
        self.assertEqual(risk["violations"], [])
        self.assertEqual(risk["scenarios"][0]["pnl"], -200)
        self.assertEqual(risk["scenarios"][0]["nav_return"], -.02)

    def test_missing_factors_and_sectors_are_unknown_and_fail_requested_limits(self):
        risk = self.risk(metadata={}, limits={"max_sector_gross": .5, "max_factor_exposure": {"market": .5}})
        self.assertEqual(risk["sector_state"], "UNKNOWN")
        self.assertIsNone(risk["factor_exposure"]["market"]["value"])
        self.assertIn("MISSING_SECTOR", risk["violations"])
        self.assertIn("MISSING_FACTOR:market", risk["violations"])

    def test_unknown_historical_tail_is_null_not_zero(self):
        risk = self.risk()
        self.assertEqual(risk["historical_loss_risk"]["state"], "UNAVAILABLE")
        self.assertIsNone(risk["historical_loss_risk"]["var"])

    def test_historical_nav_tail_and_drawdown(self):
        history = [((NOW - timedelta(minutes=22-i)).isoformat(), str(10000 + i*10)) for i in range(21)]
        risk = self.risk(nav_history=history)
        self.assertEqual(risk["historical_loss_risk"]["state"], "AVAILABLE")
        self.assertAlmostEqual(risk["current_drawdown"], 10000 / 10200 - 1)

    def test_future_nav_is_not_used_in_historical_peak_or_tail(self):
        history = [((NOW - timedelta(seconds=1)).isoformat(), "10000"),
                   ((NOW + timedelta(days=1)).isoformat(), "1000000")]
        risk = self.risk(nav_history=history)
        self.assertEqual(risk["current_drawdown"], 0)
        self.assertEqual(risk["peak_nav"], "10000")
        self.assertEqual(risk["historical_loss_risk"]["observations"], 1)

    def test_nonpositive_nav_bases_do_not_silently_gain_tail_coverage(self):
        history = [((NOW - timedelta(minutes=22-i)).isoformat(), str(10000 + i)) for i in range(21)]
        history[5] = (history[5][0], "0")
        risk = self.risk(nav_history=history)
        self.assertEqual(risk["historical_loss_risk"]["state"], "UNAVAILABLE")
        self.assertEqual(risk["historical_loss_risk"]["nonpositive_return_bases"], 1)

    def test_conflicting_same_time_nav_history_fails(self):
        stamp = (NOW - timedelta(seconds=1)).isoformat()
        with self.assertRaisesRegex(ValueError, "conflicting NAV"):
            self.risk(nav_history=[(stamp, "10000"), (stamp, "10001")])

    def test_scenarios_must_explicitly_cover_every_position(self):
        with self.assertRaises(ValueError):
            self.risk(scenarios=[{"name": "missing short", "shocks": {"A": -.1}}])


class PortfolioWorkspaceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.base = Path(self.temporary.name) / "state"
        self.workspace = PortfolioWorkspace(self.base)

    def tearDown(self):
        self.temporary.cleanup()

    def decode(self, contract):
        data = contract.to_dict()
        EnvelopeSchema.model_validate(data)
        return data

    def test_missing_workspace_and_reads_create_no_state(self):
        result = self.decode(self.workspace.run(request()))
        self.assertEqual(result["state"], "UNAVAILABLE")
        self.assertIsNone(self.workspace.section("portfolio", NOW, 60))
        self.assertFalse(self.base.exists())

    def test_actual_oms_fill_restart_replay_cash_and_risk(self):
        populate(self.base)
        populate(self.base, client="short-b", symbol="B", side="sell", quantity="4", price="50", fee="2")
        payload = request(source_identity="a"*64, input_classification="SYNTHETIC")
        result = self.decode(self.workspace.run(payload))
        self.assertEqual(result["state"], "AVAILABLE")
        self.assertEqual(Decimal(result["data"]["metrics"]["cash"]), 9197)
        self.assertEqual(Decimal(result["data"]["metrics"]["nav"]), 10117)
        self.assertEqual(self.decode(PortfolioWorkspace(self.base).run(payload)), result)
        identity = result["data"]["checkpoint_id"]
        saved = self.workspace.load(identity)
        self.assertEqual(saved["nav"], result["data"]["metrics"]["nav"])
        self.assertEqual(len(saved["positions"]), 2)
        for section in ("portfolio", "risk"):
            self.assertEqual(self.decode(PortfolioWorkspace(self.base).section(section, NOW, 60))["state"], "AVAILABLE")

    def test_error_limits_persist_evidence_and_expose_no_success_data(self):
        populate(self.base)
        payload = request(limits={"max_gross": .01})
        result = self.decode(self.workspace.run(payload))
        self.assertEqual(result["state"], "ERROR")
        self.assertIsNone(result["data"])
        identity = result["provenance"]["source_identity"]
        saved = self.workspace.load(identity)
        self.assertIn("MAX_GROSS", saved["risk"]["violations"])
        self.assertEqual(self.decode(self.workspace.section("risk", NOW, 60))["state"], "ERROR")

    def test_missing_unknown_stale_future_and_nanosecond_marks(self):
        populate(self.base)
        self.assertEqual(self.decode(self.workspace.run(request(marks={})))['state'], "UNAVAILABLE")
        self.assertEqual(self.decode(self.workspace.run(request(max_age_seconds=None)))['state'], "UNKNOWN")
        payload = request(now=NOW + timedelta(seconds=61))
        payload["marks"]["A"]["as_of"] = NOW.isoformat()
        result = self.decode(self.workspace.run(payload))
        self.assertEqual(result["state"], "STALE")
        self.assertEqual(result["provenance"]["as_of"], NOW.isoformat())
        payload = request()
        payload["marks"]["A"]["as_of"] = (NOW + timedelta(seconds=1)).isoformat()
        with self.assertRaises(ValueError):
            self.workspace.run(payload)
        payload = request()
        payload["now"] = "2026-11-02T14:02:00.123456789Z"
        for mark in payload["marks"].values():
            mark["as_of"] = payload["now"]
        actual = self.decode(self.workspace.run(payload))
        self.assertIn("123456789", actual["provenance"]["as_of"])

    def test_read_cutoff_freshness_and_initial_cash_is_fixed(self):
        populate(self.base)
        self.workspace.run(request())
        self.assertIsNone(self.workspace.section("portfolio", NOW - timedelta(microseconds=1), 60))
        self.assertEqual(self.decode(self.workspace.section("portfolio", NOW + timedelta(seconds=61), 60))["state"], "STALE")
        self.assertEqual(self.decode(self.workspace.section("portfolio", NOW, None))["state"], "UNKNOWN")
        with self.assertRaises(ValueError):
            self.workspace.run(request(initial_cash="10001"))

    def test_changed_execution_history_and_checkpoint_bytes_fail_closed(self):
        populate(self.base)
        result = self.decode(self.workspace.run(request()))
        path = self.base / "execution-engine/oms.sqlite"
        with sqlite3.connect(path) as connection:
            connection.execute("UPDATE executions SET fee='2'")
        with self.assertRaisesRegex(ValueError, "history changed"):
            self.workspace.run(request())
        identity = result["data"]["checkpoint_id"]
        (self.base / f"portfolio-ledger/checkpoints/{identity}.json").write_text("{}")
        with self.assertRaisesRegex(ValueError, "integrity"):
            self.workspace.load(identity)

    def test_position_mismatch_is_recorded_error(self):
        populate(self.base)
        with sqlite3.connect(self.base / "execution-engine/oms.sqlite") as connection:
            connection.execute("UPDATE positions SET quantity='9'")
        result = self.decode(self.workspace.run(request()))
        self.assertEqual(result["state"], "ERROR")
        evidence = self.workspace.load(result["provenance"]["source_identity"])
        self.assertIn("OMS_POSITION_MISMATCH", evidence["risk"]["violations"])

    def test_delayed_fill_knowledge_never_rewrites_an_earlier_account_checkpoint(self):
        with OfflineOMS(self.base / "execution-engine/oms.sqlite") as oms:
            outbox = ExecutionOutbox(oms)
            adapter = OfflinePaperAdapter(self.base / "execution-engine/counterparty.sqlite")
            try:
                outbox.admit_order("late", "A", "buy", "1", reference_price="100", price_asof=NOW,
                                   max_notional="1000", max_position_quantity="10", now=NOW)
                outbox.dispatch(adapter, now=NOW)
                original = self.decode(self.workspace.run(request()))
                self.assertEqual(Decimal(original["data"]["metrics"]["nav"]), 10000)
                event = adapter.fill("late", "1", "100", now=NOW + timedelta(seconds=1))
                outbox.ingest_event(event, received_at=NOW + timedelta(seconds=10))
            finally:
                adapter.close()
        intermediate = self.decode(PortfolioWorkspace(self.base).run(request(now=NOW + timedelta(seconds=5))))
        self.assertEqual(intermediate["state"], "UNKNOWN")
        self.assertIsNone(intermediate["data"])
        observed = self.decode(PortfolioWorkspace(self.base).run(request(now=NOW + timedelta(seconds=10))))
        self.assertEqual(Decimal(observed["data"]["metrics"]["nav"]), 10010)
        historical = self.decode(PortfolioWorkspace(self.base).section("portfolio", NOW + timedelta(seconds=5), 60))
        self.assertEqual(historical, original)

    def test_future_durable_clock_prevents_backdated_account_admission(self):
        populate(self.base)
        with sqlite3.connect(self.base / "execution-engine/oms.sqlite") as connection:
            connection.execute("CREATE TABLE twap_clock(singleton INTEGER PRIMARY KEY, timestamp TEXT NOT NULL)")
            connection.execute("INSERT INTO twap_clock VALUES(1,?)", ((NOW + timedelta(seconds=30)).isoformat(),))
        result = self.decode(self.workspace.run(request()))
        self.assertEqual(result["state"], "UNKNOWN")
        self.assertFalse((self.base / "portfolio-ledger").exists())

    def test_new_backdated_accounting_request_cannot_republish_saved_cash(self):
        populate(self.base)
        now = NOW + timedelta(seconds=30)
        original = self.decode(self.workspace.run(request(now=now)))
        with self.assertRaisesRegex(ValueError, "clock cannot move backward"):
            self.workspace.run(request(now=NOW + timedelta(seconds=20)))
        with self.assertRaisesRegex(ValueError, "initial simulation cash"):
            self.workspace.run(request(now=NOW + timedelta(seconds=40), initial_cash="11000"))
        replayed = self.decode(PortfolioWorkspace(self.base).run(request(now=now)))
        self.assertEqual(replayed, original)

    def test_symlink_paths_unknown_fields_and_invalid_identity_reject(self):
        payload = request(path="../escape")
        with self.assertRaises(ValueError):
            self.workspace.run(payload)
        with self.assertRaises(ValueError):
            self.workspace.load("../escape")
        self.base.mkdir()
        target = Path(self.temporary.name) / "elsewhere"
        target.mkdir()
        (self.base / "execution-engine").symlink_to(target, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "symbolic"):
            self.workspace.run(request())
        self.assertEqual(list(target.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
