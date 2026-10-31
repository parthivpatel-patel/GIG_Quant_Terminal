"""Broker-free regressions for admission, unknown outcomes, and crash replay."""
from datetime import UTC, datetime, timedelta
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import Mock, patch

import pandas as pd

from gig.execution.algo import ExecutionConfig
from gig.execution.alpaca import AlpacaBroker
from gig.execution.base import Broker
from gig.execution.oms import OfflineOMS, OrderError
from gig.execution.paper import PaperBroker
from gig.execution.pretrade import PreTradeConfig, PreTradeReport, check
from gig.execution.reconcile import build_trade_plan
from gig.execution.simulator import SimulationLimits, simulate_signals
from gig.execution.targets import TargetBook
from gig.execution.trader import TradeRun, _submit, account_status, flatten, run_once

NOW = datetime(2026, 10, 31, 12, tzinfo=UTC)


def book_and_plan():
    book = TargetBook(NOW.date(), pd.Series({"AAA": .05, "BBB": -.05}),
                      pd.Series({"AAA": "same", "BBB": "same"}),
                      pd.Series({"AAA": 100., "BBB": 100.}),
                      pd.Series(dtype=float), "test", 2)
    return book, build_trade_plan(book.weights, {}, book.prices, 1_000_000)


class AdapterSafetyTests(unittest.TestCase):
    def broker(self):
        return AlpacaBroker.__new__(AlpacaBroker)

    def test_invalid_intents_cannot_reach_sdk(self):
        broker = self.broker()
        broker._client = Mock()
        for symbol, quantity, price, client_id in [
            ("", 1, 100, "x"), ("AAA", True, 100, "x"),
            ("AAA", 1, True, "x"), ("AAA", 1, 100, ""),
            ("AAA", 1, 100, "secret\n"), ("AAA", 1, 100, 42),
        ]:
            with self.subTest(symbol=symbol, quantity=quantity, price=price, client_id=client_id):
                with self.assertRaises(ValueError):
                    broker.submit(symbol, "buy", quantity, limit_price=price, client_order_id=client_id)
        broker._client.submit_order.assert_not_called()

    def test_unknown_clock_remains_unknown(self):
        broker = self.broker()
        for state in (None, "true", 1):
            broker._client = SimpleNamespace(get_clock=lambda: SimpleNamespace(is_open=state))
            self.assertIsNone(broker.is_open())
        broker._client = SimpleNamespace(get_clock=lambda: SimpleNamespace(is_open=False))
        self.assertIs(broker.is_open(), False)

    def test_cancellation_requires_known_success_for_every_response(self):
        broker = self.broker()
        for responses in (None, {}, [SimpleNamespace(status=500)], [SimpleNamespace()]):
            broker._client = SimpleNamespace(cancel_orders=lambda: responses)
            with self.subTest(responses=responses), self.assertRaisesRegex(RuntimeError, "cancellation failed"):
                broker.cancel_open_orders()
        broker._client = SimpleNamespace(cancel_orders=lambda: [])
        self.assertEqual(broker.cancel_open_orders(), 0)
        broker._client = SimpleNamespace(cancel_orders=lambda: [SimpleNamespace(status=200)])
        self.assertEqual(broker.cancel_open_orders(), 1)

    def test_flatten_acknowledges_each_outcome_without_claiming_fills(self):
        broker = self.broker()
        broker.positions = lambda: {"AAA": 2., "BBB": -3., "CCC": 4.}
        responses = [SimpleNamespace(symbol="AAA", status=200, body={"id": "ack1"}),
                     SimpleNamespace(symbol="BBB", status=403, body={})]
        broker._client = SimpleNamespace(close_all_positions=lambda **kwargs: responses)
        report = flatten(broker, dry_run=False)
        self.assertEqual(report["n"], 0)
        self.assertEqual(report["closed"], [])
        states = {row["symbol"]: row["classification"] for row in report["close_requests"]}
        self.assertEqual(states, {"AAA": "BROKER_ACKNOWLEDGED", "BBB": "BROKER_REJECTED", "CCC": "UNKNOWN"})

    def test_flatten_error_is_sanitized(self):
        broker = self.broker()
        broker.positions = lambda: {"AAA": 1.}
        broker._client = SimpleNamespace(close_all_positions=Mock(side_effect=RuntimeError("FAKE_SECRET")))
        with self.assertRaises(RuntimeError) as result:
            broker.close_all()
        self.assertNotIn("FAKE_SECRET", str(result.exception))

    def test_paper_numeric_overflow_never_mutates_state(self):
        broker = PaperBroker(lambda _: 1e308)
        for quantity in (True, 1e308):
            with self.subTest(quantity=quantity), self.assertRaises(ValueError):
                broker.submit("AAA", "buy", quantity)
        self.assertEqual(broker.positions(), {})
        self.assertEqual(broker.fills, [])

    def test_unknown_borrow_and_cancellation_are_not_authority(self):
        class MinimalBroker(Broker):
            def submit(self, *args, **kwargs):
                raise AssertionError("must not submit")
            def positions(self):
                return {}
            def nav(self):
                return 1000.
        broker = MinimalBroker()
        self.assertEqual(broker.shortable(["AAA"]), {"AAA": False})
        with self.assertRaises(NotImplementedError):
            broker.cancel_open_orders()


class TraderSafetyTests(unittest.TestCase):
    def run_value(self):
        book, plan = book_and_plan()
        return TradeRun("test", NOW, False, plan.nav, book, plan, PreTradeReport())

    def test_cancellation_failure_halts_all_submission(self):
        run = self.run_value()
        broker = SimpleNamespace(cancel_open_orders=Mock(side_effect=RuntimeError("FAKE_SECRET")), submit=Mock())
        _submit(run, broker, ExecutionConfig(slice_pause_s=0), Mock())
        broker.submit.assert_not_called()
        self.assertEqual(run.orders, [])
        self.assertNotIn("FAKE_SECRET", repr(run.errors))

    def test_ambiguous_submission_halts_without_retry(self):
        run = self.run_value()
        broker = SimpleNamespace(cancel_open_orders=lambda: 0, submit=Mock(side_effect=TimeoutError("FAKE_SECRET")))
        _submit(run, broker, ExecutionConfig(slice_pause_s=0), Mock())
        self.assertEqual(broker.submit.call_count, 1)
        self.assertEqual(run.orders[0]["status"], "submission_unknown")
        self.assertEqual(run.orders[0]["classification"], "UNKNOWN")
        self.assertNotIn("FAKE_SECRET", repr(run.errors))

    def test_nonfinite_high_water_and_invalid_policy_fail_closed(self):
        book, plan = book_and_plan()
        for high_water in (float("nan"), float("inf"), -1, 0, True):
            report = check(plan, book, SimulationLimits(), market_open=True, nav_high_water=high_water, now=NOW.date())
            self.assertTrue(report.blocked)
            self.assertEqual(report.blocking[0].code, "invalid_risk_input")
        for policy in (PreTradeConfig(max_stale_days=-1), PreTradeConfig(max_orders=True)):
            self.assertTrue(check(plan, book, SimulationLimits(), config=policy, market_open=True, now=NOW.date()).blocked)

    def test_positions_failure_cannot_look_like_empty_account(self):
        book, _ = book_and_plan()
        book.asof = datetime.now(UTC).date()
        broker = SimpleNamespace(nav=lambda: 1_000_000., positions=Mock(side_effect=RuntimeError("FAKE_SECRET")),
                                 is_open=lambda: True, shortable=lambda symbols: {s: True for s in symbols}, submit=Mock())
        settings = SimpleNamespace(**asdict(SimulationLimits()), log_level="INFO")
        with patch("gig.execution.trader.get_settings", return_value=settings), \
             patch("gig.execution.trader.build_target_book", return_value=book), \
             patch("gig.execution.trader._reference_prices", return_value=book.prices), \
             patch("gig.ops.killswitch.kill_switch", return_value=SimpleNamespace(status=lambda: {})):
            result = run_once(dry_run=False, broker=broker, record=False)
        broker.submit.assert_not_called()
        self.assertTrue(result.blocked)
        self.assertIn("account_unavailable", [finding.code for finding in result.pretrade.blocking])
        self.assertNotIn("FAKE_SECRET", repr(result.errors))

    def test_unavailable_target_does_not_report_zero_drift(self):
        book, _ = book_and_plan()
        book.available = False
        broker = SimpleNamespace(nav=lambda: 1000., positions=lambda: {}, is_open=lambda: None)
        with patch("gig.execution.trader.build_target_book", return_value=book):
            result = account_status(broker)
        self.assertIs(result["available"], False)
        self.assertIsNone(result["worst_drift"])


class SimulatorReplayTests(unittest.TestCase):
    def run_scenario(self, oms, **kwargs):
        options = dict(scores={"AAA": 1., "BBB": -1.}, prices={"AAA": 100., "BBB": 100.},
                       sectors={"AAA": "same", "BBB": "same"}, oms=oms,
                       run_id="scenario1", asof=NOW, now=NOW)
        options.update(kwargs)
        return simulate_signals(**options)

    def test_complete_scenario_replay_and_changed_intent(self):
        with tempfile.TemporaryDirectory() as folder, OfflineOMS(Path(folder)/"oms.sqlite") as oms:
            first = self.run_scenario(oms)
            replay = self.run_scenario(oms, now=NOW+timedelta(seconds=1))
            self.assertTrue(replay["replayed"])
            self.assertEqual(replay["intent_sha256"], first["intent_sha256"])
            self.assertEqual(replay["orders"], [])
            with self.assertRaisesRegex(OrderError, "run_id conflicts"):
                self.run_scenario(oms, scores={"AAA": -1., "BBB": 1.})
            self.assertEqual(len(oms.events()), 8)

    def test_interrupted_scenario_requires_reconciliation_without_resubmission(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"oms.sqlite"
            with OfflineOMS(path) as oms:
                original = oms.fill
                calls = []
                def interrupted(*args, **kwargs):
                    calls.append(args)
                    if len(calls) == 2:
                        raise RuntimeError("simulated crash")
                    return original(*args, **kwargs)
                with patch.object(oms, "fill", side_effect=interrupted), self.assertRaises(RuntimeError):
                    self.run_scenario(oms)
                count = len(oms.events())
            with OfflineOMS(path) as oms:
                self.assertTrue(oms.recover()["requires_reconciliation"])
                self.assertEqual(oms.recover()["incomplete_simulations"], ["scenario1"])
                replay = self.run_scenario(oms)
                self.assertEqual(replay["code"], "RECONCILIATION_REQUIRED")
                self.assertEqual(len(oms.events()), count)

    def test_kill_switch_prevents_scenario_side_effects(self):
        with tempfile.TemporaryDirectory() as folder, OfflineOMS(Path(folder)/"oms.sqlite") as oms:
            oms.set_kill_switch(True, now=NOW)
            count = len(oms.events())
            self.assertEqual(self.run_scenario(oms)["code"], "KILL_SWITCH")
            self.assertEqual(len(oms.events()), count)
            self.assertEqual(oms.connection.execute("SELECT COUNT(*) FROM simulation_runs").fetchone()[0], 0)

    def test_execution_replay_cannot_rewrite_timestamp(self):
        with tempfile.TemporaryDirectory() as folder, OfflineOMS(Path(folder)/"oms.sqlite") as oms:
            self.run_scenario(oms)
            row = oms.connection.execute("SELECT * FROM executions ORDER BY execution_id LIMIT 1").fetchone()
            with self.assertRaisesRegex(OrderError, "execution_id conflicts"):
                oms.fill(row["client_id"], row["execution_id"], row["quantity"], row["price"],
                         now=NOW+timedelta(microseconds=1))


if __name__ == "__main__":
    unittest.main()
