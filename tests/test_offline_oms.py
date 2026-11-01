"""Offline numerical and persistence checks; run with Python unittest."""

import sqlite3
import sys
import tempfile
import unittest
from datetime import UTC, datetime, timedelta, timezone
from decimal import Decimal, localcontext
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import pandas as pd

from gig.execution.alpaca import AlpacaBroker
from gig.execution.oms import OfflineOMS, OrderError
from gig.execution.paper import PaperBroker
from gig.execution.pretrade import check as risk_check
from gig.execution.reconcile import build_trade_plan
from gig.execution.simulator import SimulationLimits, simulate_signals
from gig.execution.targets import TargetBook
from gig.execution.twap import plan_twap
from gig.types import Fill, OrderAcknowledgement

NOW = datetime(2026, 10, 31, 12, tzinfo=UTC)


class OMSTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "oms.sqlite"
        self.oms = OfflineOMS(self.path)

    def tearDown(self):
        self.oms.close()
        self.tmp.cleanup()

    def approved(self, client="o1", side="buy", qty="10", cap="100", limit=None):
        self.oms.create(client, "ABC", side, qty, limit_price=limit, now=NOW)
        return self.oms.approve(client, reference_price="100", price_asof=NOW,
                                max_notional="2000", max_position_quantity=cap, now=NOW)

    def working(self, **kwargs):
        order = self.approved(**kwargs)
        return self.oms.acknowledge(order["client_id"], now=NOW)

    def test_intent_idempotency_and_conflict(self):
        self.oms.create("x", "ABC", "buy", "10", now=NOW)
        self.oms.create("x", "ABC", "buy", "10.00", now=NOW)
        self.assertEqual(len(self.oms.events()), 1)
        with self.assertRaises(OrderError):
            self.oms.create("x", "ABC", "sell", "10", now=NOW)
        self.assertEqual(self.oms.order("x")["side"], "buy")

    def test_partial_fill_replay_and_exact_position(self):
        self.working(qty="0.3")
        order = self.oms.fill("o1", "e1", "0.1", "100.01", now=NOW)
        self.assertEqual(order["state"], "PARTIALLY_FILLED")
        self.oms.fill("o1", "e1", "0.10", "100.010", now=NOW)
        self.assertEqual(self.oms.positions(), {"ABC": "0.1"})
        order = self.oms.fill("o1", "e2", "0.2", "99.99", fee="0.01", now=NOW)
        self.assertEqual(order["state"], "FILLED")
        self.assertEqual(Decimal(self.oms.positions()["ABC"]), Decimal("0.3"))
        self.assertEqual(self.oms.connection.execute("SELECT COUNT(*) FROM executions").fetchone()[0], 2)

    def test_execution_conflict_and_overfill_do_not_mutate(self):
        self.working()
        self.oms.fill("o1", "e1", "4", "100", now=NOW)
        with self.assertRaises(OrderError):
            self.oms.fill("o1", "e1", "5", "100", now=NOW)
        with self.assertRaises(OrderError):
            self.oms.fill("o1", "e2", "7", "100", now=NOW)
        self.assertEqual(self.oms.positions(), {"ABC": "4"})
        self.assertEqual(self.oms.order("o1")["filled"], "4")

    def test_limit_price_and_approved_notional_enforced(self):
        self.working(limit="101")
        with self.assertRaises(OrderError):
            self.oms.fill("o1", "bad", "1", "101.01", now=NOW)
        self.working(client="sell", side="sell", limit="99")
        with self.assertRaises(OrderError):
            self.oms.fill("sell", "bad", "1", "98.99", now=NOW)
        with self.assertRaises(OrderError):
            self.oms.fill("sell", "too_much", "10", "201", now=NOW)
        self.assertEqual(self.oms.positions(), {})

    def test_submit_requires_approval(self):
        self.oms.create("x", "ABC", "buy", "1", now=NOW)
        with self.assertRaises(OrderError):
            self.oms.acknowledge("x", now=NOW)
        with self.assertRaises(OrderError):
            self.oms.fill("x", "e", "1", "100", now=NOW)

    def test_stale_future_and_naive_risk_evidence_rejected(self):
        for asof in [NOW - timedelta(minutes=2), NOW + timedelta(microseconds=1), NOW.replace(tzinfo=None)]:
            self.oms.create(str(asof), "ABC", "buy", "1", now=NOW)
            with self.assertRaises(OrderError):
                self.oms.approve(str(asof), reference_price="100", price_asof=asof,
                                 max_notional="1000", max_position_quantity="10", now=NOW)
        self.approved()
        with self.assertRaises(OrderError):
            self.oms.acknowledge("o1", now=NOW + timedelta(seconds=61))

    def test_risk_position_reservation_includes_working_orders(self):
        self.working(qty="6", cap="10")
        self.approved(client="second", qty="5", cap="10")
        with self.assertRaises(OrderError):
            self.oms.acknowledge("second", now=NOW)
        self.oms.terminate("o1", "CANCELLED", now=NOW)
        self.assertEqual(self.oms.acknowledge("second", now=NOW)["state"], "ACKNOWLEDGED")

    def test_kill_switch_persists_and_blocks_submission(self):
        self.approved()
        self.oms.set_kill_switch(True, now=NOW)
        self.oms.close()
        self.oms = OfflineOMS(self.path)
        self.assertTrue(self.oms.kill_switch)
        with self.assertRaises(OrderError):
            self.oms.acknowledge("o1", now=NOW)

    def test_restart_recovery_never_resubmits(self):
        self.working()
        self.oms.fill("o1", "e1", "2", "100", now=NOW)
        self.oms.close()
        self.oms = OfflineOMS(self.path)
        recovery = self.oms.recover()
        self.assertEqual(recovery["auto_resubmitted"], 0)
        self.assertTrue(recovery["requires_reconciliation"])
        self.assertEqual(recovery["positions"], {"ABC": "2"})
        self.assertEqual(recovery["working_orders"][0]["filled"], "2")
        self.oms.fill("o1", "e1", "2", "100", now=NOW)
        self.assertEqual(self.oms.positions(), {"ABC": "2"})

    def test_atomic_rollback_of_fill_and_position_on_event_failure(self):
        self.working()
        self.oms.connection.execute("""CREATE TRIGGER fail_event BEFORE INSERT ON events
            WHEN NEW.state='PARTIALLY_FILLED' BEGIN SELECT RAISE(ABORT,'injected failure'); END""")
        with self.assertRaises(sqlite3.IntegrityError):
            self.oms.fill("o1", "e1", "2", "100", now=NOW)
        self.assertEqual(self.oms.order("o1")["filled"], "0")
        self.assertEqual(self.oms.positions(), {})
        self.assertEqual(self.oms.connection.execute("SELECT COUNT(*) FROM executions").fetchone()[0], 0)

    def test_reconcile_is_read_only_and_exact(self):
        self.working()
        self.oms.fill("o1", "e1", "10", "100", now=NOW)
        self.assertTrue(self.oms.reconcile({"ABC": "10.00"})["matched"])
        report = self.oms.reconcile({"ABC": "9", "DEF": "-2"})
        self.assertFalse(report["matched"])
        self.assertEqual(report["differences"]["DEF"]["observed"], "-2")
        self.assertEqual(self.oms.positions(), {"ABC": "10"})

    def test_terminal_states_and_time_order(self):
        self.working()
        self.oms.terminate("o1", "EXPIRED", now=NOW)
        self.oms.terminate("o1", "EXPIRED", now=NOW)
        with self.assertRaises(OrderError):
            self.oms.fill("o1", "e1", "1", "100", now=NOW)
        with self.assertRaises(OrderError):
            self.oms.terminate("o1", "CANCELLED", now=NOW)
        self.working(client="new")
        with self.assertRaises(OrderError):
            self.oms.fill("new", "e1", "1", "100", now=NOW - timedelta(seconds=1))

    def test_invalid_numbers_and_unsupported_precision(self):
        for quantity in ["NaN", "Infinity", "-1", "0", "1e-50", "1" * 30, True, 0.1]:
            with self.subTest(quantity=quantity), self.assertRaises(OrderError):
                self.oms.create("x", "ABC", "buy", quantity, now=NOW)
        self.assertEqual(self.oms.events(), [])

    def test_decimal_context_cannot_round_admitted_orders(self):
        with localcontext() as context:
            context.prec = 2
            self.working(qty="0.123456789123")
            self.oms.fill("o1", "e", "0.123456789123", "100.123456789", now=NOW)
        self.assertEqual(self.oms.positions(), {"ABC": "0.123456789123"})

    def test_two_connections_observe_same_idempotent_intent(self):
        with OfflineOMS(self.path) as other:
            self.oms.create("x", "ABC", "buy", "2", now=NOW)
            other.create("x", "ABC", "buy", "2", now=NOW)
            self.assertEqual(len(other.events()), 1)
            with self.assertRaises(OrderError):
                other.create("x", "ABC", "buy", "3", now=NOW)


class TWAPTests(unittest.TestCase):
    def test_whole_lot_reference_and_exclusive_end(self):
        plan = plan_twap("parent", "10", NOW, NOW + timedelta(seconds=12), 3)
        self.assertEqual([child.quantity for child in plan], [Decimal(4), Decimal(3), Decimal(3)])
        self.assertEqual([child.timestamp for child in plan], [NOW + timedelta(seconds=i) for i in [0, 4, 8]])
        self.assertEqual([child.client_id for child in plan], ["parent:1", "parent:2", "parent:3"])

    def test_fractional_lots_fewer_children_and_timezone_equivalence(self):
        local = NOW.astimezone(timezone(timedelta(hours=-4)))
        plan = plan_twap("p", "0.03", local, local + timedelta(minutes=1), 10, lot_size="0.01")
        self.assertEqual(len(plan), 3)
        self.assertEqual(sum(child.quantity for child in plan), Decimal("0.03"))
        self.assertEqual(plan[0].timestamp, NOW)

    def test_schedule_bounds_and_non_lot_quantities(self):
        invalid = [dict(quantity="1.5", slices=2), dict(quantity="1", slices=0),
                   dict(quantity="1", slices=True), dict(quantity="1", slices=10001)]
        for args in invalid:
            with self.subTest(args=args), self.assertRaises(OrderError):
                plan_twap("p", start=NOW, end=NOW + timedelta(minutes=1), **args)
        with self.assertRaises(OrderError):
            plan_twap("p", "2", NOW, NOW + timedelta(microseconds=1), 2)


class BrokerGuardTests(unittest.TestCase):
    def test_credentials_do_not_authorize_connectivity(self):
        settings = SimpleNamespace(alpaca_base_url="https://paper-api.alpaca.markets", broker_connectivity_enabled=False)
        with patch("gig.execution.alpaca.get_settings", return_value=settings):
            with self.assertRaisesRegex(RuntimeError, "connectivity is disabled"):
                AlpacaBroker()
            with self.assertRaisesRegex(RuntimeError, "live-money"):
                AlpacaBroker(allow_live=True)

    def test_live_and_spoofed_urls_rejected_before_sdk_import(self):
        for endpoint in ["https://api.alpaca.markets", "https://paper-api.alpaca.markets.evil.invalid", "http://paper-api.alpaca.markets",
                         "https://paper-api.alpaca.markets@evil.invalid", "https://paper-api.alpaca.markets/other", "https://paper-api.alpaca.markets?x=1"]:
            settings = SimpleNamespace(alpaca_base_url=endpoint, broker_connectivity_enabled=True)
            with self.subTest(endpoint=endpoint), patch("gig.execution.alpaca.get_settings", return_value=settings), self.assertRaisesRegex(RuntimeError, "approved Alpaca paper endpoint"):
                AlpacaBroker()

    def test_in_memory_paper_rejects_nonfinite_and_negative_intents(self):
        broker = PaperBroker(lambda symbol: 100)
        for side, qty, price in [("bad", 1, 100), ("buy", float("nan"), 100), ("buy", 1, float("inf")), ("buy", -1, 100)]:
            with self.subTest(side=side, qty=qty, price=price), self.assertRaises(ValueError):
                broker.submit("ABC", side, qty, limit_price=price)
        self.assertEqual(broker.positions(), {})

    def test_sdk_acknowledgement_is_never_a_fill(self):
        enums = ModuleType("alpaca.trading.enums")
        enums.OrderSide = SimpleNamespace(BUY="buy", SELL="sell")
        enums.TimeInForce = SimpleNamespace(DAY="day")
        requests = ModuleType("alpaca.trading.requests")
        requests.LimitOrderRequest = lambda **kwargs: SimpleNamespace(**kwargs)
        requests.MarketOrderRequest = lambda **kwargs: SimpleNamespace(**kwargs)
        broker = AlpacaBroker.__new__(AlpacaBroker)
        received = []
        def submit_order(request):
            received.append(request)
            return SimpleNamespace(id="broker-observed-ack-id", status="accepted")
        broker._client = SimpleNamespace(submit_order=submit_order)
        with patch.dict(sys.modules, {"alpaca.trading.enums": enums, "alpaca.trading.requests": requests}):
            ack = broker.submit("ABC", "buy", 3, limit_price=100, client_order_id="test-safe-id")
        self.assertIsInstance(ack, OrderAcknowledgement)
        self.assertNotIsInstance(ack, Fill)
        self.assertFalse(hasattr(ack, "price"))
        self.assertFalse(hasattr(ack, "quantity"))
        self.assertEqual(ack.requested_quantity, 3)
        self.assertEqual(received[0].client_order_id, "test-safe-id")


class IntegratedSimulationTests(unittest.TestCase):
    def test_signal_portfolio_risk_oms_fills_positions_and_reconciliation(self):
        with tempfile.TemporaryDirectory() as folder, OfflineOMS(Path(folder) / "oms.sqlite") as oms:
            result = simulate_signals({"AAA": 1, "BBB": -1}, {"AAA": 100, "BBB": 100},
                                      {"AAA": "tech", "BBB": "tech"}, oms=oms, run_id="scenario1",
                                      asof=NOW, now=NOW)
            self.assertEqual(result["state"], "AVAILABLE")
            self.assertFalse(result["broker_connected"])
            self.assertEqual(result["execution_classification"], "SIMULATED")
            self.assertEqual(result["portfolio"], {"AAA": 0.05, "BBB": -0.05})
            self.assertEqual(result["positions"], {"AAA": "500.0", "BBB": "-500.0"})
            self.assertTrue(result["reconciliation"]["matched"])
            self.assertEqual(len(result["orders"]), 2)
            replay = simulate_signals({"AAA": 1, "BBB": -1}, {"AAA": 100, "BBB": 100},
                                      {"AAA": "tech", "BBB": "tech"}, oms=oms, run_id="scenario1",
                                      asof=NOW, now=NOW)
            self.assertEqual(replay["orders"], [])
            self.assertEqual(replay["positions"], result["positions"])

    def test_stale_scenario_cannot_create_orders(self):
        with tempfile.TemporaryDirectory() as folder, OfflineOMS(Path(folder) / "oms.sqlite") as oms:
            result = simulate_signals({"AAA": 1, "BBB": -1}, {"AAA": 100, "BBB": 100}, {},
                                      oms=oms, run_id="stale", asof=NOW-timedelta(minutes=2), now=NOW)
            self.assertEqual(result["state"], "STALE")
            self.assertEqual(oms.events(), [])

    def test_nonfinite_risk_inputs_unknown_market_and_future_data_block(self):
        book = TargetBook(NOW.date(), pd.Series({"AAA": .05, "BBB": -.05}), pd.Series({"AAA": "x", "BBB": "x"}),
                          pd.Series({"AAA": 100, "BBB": 100}), pd.Series(dtype=float), "test", 2)
        plan = build_trade_plan(book.weights, {}, book.prices, 1_000_000)
        self.assertTrue(risk_check(plan, book, SimulationLimits(), market_open=None, now=NOW.date()).blocked)
        book.asof = NOW.date() + timedelta(days=1)
        self.assertTrue(risk_check(plan, book, SimulationLimits(), market_open=True, now=NOW.date()).blocked)
        book.asof = NOW.date()
        plan.nav = float("nan")
        self.assertEqual(risk_check(plan, book, SimulationLimits(), market_open=True, now=NOW.date()).blocking[0].code, "invalid_risk_input")


if __name__ == "__main__":
    unittest.main()
