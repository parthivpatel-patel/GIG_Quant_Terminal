"""Offline research strategy simulation stays unapproved and brokerless."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import httpx
import pandas as pd

from gig.service.app import create_app
from gig.service.calculator_journal import CalculatorJournalWorkspace
from gig.service.contracts import EnvelopeSchema
from gig.service.dataset_workspace import PersistedDatasetWorkspace
from gig.service.financial_workspace import FinancialIntelligenceWorkspace
from gig.service.options_workspace import OptionsWorkspace
from gig.service.persisted_research import PersistedResearchWorkspace
from gig.service.research_screen import ResearchScreenWorkspace
from gig.service.research_strategy import ResearchStrategyWorkspace
from gig.service.signal_workspace import SignalWorkspace
from test_options_workspace import option_fixture, option_request
from test_persisted_flow_http import allocation_request, seed_flow


class StrategyTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)
        self.request = seed_flow(self.base)
        self.now = self.request["as_of"]
        dataset = PersistedDatasetWorkspace(self.base).build(self.request).to_dict()["data"]
        run = PersistedResearchWorkspace(self.base).run({
            "dataset_id": dataset["dataset_id"], "model": "lightgbm",
            "min_train": 30, "test_size": 12, "embargo": 1, "seed": 17,
        }).to_dict()["data"]
        signal = SignalWorkspace(self.base).run({
            "run_ids": [run["run_id"]], "now": self.now, "lookback": 20, "min_ic_observations": 5,
        }).to_dict()["data"]
        self.screen = ResearchScreenWorkspace(self.base).run({
            "signal_id": signal["signal_id"], "buy_threshold": 0.0, "sell_threshold": -0.01,
        }).to_dict()["data"]
        self.strategies = ResearchStrategyWorkspace(self.base)
        self.payload = allocation_request("", self.now)
        del self.payload["run_id"]
        self.payload.update(screen_id=self.screen["screen_id"], max_turnover=0.3, max_transaction_cost="100")

    def test_simulate_prepares_without_filling(self):
        prepared = self.strategies.run(self.payload).to_dict()
        EnvelopeSchema.model_validate(prepared)
        self.assertEqual(prepared["state"], "AVAILABLE")
        self.assertFalse(prepared["data"]["trading_approved"])
        self.assertFalse(prepared["data"]["broker_connected"])
        self.assertEqual(prepared["data"]["fill_requested"], False)
        self.assertEqual(prepared["data"]["fills"], [])
        self.assertTrue(prepared["data"]["prepared_client_ids"])
        replayed = self.strategies.replay({"strategy_id": prepared["data"]["strategy_id"]}).to_dict()
        self.assertEqual(replayed["data"]["strategy_id"], prepared["data"]["strategy_id"])
        missing = self.strategies.replay({"strategy_id": "0" * 64}).to_dict()
        self.assertEqual(missing["state"], "UNAVAILABLE")

    def test_simulate_can_fill_in_a_fresh_offline_oms(self):
        filled = self.strategies.run({**self.payload, "fill": True}).to_dict()
        self.assertEqual(filled["state"], "AVAILABLE", filled)
        self.assertTrue(filled["data"]["fills"])
        self.assertEqual(len(filled["data"]["fills"]), len(filled["data"]["prepared_client_ids"]))
        self.assertFalse(filled["data"]["trading_approved"])

    def test_size_caps_name_notional_and_replays(self):
        sized = self.strategies.size({
            "screen_id": self.screen["screen_id"], "now": self.now,
            "portfolio_notional": "100000", "max_name_notional": "5000",
            "lot_size": 1, "max_age_seconds": 86_400, "vol_scale": "1",
        }).to_dict()
        EnvelopeSchema.model_validate(sized)
        self.assertEqual(sized["state"], "AVAILABLE", sized)
        self.assertFalse(sized["data"]["trading_approved"])
        self.assertFalse(sized["data"]["broker_connected"])
        self.assertTrue(sized["data"]["rows"])
        for row in sized["data"]["rows"]:
            self.assertLessEqual(float(row["target_notional"]), 5000.0 + 1e-9)
            self.assertFalse(row["trading_approved"])
        replayed = self.strategies.replay_size({"size_id": sized["data"]["size_id"]}).to_dict()
        self.assertEqual(replayed["data"]["size_id"], sized["data"]["size_id"])
        missing = self.strategies.replay_size({"size_id": "0" * 64}).to_dict()
        self.assertEqual(missing["state"], "UNAVAILABLE")

    def test_size_overlays_regime_risk_parity_calculator_weights(self):
        journal = CalculatorJournalWorkspace(self.base)
        instruments = [row["instrument_id"] for row in self.screen["rows"][:2]]
        self.assertEqual(len(instruments), 2)
        covariance = journal.run({
            "operation": "from-dataset", "dataset_id": self.screen["dataset_id"],
            "kind": "ledoit-wolf", "instruments": instruments, "field": "return_1", "inputs": {},
        }).to_dict()["data"]
        parity = journal.run({
            "operation": "from-screen", "screen_id": self.screen["screen_id"],
            "record_id": covariance["record_id"], "kind": "risk-parity", "inputs": {},
            "regime_short_bars": 20, "regime_long_bars": 60,
        }).to_dict()
        self.assertEqual(parity["state"], "AVAILABLE", parity)
        sized = self.strategies.size({
            "screen_id": self.screen["screen_id"], "now": self.now,
            "portfolio_notional": "100000", "max_name_notional": "50000",
            "lot_size": 1, "max_age_seconds": 86_400, "vol_scale": "1",
            "calculator_record_id": parity["data"]["record_id"],
        }).to_dict()
        self.assertEqual(sized["state"], "AVAILABLE", sized)
        self.assertTrue(sized["data"]["metrics"]["allocation_weights_applied"])
        self.assertEqual(sized["data"]["calculator_record_id"], parity["data"]["record_id"])
        self.assertEqual(sized["data"]["request"]["calculator_kind"], "risk-parity")
        self.assertIn("allocation_regime", sized["data"]["request"])
        self.assertFalse(sized["data"]["trading_approved"])
        with self.assertRaisesRegex(ValueError, "weight-producing"):
            self.strategies.size({
                "screen_id": self.screen["screen_id"], "now": self.now,
                "portfolio_notional": "100000", "max_name_notional": "5000",
                "lot_size": 1, "calculator_record_id": covariance["record_id"],
            })

    def test_factor_exposure_from_size_and_replays(self):
        sized = self.strategies.size({
            "screen_id": self.screen["screen_id"], "now": self.now,
            "portfolio_notional": "100000", "max_name_notional": "50000",
            "lot_size": 1, "max_age_seconds": 86_400, "vol_scale": "1",
        }).to_dict()
        self.assertEqual(sized["state"], "AVAILABLE", sized)
        exposure = self.strategies.factor_exposure({
            "size_id": sized["data"]["size_id"], "now": self.now,
            "n_factors": 2, "min_obs": 40, "lookback": 40, "shrinkage": 0.15,
            "max_ex_ante_vol": "5", "max_systematic_share": 1.0,
        }).to_dict()
        self.assertEqual(exposure["state"], "AVAILABLE", exposure)
        self.assertFalse(exposure["data"]["trading_approved"])
        self.assertTrue(exposure["data"]["report"]["available"])
        self.assertIn("factor_exposure", exposure["data"]["report"])
        self.assertIn("trailing_beta", exposure["data"]["trailing"])
        self.assertEqual(exposure["data"]["size_id"], sized["data"]["size_id"])
        replayed = self.strategies.replay_factor_exposure({
            "exposure_id": exposure["data"]["exposure_id"],
        }).to_dict()
        self.assertEqual(replayed["data"]["exposure_id"], exposure["data"]["exposure_id"])
        missing = self.strategies.replay_factor_exposure({"exposure_id": "0" * 64}).to_dict()
        self.assertEqual(missing["state"], "UNAVAILABLE")
        with self.assertRaisesRegex(ValueError, "n_factors"):
            self.strategies.factor_exposure({
                "size_id": sized["data"]["size_id"], "now": self.now, "n_factors": 99,
            })

    def test_size_derives_vol_scale_from_target_volatility(self):
        sized = self.strategies.size({
            "screen_id": self.screen["screen_id"], "now": self.now,
            "portfolio_notional": "100000", "max_name_notional": "50000",
            "lot_size": 1, "max_age_seconds": 86_400, "target_volatility": "0.15",
            "annualization": 252,
        }).to_dict()
        self.assertEqual(sized["state"], "AVAILABLE", sized)
        self.assertEqual(sized["data"]["request"]["vol_scale_source"], "TARGET_VOLATILITY")
        self.assertEqual(sized["data"]["risk"]["state"], "AVAILABLE")
        self.assertGreater(float(sized["data"]["request"]["vol_scale"]), 0.0)
        self.assertLessEqual(float(sized["data"]["request"]["vol_scale"]), 10.0)
        with self.assertRaisesRegex(ValueError, "not both"):
            self.strategies.size({
                "screen_id": self.screen["screen_id"], "now": self.now,
                "portfolio_notional": "100000", "max_name_notional": "5000",
                "lot_size": 1, "vol_scale": "1", "target_volatility": "0.15",
            })

    def test_size_applies_max_gross_exposure_cap(self):
        baseline = self.strategies.size({
            "screen_id": self.screen["screen_id"], "now": self.now,
            "portfolio_notional": "100000", "max_name_notional": "50000",
            "lot_size": 1, "max_age_seconds": 86_400, "vol_scale": "1",
        }).to_dict()
        self.assertEqual(baseline["state"], "AVAILABLE", baseline)
        gross = float(baseline["data"]["metrics"]["gross_exposure"])
        self.assertGreater(gross, 0.0)
        capped = self.strategies.size({
            "screen_id": self.screen["screen_id"], "now": self.now,
            "portfolio_notional": "100000", "max_name_notional": "50000",
            "lot_size": 1, "max_age_seconds": 86_400, "vol_scale": "1",
            "max_gross_exposure": str(max(gross * 0.5, 0.01)),
        }).to_dict()
        self.assertEqual(capped["state"], "AVAILABLE", capped)
        self.assertTrue(capped["data"]["request"]["gross_exposure_cap_applied"])
        self.assertLessEqual(float(capped["data"]["metrics"]["gross_exposure"]) + 1e-9,
                             float(capped["data"]["request"]["max_gross_exposure"]) + 1e-6)
        self.assertFalse(capped["data"]["trading_approved"])

    def test_exit_eval_and_regime_aware_sizing(self):
        sized = self.strategies.size({
            "screen_id": self.screen["screen_id"], "now": self.now,
            "portfolio_notional": "100000", "max_name_notional": "8000",
            "lot_size": 1, "max_age_seconds": 86_400, "vol_scale": "1",
            "regime_short_bars": 10, "regime_long_bars": 40, "annualization": 252,
        }).to_dict()
        self.assertEqual(sized["state"], "AVAILABLE", sized)
        self.assertIn("TRAILING_VOL_REGIME", sized["data"]["request"]["vol_scale_source"])
        self.assertEqual(sized["data"]["risk"]["data"]["regime"]["method"],
                         "DESCRIPTIVE_TRAILING_VOLATILITY_NOT_HMM")
        evaluated = self.strategies.exit_eval({
            "size_id": sized["data"]["size_id"],
            "stop_loss_pct": "0.5", "take_profit_pct": "0.5", "horizon_bars": 5,
        }).to_dict()
        EnvelopeSchema.model_validate(evaluated)
        self.assertEqual(evaluated["state"], "AVAILABLE", evaluated)
        self.assertFalse(evaluated["data"]["trading_approved"])
        self.assertTrue(evaluated["data"]["rows"])
        self.assertTrue(all(row["outcome"] in {
            "STOP_LOSS", "TAKE_PROFIT", "HORIZON", "NO_FORWARD_PATH",
        } for row in evaluated["data"]["rows"]))
        replayed = self.strategies.replay_exit_eval({
            "exit_eval_id": evaluated["data"]["exit_eval_id"],
        }).to_dict()
        self.assertEqual(replayed["data"]["exit_eval_id"], evaluated["data"]["exit_eval_id"])
        feedback = self.strategies.feedback({
            "exit_eval_ids": [evaluated["data"]["exit_eval_id"]],
        }).to_dict()
        EnvelopeSchema.model_validate(feedback)
        self.assertEqual(feedback["state"], "AVAILABLE", feedback)
        self.assertFalse(feedback["data"]["trading_approved"])
        self.assertEqual(feedback["data"]["metrics"]["evaluations"], 1)
        self.assertGreaterEqual(feedback["data"]["metrics"]["names"], 1)
        again = self.strategies.replay_feedback({
            "feedback_id": feedback["data"]["feedback_id"],
        }).to_dict()
        self.assertEqual(again["data"]["feedback_id"], feedback["data"]["feedback_id"])

    def test_invalidate_blocks_later_size_and_simulate(self):
        sized = self.strategies.size({
            "screen_id": self.screen["screen_id"], "now": self.now,
            "portfolio_notional": "100000", "max_name_notional": "8000",
            "lot_size": 1, "max_age_seconds": 86_400, "vol_scale": "1",
        }).to_dict()
        self.assertEqual(sized["state"], "AVAILABLE", sized)
        invalidated = self.strategies.invalidate({
            "size_id": sized["data"]["size_id"], "reason": "MANUAL", "now": self.now,
            "notes": "research stop after feedback review",
        }).to_dict()
        EnvelopeSchema.model_validate(invalidated)
        self.assertEqual(invalidated["state"], "AVAILABLE", invalidated)
        self.assertFalse(invalidated["data"]["usable_for_simulation"])
        self.assertFalse(invalidated["data"]["trading_approved"])
        blocked_size = self.strategies.size({
            "screen_id": self.screen["screen_id"], "now": self.now,
            "portfolio_notional": "100000", "max_name_notional": "8000",
            "lot_size": 1, "vol_scale": "1",
        }).to_dict()
        self.assertEqual(blocked_size["state"], "UNAVAILABLE")
        blocked_sim = self.strategies.run(self.payload).to_dict()
        self.assertEqual(blocked_sim["state"], "UNAVAILABLE")
        blocked_exit = self.strategies.exit_eval({
            "size_id": sized["data"]["size_id"],
            "stop_loss_pct": "0.1", "take_profit_pct": "0.2", "horizon_bars": 3,
        }).to_dict()
        self.assertEqual(blocked_exit["state"], "UNAVAILABLE")
        replayed = self.strategies.replay_invalidation({
            "invalidation_id": invalidated["data"]["invalidation_id"],
        }).to_dict()
        self.assertEqual(replayed["data"]["invalidation_id"],
                         invalidated["data"]["invalidation_id"])
        with self.assertRaisesRegex(ValueError, "already invalidated"):
            self.strategies.invalidate({
                "screen_id": self.screen["screen_id"], "reason": "MANUAL", "now": self.now,
            })

    def test_calculator_journal_binds_historical_var_from_size(self):
        sized = self.strategies.size({
            "screen_id": self.screen["screen_id"], "now": self.now,
            "portfolio_notional": "100000", "max_name_notional": "8000",
            "lot_size": 1, "max_age_seconds": 86_400, "vol_scale": "1",
        }).to_dict()
        self.assertEqual(sized["state"], "AVAILABLE", sized)
        journaled = CalculatorJournalWorkspace(self.base).run({
            "operation": "from-size", "size_id": sized["data"]["size_id"],
            "kind": "historical-var-components", "field": "return_1",
            "inputs": {"confidence": 0.95},
        }).to_dict()
        EnvelopeSchema.model_validate(journaled)
        self.assertEqual(journaled["state"], "AVAILABLE", journaled)
        body = journaled["data"]
        self.assertFalse(body["trading_approved"])
        self.assertEqual(body["source"]["size_id"], sized["data"]["size_id"])
        self.assertEqual(body["kind"], "historical-var-components")
        self.assertIn("expected_shortfall", body["result"])
        replayed = CalculatorJournalWorkspace(self.base).run({
            "operation": "replay", "record_id": body["record_id"],
        }).to_dict()
        self.assertEqual(replayed["data"]["result"], body["result"])

    def test_estimate_cost_from_size_and_delta(self):
        first = self.strategies.size({
            "screen_id": self.screen["screen_id"], "now": self.now,
            "portfolio_notional": "100000", "max_name_notional": "50000",
            "lot_size": 1, "max_age_seconds": 86_400, "vol_scale": "0.2",
        }).to_dict()
        self.assertEqual(first["state"], "AVAILABLE", first)
        opened = self.strategies.estimate_cost({
            "size_id": first["data"]["size_id"], "now": self.now,
            "spread_bps": "4", "commission_bps": "1", "impact_eta": "0.10",
            "adv_lookback_bars": 20,
        }).to_dict()
        EnvelopeSchema.model_validate(opened)
        self.assertEqual(opened["state"], "AVAILABLE", opened)
        self.assertFalse(opened["data"]["trading_approved"])
        self.assertFalse(opened["data"]["broker_connected"])
        self.assertEqual(opened["data"]["request"]["trade_basis"], "OPEN_FROM_FLAT")
        self.assertEqual(opened["data"]["metrics"]["evidence_class"], "ESTIMATED")
        self.assertGreater(float(opened["data"]["metrics"]["total_estimated_cost"]), 0.0)
        self.assertTrue(opened["data"]["rows"])
        for row in opened["data"]["rows"]:
            self.assertEqual(row["evidence_class"], "ESTIMATED")
            self.assertGreater(float(row["one_way_bps"]), 0.0)
        second = self.strategies.size({
            "screen_id": self.screen["screen_id"], "now": self.now,
            "portfolio_notional": "100000", "max_name_notional": "50000",
            "lot_size": 1, "max_age_seconds": 86_400, "vol_scale": "1",
        }).to_dict()
        self.assertEqual(second["state"], "AVAILABLE", second)
        self.assertGreater(
            float(second["data"]["metrics"]["gross_notional"]),
            float(first["data"]["metrics"]["gross_notional"]) + 1e-9,
        )
        delta = self.strategies.estimate_cost({
            "size_id": second["data"]["size_id"], "now": self.now,
            "previous_size_id": first["data"]["size_id"],
            "adv_lookback_bars": 20,
        }).to_dict()
        self.assertEqual(delta["state"], "AVAILABLE", delta)
        self.assertEqual(delta["data"]["request"]["trade_basis"], "DELTA_VS_PREVIOUS")
        self.assertEqual(delta["data"]["previous_size_id"], first["data"]["size_id"])
        self.assertGreater(float(delta["data"]["metrics"]["trade_notional"]), 0.0)
        self.assertLess(
            float(delta["data"]["metrics"]["trade_notional"]) + 1e-9,
            float(opened["data"]["metrics"]["trade_notional"])
            + float(second["data"]["metrics"]["gross_notional"]),
        )
        replayed = self.strategies.replay_cost({"cost_id": opened["data"]["cost_id"]}).to_dict()
        self.assertEqual(replayed["data"]["cost_id"], opened["data"]["cost_id"])
        missing = self.strategies.replay_cost({"cost_id": "0" * 64}).to_dict()
        self.assertEqual(missing["state"], "UNAVAILABLE")


    def test_cost_rebalance_skip_shrink_and_alpha_gate(self):
        first = self.strategies.size({
            "screen_id": self.screen["screen_id"], "now": self.now,
            "portfolio_notional": "100000", "max_name_notional": "50000",
            "lot_size": 1, "max_age_seconds": 86_400, "vol_scale": "0.2",
        }).to_dict()
        self.assertEqual(first["state"], "AVAILABLE", first)
        second = self.strategies.size({
            "screen_id": self.screen["screen_id"], "now": self.now,
            "portfolio_notional": "100000", "max_name_notional": "50000",
            "lot_size": 1, "max_age_seconds": 86_400, "vol_scale": "1",
            "previous_size_id": first["data"]["size_id"],
        }).to_dict()
        self.assertEqual(second["state"], "AVAILABLE", second)
        cost = self.strategies.estimate_cost({
            "size_id": second["data"]["size_id"], "now": self.now,
            "previous_size_id": first["data"]["size_id"],
            "adv_lookback_bars": 20,
        }).to_dict()
        self.assertEqual(cost["state"], "AVAILABLE", cost)
        # Force skips via huge min_trade_notional
        skipped = self.strategies.cost_rebalance({
            "size_id": second["data"]["size_id"], "cost_id": cost["data"]["cost_id"],
            "now": self.now, "mode": "SKIP_BELOW_COST",
            "min_trade_notional": "1e18",
        }).to_dict()
        EnvelopeSchema.model_validate(skipped)
        self.assertEqual(skipped["state"], "AVAILABLE", skipped)
        self.assertFalse(skipped["data"]["trading_approved"])
        self.assertGreaterEqual(skipped["data"]["metrics"]["trades_skipped"], 1)
        self.assertTrue(any(r["action"] == "SKIP" for r in skipped["data"]["rows"]))
        # Shrink when max_cost_nav is tiny
        shrunk = self.strategies.cost_rebalance({
            "size_id": second["data"]["size_id"], "cost_id": cost["data"]["cost_id"],
            "now": self.now, "mode": "SHRINK_TO_COST", "max_cost_nav": "1e-12",
        }).to_dict()
        self.assertEqual(shrunk["state"], "AVAILABLE", shrunk)
        self.assertLess(
            float(shrunk["data"]["request"]["shrink_scale"]) + 1e-18, 1.0)
        self.assertTrue(any(r["action"] == "SHRINK" for r in shrunk["data"]["rows"]))
        # Alpha gate with tiny scale skips
        gated = self.strategies.cost_rebalance({
            "size_id": second["data"]["size_id"], "cost_id": cost["data"]["cost_id"],
            "now": self.now, "mode": "ALPHA_COST_GATE", "alpha_scale": "1e-18",
        }).to_dict()
        self.assertEqual(gated["state"], "AVAILABLE", gated)
        self.assertGreaterEqual(gated["data"]["metrics"]["trades_skipped"], 1)
        replayed = self.strategies.replay_rebalance({
            "rebalance_id": skipped["data"]["rebalance_id"],
        }).to_dict()
        self.assertEqual(replayed["data"]["rebalance_id"], skipped["data"]["rebalance_id"])
        missing = self.strategies.replay_rebalance({"rebalance_id": "0" * 64}).to_dict()
        self.assertEqual(missing["state"], "UNAVAILABLE")

    def test_cycle_can_attach_cost_rebalance(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            request = seed_flow(root)
            now = request["as_of"]
            dataset = PersistedDatasetWorkspace(root).build(request).to_dict()["data"]
            run = PersistedResearchWorkspace(root).run({
                "dataset_id": dataset["dataset_id"], "model": "lightgbm",
                "min_train": 30, "test_size": 12, "embargo": 1, "seed": 17,
            }).to_dict()["data"]
            payload = allocation_request("", now)
            del payload["run_id"]
            payload.update(
                run_ids=[run["run_id"]], lookback=20, min_ic_observations=5,
                buy_threshold=0.0, sell_threshold=-0.01,
                portfolio_notional="100000", max_name_notional="8000",
                max_turnover=0.3, max_transaction_cost="100", fill=True,
                cost_rebalance=True, rebalance_mode="SKIP_BELOW_COST",
                min_trade_notional="1", adv_lookback_bars=20,
            )
            cycled = ResearchStrategyWorkspace(root).cycle(payload).to_dict()
            EnvelopeSchema.model_validate(cycled)
            self.assertEqual(cycled["state"], "AVAILABLE", cycled)
            body = cycled["data"]
            self.assertTrue(body["cost_id"])
            self.assertTrue(body["rebalance_id"])
            self.assertIsNotNone(body["rebalance_metrics"])
            self.assertFalse(body["trading_approved"])

    def test_calculator_journal_binds_liquidity_and_almgren_from_size(self):
        sized = self.strategies.size({
            "screen_id": self.screen["screen_id"], "now": self.now,
            "portfolio_notional": "100000", "max_name_notional": "8000",
            "lot_size": 1, "max_age_seconds": 86_400, "vol_scale": "1",
        }).to_dict()
        self.assertEqual(sized["state"], "AVAILABLE", sized)
        liquidity = CalculatorJournalWorkspace(self.base).run({
            "operation": "from-size", "size_id": sized["data"]["size_id"],
            "kind": "liquidity-var", "adv_lookback_bars": 20,
            "inputs": {"max_participation": 0.2, "confidence": 0.99},
        }).to_dict()
        self.assertEqual(liquidity["state"], "AVAILABLE", liquidity)
        self.assertFalse(liquidity["data"]["trading_approved"])
        self.assertIn("liquidity_adjusted_var", liquidity["data"]["result"])
        self.assertEqual(liquidity["data"]["source"]["extraction"],
                         "RESEARCH_SIZE_LARGEST_NAME_LIQUIDITY_VAR")
        schedule = CalculatorJournalWorkspace(self.base).run({
            "operation": "from-size", "size_id": sized["data"]["size_id"],
            "kind": "almgren-chriss", "adv_lookback_bars": 20,
            "inputs": {
                "horizon": 5, "intervals": 5, "temporary_impact": 2.5e-6,
                "permanent_impact": 2.5e-7, "risk_aversion": 2e-6,
            },
        }).to_dict()
        self.assertEqual(schedule["state"], "AVAILABLE", schedule)
        self.assertFalse(schedule["data"]["trading_approved"])
        self.assertIn("expected_cost", schedule["data"]["result"])
        self.assertEqual(schedule["data"]["source"]["extraction"],
                         "RESEARCH_SIZE_LARGEST_NAME_ALMGREN_CHRISS")
        participation = CalculatorJournalWorkspace(self.base).run({
            "operation": "from-size", "size_id": sized["data"]["size_id"],
            "kind": "participation-schedule", "adv_lookback_bars": 20,
            "inputs": {"max_participation": 0.1},
        }).to_dict()
        self.assertEqual(participation["state"], "AVAILABLE", participation)
        self.assertFalse(participation["data"]["trading_approved"])
        self.assertIn("scheduled_shares", participation["data"]["result"])
        self.assertEqual(participation["data"]["source"]["extraction"],
                         "RESEARCH_SIZE_LARGEST_NAME_PARTICIPATION_SCHEDULE")
        shortfall = CalculatorJournalWorkspace(self.base).run({
            "operation": "from-size", "size_id": sized["data"]["size_id"],
            "kind": "shortfall-simulation", "adv_lookback_bars": 20,
            "inputs": {
                "intervals": 5, "interval": 1, "temporary_impact": 2.5e-6,
                "permanent_impact": 2.5e-7, "paths": 200, "seed": 7,
            },
        }).to_dict()
        self.assertEqual(shortfall["state"], "AVAILABLE", shortfall)
        self.assertFalse(shortfall["data"]["trading_approved"])
        self.assertIn("mean", shortfall["data"]["result"])
        self.assertEqual(shortfall["data"]["source"]["extraction"],
                         "RESEARCH_SIZE_LARGEST_NAME_SHORTFALL_SIMULATION")

    def test_cycle_can_attach_pretrade_cost_estimate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            request = seed_flow(root)
            now = request["as_of"]
            dataset = PersistedDatasetWorkspace(root).build(request).to_dict()["data"]
            run = PersistedResearchWorkspace(root).run({
                "dataset_id": dataset["dataset_id"], "model": "lightgbm",
                "min_train": 30, "test_size": 12, "embargo": 1, "seed": 17,
            }).to_dict()["data"]
            payload = allocation_request("", now)
            del payload["run_id"]
            payload.update(
                run_ids=[run["run_id"]], lookback=20, min_ic_observations=5,
                buy_threshold=0.0, sell_threshold=-0.01,
                portfolio_notional="100000", max_name_notional="8000",
                max_turnover=0.3, max_transaction_cost="100", fill=True,
                estimate_cost=True, adv_lookback_bars=20,
            )
            cycled = ResearchStrategyWorkspace(root).cycle(payload).to_dict()
            EnvelopeSchema.model_validate(cycled)
            self.assertEqual(cycled["state"], "AVAILABLE", cycled)
            body = cycled["data"]
            self.assertTrue(body["cost_id"])
            self.assertIsNotNone(body["cost_metrics"])
            self.assertEqual(body["cost_metrics"]["evidence_class"], "ESTIMATED")
            self.assertIsNotNone(body["metrics"]["estimated_cost_nav"])
            self.assertFalse(body["trading_approved"])

    def test_size_applies_max_expected_shortfall_bound(self):
        probe = self.strategies.size({
            "screen_id": self.screen["screen_id"], "now": self.now,
            "portfolio_notional": "100000", "max_name_notional": "50000",
            "lot_size": 1, "max_age_seconds": 86_400, "vol_scale": "1",
            "max_expected_shortfall": "5", "annualization": 252,
        }).to_dict()
        self.assertEqual(probe["state"], "AVAILABLE", probe)
        expected = float(probe["data"]["risk"]["data"]["loss_risk"]["expected_shortfall"])
        self.assertGreater(expected, 0.0)
        bound = str(max(expected * 0.5, 1e-6))
        gated = self.strategies.size({
            "screen_id": self.screen["screen_id"], "now": self.now,
            "portfolio_notional": "100000", "max_name_notional": "50000",
            "lot_size": 1, "max_age_seconds": 86_400, "vol_scale": "1",
            "max_expected_shortfall": bound, "annualization": 252,
        }).to_dict()
        self.assertEqual(gated["state"], "AVAILABLE", gated)
        self.assertEqual(gated["data"]["risk"]["state"], "AVAILABLE")
        self.assertIn("MAX_EXPECTED_SHORTFALL", gated["data"]["request"]["vol_scale_source"])
        self.assertLess(float(gated["data"]["request"]["vol_scale"]),
                        float(probe["data"]["request"]["vol_scale"]) + 1e-12)
        self.assertTrue(gated["data"]["risk"]["data"]["expected_shortfall_bound_applied"])
        self.assertFalse(gated["data"]["trading_approved"])

    def test_size_applies_max_size_turnover_blend(self):
        first = self.strategies.size({
            "screen_id": self.screen["screen_id"], "now": self.now,
            "portfolio_notional": "100000", "max_name_notional": "50000",
            "lot_size": 1, "max_age_seconds": 86_400, "vol_scale": "0.2",
        }).to_dict()
        self.assertEqual(first["state"], "AVAILABLE", first)
        measured = self.strategies.size({
            "screen_id": self.screen["screen_id"], "now": self.now,
            "portfolio_notional": "100000", "max_name_notional": "50000",
            "lot_size": 1, "max_age_seconds": 86_400, "vol_scale": "1",
            "previous_size_id": first["data"]["size_id"],
        }).to_dict()
        self.assertEqual(measured["state"], "AVAILABLE", measured)
        self.assertGreater(float(measured["data"]["metrics"]["size_turnover_nav"]), 0.05)
        second = self.strategies.size({
            "screen_id": self.screen["screen_id"], "now": self.now,
            "portfolio_notional": "100000", "max_name_notional": "50000",
            "lot_size": 1, "max_age_seconds": 86_400, "vol_scale": "1",
            "previous_size_id": first["data"]["size_id"],
            "max_size_turnover": "0.05",
        }).to_dict()
        EnvelopeSchema.model_validate(second)
        self.assertEqual(second["state"], "AVAILABLE", second)
        self.assertTrue(second["data"]["request"]["size_turnover_cap_applied"])
        self.assertLessEqual(float(second["data"]["metrics"]["size_turnover_nav"]), 0.05 + 1e-9)
        self.assertEqual(second["data"]["previous_size_id"], first["data"]["size_id"])
        self.assertFalse(second["data"]["trading_approved"])

    def test_size_applies_max_adv_participation(self):
        probe = self.strategies.size({
            "screen_id": self.screen["screen_id"], "now": self.now,
            "portfolio_notional": "100000", "max_name_notional": "50000",
            "lot_size": 1, "max_age_seconds": 86_400, "vol_scale": "1",
        }).to_dict()
        self.assertEqual(probe["state"], "AVAILABLE", probe)
        # First call without a cap to materialize ADV, then choose a tight but
        # non-zero participation that still leaves lot sizes.
        soft = self.strategies.size({
            "screen_id": self.screen["screen_id"], "now": self.now,
            "portfolio_notional": "100000", "max_name_notional": "50000",
            "lot_size": 1, "max_age_seconds": 86_400, "vol_scale": "1",
            "max_adv_participation": "1", "adv_lookback_bars": 20,
        }).to_dict()
        self.assertEqual(soft["state"], "AVAILABLE", soft)
        adv_values = [float(row["adv_dollar"]) for row in soft["data"]["rows"]]
        self.assertTrue(adv_values)
        # Cap at ~10% of the smallest ADV so at least one name is reduced.
        participation = max(min(adv_values) * 0.1 / max(adv_values), 1e-6)
        gated = self.strategies.size({
            "screen_id": self.screen["screen_id"], "now": self.now,
            "portfolio_notional": "100000", "max_name_notional": "50000",
            "lot_size": 1, "max_age_seconds": 86_400, "vol_scale": "1",
            "max_adv_participation": str(participation), "adv_lookback_bars": 20,
        }).to_dict()
        self.assertEqual(gated["state"], "AVAILABLE", gated)
        self.assertTrue(gated["data"]["request"]["adv_participation_cap_applied"])
        self.assertTrue(any(row.get("adv_capped") for row in gated["data"]["rows"]))
        self.assertLess(
            float(gated["data"]["metrics"]["gross_notional"]),
            float(probe["data"]["metrics"]["gross_notional"]) + 1e-12,
        )
        self.assertFalse(gated["data"]["trading_approved"])

    def test_cycle_discovers_sizes_and_simulates(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            request = seed_flow(root)
            now = request["as_of"]
            dataset = PersistedDatasetWorkspace(root).build(request).to_dict()["data"]
            run = PersistedResearchWorkspace(root).run({
                "dataset_id": dataset["dataset_id"], "model": "lightgbm",
                "min_train": 30, "test_size": 12, "embargo": 1, "seed": 17,
            }).to_dict()["data"]
            payload = allocation_request("", now)
            del payload["run_id"]
            payload.update(
                run_ids=[run["run_id"]], lookback=20, min_ic_observations=5,
                buy_threshold=0.0, sell_threshold=-0.01,
                portfolio_notional="100000", max_name_notional="8000",
                max_turnover=0.3, max_transaction_cost="100", fill=True,
            )
            cycled = ResearchStrategyWorkspace(root).cycle(payload).to_dict()
            EnvelopeSchema.model_validate(cycled)
            self.assertEqual(cycled["state"], "AVAILABLE", cycled)
            body = cycled["data"]
            self.assertFalse(body["trading_approved"])
            self.assertFalse(body["broker_connected"])
            self.assertTrue(body["screen_id"])
            self.assertTrue(body["size_id"])
            self.assertTrue(body["strategy_id"])
            self.assertTrue(body["fills"])
            replayed = ResearchStrategyWorkspace(root).replay_cycle(
                {"cycle_id": body["cycle_id"]}).to_dict()
            self.assertEqual(replayed["data"]["cycle_id"], body["cycle_id"])

    def test_cycle_market_neutral_factor_and_schedule_tick(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            request = seed_flow(root)
            now = request["as_of"]
            dataset = PersistedDatasetWorkspace(root).build(request).to_dict()["data"]
            strategies = ResearchStrategyWorkspace(root)
            payload = allocation_request("", now)
            del payload["run_id"]
            payload.update(
                dataset_id=dataset["dataset_id"], mode="market_neutral_momentum",
                momentum_lookback=10, reversion_lookback=5,
                buy_threshold=0.0, sell_threshold=-0.01,
                portfolio_notional="100000", max_name_notional="8000",
                max_turnover=0.3, max_transaction_cost="100", fill=True, vol_scale="1",
            )
            cycled = strategies.cycle(payload).to_dict()
            EnvelopeSchema.model_validate(cycled)
            self.assertEqual(cycled["state"], "AVAILABLE", cycled)
            self.assertEqual(cycled["data"]["discovery"]["mode"], "market_neutral_momentum")
            self.assertFalse(cycled["data"]["trading_approved"])
            self.assertTrue(cycled["data"]["fills"])

            def _scheduled_workspace(name: str):
                path = root / name
                path.mkdir()
                seed = seed_flow(path)
                stamp = seed["as_of"]
                saved = PersistedDatasetWorkspace(path).build(seed).to_dict()["data"]
                workspace = ResearchStrategyWorkspace(path)
                body = allocation_request("", stamp)
                del body["run_id"]
                body.update(
                    dataset_id=saved["dataset_id"], mode="market_neutral_momentum",
                    momentum_lookback=10, reversion_lookback=5,
                    buy_threshold=0.0, sell_threshold=-0.01,
                    portfolio_notional="100000", max_name_notional="8000",
                    max_turnover=0.3, max_transaction_cost="100", fill=True, vol_scale="1",
                )
                return workspace, stamp, {key: value for key, value in body.items() if key != "now"}

            scheduled, schedule_now, schedule_request = _scheduled_workspace("scheduled")
            planned = scheduled.schedule({
                "action": "cycle", "cadence_seconds": 3600, "max_ticks": 1,
                "now": schedule_now, "request": schedule_request,
            }).to_dict()
            EnvelopeSchema.model_validate(planned)
            self.assertEqual(planned["state"], "AVAILABLE", planned)
            schedule_id = planned["data"]["schedule_id"]
            due_before = scheduled.due_schedules({"now": schedule_now, "limit": 8}).to_dict()
            self.assertEqual(due_before["state"], "AVAILABLE")
            self.assertTrue(any(row["schedule_id"] == schedule_id for row in due_before["data"]["rows"]))
            early = scheduled.tick_schedule({
                "schedule_id": schedule_id, "now": schedule_now,
            }).to_dict()
            self.assertEqual(early["state"], "AVAILABLE", early)
            self.assertEqual(early["data"]["tick_index"], 1)
            self.assertEqual(early["data"]["action"], "cycle")
            self.assertFalse(early["data"]["trading_approved"])
            exhausted = scheduled.tick_schedule({
                "schedule_id": schedule_id,
                "now": (pd.Timestamp(schedule_now) + pd.Timedelta(hours=2)).isoformat(),
            }).to_dict()
            self.assertEqual(exhausted["state"], "UNAVAILABLE")
            due_after = scheduled.due_schedules({
                "now": (pd.Timestamp(schedule_now) + pd.Timedelta(hours=2)).isoformat(),
                "limit": 8,
            }).to_dict()
            EnvelopeSchema.model_validate(due_after)
            self.assertFalse(due_after["data"]["trading_approved"])
            self.assertTrue(all(row["schedule_id"] != schedule_id for row in due_after["data"]["rows"]))

            stale_ws, stale_now, stale_request = _scheduled_workspace("stale-sched")
            pending = stale_ws.schedule({
                "action": "cycle", "cadence_seconds": 7200, "max_ticks": 2,
                "now": stale_now, "request": stale_request,
            }).to_dict()
            self.assertEqual(pending["state"], "AVAILABLE", pending)
            first = stale_ws.tick_schedule({
                "schedule_id": pending["data"]["schedule_id"], "now": stale_now,
            }).to_dict()
            self.assertEqual(first["state"], "AVAILABLE", first)
            stale = stale_ws.tick_schedule({
                "schedule_id": pending["data"]["schedule_id"],
                "now": (pd.Timestamp(stale_now) + pd.Timedelta(hours=1)).isoformat(),
            }).to_dict()
            self.assertEqual(stale["state"], "STALE")
            self.assertEqual(stale["data"]["tick_state"], "NOT_DUE")

            batch_ws, batch_now, batch_request = _scheduled_workspace("tick-due")
            planned_a = batch_ws.schedule({
                "action": "cycle", "cadence_seconds": 3600, "max_ticks": 1,
                "now": batch_now, "request": batch_request,
            }).to_dict()
            planned_b = batch_ws.schedule({
                "action": "cycle", "cadence_seconds": 7200, "max_ticks": 1,
                "now": batch_now, "request": {**batch_request, "n_long": 1, "n_short": 1},
            }).to_dict()
            self.assertEqual(planned_a["state"], "AVAILABLE", planned_a)
            self.assertEqual(planned_b["state"], "AVAILABLE", planned_b)
            # One tick at a time: filled cycles leave OMS state that blocks a second same-clock cycle.
            batched = batch_ws.tick_due_schedules({"now": batch_now, "limit": 1}).to_dict()
            EnvelopeSchema.model_validate(batched)
            self.assertEqual(batched["state"], "AVAILABLE", batched)
            self.assertEqual(batched["data"]["metrics"]["attempted"], 1)
            self.assertEqual(batched["data"]["metrics"]["available"], 1)
            self.assertFalse(batched["data"]["trading_approved"])
            remaining = batch_ws.due_schedules({"now": batch_now, "limit": 8}).to_dict()
            self.assertEqual(remaining["data"]["metrics"]["due"], 1)


    def test_schedule_ticks_attach_exit_eval_and_feedback(self):
        first_signal_id = self.screen["signal_id"]
        signal_doc = SignalWorkspace(self.base).load(first_signal_id)
        later = (pd.Timestamp(self.now) + pd.Timedelta(days=1)).isoformat()
        later_signal = SignalWorkspace(self.base).run({
            "run_ids": signal_doc["run_ids"], "now": later,
            "lookback": 20, "min_ic_observations": 5,
        }).to_dict()["data"]
        watch_request = {
            "signal_ids": [first_signal_id, later_signal["signal_id"]],
            "buy_threshold": 0.0, "sell_threshold": -0.01,
            "portfolio_notional": "100000", "max_name_notional": "50000",
            "lot_size": 1, "vol_scale": "1",
        }
        planned = self.strategies.schedule({
            "action": "watch", "cadence_seconds": 3600, "max_ticks": 2,
            "now": self.now, "request": watch_request,
            "exit_eval": True, "stop_loss_pct": "0.05", "take_profit_pct": "0.10",
            "horizon_bars": 5, "feedback_every_n_ticks": 2,
        }).to_dict()
        self.assertEqual(planned["state"], "AVAILABLE", planned)
        self.assertTrue(planned["data"]["exit_eval"])
        self.assertEqual(planned["data"]["feedback_every_n_ticks"], 2)
        schedule_id = planned["data"]["schedule_id"]
        first = self.strategies.tick_schedule({
            "schedule_id": schedule_id, "now": self.now,
        }).to_dict()
        self.assertEqual(first["state"], "AVAILABLE", first)
        self.assertTrue(first["data"]["size_id"])
        self.assertTrue(first["data"]["exit_eval_id"])
        self.assertIsNone(first["data"]["feedback_id"])
        self.assertTrue(first["data"]["metrics"]["exit_eval_attached"])
        self.assertFalse(first["data"]["metrics"]["feedback_attached"])
        self.assertFalse(first["data"]["trading_approved"])
        replayed = self.strategies.replay_exit_eval({
            "exit_eval_id": first["data"]["exit_eval_id"],
        }).to_dict()
        self.assertEqual(replayed["data"]["exit_eval_id"], first["data"]["exit_eval_id"])
        second_now = (pd.Timestamp(self.now) + pd.Timedelta(hours=1)).isoformat()
        second = self.strategies.tick_schedule({
            "schedule_id": schedule_id, "now": second_now,
        }).to_dict()
        self.assertEqual(second["state"], "AVAILABLE", second)
        self.assertTrue(second["data"]["exit_eval_id"])
        self.assertTrue(second["data"]["feedback_id"])
        self.assertTrue(second["data"]["metrics"]["feedback_attached"])
        self.assertEqual(second["data"]["metrics"]["exit_eval_count"], 2)
        feedback = self.strategies.replay_feedback({
            "feedback_id": second["data"]["feedback_id"],
        }).to_dict()
        self.assertEqual(feedback["state"], "AVAILABLE", feedback)
        self.assertEqual(feedback["data"]["metrics"]["evaluations"], 2)
        state = self.strategies.load_schedule_state(schedule_id)
        self.assertEqual(state["last_exit_eval_id"], second["data"]["exit_eval_id"])
        self.assertEqual(len(state["exit_eval_ids"]), 2)
        self.assertEqual(state["last_feedback_id"], second["data"]["feedback_id"])
        # Default schedules still omit exit-eval hooks.
        plain = self.strategies.schedule({
            "action": "watch", "cadence_seconds": 3600, "max_ticks": 1,
            "now": self.now, "request": watch_request,
        }).to_dict()
        plain_tick = self.strategies.tick_schedule({
            "schedule_id": plain["data"]["schedule_id"], "now": self.now,
        }).to_dict()
        self.assertEqual(plain_tick["state"], "AVAILABLE", plain_tick)
        self.assertIsNone(plain_tick["data"]["exit_eval_id"])
        self.assertIsNone(plain_tick["data"]["feedback_id"])

    def test_schedule_ticks_attach_cost_rebalance(self):
        first_signal_id = self.screen["signal_id"]
        signal_doc = SignalWorkspace(self.base).load(first_signal_id)
        later = (pd.Timestamp(self.now) + pd.Timedelta(days=1)).isoformat()
        later_signal = SignalWorkspace(self.base).run({
            "run_ids": signal_doc["run_ids"], "now": later,
            "lookback": 20, "min_ic_observations": 5,
        }).to_dict()["data"]
        watch_request = {
            "signal_ids": [first_signal_id, later_signal["signal_id"]],
            "buy_threshold": 0.0, "sell_threshold": -0.01,
            "portfolio_notional": "100000", "max_name_notional": "50000",
            "lot_size": 1, "vol_scale": "1",
        }
        planned = self.strategies.schedule({
            "action": "watch", "cadence_seconds": 3600, "max_ticks": 1,
            "now": self.now, "request": watch_request,
            "cost_rebalance": True, "rebalance_mode": "SKIP_BELOW_COST",
            "min_trade_notional": "100",
        }).to_dict()
        self.assertEqual(planned["state"], "AVAILABLE", planned)
        self.assertTrue(planned["data"]["cost_rebalance"])
        self.assertEqual(planned["data"]["cost_rebalance_request"]["mode"], "SKIP_BELOW_COST")
        tick = self.strategies.tick_schedule({
            "schedule_id": planned["data"]["schedule_id"], "now": self.now,
        }).to_dict()
        self.assertEqual(tick["state"], "AVAILABLE", tick)
        self.assertTrue(tick["data"]["size_id"])
        self.assertTrue(tick["data"]["rebalance_id"])
        self.assertTrue(tick["data"]["metrics"]["cost_rebalance_attached"])
        self.assertFalse(tick["data"]["trading_approved"])
        replayed = self.strategies.replay_rebalance({
            "rebalance_id": tick["data"]["rebalance_id"],
        }).to_dict()
        self.assertEqual(replayed["state"], "AVAILABLE", replayed)
        self.assertEqual(replayed["data"]["rebalance_id"], tick["data"]["rebalance_id"])
        state = self.strategies.load_schedule_state(planned["data"]["schedule_id"])
        self.assertEqual(state["last_rebalance_id"], tick["data"]["rebalance_id"])
        with self.assertRaisesRegex(ValueError, "rebalance_mode"):
            self.strategies.schedule({
                "action": "watch", "cadence_seconds": 3600, "max_ticks": 1,
                "now": self.now, "request": watch_request,
                "cost_rebalance": True, "rebalance_mode": "NOT_A_MODE",
            })

    def test_schedule_ticks_attach_factor_exposure(self):
        first_signal_id = self.screen["signal_id"]
        signal_doc = SignalWorkspace(self.base).load(first_signal_id)
        later = (pd.Timestamp(self.now) + pd.Timedelta(days=1)).isoformat()
        later_signal = SignalWorkspace(self.base).run({
            "run_ids": signal_doc["run_ids"], "now": later,
            "lookback": 20, "min_ic_observations": 5,
        }).to_dict()["data"]
        watch_request = {
            "signal_ids": [first_signal_id, later_signal["signal_id"]],
            "buy_threshold": 0.0, "sell_threshold": -0.01,
            "portfolio_notional": "100000", "max_name_notional": "50000",
            "lot_size": 1, "vol_scale": "1",
        }
        planned = self.strategies.schedule({
            "action": "watch", "cadence_seconds": 3600, "max_ticks": 1,
            "now": self.now, "request": watch_request,
            "factor_exposure": True, "n_factors": 2, "min_obs": 40, "lookback": 40,
        }).to_dict()
        self.assertEqual(planned["state"], "AVAILABLE", planned)
        self.assertTrue(planned["data"]["factor_exposure"])
        self.assertEqual(planned["data"]["factor_exposure_request"]["n_factors"], 2)
        tick = self.strategies.tick_schedule({
            "schedule_id": planned["data"]["schedule_id"], "now": self.now,
        }).to_dict()
        self.assertEqual(tick["state"], "AVAILABLE", tick)
        self.assertTrue(tick["data"]["size_id"])
        self.assertTrue(tick["data"]["exposure_id"])
        self.assertTrue(tick["data"]["metrics"]["factor_exposure_attached"])
        self.assertFalse(tick["data"]["trading_approved"])
        replayed = self.strategies.replay_factor_exposure({
            "exposure_id": tick["data"]["exposure_id"],
        }).to_dict()
        self.assertEqual(replayed["state"], "AVAILABLE", replayed)
        self.assertEqual(replayed["data"]["exposure_id"], tick["data"]["exposure_id"])
        state = self.strategies.load_schedule_state(planned["data"]["schedule_id"])
        self.assertEqual(state["last_exposure_id"], tick["data"]["exposure_id"])

    def test_schedule_tick_soft_skips_when_ic_ir_decayed(self):
        first_signal_id = self.screen["signal_id"]
        signal_doc = SignalWorkspace(self.base).load(first_signal_id)
        later = (pd.Timestamp(self.now) + pd.Timedelta(days=1)).isoformat()
        later_signal = SignalWorkspace(self.base).run({
            "run_ids": signal_doc["run_ids"], "now": later,
            "lookback": 20, "min_ic_observations": 5,
        }).to_dict()["data"]
        watch_request = {
            "signal_ids": [first_signal_id, later_signal["signal_id"]],
            "buy_threshold": 0.0, "sell_threshold": -0.01,
            "portfolio_notional": "100000", "max_name_notional": "50000",
            "lot_size": 1, "vol_scale": "1",
        }
        planned = self.strategies.schedule({
            "action": "watch", "cadence_seconds": 3600, "max_ticks": 2,
            "now": self.now, "request": watch_request,
            "min_ic_ir": 1e9, "alpha_decay_action": "soft_skip",
        }).to_dict()
        self.assertEqual(planned["state"], "AVAILABLE", planned)
        schedule_id = planned["data"]["schedule_id"]
        first = self.strategies.tick_schedule({
            "schedule_id": schedule_id, "now": self.now,
        }).to_dict()
        self.assertEqual(first["state"], "AVAILABLE", first)
        self.assertTrue(first["data"]["metrics"]["alpha_decayed"])
        self.assertEqual(first["data"]["metrics"]["alpha_decay_action"], "soft_skip")
        self.assertTrue(first["data"]["size_id"])
        second_now = (pd.Timestamp(self.now) + pd.Timedelta(hours=1)).isoformat()
        second = self.strategies.tick_schedule({
            "schedule_id": schedule_id, "now": second_now,
        }).to_dict()
        self.assertEqual(second["state"], "STALE")
        self.assertEqual(second["data"]["tick_state"], "ALPHA_DECAYED")
        due = self.strategies.due_schedules({"now": second_now, "limit": 8}).to_dict()
        self.assertTrue(all(row["schedule_id"] != schedule_id for row in due["data"]["rows"]))

    def test_schedule_tick_invalidates_on_alpha_decay(self):
        first_signal_id = self.screen["signal_id"]
        signal_doc = SignalWorkspace(self.base).load(first_signal_id)
        later = (pd.Timestamp(self.now) + pd.Timedelta(days=1)).isoformat()
        later_signal = SignalWorkspace(self.base).run({
            "run_ids": signal_doc["run_ids"], "now": later,
            "lookback": 20, "min_ic_observations": 5,
        }).to_dict()["data"]
        watch_request = {
            "signal_ids": [first_signal_id, later_signal["signal_id"]],
            "buy_threshold": 0.0, "sell_threshold": -0.01,
            "portfolio_notional": "100000", "max_name_notional": "50000",
            "lot_size": 1, "vol_scale": "1",
        }
        planned = self.strategies.schedule({
            "action": "watch", "cadence_seconds": 3600, "max_ticks": 2,
            "now": self.now, "request": watch_request,
            "min_ic_half_life_steps": 100.0, "alpha_decay_action": "invalidate",
        }).to_dict()
        self.assertEqual(planned["state"], "AVAILABLE", planned)
        first = self.strategies.tick_schedule({
            "schedule_id": planned["data"]["schedule_id"], "now": self.now,
        }).to_dict()
        self.assertEqual(first["state"], "AVAILABLE", first)
        self.assertTrue(first["data"]["metrics"]["alpha_decayed"])
        self.assertTrue(first["data"]["invalidation_id"])
        blocked = self.strategies.size({
            "screen_id": first["data"]["screen_id"], "now": self.now,
            "portfolio_notional": "100000", "max_name_notional": "50000",
            "lot_size": 1, "max_age_seconds": 86_400, "vol_scale": "1",
        }).to_dict()
        self.assertEqual(blocked["state"], "UNAVAILABLE")

    def test_schedule_ticks_chain_previous_size_for_delta_cost(self):
        # Watch ticks avoid OMS prepare conflicts across ticks while still chaining sizes.
        first_signal_id = self.screen["signal_id"]
        signal_doc = SignalWorkspace(self.base).load(first_signal_id)
        later = (pd.Timestamp(self.now) + pd.Timedelta(days=1)).isoformat()
        later_signal = SignalWorkspace(self.base).run({
            "run_ids": signal_doc["run_ids"], "now": later,
            "lookback": 20, "min_ic_observations": 5,
        }).to_dict()["data"]
        watch_request = {
            "signal_ids": [first_signal_id, later_signal["signal_id"]],
            "buy_threshold": 0.0, "sell_threshold": -0.01,
            "portfolio_notional": "100000", "max_name_notional": "50000",
            "lot_size": 1, "vol_scale": "1",
        }
        planned = self.strategies.schedule({
            "action": "watch", "cadence_seconds": 3600, "max_ticks": 2,
            "now": self.now, "request": watch_request,
        }).to_dict()
        self.assertEqual(planned["state"], "AVAILABLE", planned)
        schedule_id = planned["data"]["schedule_id"]
        first = self.strategies.tick_schedule({
            "schedule_id": schedule_id, "now": self.now,
        }).to_dict()
        self.assertEqual(first["state"], "AVAILABLE", first)
        self.assertTrue(first["data"]["size_id"])
        self.assertIsNone(first["data"]["previous_size_id"])
        self.assertFalse(first["data"]["metrics"]["size_chained"])
        open_cost = self.strategies.estimate_cost({
            "size_id": first["data"]["size_id"], "now": self.now, "adv_lookback_bars": 20,
        }).to_dict()
        self.assertEqual(open_cost["state"], "AVAILABLE", open_cost)
        self.assertEqual(open_cost["data"]["request"]["trade_basis"], "OPEN_FROM_FLAT")
        second_now = (pd.Timestamp(self.now) + pd.Timedelta(hours=1)).isoformat()
        second = self.strategies.tick_schedule({
            "schedule_id": schedule_id, "now": second_now,
        }).to_dict()
        self.assertEqual(second["state"], "AVAILABLE", second)
        self.assertEqual(second["data"]["previous_size_id"], first["data"]["size_id"])
        self.assertTrue(second["data"]["metrics"]["size_chained"])
        delta = self.strategies.estimate_cost({
            "size_id": second["data"]["size_id"], "now": second_now,
            "previous_size_id": first["data"]["size_id"], "adv_lookback_bars": 20,
        }).to_dict()
        self.assertEqual(delta["state"], "AVAILABLE", delta)
        self.assertEqual(delta["data"]["request"]["trade_basis"], "DELTA_VS_PREVIOUS")
        self.assertEqual(delta["data"]["previous_size_id"], first["data"]["size_id"])
        self.assertFalse(second["data"]["trading_approved"])
        state = self.strategies.load_schedule_state(schedule_id)
        self.assertEqual(state["last_size_id"], second["data"]["size_id"])
        self.assertEqual(int(state["ticks_completed"]), 2)

    def test_watch_monitors_sizes_and_optionally_scans(self):
        first_signal_id = self.screen["signal_id"]
        signal_doc = SignalWorkspace(self.base).load(first_signal_id)
        later = (pd.Timestamp(self.now) + pd.Timedelta(days=1)).isoformat()
        later_signal = SignalWorkspace(self.base).run({
            "run_ids": signal_doc["run_ids"], "now": later,
            "lookback": 20, "min_ic_observations": 5,
        }).to_dict()["data"]
        underlying = self.screen["rows"][0]["instrument_id"]
        options = OptionsWorkspace(self.base)
        fixture = option_fixture(underlying_id=underlying)
        options.ingest(fixture)
        option_run = options.run(option_request(underlying_id=underlying)).to_dict()["data"]
        names = " and ".join(row["instrument_id"].replace("US:", "NYSE:")
                             for row in self.screen["rows"][:2])
        admitted = FinancialIntelligenceWorkspace(self.base).run({
            "operation": "admit-authored",
            "source_id": "watch-enrich-fixture",
            "title": "Authored watch enrichment fixture",
            "text": f"Analysts mentioned {names} during earnings.",
            "license": "GIG-AUTHORED",
        }).to_dict()
        self.assertEqual(admitted["state"], "AVAILABLE", admitted)
        watched = self.strategies.watch({
            "signal_ids": [first_signal_id, later_signal["signal_id"]],
            "buy_threshold": 0.0, "sell_threshold": -0.01,
            "now": later, "portfolio_notional": "100000", "max_name_notional": "8000",
            "lot_size": 1, "max_age_seconds": 86_400, "vol_scale": "1",
            "options_run_ids": [option_run["run_id"]],
            "document_id": admitted["data"]["document_id"], "nlp_profile": "v2",
        }).to_dict()
        EnvelopeSchema.model_validate(watched)
        self.assertEqual(watched["state"], "AVAILABLE", watched)
        body = watched["data"]
        self.assertFalse(body["trading_approved"])
        self.assertFalse(body["broker_connected"])
        self.assertTrue(body["monitor_id"])
        self.assertTrue(body["size_id"])
        self.assertTrue(body["scan_id"])
        self.assertTrue(body["enrichment_id"])
        self.assertEqual(body["metrics"]["screens"], 2)
        self.assertGreaterEqual(body["metrics"]["fi_matched"], 0)
        self.assertGreaterEqual(body["metrics"]["options_actionable"], 0)
        replayed = self.strategies.replay_watch({"watch_id": body["watch_id"]}).to_dict()
        self.assertEqual(replayed["data"]["watch_id"], body["watch_id"])
        missing = self.strategies.replay_watch({"watch_id": "0" * 64}).to_dict()
        self.assertEqual(missing["state"], "UNAVAILABLE")


class StrategyHTTPTests(unittest.IsolatedAsyncioTestCase):
    async def test_http_simulate_and_replay(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            request = seed_flow(root)
            dataset = PersistedDatasetWorkspace(root).build(request).to_dict()["data"]
            run = PersistedResearchWorkspace(root).run({
                "dataset_id": dataset["dataset_id"], "model": "lightgbm",
                "min_train": 30, "test_size": 12, "embargo": 1, "seed": 17,
            }).to_dict()["data"]
            transport = httpx.ASGITransport(app=create_app(state_dir=root))
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                screen = await client.post("/api/v1/workspace/research-screen", json={
                    "operation": "from-runs", "run_ids": [run["run_id"]], "now": request["as_of"],
                    "lookback": 20, "min_ic_observations": 5,
                    "buy_threshold": 0.0, "sell_threshold": -0.01,
                })
                self.assertEqual(screen.status_code, 200, screen.text)
                screen_id = screen.json()["data"]["screen_id"]
                payload = allocation_request("", request["as_of"])
                del payload["run_id"]
                payload.update(operation="simulate", screen_id=screen_id, fill=True,
                               max_turnover=0.3, max_transaction_cost="100")
                simulated = await client.post("/api/v1/workspace/research-strategy", json=payload)
                self.assertEqual(simulated.status_code, 200, simulated.text)
                body = simulated.json()["data"]
                self.assertTrue(body["fills"])
                self.assertFalse(body["trading_approved"])
                replay = await client.post("/api/v1/workspace/research-strategy", json={
                    "operation": "replay", "strategy_id": body["strategy_id"],
                })
                self.assertEqual(replay.status_code, 200, replay.text)
                self.assertEqual(replay.json()["data"]["strategy_id"], body["strategy_id"])
                sized = await client.post("/api/v1/workspace/research-strategy", json={
                    "operation": "size", "screen_id": screen_id, "now": request["as_of"],
                    "portfolio_notional": "100000", "max_name_notional": "8000",
                    "lot_size": 1, "max_age_seconds": 86400,
                })
                self.assertEqual(sized.status_code, 200, sized.text)
                size_body = sized.json()["data"]
                self.assertTrue(size_body["rows"])
                self.assertFalse(size_body["trading_approved"])
                size_replay = await client.post("/api/v1/workspace/research-strategy", json={
                    "operation": "size-replay", "size_id": size_body["size_id"],
                })
                self.assertEqual(size_replay.status_code, 200, size_replay.text)
                self.assertEqual(size_replay.json()["data"]["size_id"], size_body["size_id"])
                costed = await client.post("/api/v1/workspace/research-strategy", json={
                    "operation": "estimate-cost", "size_id": size_body["size_id"],
                    "now": request["as_of"], "adv_lookback_bars": 20,
                })
                self.assertEqual(costed.status_code, 200, costed.text)
                cost_body = costed.json()["data"]
                self.assertFalse(cost_body["trading_approved"])
                self.assertEqual(cost_body["metrics"]["evidence_class"], "ESTIMATED")
                cost_replay = await client.post("/api/v1/workspace/research-strategy", json={
                    "operation": "estimate-cost-replay", "cost_id": cost_body["cost_id"],
                })
                self.assertEqual(cost_replay.status_code, 200, cost_replay.text)
                self.assertEqual(cost_replay.json()["data"]["cost_id"], cost_body["cost_id"])
                rebalanced = await client.post("/api/v1/workspace/research-strategy", json={
                    "operation": "cost-rebalance", "size_id": size_body["size_id"],
                    "cost_id": cost_body["cost_id"], "now": request["as_of"],
                    "mode": "SKIP_BELOW_COST", "min_trade_notional": "1",
                })
                self.assertEqual(rebalanced.status_code, 200, rebalanced.text)
                reb_body = rebalanced.json()["data"]
                self.assertFalse(reb_body["trading_approved"])
                reb_replay = await client.post("/api/v1/workspace/research-strategy", json={
                    "operation": "cost-rebalance-replay",
                    "rebalance_id": reb_body["rebalance_id"],
                })
                self.assertEqual(reb_replay.status_code, 200, reb_replay.text)
                self.assertEqual(reb_replay.json()["data"]["rebalance_id"], reb_body["rebalance_id"])


if __name__ == "__main__":
    unittest.main()
