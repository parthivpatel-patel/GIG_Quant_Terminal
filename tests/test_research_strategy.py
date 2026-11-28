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


if __name__ == "__main__":
    unittest.main()
