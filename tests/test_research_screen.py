"""Research buy/hold/sell screens stay unapproved and replay from pinned signals."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import httpx
import pandas as pd

from gig.service.app import create_app
from gig.service.contracts import EnvelopeSchema
from gig.service.dataset_workspace import PersistedDatasetWorkspace
from gig.service.financial_workspace import FinancialIntelligenceWorkspace
from gig.service.options_workspace import OptionsWorkspace
from gig.service.persisted_research import PersistedResearchWorkspace
from gig.service.research_screen import (
    ResearchScreenWorkspace,
    label_rows,
    match_mentions,
    option_candidate,
    timing_events,
)
from gig.service.signal_workspace import SignalWorkspace
from test_options_workspace import option_fixture, option_request
from test_persisted_flow_http import allocation_request, seed_flow


class ScreenUnitTests(unittest.TestCase):
    def test_thresholds_and_timing_are_deterministic(self):
        rows = [
            {"instrument_id": "US:AAA", "timestamp": "t", "prediction": 0.4,
             "prediction_kind": "IC_WEIGHTED_CENTERED_RANK_SCORE"},
            {"instrument_id": "US:BBB", "timestamp": "t", "prediction": -0.1,
             "prediction_kind": "IC_WEIGHTED_CENTERED_RANK_SCORE"},
            {"instrument_id": "US:CCC", "timestamp": "t", "prediction": -0.5,
             "prediction_kind": "IC_WEIGHTED_CENTERED_RANK_SCORE"},
        ]
        labeled = label_rows(rows, buy_threshold=0.25, sell_threshold=-0.25)
        self.assertEqual([row["stance"] for row in labeled],
                         ["RESEARCH_BUY", "RESEARCH_HOLD", "RESEARCH_SELL"])
        self.assertGreater(labeled[0]["research_weight"], 0.0)
        self.assertLess(labeled[2]["research_weight"], 0.0)
        self.assertAlmostEqual(abs(labeled[0]["research_weight"]) + abs(labeled[2]["research_weight"]), 1.0)
        self.assertEqual(labeled[1]["research_weight"], 0.0)
        self.assertEqual(labeled[2]["opportunity_rank"], 1)  # |−0.5| > |0.4|
        self.assertEqual(labeled[0]["opportunity_rank"], 2)
        self.assertIsNone(labeled[1]["opportunity_rank"])
        self.assertEqual(labeled[2]["confidence"]["state"], "UNCALIBRATED_RELATIVE_ABS_SCORE")
        self.assertAlmostEqual(labeled[2]["confidence"]["value"], 1.0)
        self.assertFalse(any(row["trading_approved"] for row in labeled))
        expired = label_rows(
            rows, buy_threshold=0.25, sell_threshold=-0.25, ttl_seconds=60,
            signal_as_of="2024-01-01T00:00:00Z", now="2024-01-01T00:02:00Z",
        )
        self.assertEqual(expired[0]["expiration_state"], "EXPIRED")
        with self.assertRaisesRegex(ValueError, "buy_threshold"):
            label_rows(rows, buy_threshold=-0.1, sell_threshold=0.1)
        previous = [{"instrument_id": "US:AAA", "stance": "RESEARCH_HOLD"},
                    {"instrument_id": "US:CCC", "stance": "RESEARCH_SELL"}]
        events = {row["instrument_id"]: row["event"] for row in timing_events(labeled, previous)}
        self.assertEqual(events["US:AAA"], "ENTER_LONG")
        self.assertEqual(events["US:BBB"], "ENTER_HOLD")
        self.assertEqual(events["US:CCC"], "CONTINUE")
        self.assertEqual(option_candidate(labeled, "US:AAA")["option_candidate"], "RESEARCH_CALL_CANDIDATE")
        self.assertEqual(option_candidate(labeled, "US:CCC")["option_candidate"], "RESEARCH_PUT_CANDIDATE")
        self.assertEqual(option_candidate(labeled, "US:BBB")["option_candidate"], "RESEARCH_HOLD_NO_OPTION")
        self.assertEqual(option_candidate(labeled, "US:ZZZ")["option_candidate"], "RESEARCH_NO_OPTION_MATCH")
        matches = match_mentions(labeled, [
            {"mention": "NYSE:AAA", "symbol_mention": "AAA",
             "link_state": "UNKNOWN_NO_ISSUER_AUTHORITY"},
            {"mention": "Federal Reserve", "link_state": "UNKNOWN_NO_ISSUER_AUTHORITY"},
        ])
        self.assertEqual([row["instrument_id"] for row in matches], ["US:AAA"])
        self.assertEqual(matches[0]["stance"], "RESEARCH_BUY")
        self.assertFalse(matches[0]["trading_approved"])


class ScreenFlowTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)
        self.request = seed_flow(self.base)
        self.now = self.request["as_of"]
        self.dataset = PersistedDatasetWorkspace(self.base).build(self.request).to_dict()["data"]
        run = PersistedResearchWorkspace(self.base).run({
            "dataset_id": self.dataset["dataset_id"], "model": "lightgbm",
            "min_train": 30, "test_size": 12, "embargo": 1, "seed": 17,
        }).to_dict()["data"]
        self.signal = SignalWorkspace(self.base).run({
            "run_ids": [run["run_id"]], "now": self.now, "lookback": 20, "min_ic_observations": 5,
        }).to_dict()["data"]
        self.screens = ResearchScreenWorkspace(self.base)

    def test_discover_from_dataset_factor_ranks(self):
        discovered = self.screens.discover({
            "dataset_id": self.dataset["dataset_id"], "now": self.now,
            "mode": "momentum", "momentum_lookback": 10, "reversion_lookback": 5,
            "buy_threshold": 0.0, "sell_threshold": -0.01,
        }).to_dict()
        EnvelopeSchema.model_validate(discovered)
        self.assertEqual(discovered["state"], "AVAILABLE", discovered)
        self.assertEqual(discovered["data"]["discovery"]["operation"], "from-dataset")
        self.assertTrue(discovered["data"]["opportunity_ranking"])
        self.assertFalse(discovered["data"]["trading_approved"])
        signal = SignalWorkspace(self.base).load(discovered["data"]["signal_id"])
        self.assertEqual(signal["schema_version"], "cs-factor-rank-v1")
        self.assertTrue(all(
            row["prediction_kind"] == "CROSS_SECTIONAL_FACTOR_CENTERED_RANK_SCORE"
            for row in signal["rows"]
        ))

    def test_enrich_mentions_matches_exchange_qualified_symbols(self):
        screen = self.screens.run({
            "signal_id": self.signal["signal_id"], "buy_threshold": 0.0, "sell_threshold": -0.01,
        }).to_dict()["data"]
        fi = FinancialIntelligenceWorkspace(self.base)
        admitted = fi.run({
            "operation": "admit-authored",
            "source_id": "screen-enrich-fixture",
            "title": "Authored screen enrichment fixture",
            "text": "Analysts mentioned NYSE:AAA and NASDAQ:BBB during earnings.",
            "license": "GIG-AUTHORED",
        }).to_dict()
        self.assertEqual(admitted["state"], "AVAILABLE", admitted)
        document_id = admitted["data"]["document_id"]
        enriched = self.screens.enrich_mentions({
            "screen_id": screen["screen_id"], "document_id": document_id, "nlp_profile": "v2",
        }).to_dict()
        EnvelopeSchema.model_validate(enriched)
        self.assertEqual(enriched["state"], "AVAILABLE", enriched)
        self.assertFalse(enriched["data"]["trading_approved"])
        instruments = {row["instrument_id"] for row in enriched["data"]["matched_mentions"]}
        self.assertTrue({"US:AAA", "US:BBB"} & instruments)
        self.assertGreaterEqual(enriched["data"]["matched_count"], 1)

    def test_screen_replays_and_refuses_a_future_previous_screen(self):
        first = self.screens.run({
            "signal_id": self.signal["signal_id"], "buy_threshold": 0.0, "sell_threshold": -0.01,
        }).to_dict()
        EnvelopeSchema.model_validate(first)
        self.assertEqual(first["state"], "AVAILABLE")
        self.assertEqual(first["provenance"]["readiness"], "RESEARCH_ONLY")
        self.assertFalse(first["data"]["trading_approved"])
        self.assertTrue(first["data"]["rows"])
        self.assertTrue(all(row["stance"].startswith("RESEARCH_") for row in first["data"]["rows"]))
        replayed = self.screens.replay({"screen_id": first["data"]["screen_id"]}).to_dict()
        self.assertEqual(replayed, first)
        missing = self.screens.replay({"screen_id": "0" * 64}).to_dict()
        self.assertEqual(missing["state"], "UNAVAILABLE")
        with self.assertRaisesRegex(ValueError, "precede"):
            self.screens.run({
                "signal_id": self.signal["signal_id"], "previous_screen_id": first["data"]["screen_id"],
                "buy_threshold": 0.0, "sell_threshold": -0.01,
            })

    def test_from_runs_discovers_a_screen_in_one_step(self):
        discovered = self.screens.discover({
            "run_ids": [self.signal["run_ids"][0]], "now": self.now,
            "lookback": 20, "min_ic_observations": 5,
            "buy_threshold": 0.0, "sell_threshold": -0.01,
        }).to_dict()
        self.assertEqual(discovered["state"], "AVAILABLE")
        self.assertEqual(discovered["data"]["discovery"]["operation"], "from-runs")
        self.assertEqual(discovered["data"]["discovery"]["signal_id"], discovered["data"]["signal_id"])
        self.assertFalse(discovered["data"]["trading_approved"])
        again = self.screens.replay({"screen_id": discovered["data"]["screen_id"]}).to_dict()
        self.assertEqual(again["data"]["rows"], discovered["data"]["rows"])

    def test_monitor_chains_successive_saved_signals(self):
        first_screen = self.screens.run({
            "signal_id": self.signal["signal_id"], "buy_threshold": 0.0, "sell_threshold": -0.01,
        }).to_dict()["data"]
        later = (pd.Timestamp(self.now) + pd.Timedelta(days=1)).isoformat()
        later_signal = SignalWorkspace(self.base).run({
            "run_ids": [self.signal["run_ids"][0]], "now": later,
            "lookback": 20, "min_ic_observations": 5,
        }).to_dict()["data"]
        self.assertNotEqual(later_signal["signal_id"], self.signal["signal_id"])
        monitored = self.screens.monitor({
            "signal_ids": [self.signal["signal_id"], later_signal["signal_id"]],
            "buy_threshold": 0.0, "sell_threshold": -0.01,
        }).to_dict()
        self.assertEqual(monitored["state"], "AVAILABLE")
        self.assertEqual(monitored["data"]["metrics"]["screens"], 2)
        self.assertFalse(monitored["data"]["trading_approved"])
        self.assertEqual(len(monitored["data"]["timeline"]), 2)
        self.assertEqual(monitored["data"]["screen_ids"][0], first_screen["screen_id"])

    def test_options_link_follows_the_matching_equity_stance(self):
        screen = self.screens.run({
            "signal_id": self.signal["signal_id"], "buy_threshold": 0.0, "sell_threshold": -0.01,
        }).to_dict()["data"]
        underlying = screen["rows"][0]["instrument_id"]
        options = OptionsWorkspace(self.base)
        options.ingest(option_fixture(underlying_id=underlying))
        run = options.run(option_request(underlying_id=underlying)).to_dict()["data"]
        linked = self.screens.link_options({
            "screen_id": screen["screen_id"], "options_run_id": run["run_id"],
        }).to_dict()
        self.assertEqual(linked["state"], "AVAILABLE")
        self.assertFalse(linked["data"]["trading_approved"])
        self.assertEqual(linked["data"]["underlying_id"], underlying)
        expected = option_candidate(screen["rows"], underlying)["option_candidate"]
        self.assertEqual(linked["data"]["option_candidate"], expected)
        missing = self.screens.link_options({
            "screen_id": screen["screen_id"], "options_run_id": "0" * 64,
        }).to_dict()
        self.assertEqual(missing["state"], "UNAVAILABLE")

    def test_options_scan_lists_candidates_across_saved_runs(self):
        screen = self.screens.run({
            "signal_id": self.signal["signal_id"], "buy_threshold": 0.0, "sell_threshold": -0.01,
        }).to_dict()["data"]
        options = OptionsWorkspace(self.base)
        runs = []
        for row in screen["rows"][:2]:
            underlying = row["instrument_id"]
            fixture = option_fixture(underlying_id=underlying, source=f"authored-{underlying}")
            for quote in fixture["quotes"]:
                quote["contract_id"] = quote["contract_id"].replace("SYN100", underlying.replace(":", ""))
            request = option_request(underlying_id=underlying, source=fixture["source"], positions=[
                {"contract_id": fixture["quotes"][0]["contract_id"], "quantity": 1},
                {"contract_id": fixture["quotes"][1]["contract_id"], "quantity": -1},
            ])
            options.ingest(fixture)
            result = options.run(request).to_dict()
            self.assertEqual(result["state"], "AVAILABLE", result)
            runs.append(result["data"]["run_id"])
        scanned = self.screens.scan_options({
            "screen_id": screen["screen_id"], "options_run_ids": runs,
        }).to_dict()
        EnvelopeSchema.model_validate(scanned)
        self.assertEqual(scanned["state"], "AVAILABLE", scanned)
        self.assertEqual(scanned["data"]["metrics"]["scanned"], 2)
        self.assertFalse(scanned["data"]["trading_approved"])
        self.assertEqual(len(scanned["data"]["rows"]), 2)
        self.assertTrue(all("iv_opportunity_score" in row for row in scanned["data"]["rows"]))
        ranked = [row["iv_opportunity_rank"] for row in scanned["data"]["rows"]
                  if row["iv_opportunity_rank"] is not None]
        self.assertEqual(sorted(ranked), list(range(1, len(ranked) + 1)))
        structures = self.screens.discover_option_structures({
            "screen_id": screen["screen_id"], "options_run_ids": runs, "width_strikes": 1,
        }).to_dict()
        EnvelopeSchema.model_validate(structures)
        self.assertEqual(structures["state"], "AVAILABLE", structures)
        self.assertFalse(structures["data"]["trading_approved"])
        self.assertGreaterEqual(structures["data"]["metrics"]["structures"], 1)
        self.assertTrue(all(row["structure"] in {"DEBIT_CALL_VERTICAL", "DEBIT_PUT_VERTICAL"}
                            for row in structures["data"]["rows"]))
        self.assertEqual(
            [row["structure_rank"] for row in structures["data"]["rows"]],
            list(range(1, len(structures["data"]["rows"]) + 1)),
        )


class ScreenHTTPTests(unittest.IsolatedAsyncioTestCase):
    async def test_discover_allocates_and_prepares_offline_targets(self):
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
                discovered = await client.post("/api/v1/workspace/research-screen", json={
                    "operation": "from-runs", "run_ids": [run["run_id"]], "now": request["as_of"],
                    "lookback": 20, "min_ic_observations": 5,
                    "buy_threshold": 0.0, "sell_threshold": -0.01,
                })
                self.assertEqual(discovered.status_code, 200, discovered.text)
                body = discovered.json()["data"]
                self.assertEqual(body["discovery"]["operation"], "from-runs")
                self.assertFalse(body["trading_approved"])
                alloc = allocation_request("", request["as_of"])
                del alloc["run_id"]
                alloc.update(screen_id=body["screen_id"], max_turnover=0.3, max_transaction_cost="100")
                allocated = await client.post("/api/v1/workspace/research-allocate", json=alloc)
                self.assertEqual(allocated.status_code, 200, allocated.text)
                plan = allocated.json()["data"]
                self.assertEqual(plan["source_kind"], "screen_id")
                self.assertFalse(plan["trading_approved"])
                prepared = await client.post("/api/v1/workspace/research-prepare", json={
                    "allocation_id": plan["allocation_id"], "now": request["as_of"],
                })
                self.assertEqual(prepared.status_code, 200, prepared.text)
                self.assertTrue(prepared.json()["data"]["prepared_client_ids"])
                self.assertFalse(prepared.json()["data"].get("trading_approved", False))

    async def test_http_screen_hands_the_signal_to_offline_allocation(self):
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
                signal = await client.post("/api/v1/workspace/research-signals", json={
                    "run_ids": [run["run_id"]], "now": request["as_of"],
                    "lookback": 20, "min_ic_observations": 5,
                })
                self.assertEqual(signal.status_code, 200, signal.text)
                signal_id = signal.json()["data"]["signal_id"]
                screen = await client.post("/api/v1/workspace/research-screen", json={
                    "signal_id": signal_id, "buy_threshold": 0.0, "sell_threshold": -0.01,
                })
                self.assertEqual(screen.status_code, 200, screen.text)
                body = screen.json()
                self.assertEqual(body["data"]["signal_id"], signal_id)
                self.assertFalse(body["data"]["trading_approved"])
                replay = await client.post("/api/v1/workspace/screen-replay",
                                           json={"screen_id": body["data"]["screen_id"]})
                self.assertEqual(replay.json()["data"]["screen_id"], body["data"]["screen_id"])
                alloc = allocation_request("", request["as_of"])
                del alloc["run_id"]
                alloc.update(signal_id=signal_id, max_turnover=0.3, max_transaction_cost="100")
                allocated = await client.post("/api/v1/workspace/research-allocate", json=alloc)
                self.assertEqual(allocated.status_code, 200, allocated.text)
                self.assertFalse(allocated.json()["data"]["trading_approved"])
                for bad in ({"signal_id": signal_id, "buy_threshold": True},
                            {"signal_id": signal_id, "broker": "live"},
                            {"screen_id": body["data"]["screen_id"], "extra": 1}):
                    endpoint = "research-screen" if "signal_id" in bad else "screen-replay"
                    self.assertEqual((await client.post(f"/api/v1/workspace/{endpoint}", json=bad)).status_code, 422)


if __name__ == "__main__":
    unittest.main()
