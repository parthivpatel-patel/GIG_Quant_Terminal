"""Research buy/hold/sell screens stay unapproved and replay from pinned signals."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import httpx

from gig.service.app import create_app
from gig.service.contracts import EnvelopeSchema
from gig.service.dataset_workspace import PersistedDatasetWorkspace
from gig.service.persisted_research import PersistedResearchWorkspace
from gig.service.research_screen import ResearchScreenWorkspace, label_rows, timing_events
from gig.service.signal_workspace import SignalWorkspace
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
        self.assertFalse(any(row["trading_approved"] for row in labeled))
        with self.assertRaisesRegex(ValueError, "buy_threshold"):
            label_rows(rows, buy_threshold=-0.1, sell_threshold=0.1)
        previous = [{"instrument_id": "US:AAA", "stance": "RESEARCH_HOLD"},
                    {"instrument_id": "US:CCC", "stance": "RESEARCH_SELL"}]
        events = {row["instrument_id"]: row["event"] for row in timing_events(labeled, previous)}
        self.assertEqual(events["US:AAA"], "ENTER_LONG")
        self.assertEqual(events["US:BBB"], "ENTER_HOLD")
        self.assertEqual(events["US:CCC"], "CONTINUE")


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


class ScreenHTTPTests(unittest.IsolatedAsyncioTestCase):
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
