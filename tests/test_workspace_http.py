"""Persist real admitted values through HTTP, reopen, and compare terminal truth."""
from __future__ import annotations

import asyncio
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

import httpx
import pandas as pd

from gig.service.app import create_app
from gig.service.contracts import EnvelopeSchema
from test_terminal_workspace import (
    NOW,
    bar_request,
    news_request,
    research_request,
    simulation_request,
)


class WorkspaceHTTPTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.directory = Path(self.temporary.name) / "admitted"
        async def clock():
            while True:
                await asyncio.sleep(.01)
        self.clock = asyncio.create_task(clock())
        self.app = create_app(state_dir=self.directory)
        self.client = httpx.AsyncClient(transport=httpx.ASGITransport(app=self.app), base_url="http://test")

    async def asyncTearDown(self):
        await self.client.aclose()
        self.clock.cancel()
        self.temporary.cleanup()

    def admitted(self, response, *, status=200):
        self.assertEqual(response.status_code, status, response.text)
        payload = response.json()
        EnvelopeSchema.model_validate(payload)
        return payload

    async def view(self, name, *, stamp=NOW, age=60, client=None):
        query = {"mode": "local", "as_of": stamp.isoformat()}
        if age is not None:
            query["max_age_seconds"] = age
        return self.admitted(await (client or self.client).get(f"/api/v1/terminal/{name}", params=query))

    async def test_read_missing_workspace_creates_no_state(self):
        for name in ("markets", "signals", "portfolio", "risk", "research", "execution", "news"):
            with self.subTest(name=name):
                self.assertEqual((await self.view(name))["state"], "UNAVAILABLE")
        self.assertFalse(self.directory.exists())

    async def test_bars_ingress_then_actual_market_and_factor_views_after_restart(self):
        request = bar_request()
        receipt = self.admitted(await self.client.post("/api/v1/workspace/bars", json=request))
        self.assertEqual(receipt["state"], "AVAILABLE")
        reopened = httpx.AsyncClient(transport=httpx.ASGITransport(app=create_app(state_dir=self.directory)), base_url="http://test")
        async with reopened:
            markets = await self.view("markets", client=reopened)
            expected = {row["instrument_id"]: row for row in request["rows"][-2:]}
            for row in markets["data"]["rows"]:
                self.assertEqual(row["close"], expected[row["instrument_id"]]["close"])
            self.assertEqual(markets["provenance"]["as_of"], NOW.isoformat())
            signals = await self.view("signals", client=reopened)
            self.assertEqual(signals["state"], "AVAILABLE")
            self.assertEqual(len(signals["data"]["chart"]["points"]), 2)
            self.assertEqual((await self.view("stat-arb", client=reopened))["state"], "AVAILABLE")

    async def test_point_in_time_revision_and_freshness_are_visible(self):
        request = bar_request()
        self.admitted(await self.client.post("/api/v1/workspace/bars", json=request))
        corrected = deepcopy(request)
        corrected["rows"] = [deepcopy(request["rows"][-2])]
        corrected["rows"][0].update(close=123, open=123, low=122, high=124,
                                    available_at=(NOW + pd.Timedelta(seconds=30)).isoformat())
        self.admitted(await self.client.post("/api/v1/workspace/bars", json=corrected))
        before = await self.view("markets")
        after = await self.view("markets", stamp=NOW + pd.Timedelta(seconds=30))
        self.assertNotEqual(before["data"]["rows"][0]["close"], 123)
        self.assertEqual(after["data"]["rows"][0]["close"], 123)
        self.assertEqual((await self.view("markets", stamp=NOW + pd.Timedelta(seconds=61)))["state"], "STALE")
        unknown = await self.view("markets", age=None)
        self.assertEqual(unknown["state"], "UNKNOWN")
        self.assertIsNone(unknown["data"])

    async def test_news_source_text_and_licensing_survive_http(self):
        request = news_request()
        admitted = self.admitted(await self.client.post("/api/v1/workspace/news", json=request))
        self.assertEqual(admitted["state"], "AVAILABLE")
        news = await self.view("news", age=120)
        row = news["data"]["rows"][0]
        self.assertEqual(row["body"], request["articles"][0]["body"])
        self.assertEqual(row["license"], request["articles"][0]["license"])
        self.assertEqual(row["sentiment"]["method"], "LEXICON")
        self.assertIn("ID:AAA", row["linked_symbols"])

    async def test_purged_research_replay_and_terminal_model_identity(self):
        request = research_request()
        first = self.admitted(await self.client.post("/api/v1/workspace/research", json=request))
        second = self.admitted(await self.client.post("/api/v1/workspace/research", json=request))
        self.assertEqual(first["content_sha256"], second["content_sha256"])
        visible = await self.view("research")
        self.assertEqual(visible["data"]["run_id"], first["data"]["run_id"])
        self.assertEqual(visible["provenance"]["model_identity"], first["data"]["run_id"])
        self.assertFalse(visible["data"]["trading_approved"])
        models = await self.view("models")
        self.assertEqual(models["state"], "AVAILABLE")
        self.assertTrue(all(row["trained"] for row in models["data"]["rows"]))
        self.assertTrue(all(not row["trading_approved"] for row in models["data"]["rows"]))
        for fold in visible["data"]["folds"]:
            self.assertLess(pd.Timestamp(fold["max_training_label_end"]), pd.Timestamp(fold["test_start"]))

    async def test_offline_flow_positions_risk_and_replay_never_import_broker(self):
        with patch("gig.execution.alpaca.AlpacaBroker", side_effect=AssertionError("no broker")):
            self.admitted(await self.client.post("/api/v1/workspace/bars", json=bar_request()))
            request = simulation_request()
            simulated = self.admitted(await self.client.post("/api/v1/workspace/simulate", json=request))
            self.assertEqual(simulated["state"], "AVAILABLE")
            self.assertFalse(simulated["data"]["result"]["broker_connected"])
            self.assertEqual(simulated["data"]["result"]["execution_classification"], "SIMULATED")
            replay = self.admitted(await self.client.post("/api/v1/workspace/simulate", json=request))
            self.assertTrue(replay["data"]["result"]["replayed"])
            self.assertEqual(replay["data"]["rows"], [])
            for name in ("portfolio", "risk", "execution"):
                result = await self.view(name)
                self.assertEqual(result["state"], "AVAILABLE")
                self.assertEqual(result["provenance"]["evidence_state"], "CALLER_SUPPLIED_NOT_VENDOR_VERIFIED")

    async def test_invalid_and_cross_origin_writes_leave_missing_workspace_absent(self):
        cases = [({"content": "{}"}, 415),
                 ({"json": bar_request(), "headers": {"Origin": "https://unrelated.invalid"}}, 403),
                 ({"json": {**bar_request(), "path": "../escape"}}, 422),
                 ({"content": b'{"rows":[],"rows":[]}', "headers": {"Content-Type": "application/json"}}, 400)]
        for arguments, status in cases:
            with self.subTest(status=status):
                response = await self.client.post("/api/v1/workspace/bars", **arguments)
                self.assertEqual(self.admitted(response, status=status)["state"], "ERROR")
                self.assertFalse(self.directory.exists())

    async def test_request_timestamp_precision_rejected_before_state_write(self):
        request = simulation_request()
        request["asof"] = NOW.isoformat()
        result = self.admitted(await self.client.post("/api/v1/workspace/simulate", json=request), status=422)
        self.assertIsNone(result["data"])
        self.assertFalse(self.directory.exists())
        for stamp in ("2026-01-01T00:00:00", "2026-01-01T00:00:00.1234567891Z"):
            response = await self.client.get("/api/v1/terminal/markets", params={"as_of": stamp})
            self.admitted(response, status=422)


if __name__ == "__main__":
    unittest.main()
