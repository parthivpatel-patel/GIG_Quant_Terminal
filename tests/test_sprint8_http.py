"""Sprint 8 HTTP contracts preserve saved evaluation units and availability."""
from __future__ import annotations

import asyncio
import tempfile
import unittest
from pathlib import Path

import httpx

from gig.service.app import create_app
from gig.service.contracts import EnvelopeSchema
from test_observed_reference import reference_fixture


class Sprint8HTTPTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.base = Path(self.temporary.name)/"state"
        async def heartbeat():
            while True:
                await asyncio.sleep(.01)
        self.heartbeat = asyncio.create_task(heartbeat())
        self.client = httpx.AsyncClient(transport=httpx.ASGITransport(app=create_app(state_dir=self.base)), base_url="http://test")

    async def asyncTearDown(self):
        await self.client.aclose()
        self.heartbeat.cancel()
        self.temporary.cleanup()

    def decoded(self, response):
        self.assertEqual(response.status_code, 200, response.text)
        value = response.json()
        EnvelopeSchema.model_validate(value)
        return value

    async def test_saved_evaluation_is_research_in_every_terminal_view(self):
        endpoint = "/api/v1/workspace/observed-reference"
        source = self.decoded(await self.client.post(endpoint, json={"operation": "admit", **reference_fixture()}))["data"]
        report = self.decoded(await self.client.post(endpoint, json={"operation": "research", "snapshot_id": source["snapshot_id"],
                              "model": "ridge", "training_observations": 30, "test_observations": 10}))["data"]
        saved = self.decoded(await self.client.post(endpoint, json={"operation": "evaluate", "report_id": report["report_id"], "bootstrap_replications": 100}))
        for section in ("research", "models", "signals"):
            selected = self.decoded(await self.client.get("/api/v1/terminal/"+section, params={
                "engine": "observed-reference", "as_of": saved["data"]["as_of"], "max_age_seconds": 60}))
            self.assertEqual(selected["state"], "AVAILABLE")
            self.assertEqual(selected["data"]["evaluation_id"], saved["data"]["evaluation_id"])
            self.assertEqual(selected["provenance"]["classification"], "MODEL_ESTIMATE")
            self.assertFalse(selected["data"]["trading_approved"])
        earlier = self.decoded(await self.client.get("/api/v1/terminal/research", params={
            "engine": "observed-reference", "as_of": report["as_of"], "max_age_seconds": 60}))
        self.assertNotIn("evaluation_id", earlier["data"])
        self.assertEqual(earlier["data"]["report_id"], report["report_id"])

    async def test_equity_archive_http_provenance_sections_and_replay(self):
        from test_public_equity import ACQUIRED, REQUEST, fixture
        endpoint = "/api/v1/workspace/equity-reference"
        source = self.decoded(await self.client.post(endpoint, json={"operation": "admit", **REQUEST,
                              "csv_text": fixture(), "acquired_at": ACQUIRED}))
        for section in ("markets", "research", "operations"):
            selected = self.decoded(await self.client.get("/api/v1/terminal/"+section, params={
                "engine": "observed-equity", "as_of": ACQUIRED, "max_age_seconds": 60}))
            self.assertEqual(selected["data"]["snapshot_id"], source["data"]["snapshot_id"])
            self.assertEqual(selected["provenance"]["classification"], "SYNTHETIC")
            self.assertFalse(selected["data"]["funded_allocation_eligible"])
        replay = self.decoded(await self.client.post(endpoint, json={"operation": "replay", "snapshot_id": source["data"]["snapshot_id"]}))
        self.assertEqual(source, replay)
        blocked = await self.client.post(endpoint, json={"operation": "fetch"}, headers={"Origin": "https://outside.invalid"})
        self.assertEqual(blocked.status_code, 403)
