"""New workspace boundaries retain explicit absence, scope and origin controls."""
from __future__ import annotations

import asyncio
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx

from gig.service.app import create_app
from gig.service.contracts import EnvelopeSchema


class Sprint7HTTPTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.base = Path(self.temporary.name)/"state"
        async def clock():
            while True:
                await asyncio.sleep(.01)
        self.clock = asyncio.create_task(clock())
        self.client = httpx.AsyncClient(transport=httpx.ASGITransport(app=create_app(state_dir=self.base)), base_url="http://test")

    async def asyncTearDown(self):
        await self.client.aclose()
        self.clock.cancel()
        self.temporary.cleanup()

    def decoded(self, response, status=200):
        self.assertEqual(response.status_code, status, response.text)
        result = response.json()
        EnvelopeSchema.model_validate(result)
        return result

    async def test_missing_new_engines_do_not_create_state_or_synthetic_results(self):
        for engine, section in (("funded-lifecycle", "portfolio"), ("independent-statement", "execution"), ("observed-reference", "models")):
            value = self.decoded(await self.client.get("/api/v1/terminal/"+section,
                params={"engine": engine, "as_of": "2026-11-03T15:00:00.123456789Z", "max_age_seconds": 60}))
            self.assertEqual(value["state"], "UNAVAILABLE")
            self.assertIsNone(value["data"])
        self.assertFalse(self.base.exists())

    async def test_missing_replay_remains_unavailable(self):
        requests = (("statement-replay", {"report_id": "0"*64}),
                    ("observed-reference", {"operation": "replay", "report_id": "0"*64}))
        for action, payload in requests:
            value = self.decoded(await self.client.post("/api/v1/workspace/"+action, json=payload))
            self.assertEqual(value["state"], "UNAVAILABLE")
            self.assertIsNone(value["data"])
        self.assertFalse(self.base.exists())

    async def test_origins_block_before_lifecycle_statement_or_network_calls(self):
        actions = {"funded-lifecycle": "funded_lifecycle", "statement-admit": "statement_admit",
                   "statement-reconcile": "statement_reconcile", "observed-reference": "observed_reference"}
        for action, method in actions.items():
            with self.subTest(action=action), patch("gig.service.cycle_services.CycleServices."+method) as operation:
                self.decoded(await self.client.post("/api/v1/workspace/"+action, json={}, headers={"Origin": "https://outside.invalid"}), 403)
                operation.assert_not_called()
        self.assertFalse(self.base.exists())

    async def test_reference_source_cannot_be_arbitrary_url_or_broker_connection(self):
        with patch("gig.data.public_reference.urllib.request.build_opener") as opener:
            payload = {"operation": "fetch", "provider": "https://outside.invalid", "currencies": ["USD"], "start": "2025-01-01", "end": "2025-12-31"}
            self.decoded(await self.client.post("/api/v1/workspace/observed-reference", json=payload), 422)
            opener.assert_not_called()
        self.assertFalse(self.base.exists())

    async def test_execution_precision_and_external_errors_are_preflighted_and_sanitized(self):
        payload = {"operation": "configure", "now": "2026-11-03T15:00:00.000000001Z", "settlement_policy": "IMMEDIATE"}
        self.decoded(await self.client.post("/api/v1/workspace/funded-lifecycle", json=payload), 422)
        self.assertFalse(self.base.exists())
        with patch("gig.service.cycle_services.CycleServices.statement_reconcile", side_effect=ValueError("private-statement-and-secret-path")):
            response = await self.client.post("/api/v1/workspace/statement-reconcile", json={})
            self.decoded(response, 422)
            self.assertNotIn("private-statement", response.text)
