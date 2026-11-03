"""Sprint 6 service boundaries: isolated books, serialization and explicit states."""
from __future__ import annotations

import asyncio
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx

from gig.service.app import create_app
from gig.service.contracts import EnvelopeSchema


class Sprint6HTTPTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.base = Path(self.temporary.name) / "state"
        async def clock():
            while True:
                await asyncio.sleep(.01)
        self.clock = asyncio.create_task(clock())
        self.client = httpx.AsyncClient(transport=httpx.ASGITransport(app=create_app(state_dir=self.base)), base_url="http://test")

    async def asyncTearDown(self):
        await self.client.aclose()
        self.clock.cancel()
        self.temporary.cleanup()

    def contract(self, response, *, status=200):
        self.assertEqual(response.status_code, status, response.text)
        value = response.json()
        EnvelopeSchema.model_validate(value)
        return value

    async def test_empty_new_engines_are_unavailable_and_reads_create_no_account(self):
        for engine, section in (("funded-account", "portfolio"), ("funded-risk", "risk"),
                                ("economic-research", "research"), ("microstructure-research", "execution")):
            with self.subTest(engine=engine):
                value = self.contract(await self.client.get("/api/v1/terminal/" + section,
                    params={"engine": engine, "mode": "local", "as_of": "2026-11-02T15:49:00.123456789Z", "max_age_seconds": 60}))
                self.assertEqual(value["state"], "UNAVAILABLE")
                self.assertIsNone(value["data"])
        self.assertFalse(self.base.exists())

    async def test_cross_origin_research_writes_never_invoke_domain_methods(self):
        actions = {"funded-account": "funded_account", "economic-research": "economic_research",
                   "microstructure-admit": "microstructure_admit", "microstructure-research": "microstructure_research",
                   "funded-risk": "funded_risk"}
        for action, method in actions.items():
            with self.subTest(action=action), patch("gig.service.cycle_services.CycleServices." + method) as operation:
                self.contract(await self.client.post("/api/v1/workspace/" + action, json={},
                    headers={"Origin": "https://outside.invalid"}), status=403)
                operation.assert_not_called()
        self.assertFalse(self.base.exists())

    async def test_new_errors_are_sanitized_before_external_serialization(self):
        for action, method in (("funded-account", "funded_account"), ("economic-research", "economic_research"),
                               ("microstructure-research", "microstructure_research"), ("funded-risk", "funded_risk")):
            with self.subTest(action=action), patch("gig.service.cycle_services.CycleServices." + method,
                                                   side_effect=ValueError("private-path-provider-secret")):
                response = await self.client.post("/api/v1/workspace/" + action, json={})
                value = self.contract(response, status=422)
                self.assertEqual(value["error"]["code"], "INVALID_REQUEST")
                self.assertNotIn("private-path-provider-secret", response.text)

    async def test_execution_precision_preflight_creates_no_funded_state(self):
        for now in ("2026-11-02T15:49:00.000000001Z", "2026-11-02T15:49:00.1234567890Z", "2026-11-02T15:49:00"):
            with self.subTest(now=now):
                self.contract(await self.client.post("/api/v1/workspace/funded-account",
                    json={"operation": "initialize", "now": now, "initial_cash": "100000"}), status=422)
                self.assertFalse((self.base / "funded-account").exists())

    async def test_missing_saved_research_is_unavailable_without_synthetic_fallback(self):
        for action in ("economic-replay", "microstructure-replay", "funded-risk-replay"):
            with self.subTest(action=action):
                result = self.contract(await self.client.post("/api/v1/workspace/" + action, json={"report_id": "0"*64}))
                self.assertEqual(result["state"], "UNAVAILABLE")
                self.assertIsNone(result["data"])
        self.assertFalse(self.base.exists())

    async def test_nonfinite_payloads_do_not_reach_domain_or_mutate_storage(self):
        for text in ('{"operation":"initialize","initial_cash":NaN}', '{"x":1e999}', '{"x":1,"x":2}'):
            with patch("gig.service.cycle_services.CycleServices.funded_account") as operation:
                self.contract(await self.client.post("/api/v1/workspace/funded-account",
                    content=text, headers={"Content-Type": "application/json"}), status=400)
                operation.assert_not_called()
        self.assertFalse(self.base.exists())
