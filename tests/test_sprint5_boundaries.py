"""Service-side mutation preflight, offline limits and sanitized research errors."""
from __future__ import annotations

import asyncio
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx

from gig.execution.oms import OfflineOMS
from gig.service.app import create_app
from gig.service.contracts import EnvelopeSchema
from test_portfolio_ledger import populate, request


class Sprint5BoundaryTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.base = Path(self.temporary.name) / "state"
        async def clock():
            while True:
                await asyncio.sleep(.01)
        self.clock = asyncio.create_task(clock())
        self.client = httpx.AsyncClient(transport=httpx.ASGITransport(app=create_app(state_dir=self.base)),
                                        base_url="http://test")

    async def asyncTearDown(self):
        await self.client.aclose()
        self.clock.cancel()
        self.temporary.cleanup()

    async def test_nanosecond_execution_risk_time_rejects_before_ledger_publication(self):
        populate(self.base)
        payload = request(limits={"max_gross": .001})
        payload["now"] = "2026-11-02T14:02:00.123456789Z"
        response = await self.client.post("/api/v1/workspace/portfolio-accounting", json=payload)
        self.assertEqual(response.status_code, 422)
        self.assertFalse((self.base / "portfolio-ledger").exists())
        with OfflineOMS(self.base / "execution-engine/oms.sqlite") as oms:
            self.assertFalse(oms.kill_switch)
        payload["now"] = "2026-11-02T14:02:00.123456Z"
        with patch("gig.execution.alpaca.AlpacaBroker", side_effect=AssertionError("no broker")):
            result = await self.client.post("/api/v1/workspace/portfolio-accounting", json=payload)
        self.assertEqual(result.status_code, 200)
        self.assertEqual(result.json()["state"], "ERROR")
        with OfflineOMS(self.base / "execution-engine/oms.sqlite") as oms:
            self.assertTrue(oms.kill_switch)
        self.assertEqual(len(list((self.base / "portfolio-ledger/checkpoints").glob("*.json"))), 1)

    async def test_new_engines_missing_data_is_explicit_and_read_only(self):
        for engine, section in (("pairs", "stat-arb"), ("options-research", "options"),
                                ("signals-research", "signals")):
            with self.subTest(engine=engine):
                response = await self.client.get("/api/v1/terminal/" + section,
                    params={"mode": "local", "engine": engine, "as_of": "2026-11-02T14:00:00Z", "max_age_seconds": 60})
                self.assertEqual(response.status_code, 200)
                result = response.json()
                EnvelopeSchema.model_validate(result)
                self.assertEqual(result["state"], "UNAVAILABLE")
                self.assertIsNone(result["data"])
        self.assertFalse(self.base.exists())

    async def test_source_errors_are_sanitized_and_origin_guard_precedes_research(self):
        for action, method in (("pairs-research", "pairs_research"), ("options-research", "options_research"),
                               ("research-signals", "research_signals")):
            with self.subTest(action=action), patch("gig.service.cycle_services.CycleServices." + method,
                                                   side_effect=ValueError("private-path-and-provider-secret")) as operation:
                response = await self.client.post("/api/v1/workspace/" + action, json={})
                self.assertEqual(response.status_code, 422)
                self.assertNotIn("private-path", response.text)
                self.assertEqual(response.json()["error"]["code"], "INVALID_REQUEST")
                operation.reset_mock()
                cross_origin = await self.client.post("/api/v1/workspace/" + action,
                    json={}, headers={"Origin": "https://outside.invalid"})
                self.assertEqual(cross_origin.status_code, 403)
                operation.assert_not_called()
