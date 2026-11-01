"""Exercise integrated CPU, provider and execution persistence through HTTP."""

from __future__ import annotations

import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx

from gig.service.app import create_app
from gig.service.contracts import EnvelopeSchema
from test_finbert_actual import ACTUAL, ASSETS
from test_finbert_actual import request as sentiment_request
from test_provider_ingestion import NOW, config, page
from test_sequence_model import fixture, inference_rows


class CycleHTTPTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.directory = Path(self.temporary.name) / "state"

        async def clock():
            while True:
                await asyncio.sleep(.01)

        self.clock = asyncio.create_task(clock())
        self.client = self.reopen()

    def reopen(self, *, finbert=False):
        app = create_app(state_dir=self.directory, finbert_dir=ASSETS if finbert else None)
        return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")

    async def asyncTearDown(self):
        await self.client.aclose()
        self.clock.cancel()
        self.temporary.cleanup()

    def decoded(self, response, status=200):
        self.assertEqual(response.status_code, status, response.text)
        data = response.json()
        EnvelopeSchema.model_validate(data)
        return data

    async def post(self, action, payload, *, client=None, status=200):
        return self.decoded(await (client or self.client).post(
            f"/api/v1/workspace/{action}", json=payload), status)

    async def view(self, section, stamp, *, engine="default", client=None, **params):
        return self.decoded(await (client or self.client).get(
            f"/api/v1/terminal/{section}", params={
                "mode": "local", "engine": engine, "as_of": stamp,
                "max_age_seconds": 60, **params,
            }))

    async def test_fixture_ingestion_and_universe_are_shared_after_restart(self):
        payload = {"provider": "alpaca_iex", "mode": "OFFLINE_FIXTURE", "job_id": "http-one",
                   **config(), "pages": [{"body": json.loads(page().body), "observed_at": NOW}],
                   "now": NOW}
        with patch("gig.execution.alpaca.AlpacaBroker", side_effect=AssertionError("no broker")):
            receipt = await self.post("ingest", payload)
        self.assertEqual(receipt["state"], "AVAILABLE")
        self.assertEqual(receipt["provenance"]["classification"], "SYNTHETIC")
        await self.post("universe", {"universe_id": "universe-one", "source": "authored-fixture",
                                    "effective_at": NOW, "available_at": NOW,
                                    "members": config()["instruments"]})
        async with self.reopen() as reopened:
            stale = await self.view("markets", NOW, client=reopened)
            self.assertEqual(stale["state"], "STALE")
            market = await self.view("markets", NOW, client=reopened, max_age_seconds=600)
            self.assertEqual(market["state"], "AVAILABLE")
            self.assertEqual(market["data"]["rows"][0]["volume"], "9007199254740993")
            self.assertEqual(market["provenance"]["classification"], "SYNTHETIC")
            universe = await self.view("research", NOW, engine="provider", client=reopened,
                                       universe_id="universe-one")
            self.assertEqual(universe["data"]["members"][0]["instrument_id"], "US:AAA")

    async def test_actual_cpu_sequence_training_and_registry_prediction(self):
        payload = fixture()
        payload["epochs"] = 3
        trained = await self.post("train-sequence", payload)
        self.assertEqual(trained["state"], "AVAILABLE")
        model_id = trained["data"]["model_id"]
        async with self.reopen() as reopened:
            prediction = await self.post("predict-sequence", {
                "model_id": model_id, "rows": inference_rows(payload)}, client=reopened)
            self.assertEqual(prediction["state"], "AVAILABLE")
            self.assertEqual(len(prediction["data"]["rows"]), 16)
            models = await self.view("models", trained["provenance"]["as_of"],
                                     engine="sequence", client=reopened)
            self.assertEqual(models["provenance"]["model_identity"], model_id)
            self.assertFalse(models["data"]["rows"][0]["trading_approved"])

    async def test_execution_ack_partial_fill_reconciliation_and_restart(self):
        now = "2026-11-01T15:00:00Z"

        async def operation(name, **values):
            return await self.post("execution-engine", {"operation": name, "now": now, **values})

        with patch("gig.execution.alpaca.AlpacaBroker", side_effect=AssertionError("no broker")):
            await operation("prepare", client_id="http-one", symbol="SYN001", side="buy",
                            quantity="10", reference_price="100", price_asof=now,
                            max_notional="1000", max_position_quantity="10", limit_price="100")
            ack = await operation("dispatch")
            self.assertEqual(ack["data"]["positions"], {})
            filled = await operation("fill", client_id="http-one", quantity="2", price="100")
            self.assertEqual(filled["data"]["positions"]["SYN001"], "2")
            reconciled = await operation("reconcile")
            self.assertEqual(reconciled["state"], "AVAILABLE")
            await operation("kill-switch", engaged=True)
        async with self.reopen() as reopened:
            for section in ("execution", "portfolio", "risk"):
                result = await self.view(section, now, engine="execution-engine", client=reopened)
                self.assertEqual(result["state"], "AVAILABLE")
                self.assertTrue(result["data"]["metrics"]["kill_switch"])
                self.assertIsNone(result["data"]["metrics"]["nav"])
                self.assertFalse(result["data"]["metrics"]["broker_connected"])

    async def test_dependency_error_and_rejected_paths_are_sanitized(self):
        missing = await self.post("finbert", sentiment_request(), status=503)
        self.assertEqual(missing["state"], "ERROR")
        self.assertEqual(missing["error"]["code"], "DEPENDENCY_UNAVAILABLE")
        bad = await self.post("train-sequence", {**fixture(), "output_dir": "../escape"}, status=422)
        self.assertEqual(bad["state"], "ERROR")
        self.assertNotIn("escape", json.dumps(bad))
        self.assertFalse(self.directory.exists())

    @unittest.skipUnless(ACTUAL, "Actual pinned FinBERT assets/runtime not provisioned")
    async def test_actual_finbert_http_provenance_and_restart(self):
        payload = sentiment_request()
        async with self.reopen(finbert=True) as configured:
            response = await self.post("finbert", payload, client=configured)
        self.assertEqual(response["state"], "AVAILABLE")
        async with self.reopen(finbert=True) as configured:
            news = await self.view("news", payload["as_of"], engine="finbert", client=configured,
                                   max_age_seconds=3600)
            self.assertEqual(news["state"], "AVAILABLE")
            self.assertEqual(news["data"]["rows"][0]["body"], payload["articles"][0]["body"])
            models = await self.view("models", payload["as_of"], engine="finbert", client=configured)
            self.assertEqual(models["provenance"]["readiness"], "NONCOMMERCIAL_RESEARCH_ONLY")
            self.assertFalse(models["data"]["rows"][0]["commercial_use_permitted"])


if __name__ == "__main__":
    unittest.main()
