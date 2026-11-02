"""Saved data through actual models, constrained book and durable offline fills."""
from __future__ import annotations

import asyncio
import tempfile
import unittest
from decimal import Decimal
from pathlib import Path
from unittest.mock import patch

import httpx
import pandas as pd

from gig.data.validated import PointInTimeStore
from gig.service.app import create_app
from gig.service.contracts import EnvelopeSchema
from test_persisted_data import fixture_bars, seed_persisted_fixture


def seed_flow(base):
    request = seed_persisted_fixture(base)
    # Use explicit microsecond timestamps for the offline OMS; never silently
    # truncate source timestamps in the service. These are new authored inputs.
    # Recreate only this fresh test fixture directory's evidence through admission.
    # A separate source keeps the original nanosecond fixture fully preserved.
    from gig.data.ingestion import UniverseStore
    from test_persisted_data import INSTRUMENTS, SOURCE, UNIVERSE
    source = SOURCE + "-microsecond"
    rows = [{**r, "timestamp": pd.Timestamp(r["timestamp"]).floor("us").isoformat(),
             "available_at": pd.Timestamp(r["available_at"]).floor("us").isoformat()} for r in fixture_bars()]
    from gig.data.validated import admit_bars
    PointInTimeStore(base / "bars.sqlite").ingest(admit_bars(rows, source=source,
        classification="SYNTHETIC", corporate_action_policy="RAW", currency="USD"))
    request.update(source=source, start=rows[0]["timestamp"], end=rows[-1]["timestamp"],
                   as_of=rows[-1]["timestamp"])
    UniverseStore(base / "universe.sqlite").ingest({"universe_id": UNIVERSE + "-microsecond",
        "source": "authored-flow", "effective_at": "2026-01-01T00:00:00Z",
        "available_at": "2026-01-01T00:00:00Z", "members": INSTRUMENTS})
    request["universe_id"] = UNIVERSE + "-microsecond"
    return request


def allocation_request(run_id, now):
    return {"run_id": run_id, "now": now, "initial_cash": "100000", "n_long": 1,
            "n_short": 1, "gross_leverage": .2, "max_name_weight": .1, "lot_size": "1",
            "max_age_seconds": 10 * 86400, "annualization": 252, "price_tick": "0.01",
            "commission_bps": 5, "slippage_bps": 10,
            "metadata": {s: {"sector": "Authored", "factors": {"market": 1.}} for s in ("US:AAA", "US:BBB")},
            "limits": {"max_gross": .5, "max_abs_net": .3, "max_name_weight": .2,
                       "max_sector_gross": .5, "max_factor_exposure": {"market": .3}, "min_cash": "0"}}


class PersistedFlowHTTPTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.base = Path(self.temporary.name) / "state"
        self.dataset_request = seed_flow(self.base)
        self.now = self.dataset_request["as_of"]

        async def clock():
            while True:
                await asyncio.sleep(.01)
        self.clock = asyncio.create_task(clock())
        self.client = self.reopen()

    def reopen(self):
        return httpx.AsyncClient(transport=httpx.ASGITransport(app=create_app(state_dir=self.base)), base_url="http://test")

    async def asyncTearDown(self):
        await self.client.aclose()
        self.clock.cancel()
        self.temporary.cleanup()

    def decoded(self, response, status=200):
        self.assertEqual(response.status_code, status, response.text)
        result = response.json()
        EnvelopeSchema.model_validate(result)
        return result

    async def post(self, action, request, client=None):
        return self.decoded(await (client or self.client).post("/api/v1/workspace/" + action, json=request))

    async def model_run(self):
        dataset = await self.post("research-dataset", self.dataset_request)
        self.assertEqual(dataset["state"], "AVAILABLE")
        result = await self.post("research-model", {"dataset_id": dataset["data"]["dataset_id"],
            "model": "ridge", "min_train": 30, "test_size": 12, "embargo": 1, "seed": 17})
        self.assertEqual(result["state"], "AVAILABLE")
        return dataset, result

    async def test_actual_full_flow_ack_partial_fills_accounting_and_restart(self):
        with patch("gig.execution.alpaca.AlpacaBroker", side_effect=AssertionError("no broker")):
            dataset, run = await self.model_run()
            predicted = await self.post("research-predict", {"dataset_id": dataset["data"]["dataset_id"], "run_id": run["data"]["run_id"]})
            self.assertEqual(predicted["state"], "AVAILABLE")
            allocated = await self.post("research-allocate", allocation_request(run["data"]["run_id"], self.now))
            self.assertEqual(allocated["state"], "AVAILABLE")
            self.assertEqual(allocated["data"]["risk"]["state"], "AVAILABLE")
            request = {"allocation_id": allocated["data"]["allocation_id"], "now": self.now}
            prepared = await self.post("research-prepare", request)
            self.assertEqual(prepared, await self.post("research-prepare", request))
            clients = prepared["data"]["prepared_client_ids"]
            self.assertEqual(len(clients), 2)
            ack = await self.post("execution-engine", {"operation": "dispatch", "now": self.now})
            self.assertEqual(ack["data"]["positions"], {})
            for client, row in zip(clients, prepared["data"]["rows"], strict=True):
                filled = await self.post("execution-engine", {"operation": "fill", "now": self.now,
                    "client_id": client, "quantity": "1", "price": row["reference_price"], "fee": "0.1"})
                self.assertEqual(filled["state"], "AVAILABLE")
            await self.post("execution-engine", {"operation": "reconcile", "now": self.now})
            marks = {r["instrument_id"]: {"price": r["reference_price"], "as_of": self.now} for r in prepared["data"]["rows"]}
            account_request = {"operation": "reconcile", "initial_cash": "100000", "now": self.now,
                "marks": marks, "max_age_seconds": 60, "metadata": allocation_request("", "")["metadata"],
                "limits": allocation_request("", "")["limits"]}
            account = await self.post("portfolio-accounting", account_request)
            self.assertEqual(account["state"], "AVAILABLE")
            self.assertEqual(Decimal(account["data"]["metrics"]["nav"]), Decimal("99999.8"))
            async with self.reopen() as reopened:
                self.assertEqual(account, await self.post("portfolio-accounting", account_request, client=reopened))
                for engine, section in [("persisted", "models"), ("persisted", "signals"),
                                        ("accounting", "portfolio"), ("accounting", "risk"),
                                        ("allocation", "portfolio"), ("execution-engine", "execution")]:
                    response = await reopened.get("/api/v1/terminal/" + section, params={
                        "engine": engine, "mode": "local", "as_of": self.now, "max_age_seconds": 60})
                    self.assertEqual(self.decoded(response)["state"], "AVAILABLE")

    async def test_unknown_exposure_and_unsafe_asynchronous_limits_reject(self):
        _, run = await self.model_run()
        request = allocation_request(run["data"]["run_id"], self.now)
        request["limits"]["max_abs_net"] = 0
        self.assertEqual((await self.post("research-allocate", request))["state"], "ERROR")
        self.assertFalse((self.base / "execution-engine").exists())
        request = allocation_request(run["data"]["run_id"], self.now)
        request["metadata"]["US:AAA"]["factors"] = {}
        self.decoded(await self.client.post("/api/v1/workspace/research-allocate", json=request), status=422)

    def test_read_only_archive_creates_nothing_and_refuses_admission(self):
        with self.assertRaises(FileNotFoundError):
            PointInTimeStore(self.base / "missing/bars.sqlite", read_only=True)
        self.assertFalse((self.base / "missing").exists())
        store = PointInTimeStore(self.base / "bars.sqlite", read_only=True)
        self.assertTrue(store.snapshot(self.now, max_age_seconds=60)["rows"])
        with self.assertRaises(ValueError):
            store.ingest(None)
