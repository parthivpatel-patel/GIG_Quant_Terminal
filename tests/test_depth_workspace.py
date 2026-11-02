"""Depth snapshots preserve actual levels, sequence and knowledge-time truth."""
from __future__ import annotations

import asyncio
import tempfile
import unittest
from pathlib import Path

import httpx

from gig.service.app import create_app
from gig.service.contracts import EnvelopeSchema
from gig.service.depth_workspace import DepthWorkspace

AT = "2026-11-02T15:49:00.123456789Z"


def depth_request(**changes):
    return {"instrument_id": "SYN101", "source": "authored-depth-fixture", "classification": "SYNTHETIC",
            "currency": "USD", "feed": "CALLER_SUPPLIED", "sequence": 1,
            "event_at": AT, "available_at": AT,
            "bids": [{"price": "100.00", "quantity": "9007199254740993"}, {"price": "99.90", "quantity": "10"}],
            "asks": [{"price": "100.10", "quantity": "20"}, {"price": "100.20", "quantity": "30"}], **changes}


class DepthTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.base = Path(self.temporary.name) / "depth-state"
        self.workspace = DepthWorkspace(self.base)

    def tearDown(self):
        self.temporary.cleanup()

    def view(self, at=AT, maximum=60):
        return self.workspace.section("markets", at, maximum, instrument_id="SYN101").to_dict()

    def test_immutable_exact_levels_and_reopen(self):
        request = depth_request()
        first = self.workspace.ingest(request).to_dict()
        EnvelopeSchema.model_validate(first)
        self.assertEqual(first, self.workspace.ingest(request).to_dict())
        self.workspace = DepthWorkspace(self.base)
        saved = self.view()
        self.assertEqual(saved["data"]["depth"]["bids"][0]["quantity"], "9007199254740993")
        self.assertEqual(saved["data"]["metrics"]["spread"]["value"], "0.10")
        self.assertEqual(saved["provenance"]["as_of"], "2026-11-02T15:49:00.123456789+00:00")
        saved["data"]["depth"]["bids"][0]["price"] = "999"
        self.assertEqual(self.view()["data"]["depth"]["bids"][0]["price"], "100.00")

    def test_missing_reads_and_future_snapshots_do_not_invent_book(self):
        self.assertEqual(self.view()["state"], "UNAVAILABLE")
        self.assertFalse(self.base.exists())
        self.workspace.ingest(depth_request())
        self.assertEqual(self.view("2026-11-02T15:48:59Z")["state"], "UNAVAILABLE")
        self.assertEqual(self.view("2026-11-02T15:51:00Z")["state"], "STALE")
        self.assertEqual(self.view(maximum=None)["state"], "UNKNOWN")

    def test_out_of_order_sequences_do_not_rewind_current_book(self):
        self.workspace.ingest(depth_request(sequence=3))
        older = depth_request(sequence=2, available_at="2026-11-02T15:49:01Z")
        older["bids"][0]["quantity"] = "1"
        self.workspace.ingest(older)
        self.assertEqual(self.view("2026-11-02T15:49:01Z")["data"]["depth"]["sequence"], 3)

    def test_ambiguous_sources_and_zero_queue_stay_unknown(self):
        request = depth_request()
        for side in ("bids", "asks"):
            for row in request[side]:
                row["quantity"] = "0"
        result = self.workspace.ingest(request).to_dict()
        self.assertIsNone(result["data"]["metrics"]["displayed_depth_imbalance"]["value"])
        self.assertEqual(result["data"]["metrics"]["displayed_depth_imbalance"]["state"], "UNKNOWN")
        self.workspace.ingest(depth_request(source="another-declared-feed"))
        self.assertEqual(self.view()["state"], "UNKNOWN")

    def test_bad_prices_levels_times_and_conflicting_identity_reject(self):
        for changes in [{"bids": []}, {"sequence": True}, {"available_at": "2026-11-02T00:00:00Z"},
                        {"event_at": "2026-11-02"}, {"asks": [{"price": "99", "quantity": "1"}]},
                        {"bids": [{"price": "NaN", "quantity": "1"}]}]:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.workspace.ingest(depth_request(**changes))
        self.assertFalse(self.base.exists())
        request = depth_request()
        self.workspace.ingest(request)
        request["bids"][0]["quantity"] = "12"
        with self.assertRaises(ValueError):
            self.workspace.ingest(request)


class DepthHTTPTests(unittest.IsolatedAsyncioTestCase):
    async def test_contract_through_http_and_restart(self):
        async def clock():
            while True:
                await asyncio.sleep(.01)
        ticker = asyncio.create_task(clock())
        try:
            with tempfile.TemporaryDirectory() as directory:
                base = Path(directory) / "state"
                async with httpx.AsyncClient(transport=httpx.ASGITransport(app=create_app(state_dir=base)), base_url="http://test") as client:
                    response = await client.post("/api/v1/workspace/depth", json=depth_request())
                    self.assertEqual(response.status_code, 200, response.text)
                async with httpx.AsyncClient(transport=httpx.ASGITransport(app=create_app(state_dir=base)), base_url="http://test") as client:
                    response = await client.get("/api/v1/terminal/markets", params={"engine": "depth", "instrument_id": "SYN101", "as_of": AT, "max_age_seconds": 60})
                    result = response.json()
                    EnvelopeSchema.model_validate(result)
                    self.assertEqual(result["state"], "AVAILABLE")
                    self.assertEqual(result["provenance"]["classification"], "SYNTHETIC")
                    self.assertEqual(len(result["data"]["rows"]), 4)
        finally:
            ticker.cancel()
