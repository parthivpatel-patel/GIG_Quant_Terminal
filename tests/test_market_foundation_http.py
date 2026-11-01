"""The new market-foundation calculations cross the actual typed HTTP boundary."""
from __future__ import annotations

import asyncio
import unittest
from unittest.mock import patch

import httpx

from gig.service.app import create_app
from gig.service.contracts import EnvelopeSchema


def split_request():
    return {
        "source": "authored-synthetic-action-fixture", "classification": "SYNTHETIC",
        "currency": "USD", "as_of": "2026-09-29T20:00:00Z", "mode": "SPLIT_ADJUSTED",
        "rows": [{"instrument_id": "SYN001", "timestamp": "2026-09-28T20:00:00Z",
                  "available_at": "2026-09-28T20:00:00Z", "open": 100, "high": 101,
                  "low": 99, "close": 100, "volume": 9007199254740993},
                 {"instrument_id": "SYN001", "timestamp": "2026-09-29T20:00:00Z",
                  "available_at": "2026-09-29T20:00:00Z", "open": 50, "high": 51,
                  "low": 49, "close": 50, "volume": 2000000}],
        "actions": [{"action_id": "split-1", "instrument_id": "SYN001",
                     "source": "authored-synthetic-action-fixture", "kind": "SPLIT",
                     "effective_at": "2026-09-29T13:30:00Z", "available_at": "2026-09-27T21:00:00Z",
                     "ratio_new": 2, "ratio_old": 1}],
    }


class MarketFoundationHTTPTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        async def clock():
            while True:
                await asyncio.sleep(.01)
        self.clock = asyncio.create_task(clock())
        self.client = httpx.AsyncClient(transport=httpx.ASGITransport(app=create_app()), base_url="http://test")

    async def asyncTearDown(self):
        await self.client.aclose()
        self.clock.cancel()

    async def test_calendar_scheduled_session_and_freshness_are_estimates(self):
        response = await self.client.post("/api/v1/analytics/calendar", json={
            "calendar": "XNYS", "start": "2026-11-23", "end": "2026-11-30",
            "as_of": "2026-11-27T18:00:00Z", "last_bar_close": "2026-11-25T21:00:00Z"})
        self.assertEqual(response.status_code, 200, response.text)
        result = response.json()
        EnvelopeSchema.model_validate(result)
        self.assertEqual(result["state"], "AVAILABLE")
        self.assertEqual(result["provenance"]["classification"], "MODEL_ESTIMATE")
        early = next(row for row in result["data"]["rows"] if row["session"] == "2026-11-27")
        self.assertEqual(early["close"], "2026-11-27T18:00:00+00:00")
        self.assertEqual(result["data"]["freshness"]["state"], "STALE")

    async def test_split_math_and_identity_survive_final_response_bytes(self):
        response = await self.client.post("/api/v1/analytics/corporate-actions", json=split_request())
        self.assertEqual(response.status_code, 200, response.text)
        result = response.json()
        EnvelopeSchema.model_validate(result)
        self.assertEqual(result["state"], "AVAILABLE")
        rows = result["data"]["rows"]
        self.assertEqual(rows[0]["close"], 100)
        self.assertEqual(rows[0]["adjusted_close"], 50)
        self.assertEqual(rows[0]["adjusted_close_exact"], {"numerator": "50", "denominator": "1"})
        self.assertEqual(rows[0]["adjusted_volume_exact"], {"numerator": "18014398509481986", "denominator": "1"})
        # The admitted bar contract uses exact decimal strings for volume.
        self.assertEqual(rows[0]["volume"], "9007199254740993")
        self.assertEqual(result["provenance"]["classification"], "MODEL_ESTIMATE")

    async def test_no_visible_data_is_unavailable_and_bad_requests_are_errors(self):
        request = split_request()
        request["as_of"] = "2026-09-01T00:00:00Z"
        response = await self.client.post("/api/v1/analytics/corporate-actions", json=request)
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["state"], "UNAVAILABLE")
        self.assertIsNone(response.json()["data"])
        response = await self.client.post("/api/v1/analytics/calendar", json={
            "calendar": "UNKNOWN-EXCHANGE", "start": "2026-01-01", "end": "2026-01-03"})
        self.assertEqual(response.status_code, 422)
        self.assertEqual(response.json()["state"], "ERROR")
        with patch("gig.data.calendars.calendar_analytics", side_effect=ImportError("FAKE_SECRET_DETAIL")):
            response = await self.client.post("/api/v1/analytics/calendar", json={})
        self.assertEqual(response.status_code, 503)
        self.assertNotIn("FAKE_SECRET_DETAIL", response.text)
        self.assertEqual(response.json()["error"]["code"], "DEPENDENCY_UNAVAILABLE")


if __name__ == "__main__":
    unittest.main()
