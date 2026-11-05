"""Chart bars: labelled synthetic fixtures and admitted local bars with point-in-time truth."""
from __future__ import annotations

import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

import httpx
import numpy as np
import pandas as pd

from gig.research import indicators as ind
from gig.service.app import create_app
from gig.service.charts import resample, synthetic_bars
from gig.service.contracts import EnvelopeSchema
from test_terminal_workspace import NOW, bar_request


class ChartsHTTPTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.directory = Path(self.temporary.name) / "state"
        self.client = httpx.AsyncClient(transport=httpx.ASGITransport(app=create_app(state_dir=self.directory)),
                                        base_url="http://test")

    async def asyncTearDown(self):
        await self.client.aclose()
        self.temporary.cleanup()

    async def get(self, **params):
        response = await self.client.get("/api/v1/charts/bars", params=params)
        payload = response.json()
        if response.status_code == 200:
            EnvelopeSchema.model_validate(payload)
        return response.status_code, payload

    async def test_synthetic_bars_are_labelled_and_indicators_recompute(self):
        status, payload = await self.get(mode="synthetic", symbol="SYN004", timeframe="1D")
        self.assertEqual(status, 200)
        self.assertEqual(payload["state"], "AVAILABLE")
        self.assertEqual(payload["provenance"]["classification"], "SYNTHETIC")
        data = payload["data"]
        bars = data["bars"]
        self.assertEqual(len(bars["t"]), 504)
        self.assertTrue(all(t.endswith("+00:00") for t in bars["t"]))
        close = np.array(bars["close"])
        expected = ind.rsi(close, 14)
        got = np.array([np.nan if v is None else v for v in data["indicators"]["rsi14"]])
        np.testing.assert_allclose(got, expected, rtol=1e-6, equal_nan=True)
        self.assertIsNone(data["indicators"]["sma50"][48])
        self.assertIsNotNone(data["indicators"]["sma50"][49])
        self.assertEqual([row["symbol"] for row in data["watchlist"]], [f"SYN{i:03d}" for i in range(1, 13)])
        for o, h, low, c in zip(bars["open"], bars["high"], bars["low"], bars["close"], strict=True):
            self.assertLessEqual(low, min(o, c))
            self.assertGreaterEqual(h, max(o, c))

    async def test_synthetic_is_deterministic_and_rejects_unknown_symbols(self):
        first = (await self.get(mode="synthetic", symbol="SYN002", timeframe="1W"))[1]
        second = (await self.get(mode="synthetic", symbol="SYN002", timeframe="1W"))[1]
        self.assertEqual(first["content_sha256"], second["content_sha256"])
        status, unknown = await self.get(mode="synthetic", symbol="AAPL")
        self.assertEqual(unknown["state"], "UNAVAILABLE")
        self.assertIsNone(unknown["data"])
        self.assertEqual((await self.get(mode="synthetic", timeframe="5m"))[0], 422)

    async def test_local_without_admitted_bars_is_unavailable_and_creates_no_state(self):
        status, payload = await self.get(mode="local", max_age_seconds=86400)
        self.assertEqual(status, 200)
        self.assertEqual(payload["state"], "UNAVAILABLE")
        self.assertEqual(list(self.directory.rglob("bars.sqlite")), [])

    async def test_local_admitted_bars_point_in_time_and_freshness(self):
        request = bar_request(60)
        response = await self.client.post("/api/v1/workspace/bars", json=request)
        self.assertEqual(response.status_code, 200, response.text)
        status, payload = await self.get(mode="local", symbol="ID:AAA", as_of=NOW.isoformat(), max_age_seconds=86400)
        self.assertEqual(payload["state"], "AVAILABLE")
        data = payload["data"]
        expected = [row["close"] for row in request["rows"] if row["instrument_id"] == "ID:AAA"]
        self.assertEqual(data["bars"]["close"], expected)
        self.assertEqual(data["admitted"]["source"], "caller-feed")
        self.assertEqual(payload["provenance"]["classification"], "SYNTHETIC")
        self.assertEqual(payload["provenance"]["evidence_state"], "CALLER_SUPPLIED_NOT_VENDOR_VERIFIED")
        self.assertEqual({row["symbol"] for row in data["watchlist"]}, {"ID:AAA", "ID:BBB"})
        earlier = NOW - pd.Timedelta(days=30)
        past = (await self.get(mode="local", symbol="ID:AAA", as_of=earlier.isoformat(), max_age_seconds=86400))[1]
        self.assertEqual(len(past["data"]["bars"]["t"]), 30)
        late = (await self.get(mode="local", symbol="ID:AAA", as_of=(NOW + pd.Timedelta(days=3)).isoformat(),
                               max_age_seconds=86400))[1]
        self.assertEqual(late["state"], "STALE")
        self.assertEqual(late["provenance"]["as_of"], payload["provenance"]["as_of"])
        self.assertEqual((await self.get(mode="local", symbol="ID:AAA", as_of=NOW.isoformat()))[1]["state"], "UNAVAILABLE")
        self.assertEqual((await self.get(mode="local", symbol="ID:ZZZ", as_of=NOW.isoformat(), max_age_seconds=86400))[1]["state"],
                         "UNAVAILABLE")

    async def test_local_revision_is_visible_only_after_its_knowledge_time(self):
        request = bar_request(30)
        await self.client.post("/api/v1/workspace/bars", json=request)
        corrected = deepcopy(request)
        corrected["rows"] = [deepcopy(request["rows"][-2])]
        corrected["rows"][0].update(close=150, open=150, low=149, high=151,
                                    available_at=(NOW + pd.Timedelta(hours=1)).isoformat())
        await self.client.post("/api/v1/workspace/bars", json=corrected)
        before = (await self.get(mode="local", symbol="ID:AAA", as_of=NOW.isoformat(), max_age_seconds=86400))[1]
        after = (await self.get(mode="local", symbol="ID:AAA", as_of=(NOW + pd.Timedelta(hours=1)).isoformat(),
                                max_age_seconds=86400))[1]
        self.assertNotEqual(before["data"]["bars"]["close"][-1], 150)
        self.assertEqual(after["data"]["bars"]["close"][-1], 150)


class ResampleTests(unittest.TestCase):
    def test_weekly_bars_aggregate_without_lookahead(self):
        frame = synthetic_bars()
        daily = frame[frame["symbol"] == "SYN001"].sort_values("timestamp")
        weekly = resample(daily, "1W")
        week = daily[(daily["timestamp"] > weekly["timestamp"].iloc[-2]) & (daily["timestamp"] <= weekly["timestamp"].iloc[-1])]
        last = weekly.iloc[-1]
        self.assertEqual(last["open"], week["open"].iloc[0])
        self.assertEqual(last["close"], week["close"].iloc[-1])
        self.assertEqual(last["high"], week["high"].max())
        self.assertEqual(last["low"], week["low"].min())
        self.assertEqual(last["volume"], week["volume"].sum())
        self.assertEqual(last["timestamp"], week["timestamp"].max())
        self.assertTrue((weekly["timestamp"].diff().dropna() > pd.Timedelta(0)).all())


if __name__ == "__main__":
    unittest.main()
