"""Observed curve math and persisted HTTP charts without provider connections."""

from __future__ import annotations

import asyncio
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

import httpx

from gig.data.ingestion import ProviderWorkspace, parse_treasury
from gig.research.rates import curve_analytics, quoted_yield
from gig.service.app import create_app
from gig.service.contracts import EnvelopeSchema
from test_provider_ingestion import treasury_xml

AT = "2026-01-02T16:00:00.123456789Z"


def row(tenor, value, event="2025-01-02"):
    return {"event_date": event, "tenor": tenor, "yield_percent": value,
            "available_at": AT, "source_sha256": "a" * 64, "classification": "SYNTHETIC"}


class CurveNumericalTests(unittest.TestCase):
    def test_exact_same_day_spreads_and_zero_are_available(self):
        rows = [row("BC_2YEAR", "4.10"), row("BC_10YEAR", "3.85"),
                row("BC_3MONTH", "3.85"), row("BC_5YEAR", "-0.10"),
                row("BC_30YEAR", "0.10")]
        original = deepcopy(rows)
        result = curve_analytics(rows)
        self.assertEqual(result["metrics"]["spread_10y_2y"]["value"], "-25.00")
        self.assertEqual(result["metrics"]["spread_10y_3m"]["value"], "0.00")
        self.assertEqual(result["metrics"]["spread_10y_3m"]["state"], "AVAILABLE")
        self.assertEqual(result["metrics"]["spread_30y_5y"]["value"], "20.00")
        self.assertFalse(result["chart"]["interpolated"])
        self.assertEqual(len(result["chart"]["points"]), len(rows))
        self.assertEqual(rows, original)

    def test_previous_date_never_supplies_a_missing_latest_tenor(self):
        result = curve_analytics([row("BC_10YEAR", "3.85"), row("BC_2YEAR", "4.10", "2025-01-01")])
        spread = result["metrics"]["spread_10y_2y"]
        self.assertIsNone(spread["value"])
        self.assertEqual(spread["state"], "UNAVAILABLE")
        self.assertEqual([p["observation_day_offset"] for p in result["chart"]["points"]], [-1, 0])

    def test_tenor_sort_uses_maturity_and_preserves_quote_provenance(self):
        result = curve_analytics([row("BC_10YEAR", "4"), row("BC_1_5MONTH", "4.1"), row("BC_2YEAR", "3.5")])
        points = result["chart"]["points"]
        self.assertEqual([p["tenor"] for p in points], ["BC_1_5MONTH", "BC_2YEAR", "BC_10YEAR"])
        self.assertEqual(points[0]["maturity_years"], .125)
        self.assertTrue(all(p["available_at"] == AT for p in points))
        self.assertTrue(all(p["classification"] == "SYNTHETIC" for p in points))

    def test_nonfinite_precision_and_oversized_yields_reject(self):
        for value in [True, 4, "NaN", "Infinity", "-101", "101", "1e-1000000", "1" * 65, ""]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                quoted_yield(value)
        self.assertEqual(str(quoted_yield("0.000000000001")), "1E-12")

    def test_xml_precision_rejects_before_provider_persistence(self):
        for value in ("1e-1000000", "0.1234567890123", "NaN"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                parse_treasury(treasury_xml(rate=value), AT)

    def test_ambiguous_dates_identity_and_tenors_fail(self):
        cases = [[], [row("BC_10YEAR", "4")] * 2, [row("SPOT_10YEAR", "4")],
                 [row("BC_10YEAR", "4", "2025-01-02T00:00:00Z")]]
        for rows in cases:
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                curve_analytics(rows)


class CurveHTTPTests(unittest.IsolatedAsyncioTestCase):
    async def test_persisted_fixture_curve_reopens_with_source_and_freshness(self):
        async def clock():
            while True:
                await asyncio.sleep(.01)

        ticker = asyncio.create_task(clock())
        try:
            with tempfile.TemporaryDirectory() as directory:
                base = Path(directory) / "state"
                ProviderWorkspace(base).ingest_treasury_bytes(treasury_xml(), year=2025, observed_at=AT)
                app = create_app(state_dir=base)
                async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
                    async def read(stamp):
                        response = await client.get("/api/v1/terminal/markets", params={
                            "mode": "local", "engine": "provider", "as_of": stamp,
                            "max_age_seconds": 60})
                        self.assertEqual(response.status_code, 200, response.text)
                        result = response.json()
                        EnvelopeSchema.model_validate(result)
                        return result

                    result = await read(AT)
                    self.assertEqual(result["state"], "AVAILABLE")
                    self.assertEqual(result["provenance"]["classification"], "SYNTHETIC")
                    self.assertEqual(len(result["data"]["chart"]["points"]), 2)
                    self.assertEqual(result["data"]["historical_publication_time"], "UNKNOWN")
                    self.assertEqual((await read("2026-01-02T16:02:00Z"))["state"], "STALE")
                    self.assertEqual((await read("2026-01-02T15:59:59Z"))["state"], "UNAVAILABLE")
        finally:
            ticker.cancel()


if __name__ == "__main__":
    unittest.main()
