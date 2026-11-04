"""Six 3D analytical modules compute real analytics on labeled synthetic fixtures."""
from __future__ import annotations

import asyncio
import json
import math
import unittest

import httpx

from gig.service import viz
from gig.service.app import create_app
from gig.service.contracts import EnvelopeSchema


class VizModuleTests(unittest.TestCase):
    def test_vol_surface_inverts_authored_prices(self):
        from gig.options import black_scholes
        data = viz.vol_surface()
        point = next(p for p in data["snapshots"]["current"] if p["strike"] == 110 and p["tenor_years"] == 0.5)
        repriced = black_scholes(100, 110, 0.03, 0, point["implied_vol"], 0.5, True)["price"]
        self.assertAlmostEqual(repriced, point["authored_price"], places=8)
        self.assertEqual(data["observed_option_quotes"], 0)
        self.assertEqual(len(data["snapshots"]["prior"]), len(data["snapshots"]["current"]))

    def test_order_book_never_crossed_and_labeled(self):
        data = viz.order_book()
        for frame in data["frames"]:
            self.assertLess(frame["bids"][0]["price"], frame["asks"][0]["price"])
        self.assertEqual(data["layers"]["displayed_depth"], "AUTHORED_SYNTHETIC_DEPTH")
        self.assertEqual(data["layers"]["inferred_liquidity"], "NOT_COMPUTED")

    def test_risk_terrain_zero_shock_is_zero_and_long_only(self):
        data = viz.risk_terrain()
        center = next(p for p in data["horizons"]["today"] if p["spot_shock"] == 0 and p["volatility_shock"] == 0.03)
        base = next(p for p in data["horizons"]["today"] if p["spot_shock"] == 0 and abs(p["volatility_shock"]) < 1e-12)
        self.assertAlmostEqual(base["model_pnl"], 0.0, places=6)
        self.assertGreater(center["model_pnl"], 0)
        self.assertAlmostEqual(sum(p["share"] for p in data["concentration"]), 1.0)
        self.assertLess(data["risk_limit"]["loss_limit"], 0)

    def test_correlation_network_symmetric_bounded(self):
        data = viz.correlation_network()
        window = data["windows"]["latest_60"]
        self.assertEqual(len(window["edges"]), 12 * 11 // 2)
        self.assertTrue(all(-1 <= e["rho"] <= 1 for e in window["edges"]))
        self.assertAlmostEqual(window["standard_error_at_zero"], 1 / math.sqrt(57))
        self.assertIn("does not imply causation", data["interpretation"])

    def test_stat_arb_is_causal(self):
        data = viz.stat_arb()
        self.assertIsNone(data["points"][0]["regime"])
        self.assertEqual(data["economic_validation"], "UNAVAILABLE")
        self.assertIn("no smoothing", data["method"])

    def test_model_evaluation_families_are_separate(self):
        data = viz.model_evaluation()
        families = {r["family"] for r in data["rows"]}
        self.assertTrue({"variance_qlike", "forecast_mse", "regime_loglik_per_obs"} <= families)
        for family in families:
            models = {r["model"] for r in data["rows"] if r["family"] == family}
            self.assertEqual(len(models), 1, family)
        self.assertIn("BiLSTM", data["not_computed"])
        self.assertIn("not trading profitability", data["interpretation"])

    def test_stored_outputs_match_recomputation(self):
        def close(a, b, path):
            if isinstance(a, dict):
                self.assertEqual(sorted(a), sorted(b), path)
                for key in a:
                    close(a[key], b[key], f"{path}.{key}")
            elif isinstance(a, list):
                self.assertEqual(len(a), len(b), path)
                for i, (x, y) in enumerate(zip(a, b, strict=True)):
                    close(x, y, f"{path}[{i}]")
            elif isinstance(a, float) and isinstance(b, (int, float)):
                self.assertTrue(math.isclose(a, b, rel_tol=1e-6, abs_tol=1e-9), f"{path}: {a} != {b}")
            else:
                self.assertEqual(a, b, path)
        stored = viz.precomputed()
        self.assertEqual(sorted(stored), sorted(viz.MODULES))
        for name in viz.MODULES:
            close(stored[name], json.loads(json.dumps(viz.BUILDERS[name]())), name)

    def test_local_mode_is_unavailable(self):
        for name in viz.MODULES:
            self.assertEqual(json.loads(viz.module(name, "local").wire)["state"], "UNAVAILABLE")


class VizHTTPTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        async def clock():
            while True:
                await asyncio.sleep(.01)
        self.clock = asyncio.create_task(clock())
        self.client = httpx.AsyncClient(transport=httpx.ASGITransport(app=create_app()), base_url="http://test")

    async def asyncTearDown(self):
        await self.client.aclose()
        self.clock.cancel()

    async def test_routes(self):
        response = await self.client.get("/api/v1/viz/order-book")
        body = response.json()
        EnvelopeSchema.model_validate(body)
        self.assertEqual(body["provenance"]["classification"], "SYNTHETIC")
        self.assertEqual((await self.client.get("/api/v1/viz/unknown")).status_code, 404)
        self.assertEqual((await self.client.get("/api/v1/viz/order-book?mode=live")).status_code, 422)
