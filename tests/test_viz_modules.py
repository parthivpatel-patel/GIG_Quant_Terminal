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


class VizInputSensitivityTests(unittest.TestCase):
    """Each module is a function of its inputs: changing an input changes exactly what it should."""

    def test_vol_surface_follows_quoted_prices(self):
        quotes = viz.authored_option_quotes()
        base = viz.build_vol_surface(quotes, spot=100.0, rate=0.03)
        bumped = {label: [dict(q) for q in rows] for label, rows in quotes.items()}
        target = next(q for q in bumped["current"] if q["strike"] == 110 and q["tenor_years"] == 0.5)
        target["price"] *= 1.2
        moved = viz.build_vol_surface(bumped, spot=100.0, rate=0.03)
        changed = [(a["strike"], a["tenor_years"]) for a, b in zip(base["snapshots"]["current"], moved["snapshots"]["current"], strict=True)
                   if a["implied_vol"] != b["implied_vol"]]
        self.assertEqual(changed, [(110, 0.5)])
        point = lambda d: next(p for p in d["snapshots"]["current"] if p["strike"] == 110 and p["tenor_years"] == 0.5)  # noqa: E731
        self.assertGreater(point(moved)["implied_vol"], point(base)["implied_vol"])
        self.assertEqual(base["snapshots"]["prior"], moved["snapshots"]["prior"])
        with self.assertRaises(ValueError):
            viz.build_vol_surface({"current": quotes["current"][:-1]}, spot=100.0, rate=0.03)

    def test_order_book_follows_depth(self):
        frames = viz.authored_depth_frames()
        base = viz.build_order_book(frames, instrument="X")
        heavier = [dict(f, bids=[dict(b, quantity=b["quantity"] * 3) for b in f["bids"]]) for f in frames]
        moved = viz.build_order_book(heavier, instrument="X")
        for a, b in zip(base["frames"], moved["frames"], strict=True):
            self.assertGreater(b["displayed_imbalance"], a["displayed_imbalance"])
        crossed = [dict(frames[0], asks=[dict(frames[0]["asks"][0], price=frames[0]["bids"][0]["price"])])]
        with self.assertRaises(ValueError):
            viz.build_order_book(crossed, instrument="X")

    def test_risk_terrain_scales_with_positions(self):
        from gig.options import OptionPosition
        grid = {"spot": 100, "rate": 0.03, "spot_shocks": [-0.2, 0.0, 0.2], "vol_shocks": [0.0, 0.05]}
        base = viz.build_risk_terrain(viz.authored_option_positions(), **grid)
        doubled = viz.build_risk_terrain([OptionPosition(p.instrument, p.underlying, p.strike, p.ttm, p.volatility,
                                                         p.quantity * 2, call=p.call, source=p.source)
                                          for p in viz.authored_option_positions()], **grid)
        for a, b in zip(base["horizons"]["today"], doubled["horizons"]["today"], strict=True):
            self.assertAlmostEqual(b["model_pnl"], 2 * a["model_pnl"], places=6)
        self.assertAlmostEqual(doubled["risk_limit"]["loss_limit"], 2 * base["risk_limit"]["loss_limit"])

    def test_correlation_network_follows_returns(self):
        returns, _, symbols = viz._fixture()
        base = viz.build_correlation_network(returns, symbols)
        flipped = returns.copy()
        flipped[:, 0] = -flipped[:, 0]
        moved = viz.build_correlation_network(flipped, symbols)
        for a, b in zip(base["windows"]["latest_60"]["edges"], moved["windows"]["latest_60"]["edges"], strict=True):
            expected = -a["rho"] if symbols[0] in (a["a"], a["b"]) else a["rho"]
            self.assertAlmostEqual(b["rho"], expected, places=12)
        with self.assertRaises(ValueError):
            viz.build_correlation_network(returns[:, :3], symbols[:3])

    def test_stat_arb_follows_pair(self):
        _, prices, symbols = viz._fixture()
        first = viz.build_stat_arb(prices[:, 0], prices[:, 1], (symbols[0], symbols[1]))
        second = viz.build_stat_arb(prices[:, 0], prices[:, 2], (symbols[0], symbols[2]))
        self.assertEqual(second["pair"], [symbols[0], symbols[2]])
        self.assertNotEqual([p["beta"] for p in first["points"]], [p["beta"] for p in second["points"]])
        truncated = viz.build_stat_arb(prices[:120, 0], prices[:120, 1], (symbols[0], symbols[1]))
        self.assertEqual([p["beta"] for p in truncated["points"]], [p["beta"] for p in first["points"][:120]])

    def test_model_evaluation_follows_returns(self):
        returns, prices, _ = viz._fixture()
        folds = ((60, 90),)
        base = viz.build_model_evaluation(returns, prices, folds)
        future = returns.copy()
        future[90:] *= 3
        self.assertEqual(viz.build_model_evaluation(future, prices, folds)["rows"], base["rows"])
        noisy = returns.copy()
        noisy[60:90, 0] *= 3
        moved = viz.build_model_evaluation(noisy, prices, folds)
        value = lambda d, fam: next(r for r in d["rows"] if r["family"] == fam)  # noqa: E731
        self.assertNotEqual(value(base, "variance_qlike")["value"], value(moved, "variance_qlike")["value"])
        self.assertGreater(value(moved, "forecast_mse")["baseline"], value(base, "forecast_mse")["baseline"])

    def test_local_mode_still_unavailable(self):
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
