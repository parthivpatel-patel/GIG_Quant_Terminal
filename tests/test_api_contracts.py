"""Exercise the actual HTTP boundary, including its final response bytes."""
from __future__ import annotations

import asyncio
import hashlib
import json
import math
import unittest
from decimal import Decimal
from unittest.mock import patch

import httpx
import numpy as np
import pandas as pd

from gig.service.app import create_app
from gig.service.contracts import EnvelopeSchema
from gig.service.serialization import dumps
from gig.service.terminal import SECTIONS


class HTTPBoundaryTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        # This executor does not wake its selector reliably from worker threads.
        # A periodic test clock drives the real async/threadpool boundary. This
        # is test infrastructure only; production handlers remain nonblocking.
        async def clock():
            while True:
                await asyncio.sleep(.01)
        self.clock = asyncio.create_task(clock())
        self.app = create_app()
        self.client = httpx.AsyncClient(transport=httpx.ASGITransport(app=self.app),
                                       base_url="http://test")

    async def asyncTearDown(self):
        await self.client.aclose()
        self.clock.cancel()

    def envelope(self, response):
        result = response.json()
        EnvelopeSchema.model_validate(result)
        digest = result.pop("content_sha256")
        self.assertEqual(digest, hashlib.sha256(dumps(result).encode()).hexdigest())
        result["content_sha256"] = digest
        return result

    async def test_all_twelve_views_in_both_modes(self):
        for section in SECTIONS:
            for mode in ("local", "synthetic"):
                with self.subTest(section=section, mode=mode):
                    response = await self.client.get(f"/api/v1/terminal/{section}", params={"mode": mode})
                    self.assertEqual(response.status_code, 200, response.text)
                    result = self.envelope(response)
                    if mode == "synthetic" and section != "operations":
                        self.assertEqual(result["provenance"]["classification"], "SYNTHETIC")
                        self.assertEqual(result["state"], "AVAILABLE")
                    if result["state"] in ("UNAVAILABLE", "UNKNOWN", "ERROR"):
                        self.assertIsNone(result["data"])
                    self.assertEqual(result["provenance"]["readiness"], "RESEARCH_ONLY")

    async def test_default_is_local_and_broker_is_never_constructed(self):
        with patch("gig.execution.alpaca.AlpacaBroker", side_effect=AssertionError("broker construction")):
            response = await self.client.get("/api/v1/terminal/execution")
            self.assertEqual(self.envelope(response)["state"], "UNAVAILABLE")
            response = await self.client.get("/api/paper")
            self.assertFalse(response.json()["available"])
            self.assertEqual(response.json()["error"]["code"], "BROKER_DISABLED")

    async def test_strict_serialization_precedes_framework_coercion(self):
        @self.app.get("/api/probe")
        def probe():
            return {"id": np.int64(2**63 - 1), "decimal": Decimal("0.100000000000000000001"),
                    "when": pd.Timestamp("2026-01-01T01:00:00.123456789+01:00"),
                    "zero": -0.0, "tiny": float.fromhex("0x0.0000000000001p-1022"), "huge": 1e308}
        response = await self.client.get("/api/probe")
        self.assertEqual(response.status_code, 200)
        result = response.json()
        self.assertEqual(result["id"], {"$integer": "9223372036854775807"})
        self.assertEqual(result["decimal"], {"$decimal": "0.100000000000000000001"})
        self.assertEqual(result["when"], "2026-01-01T00:00:00.123456789+00:00")
        self.assertEqual(math.copysign(1, result["zero"]), -1)
        self.assertEqual(result["tiny"], 5e-324)
        self.assertEqual(result["huge"], 1e308)

    async def test_nonfinite_success_result_becomes_sanitized_error(self):
        @self.app.get("/api/probe")
        def probe():
            return {"value": float("inf")}
        response = await self.client.get("/api/probe")
        self.assertEqual(response.status_code, 500)
        result = self.envelope(response)
        self.assertEqual(result["state"], "ERROR")
        self.assertEqual(result["error"]["code"], "SERIALIZATION_ERROR")
        self.assertNotIn("Infinity", response.text)

    async def test_unsupported_object_never_leaks_its_repr(self):
        class Sensitive:
            def __repr__(self):
                raise AssertionError("repr must not be called")
        @self.app.get("/api/probe")
        def probe():
            return {"value": Sensitive()}
        response = await self.client.get("/api/probe")
        self.assertEqual(response.status_code, 500)
        self.assertEqual(self.envelope(response)["state"], "ERROR")

    async def test_exception_sanitization_including_exception_groups(self):
        @self.app.get("/api/probe")
        def probe():
            raise RuntimeError("FAKE-SECRET-PROVIDER-TOKEN")
        response = await self.client.get("/api/probe")
        self.assertEqual(response.status_code, 500)
        self.assertNotIn("FAKE-SECRET", response.text)
        self.assertEqual(self.envelope(response)["error"]["code"], "INTERNAL_ERROR")

    async def test_duplicate_keys_and_nonfinite_are_rejected_before_pydantic(self):
        for raw in (b'{"spot":100,"spot":200,"strike":100}',
                    b'{"spot":NaN,"strike":100}', b'{"spot":Infinity,"strike":100}',
                    b'{"spot":-Infinity,"strike":100}', b'{"spot":1e400,"strike":100}',
                    b'{"spot":1e-400,"strike":100}'):
            with self.subTest(raw=raw):
                response = await self.client.post("/api/v1/analytics/options", content=raw)
                self.assertEqual(response.status_code, 400)
                self.assertEqual(self.envelope(response)["state"], "ERROR")

    async def test_malformed_json_utf8_and_control_characters(self):
        for raw in (b'{"spot":', b'\xff', b'{"x":"bad\x01"}', b'[', b'null'):
            with self.subTest(raw=raw):
                response = await self.client.post("/api/v1/analytics/options", content=raw)
                self.assertIn(response.status_code, (400, 422))
                self.assertEqual(self.envelope(response)["state"], "ERROR")

    async def test_resource_bounds_reject_before_work(self):
        cases = [(b'{"x":"' + b"a" * 70_000 + b'"}', 400),
                 (b'[' * 40 + b'0' + b']' * 40, 400),
                 (b" " * 1_048_577, 413),
                 (json.dumps({"x": [0] * 10_001}).encode(), 400)]
        for raw, expected in cases:
            with self.subTest(length=len(raw)):
                response = await self.client.post("/api/v1/analytics/options", content=raw)
                self.assertEqual(response.status_code, expected)
                self.assertIsNone(self.envelope(response)["data"])

    async def test_published_option_reference_through_http(self):
        response = await self.client.post("/api/v1/analytics/options", json={
            "spot": 100, "strike": 100, "rate": .05, "div": 0, "vol": .2, "ttm": 1, "call": True})
        self.assertEqual(response.status_code, 200, response.text)
        result = self.envelope(response)
        self.assertAlmostEqual(result["data"]["price"], 10.450583572185565, places=10)
        self.assertEqual(result["provenance"]["classification"], "MODEL_ESTIMATE")

    async def test_kalman_hmm_garch_and_microstructure_connected(self):
        examples = {
            "kalman": {"observations": [[1], [2], [1.5]], "transition": [[1]], "observation": [[1]],
                       "process_covariance": [[.1]], "observation_covariance": [[1]],
                       "initial_mean": [0], "initial_covariance": [[1]], "smooth": True},
            "hmm": {"observations": [-.02, -.01, .01, .02, -.01, .03, .02, -.02], "states": 2},
            "garch": {"returns": np.random.default_rng(7).normal(0, .01, 40).tolist(), "max_iterations": 30},
            "microstructure": {"quotes": [{"timestamp_ns": 2**60, "instrument": "SYN001", "bid": 99,
                            "ask": 101, "bid_size": 10, "ask_size": 12}],
                            "trades": [{"timestamp_ns": 2**60 + 1, "instrument": "SYN001", "price": 101, "size": 5}]},
        }
        for kind, payload in examples.items():
            with self.subTest(kind=kind):
                response = await self.client.post(f"/api/v1/analytics/{kind}", json=payload)
                self.assertEqual(response.status_code, 200, response.text)
                self.assertEqual(self.envelope(response)["state"], "AVAILABLE")

    async def test_bad_mode_unknown_routes_and_invalid_model_fields(self):
        for path in ("/api/v1/terminal/markets?mode=FAKE-SECRET", "/api/v1/terminal/not-a-section"):
            response = await self.client.get(path)
            self.assertIn(response.status_code, (404, 422))
            self.assertNotIn("FAKE-SECRET", response.text)
            self.envelope(response)
        response = await self.client.post("/api/v1/analytics/options", json={"spot": True, "strike": 100})
        self.assertEqual(response.status_code, 422)

    async def test_cache_headers_and_local_assets(self):
        response = await self.client.get("/healthz")
        self.assertEqual(response.json()["ok"], True)
        self.assertIn("no-store", response.headers["cache-control"])
        self.assertEqual(response.headers["x-content-type-options"], "nosniff")
        # FileResponse uses a different ASGI capability path; checking served
        # assets is intentionally separate from any browser or visual claim.
        response = await self.client.get("/static/app.js")
        self.assertEqual(response.status_code, 200)
        self.assertIn("/api/v1/terminal/", response.text)

    async def test_openapi_declares_typed_envelope(self):
        response = await self.client.get("/api/openapi.json")
        body = response.json()
        schema = body["paths"]["/api/v1/terminal/{section}"]["get"]["responses"]["200"]["content"]["application/json"]["schema"]
        self.assertEqual(schema["$ref"], "#/components/schemas/EnvelopeSchema")


if __name__ == "__main__":
    unittest.main()
