"""Integrated research approval/replay and authored-flow HTTP safety boundaries."""

from __future__ import annotations

import asyncio
import copy

import httpx
import pytest

from gig.research_org.director import FOLLOWUP_ID, OBJECTIVE_ID
from gig.research_org.treasury_followup import run_followup
from gig.service.app import create_app
from gig.service.contracts import EnvelopeSchema
from gig.service.serialization import dumps, loads


def _assumptions():
    return {
        "source": "USER_AUTHORED_ASSUMPTIONS", "classification": "SYNTHETIC",
        "valuation_at": "2026-01-01T00:00:00Z", "underlying_id": "HYPOTHETICAL:DESK",
        "currency": "USD", "spot": 100.0, "rate": 0.0, "dividend_yield": 0.0,
        "contracts": [{"contract_id": "HYPOTHETICAL:C100", "strike": 100.0,
            "right": "CALL", "expiry": "2027-01-01T00:00:00Z", "quantity": 1,
            "multiplier": 100, "volatility": 0.2}],
        "spot_shocks": [-0.2, 0.0, 0.2],
        "limits": {"max_abs_delta": 50.0, "max_abs_vega": 50.0, "max_scenario_loss": 500.0},
    }


def _contract(response, expected=200):
    assert response.status_code == expected, response.text
    body = response.json()
    EnvelopeSchema.model_validate(body)
    assert response.headers["cache-control"].startswith("no-store")
    assert response.headers["x-content-type-options"] == "nosniff"
    if expected != 200:
        assert body["state"] == "ERROR" and body["data"] is None
        assert body["error"]["code"] == "INVALID_REQUEST"
    return body


def test_market_flow_http_contract_keeps_observed_fields_unknown(tmp_path):
    async def check():
        app = create_app(state_dir=tmp_path / "state")
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://test") as client:
            register = _contract(await client.get("/api/v1/market-flow"))
            assert register["provenance"]["classification"] == "SNAPSHOT"
            assert register["data"]["analysis"] is None
            assert register["data"]["observed_flow_state"] == "UNAVAILABLE"
            payload = _assumptions()
            before = copy.deepcopy(payload)
            modeled = _contract(await client.post("/api/v1/market-flow", json=payload))
            assert payload == before
            assert modeled["provenance"]["classification"] == "MODEL_ESTIMATE"
            analysis = modeled["data"]["analysis"]
            assert analysis["classification"] == "MODEL_ESTIMATE"
            assert analysis["input_classification"] == "SYNTHETIC"
            assert analysis["open_interest"] is None
            assert analysis["observed_gamma_exposure"] is None
            assert analysis["dealer_positioning"] == "UNKNOWN"
            assert modeled["data"]["orders_submitted"] == 0
            assert modeled["data"]["execution_enabled"] is False
            assert modeled["data"]["funded_accounting_claim"] is False
            assert len(analysis["payoff"]) == 3 and len(analysis["alerts"]) == 3
            assert analysis["exposure"]["price"] > 0
            replay = _contract(await client.post("/api/v1/market-flow", json=payload))
            assert replay == modeled
            assert _contract(await client.get("/api/v1/market-flow")) == register
        assert not (tmp_path / "state").exists()

    asyncio.run(check())


@pytest.mark.parametrize("field,value", [("source", "OPRA"), ("source", "SEC"),
                                          ("classification", "REAL_TIME"),
                                          ("classification", "HISTORICAL")])
def test_market_flow_http_rejects_spoofed_observed_provenance(tmp_path, field, value):
    async def check():
        app = create_app(state_dir=tmp_path / "state")
        payload = _assumptions()
        payload[field] = value
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://test") as client:
            _contract(await client.post("/api/v1/market-flow", json=payload), 422)
        assert not (tmp_path / "state").exists()

    asyncio.run(check())


@pytest.mark.parametrize("path", ["/api/v1/market-flow", "/api/v1/research-org/director"])
@pytest.mark.parametrize("raw", [b'{"spot":NaN}', b'{"spot":Infinity}',
                                 b'{"spot":1,"spot":2}', b'{"broken":'])
def test_new_http_routes_reject_nonfinite_duplicate_or_malformed_json(tmp_path, path, raw):
    async def check():
        app = create_app(state_dir=tmp_path / "state")
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://test") as client:
            response = await client.post(path, content=raw, headers={"content-type": "application/json"})
            body = _contract(response, 400)
            assert body["error"]["message"] == "The request does not satisfy the API contract."
        assert not (tmp_path / "state").exists()

    asyncio.run(check())


@pytest.mark.parametrize("path", ["/api/v1/market-flow", "/api/v1/research-org/director"])
def test_new_http_routes_reject_cross_origin_and_non_json_writes(tmp_path, path):
    async def check():
        app = create_app(state_dir=tmp_path / "state")
        payload = _assumptions() if path.endswith("market-flow") else {"action": "approve", "objective_id": OBJECTIVE_ID}
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://test") as client:
            _contract(await client.post(path, json=payload, headers={"origin": "https://evil.invalid"}), 403)
            _contract(await client.post(path, content="{}", headers={"content-type": "text/plain"}), 415)
        assert not (tmp_path / "state").exists()

    asyncio.run(check())


@pytest.mark.parametrize("payload", [
    {"action": [], "objective_id": OBJECTIVE_ID},
    {"action": {}, "objective_id": OBJECTIVE_ID},
    {"action": "approve", "objective_id": []},
    {"action": "approve", "objective_id": "unknown-objective"},
    {"action": "run-generated-code", "objective_id": OBJECTIVE_ID},
    {"action": "approve", "objective_id": OBJECTIVE_ID, "extra": True},
    [],
])
def test_director_malformed_or_unapproved_actions_return_422_without_mutation(tmp_path, payload):
    async def check():
        app = create_app(state_dir=tmp_path / "state")
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://test") as client:
            _contract(await client.post("/api/v1/research-org/director", json=payload), 422)
            status = _contract(await client.get("/api/v1/research-org/director"))["data"]
            assert status["objectives"][0]["status"] == "NOT_APPROVED"
            assert status["investigations"] == []
        assert not (tmp_path / "state").exists()

    asyncio.run(check())


def test_director_approval_and_revocation_do_not_start_compute(tmp_path):
    async def check():
        app = create_app(state_dir=tmp_path / "state")
        async with app.router.lifespan_context(app), httpx.AsyncClient(
            transport=httpx.ASGITransport(app), base_url="http://test"
        ) as client:
            for action, expected in (("approve", "APPROVED"), ("revoke", "REVOKED")):
                control = _contract(await client.post("/api/v1/research-org/director",
                                    json={"action": action, "objective_id": OBJECTIVE_ID}))
                assert control["data"]["objectives"][0]["status"] == expected
                await asyncio.sleep(0.1)
                status = _contract(await client.get("/api/v1/research-org/status"))["data"]
                assert status["run_count"] == 0
                assert status["director"]["investigations"] == []
                assert status["scheduler_state"]["alive"] is False
                assert status["scheduler_state"]["desired"] == "STOPPED"
        restarted = create_app(state_dir=tmp_path / "state")
        async with restarted.router.lifespan_context(restarted), httpx.AsyncClient(
            transport=httpx.ASGITransport(restarted), base_url="http://test"
        ) as client:
            status = _contract(await client.get("/api/v1/research-org/status"))["data"]
            assert status["director"]["objectives"][0]["status"] == "REVOKED"
            assert status["run_count"] == 0 and not status["scheduler_state"]["alive"]

    asyncio.run(check())


def test_http_automatic_four_cycle_followup_and_normal_snapshot_survive_restart(tmp_path):
    async def check():
        state = tmp_path / "state"
        app = create_app(state_dir=state)
        latest = "2026-06-01T12:00:00+00:00"
        bars = {
            "rows": [{"instrument_id": "ID:HTTP", "timestamp": latest, "available_at": latest,
                      "open": 100, "high": 102, "low": 99, "close": 101, "volume": 1000}],
            "source": "authored-http-fixture", "classification": "SYNTHETIC",
            "corporate_action_policy": "RAW", "currency": "USD",
        }
        query = {"mode": "local", "as_of": latest, "max_age_seconds": 60}
        async with app.router.lifespan_context(app), httpx.AsyncClient(
            transport=httpx.ASGITransport(app), base_url="http://test"
        ) as client:
            _contract(await client.post("/api/v1/workspace/bars", json=bars))
            normal = _contract(await client.get("/api/v1/terminal/markets", params=query))
            assert normal["state"] == "AVAILABLE" and normal["provenance"]["classification"] == "SYNTHETIC"
            _contract(await client.post("/api/v1/research-org/director",
                      json={"action": "approve", "objective_id": OBJECTIVE_ID}))
            _contract(await client.post("/api/v1/research-org/scheduler",
                      json={"action": "start", "cadence_seconds": 0.05}))
            deadline = asyncio.get_running_loop().time() + 30
            while True:
                data = _contract(await client.get("/api/v1/research-org/status"))["data"]
                if data["run_count"] == 4 and data["scheduler_state"]["desired"] == "PAUSED":
                    break
                assert asyncio.get_running_loop().time() < deadline, data["scheduler_state"]
                await asyncio.sleep(0.05)
            followup = next(row["report"] for row in data["runs"] if row["hypothesis_id"] == FOLLOWUP_ID)
            assert followup == loads(dumps(run_followup()[1]))
            assert followup["verdict"] == "INCONCLUSIVE" and followup["economic_claim"] == "NONE"
            assert data["director"]["objectives"][0]["status"] == "COMPLETED"
            assert len(data["director"]["investigations"]) == 1
            assert data["director"]["investigations"][0]["status"] == "DONE"
            assert data["orders_submitted"] == 0
            memory = copy.deepcopy(data["director"]["memory"])
            parent = next(row["report"] for row in data["runs"] if row["report"].get("hits") == 105)
            assert parent["pairs"] == 228 and parent["hit_rate"] < 0.5
            assert _contract(await client.get("/api/v1/terminal/markets", params=query)) == normal
        restarted = create_app(state_dir=state)
        async with restarted.router.lifespan_context(restarted), httpx.AsyncClient(
            transport=httpx.ASGITransport(restarted), base_url="http://test"
        ) as client:
            restored = _contract(await client.get("/api/v1/research-org/status"))["data"]
            assert restored["run_count"] == 4
            assert restored["director"]["memory"] == memory
            assert not restored["scheduler_state"]["alive"]
            assert restored["director"]["objectives"][0]["status"] == "COMPLETED"
            replay = next(row["report"] for row in restored["runs"] if row["hypothesis_id"] == FOLLOWUP_ID)
            assert replay == followup
            assert _contract(await client.get("/api/v1/terminal/markets", params=query)) == normal

    asyncio.run(check())
