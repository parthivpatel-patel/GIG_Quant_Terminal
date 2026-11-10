"""Authored model scanner cannot impersonate observed financial flow."""

from __future__ import annotations

import asyncio
import copy
import hashlib
import math

import pytest

from gig.service.contracts import EnvelopeSchema
from gig.service.market_flow import capability_register, market_flow_contract
from gig.service.serialization import dumps


def assumptions():
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


def test_register_is_truthful_and_detached():
    body = market_flow_contract().to_dict()
    EnvelopeSchema.model_validate(body)
    assert body["data"]["analysis"] is None
    assert body["provenance"]["classification"] == "SNAPSHOT"
    by_id = {row["id"]: row for row in body["data"]["capabilities"]}
    assert by_id["observed_options_flow"]["state"] == "UNAVAILABLE"
    assert by_id["congress_disclosures"]["state"] == "UNKNOWN"
    assert by_id["finra_off_exchange"]["redistribution"] == "NOT_AUTHORIZED"
    by_id["modeled_options_risk"]["state"] = "CORRUPTED"
    assert capability_register()[0]["state"] == "AVAILABLE"
    assert body["data"]["execution_enabled"] is False


def test_independent_atm_reference_greeks_and_expiry_payoff():
    body = market_flow_contract(assumptions()).to_dict()
    report = body["data"]["analysis"]
    exposure = report["exposure"]
    # Independent closed form: r=q=0, S=K=100, T=1, sigma=.2.
    def normal(x):
        return 0.5 * (1.0 + math.erf(x / math.sqrt(2)))
    unit_price = 100 * (normal(0.1) - normal(-0.1))
    assert exposure["price"] == pytest.approx(unit_price * 100, rel=1e-12)
    assert exposure["delta"] == pytest.approx(normal(0.1) * 100)
    assert exposure["vega"] == pytest.approx(100 * math.exp(-.005) / math.sqrt(2*math.pi))
    assert report["scenarios"]["scenarios"][1]["model_pnl"] == pytest.approx(0)
    assert report["payoff"][0]["terminal_payoff"] == 0
    assert report["payoff"][2]["terminal_payoff"] == pytest.approx(2000)
    assert report["payoff"][2]["undiscounted_premium_difference"] == pytest.approx(2000-unit_price*100)
    alerts = {row["id"]: row for row in report["alerts"]}
    assert alerts["max_abs_delta"]["state"] == "BREACH"
    assert alerts["max_abs_vega"]["state"] == "WITHIN_RESEARCH_LIMIT"
    assert report["open_interest"] is None
    assert report["dealer_positioning"] == "UNKNOWN"
    assert report["observed_gamma_exposure"] is None


def test_signed_hedged_legs_reuse_common_kernel_and_put_call_parity():
    request = assumptions()
    put = copy.deepcopy(request["contracts"][0])
    put.update(contract_id="HYPOTHETICAL:P100", right="PUT", quantity=-1)
    request["contracts"].append(put)
    analysis = market_flow_contract(request).to_dict()["data"]["analysis"]
    assert analysis["exposure"]["price"] == pytest.approx(0, abs=1e-10)
    assert analysis["exposure"]["delta"] == pytest.approx(100)
    assert analysis["exposure"]["gamma"] == pytest.approx(0)
    assert analysis["exposure"]["vega"] == pytest.approx(0)
    assert analysis["payoff"][0]["terminal_payoff"] == pytest.approx(-2000)


def test_deterministic_replay_input_identity_and_defensive_boundary():
    request = assumptions()
    before = copy.deepcopy(request)
    first = market_flow_contract(request)
    assert request == before
    assert first.wire == market_flow_contract(request).wire
    report = first.to_dict()["data"]["analysis"]
    assert report["inputs_sha256"] == hashlib.sha256(dumps(report["inputs"]).encode()).hexdigest()
    report["inputs"]["spot"] = 0
    assert first.to_dict()["data"]["analysis"]["inputs"]["spot"] == 100
    request["spot_shocks"].reverse()
    assert first.wire == market_flow_contract(request).wire


@pytest.mark.parametrize("field,value", [
    ("spot", float("nan")), ("spot", float("inf")), ("spot", -1),
    ("spot", True), ("spot", "100"), ("rate", 2),
    ("classification", "REAL_TIME"), ("source", "OPRA"), ("currency", "EUR"),
    ("underlying_id", "../SECRET"), ("valuation_at", "2026-01-01"),
    ("valuation_at", "2026-01-01T00:00:00.0000000001Z"),
    ("spot_shocks", [0, 0]), ("spot_shocks", [-1, 0]),
    ("spot_shocks", [0, True]), ("contracts", []),
])
def test_invalid_or_observed_claims_fail_closed(field, value):
    request = assumptions()
    request[field] = value
    with pytest.raises(ValueError):
        market_flow_contract(request)


@pytest.mark.parametrize("field,value", [
    ("quantity", 0), ("quantity", True), ("quantity", 0.5), ("multiplier", -1),
    ("volatility", None), ("volatility", 0), ("volatility", float("nan")),
    ("expiry", "2026-01-01T00:00:00Z"), ("right", "AMERICAN_CALL"),
    ("contract_id", "<script>"), ("strike", 10**400),
])
def test_contract_precision_and_liability_boundaries(field, value):
    request = assumptions()
    request["contracts"][0][field] = value
    with pytest.raises(ValueError):
        market_flow_contract(request)


def test_duplicate_terms_mixed_expiry_and_unsupported_fields_rejected():
    request = assumptions()
    request["contracts"].append(copy.deepcopy(request["contracts"][0]))
    with pytest.raises(ValueError):
        market_flow_contract(request)
    request["contracts"][1].update(contract_id="HYPOTHETICAL:OTHER")
    with pytest.raises(ValueError):
        market_flow_contract(request)
    request["contracts"][1].update(expiry="2028-01-01T00:00:00Z", strike=105)
    with pytest.raises(ValueError):
        market_flow_contract(request)
    request = assumptions()
    request["contracts"][0]["open_interest"] = 1000
    with pytest.raises(ValueError):
        market_flow_contract(request)


def test_nanosecond_source_precision_is_retained_and_budget_enforced():
    request = assumptions()
    request["valuation_at"] = "2026-01-01T00:00:00.000000001Z"
    body = market_flow_contract(request).to_dict()
    assert body["data"]["analysis"]["inputs"]["valuation_at"].endswith("000000001+00:00")
    request["spot_shocks"] = [i / 100 for i in range(42)]
    with pytest.raises(ValueError):
        market_flow_contract(request)


def test_http_contracts_replay_and_admission_safety(tmp_path):
    import httpx

    from gig.service.app import create_app

    async def run():
        app = create_app(state_dir=tmp_path)
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            registry = await client.get("/api/v1/market-flow")
            assert registry.status_code == 200
            assert registry.json()["data"]["analysis"] is None
            first = await client.post("/api/v1/market-flow", json=assumptions())
            second = await client.post("/api/v1/market-flow", json=assumptions())
            assert first.status_code == second.status_code == 200
            assert first.content == second.content
            EnvelopeSchema.model_validate(first.json())
            foreign = await client.post("/api/v1/market-flow", json=assumptions(),
                headers={"origin": "https://foreign.invalid"})
            assert foreign.status_code == 403
            unsupported = assumptions()
            unsupported["classification"] = "REAL_TIME"
            rejected = await client.post("/api/v1/market-flow", json=unsupported)
            assert rejected.status_code == 422
            assert "REAL_TIME" not in rejected.text
            missing_type = await client.post("/api/v1/market-flow", content="{}")
            assert missing_type.status_code == 415
            nonfinite = await client.post("/api/v1/market-flow", content='{"spot":NaN}',
                headers={"content-type": "application/json"})
            assert nonfinite.status_code == 400
        restarted = create_app(state_dir=tmp_path)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=restarted),
            base_url="http://test") as client:
            replayed = await client.post("/api/v1/market-flow", json=assumptions())
            assert replayed.content == first.content
            # Stateless research must not become an admitted provider snapshot.
            assert (await client.get("/api/v1/market-flow")).json()["data"]["analysis"] is None

    asyncio.run(run())
    request = assumptions()
    request["limits"]["max_abs_delta"] = -1
    with pytest.raises(ValueError):
        market_flow_contract(request)
