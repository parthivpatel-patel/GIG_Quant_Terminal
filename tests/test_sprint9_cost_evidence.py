"""Sprint 9 executed-cost evidence keeps estimated, simulated and observed classes apart."""
from __future__ import annotations

import asyncio
import copy
import unittest
from decimal import Decimal

import httpx

from gig.execution.cost_evidence import cost_evidence_report
from gig.service.app import create_app
from gig.service.contracts import EnvelopeSchema


def request():
    return {
        "currency": "USD", "as_of": "2026-09-30T21:00:00Z", "match_tolerance_seconds": 1,
        "max_participation": "0.25",
        "estimates": [{"estimate_id": "est-1", "instrument": "SYN001", "side": "BUY", "quantity": "100",
                       "reference_price": "50", "spread_bps": "4", "commission_bps": "1",
                       "impact_eta": "0.1", "adv_value": "5000000"}],
        "fills": [
            {"fill_id": "sim-1", "instrument": "SYN001", "side": "BUY", "quantity": "100", "price": "50.03",
             "fees": "1.00", "executed_at": "2026-09-30T14:00:05Z", "knowledge_at": "2026-09-30T14:00:05Z",
             "evidence_class": "SIMULATED", "source": "offline-simulator",
             "arrival": {"bid": "49.98", "ask": "50.02", "observed_at": "2026-09-30T14:00:00Z", "source": "authored-quote"},
             "post_trade": {"mid": "50.04", "observed_at": "2026-09-30T14:05:00Z", "source": "authored-quote"}},
            {"fill_id": "obs-1", "instrument": "SYN002", "side": "SELL", "quantity": "200", "price": "20.00",
             "fees": "0.50", "executed_at": "2026-09-30T15:00:00Z", "knowledge_at": "2026-09-30T16:00:00Z",
             "evidence_class": "OBSERVED_HISTORICAL", "source": "authored-execution-report",
             "arrival": {"bid": "20.01", "ask": "20.03", "observed_at": "2026-09-30T14:59:59Z", "source": "authored-quote"}},
            {"fill_id": "obs-2", "instrument": "SYN003", "side": "BUY", "quantity": "10", "price": "10",
             "fees": "0", "executed_at": "2026-09-30T15:00:00Z", "knowledge_at": "2026-09-30T15:00:00Z",
             "evidence_class": "OBSERVED_HISTORICAL", "source": "authored-execution-report",
             "arrival": {"bid": "9.99", "ask": "10.01", "observed_at": "2026-09-30T14:59:00Z", "source": "authored-quote"}},
        ],
        "prints": [
            {"print_id": "p-1", "instrument": "SYN002", "price": "20.00", "size": "200",
             "observed_at": "2026-09-30T15:00:00.400Z", "source": "authored-tape"},
            {"print_id": "p-2", "instrument": "SYN002", "price": "20.02", "size": "600",
             "observed_at": "2026-09-30T14:59:59.500Z", "source": "authored-tape"},
            {"print_id": "p-3", "instrument": "SYN001", "price": "50.03", "size": "100",
             "observed_at": "2026-09-30T14:00:05Z", "source": "authored-tape"},
        ],
    }


class CostEvidenceTests(unittest.TestCase):
    def test_decomposition_conserves_shortfall_exactly(self):
        report = cost_evidence_report(request())
        sim = next(f for f in report["fills"] if f["fill_id"] == "sim-1")
        self.assertEqual(sim["arrival_mid"], Decimal("50.00"))
        self.assertEqual(sim["spread_cost"], Decimal("2.00"))
        self.assertEqual(sim["slippage"], Decimal("1.00"))
        self.assertEqual(sim["transaction_fees"], Decimal("1.00"))
        self.assertEqual(sim["implementation_shortfall"], Decimal("4.00"))
        self.assertEqual(sim["market_impact"], Decimal("4.00"))
        self.assertEqual(sim["market_impact_state"], "MEASURED_PRICE_MOVE_NOT_CAUSAL")
        sell = next(f for f in report["fills"] if f["fill_id"] == "obs-1")
        # Sell at 20.00 against bid 20.01: half-spread 2.00 plus 2.00 adverse slippage.
        self.assertEqual(sell["spread_cost"], Decimal("2.00"))
        self.assertEqual(sell["slippage"], Decimal("2.00"))
        self.assertIsNone(sell["market_impact"])
        self.assertEqual(sell["market_impact_state"], "UNKNOWN_NO_POST_TRADE_QUOTE")

    def test_classes_never_merge_and_simulation_is_never_promoted(self):
        report = cost_evidence_report(request())
        classes = report["classes"]
        self.assertEqual(classes["ESTIMATED"]["count"], 1)
        self.assertEqual(classes["SIMULATED"]["count"], 1)
        self.assertEqual(classes["OBSERVED_HISTORICAL"]["count"], 1)
        self.assertEqual(classes["OBSERVED_UNCORROBORATED_EXCLUDED"]["count"], 1)
        self.assertEqual(classes["BROKER_RECONCILED"]["state"], "UNAVAILABLE")
        sim = next(f for f in report["fills"] if f["fill_id"] == "sim-1")
        self.assertEqual(sim["print_corroboration"], "MATCHED")
        self.assertEqual(sim["evidence_state"], "SIMULATED_NOT_MARKET_EXECUTION")
        self.assertFalse(report["claims"]["simulated_promoted_by_prints"])
        self.assertFalse(report["claims"]["broker_reconciled"])
        self.assertTrue(report["claims"]["observed_executions_print_corroborated"])
        self.assertIsNone(report["classes"]["ESTIMATED"]["slippage"])

    def test_estimate_matches_square_root_model(self):
        est = cost_evidence_report(request())["estimates"][0]
        self.assertEqual(est["notional"], Decimal("5000"))
        self.assertEqual(est["transaction_fees"], Decimal("0.5000"))
        self.assertEqual(est["spread_cost"], Decimal("1.0000"))
        # 0.1 * sqrt(5000/5e6) * 1e4 bps = 316.227766 bps of 5000.
        self.assertAlmostEqual(float(est["market_impact"]), 5000 * 0.1 * (0.001 ** 0.5), places=9)

    def test_capacity_uses_only_independent_prints_in_arrival_window(self):
        sell = next(f for f in cost_evidence_report(request())["fills"] if f["fill_id"] == "obs-1")
        self.assertEqual(sell["observed_interval_volume"], Decimal("600"))
        self.assertEqual(sell["capacity_state"], "EXCEEDS_ASSUMPTION")
        other = next(f for f in cost_evidence_report(request())["fills"] if f["fill_id"] == "obs-2")
        self.assertEqual(other["capacity_state"], "UNKNOWN_NO_INDEPENDENT_PRINTS")
        self.assertIsNone(other["participation"])

    def test_ambiguous_and_same_source_prints_do_not_corroborate(self):
        payload = request()
        payload["prints"].append({"print_id": "p-dup", "instrument": "SYN002", "price": "20.00", "size": "200",
                                  "observed_at": "2026-09-30T14:59:59.900Z", "source": "authored-tape"})
        report = cost_evidence_report(payload)
        obs = next(f for f in report["fills"] if f["fill_id"] == "obs-1")
        self.assertEqual(obs["print_corroboration"], "AMBIGUOUS")
        self.assertEqual(report["classes"]["OBSERVED_HISTORICAL"]["count"], 0)
        self.assertFalse(report["claims"]["observed_executions_print_corroborated"])
        payload = request()
        payload["prints"][0]["source"] = "authored-execution-report"
        obs = next(f for f in cost_evidence_report(payload)["fills"] if f["fill_id"] == "obs-1")
        self.assertEqual(obs["print_corroboration"], "SAME_SOURCE_UNVERIFIED")
        self.assertEqual(obs["evidence_state"], "CALLER_ASSERTED_UNCORROBORATED")

    def test_fail_closed_inputs(self):
        cases = []
        broker = request()
        broker["fills"][0]["evidence_class"] = "BROKER_RECONCILED"
        cases.append(broker)
        relabel = request()
        relabel["fills"][0]["evidence_class"] = "OBSERVED_HISTORICAL"
        cases.append(relabel)
        future_quote = request()
        future_quote["fills"][1]["arrival"]["observed_at"] = "2026-09-30T15:00:01Z"
        cases.append(future_quote)
        late_knowledge = request()
        late_knowledge["fills"][1]["knowledge_at"] = "2026-10-01T00:00:00Z"
        cases.append(late_knowledge)
        early_post = request()
        early_post["fills"][0]["post_trade"]["observed_at"] = "2026-09-30T14:00:05Z"
        cases.append(early_post)
        crossed = request()
        crossed["fills"][0]["arrival"]["bid"] = "50.05"
        cases.append(crossed)
        duplicate = request()
        duplicate["fills"][2]["fill_id"] = "obs-1"
        cases.append(duplicate)
        currency = request()
        currency["currency"] = "EUR"
        cases.append(currency)
        nonfinite = request()
        nonfinite["fills"][0]["price"] = float("nan")
        cases.append(nonfinite)
        extra = request()
        extra["fills"][0]["venue"] = "X"
        cases.append(extra)
        naive = request()
        naive["as_of"] = "2026-09-30T21:00:00"
        cases.append(naive)
        for case in cases:
            with self.subTest(case=case), self.assertRaises((ValueError, TypeError)):
                cost_evidence_report(case)

    def test_replay_is_deterministic_and_inputs_unmodified(self):
        payload = request()
        original = copy.deepcopy(payload)
        first, second = cost_evidence_report(payload), cost_evidence_report(payload)
        self.assertEqual(payload, original)
        self.assertEqual(first["report_sha256"], second["report_sha256"])
        payload["fills"][0]["fees"] = "1.01"
        self.assertNotEqual(cost_evidence_report(payload)["report_sha256"], first["report_sha256"])


class CostEvidenceHTTPTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        async def clock():
            while True:
                await asyncio.sleep(.01)
        self.clock = asyncio.create_task(clock())
        self.client = httpx.AsyncClient(transport=httpx.ASGITransport(app=create_app()), base_url="http://test")

    async def asyncTearDown(self):
        await self.client.aclose()
        self.clock.cancel()

    async def test_envelope_and_rejections(self):
        response = await self.client.post("/api/v1/analytics/cost-evidence", json=request())
        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        EnvelopeSchema.model_validate(body)
        self.assertEqual(body["state"], "AVAILABLE")
        self.assertEqual(body["data"]["classes"]["BROKER_RECONCILED"]["reason"], "BROKER_DISCONNECTED")
        broker = request()
        broker["fills"][0]["evidence_class"] = "BROKER_RECONCILED"
        response = await self.client.post("/api/v1/analytics/cost-evidence", json=broker)
        self.assertEqual(response.status_code, 422)
        self.assertEqual(response.json()["error"]["code"], "INVALID_REQUEST")
