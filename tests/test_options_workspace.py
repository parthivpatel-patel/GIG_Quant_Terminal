"""Historical option provenance, numerical inversion and durable research replay."""
from __future__ import annotations

import copy
import sqlite3
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from gig.options import black_scholes
from gig.service.contracts import EnvelopeSchema
from gig.service.options_workspace import OptionsWorkspace

AT = "2026-11-02T15:00:00.123456789Z"
KNOWN = "2026-11-02T15:00:01.123456789Z"
LATER = "2026-11-02T15:05:00.123456789Z"


def option_fixture(**changes):
    event = pd.Timestamp(AT)
    quotes = []
    for months, days in ((6, 182.5), (12, 365.0)):
        expiry = (event+pd.Timedelta(days=days)).isoformat()
        for strike in (90, 100, 110):
            for right in ("CALL", "PUT"):
                mid = float(black_scholes(100, strike, .02, .01, .2, days/365, right == "CALL")["price"])
                quotes.append({"contract_id": f"SYN100-{months}-{strike}-{right}", "expiry": expiry,
                    "strike": strike, "right": right, "multiplier": 100, "style": "EUROPEAN",
                    "bid": mid-.01, "ask": mid+.01, "volume": 10 if right == "CALL" else 20,
                    "open_interest": 100 if right == "CALL" else 150})
    return {"source": "authored-european-chain", "classification": "SYNTHETIC", "feed": "AUTHORED_FIXTURE",
        "underlying_id": "SYN100", "currency": "USD", "event_at": AT, "acquired_at": KNOWN, "knowledge_at": KNOWN,
        "spot": 100, "spot_source_identity": "authored-synchronous-underlying-100",
        "spot_event_at": AT, "spot_knowledge_at": KNOWN, "quotes": quotes, **changes}


def option_request(**changes):
    return {"source": "authored-european-chain", "classification": "SYNTHETIC", "underlying_id": "SYN100",
        "observation_at": AT, "as_of": KNOWN, "max_age_seconds": 60, "rate": .02, "dividend_yield": .01,
        "positions": [{"contract_id": "SYN100-6-100-CALL", "quantity": 2},
                      {"contract_id": "SYN100-12-100-PUT", "quantity": -1}],
        "surface_query": {"log_forward_moneyness": 0, "ttm": .75}, **changes}


class OptionsWorkspaceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.base = Path(self.temp.name) / "state"
        self.workspace = OptionsWorkspace(self.base)

    def tearDown(self):
        self.temp.cleanup()

    def test_actual_iv_greeks_surface_and_signed_portfolio_reference(self):
        admission = self.workspace.ingest(option_fixture()).to_dict()
        self.assertEqual(admission["provenance"]["classification"], "SYNTHETIC")
        response = self.workspace.run(option_request()).to_dict()
        EnvelopeSchema.model_validate(response)
        data = response["data"]
        self.assertEqual(response["state"], "AVAILABLE")
        self.assertEqual(data["metrics"]["valid_models"], 12)
        self.assertEqual(data["metrics"]["invalid_models"], 0)
        self.assertFalse(data["trading_approved"])
        for row in data["rows"]:
            self.assertAlmostEqual(row["implied_volatility"], .2, places=10)
            self.assertAlmostEqual(row["greeks"]["price"], row["midpoint"], places=10)
        self.assertAlmostEqual(data["surface"]["query"]["volatility"], .2, places=10)
        self.assertFalse(data["surface"]["continuous_arbitrage_certified"])
        self.assertEqual(len(data["term_structure"]), 2)
        self.assertAlmostEqual(data["vts"]["rows"][0]["value"], 0, places=10)
        call = black_scholes(100, 100, .02, .01, .2, .5, True)
        put = black_scholes(100, 100, .02, .01, .2, 1, False)
        expected = 200*float(call["delta"])-100*float(put["delta"])
        self.assertAlmostEqual(data["portfolio_exposure"]["delta"], expected, places=9)
        zero = next(row for row in data["scenarios"]["scenarios"] if row["spot_shock"] == row["volatility_shock"] == 0)
        self.assertEqual(zero["model_pnl"], 0)
        self.assertEqual(data["pcr_volume"]["value"], 2)
        self.assertEqual(data["pcr_open_interest"]["value"], 1.5)
        self.assertEqual(data["csm"]["state"], "UNAVAILABLE")
        self.assertIn("UNSUPPLIED", data["csm"]["definition"])

    def test_late_revision_preserves_prior_research_and_restart_replay(self):
        self.workspace.ingest(option_fixture())
        original = self.workspace.run(option_request()).to_dict()
        revision = option_fixture(acquired_at=LATER, knowledge_at=LATER)
        revision["quotes"][0]["bid"] += .5
        revision["quotes"][0]["ask"] += .5
        self.workspace.ingest(revision)
        reopened = OptionsWorkspace(self.base)
        self.assertEqual(reopened.run(option_request()).to_dict(), original)
        self.assertEqual(reopened.replay({"run_id": original["data"]["run_id"]}).to_dict(), original)
        later = reopened.run(option_request(as_of=LATER, max_age_seconds=600)).to_dict()
        self.assertNotEqual(original["data"]["source_snapshot_id"], later["data"]["source_snapshot_id"])
        self.assertNotEqual(original["data"]["run_id"], later["data"]["run_id"])
        # Defensive reads cannot mutate durable experiment evidence.
        run = reopened.load(original["data"]["run_id"])
        run["rows"][0]["midpoint"] = -1
        self.assertGreater(reopened.load(original["data"]["run_id"])["rows"][0]["midpoint"], 0)

    def test_pipeline_matches_published_bsm_reference_and_nanosecond_staleness(self):
        # Standard European ATM reference, S=K=100,r=.05,q=0,T=1,sigma=.2.
        quote = {"contract_id": "SYN100-REF-C", "expiry": (pd.Timestamp(AT)+pd.Timedelta(days=365)).isoformat(),
            "strike": 100, "right": "CALL", "multiplier": 100, "style": "EUROPEAN",
            "bid": 10.440583572185565, "ask": 10.460583572185565, "volume": None, "open_interest": None}
        known = "2026-11-02T15:00:00.123456790Z"
        self.workspace.ingest(option_fixture(quotes=[quote], acquired_at=known, knowledge_at=known, spot_knowledge_at=known))
        request = option_request(as_of=known, max_age_seconds=0, rate=.05, dividend_yield=0,
            positions=[{"contract_id": quote["contract_id"], "quantity": 2}])
        result = self.workspace.run(request).to_dict()
        self.assertEqual(result["state"], "STALE")
        model = result["data"]["rows"][0]
        self.assertAlmostEqual(model["implied_volatility"], .2, places=10)
        self.assertAlmostEqual(model["greeks"]["delta"], .6368306511756191, places=10)
        self.assertAlmostEqual(model["greeks"]["gamma"], .018762017345846895, places=10)
        self.assertAlmostEqual(model["greeks"]["vega"], .3752403469169379, places=10)
        self.assertAlmostEqual(model["greeks"]["rho"], .5323248154537634, places=10)
        self.assertAlmostEqual(result["data"]["portfolio_exposure"]["price"], 2090.116714437113, places=8)
        self.assertEqual(self.workspace.section("risk", known, 0).to_dict()["state"], "STALE")

    def test_acquisition_and_spot_knowledge_cutoffs_nanosecond_precision(self):
        self.assertEqual(self.workspace.run(option_request()).to_dict()["state"], "UNAVAILABLE")
        self.assertFalse(self.base.exists())
        self.workspace.ingest(option_fixture())
        self.assertEqual(self.workspace.run(option_request(as_of="2026-11-02T15:00:01.123456788Z")).to_dict()["state"], "UNAVAILABLE")
        saved = self.workspace.run(option_request()).to_dict()
        self.assertEqual(saved["data"]["knowledge_at"], "2026-11-02T15:00:01.123456789+00:00")
        self.assertEqual(self.workspace.run(option_request(as_of=LATER)).to_dict()["state"], "STALE")
        self.assertEqual(self.workspace.section("options", KNOWN, None).to_dict()["state"], "UNKNOWN")
        self.assertEqual(self.workspace.section("options", LATER, 60).to_dict()["state"], "STALE")
        self.assertIsNone(self.workspace.section("execution", KNOWN, 60))

    def test_missing_counts_zero_denominator_invalid_iv_and_no_extrapolation(self):
        fixture = option_fixture()
        fixture["quotes"][0]["volume"] = None
        for quote in fixture["quotes"]:
            if quote["right"] == "CALL":
                quote["open_interest"] = 0
        # Impossible midpoint explicitly rejects that model while preserving source quote.
        fixture["quotes"][0]["bid"] = 199
        fixture["quotes"][0]["ask"] = 200
        self.workspace.ingest(fixture)
        request = option_request(positions=[], surface_query={"log_forward_moneyness": 10, "ttm": .75})
        data = self.workspace.run(request).to_dict()["data"]
        self.assertEqual(data["metrics"]["invalid_models"], 1)
        self.assertEqual(data["pcr_volume"]["state"], "UNKNOWN")
        self.assertEqual(data["pcr_open_interest"]["state"], "UNKNOWN")
        self.assertIsNone(data["pcr_volume"]["value"])
        self.assertEqual(data["surface"]["query"]["state"], "UNAVAILABLE")
        self.assertEqual(data["portfolio_exposure"]["state"], "UNAVAILABLE")
        bad = option_request(positions=[{"contract_id": fixture["quotes"][0]["contract_id"], "quantity": 1}])
        with self.assertRaises(ValueError):
            self.workspace.run(bad)
        puts_only = OptionsWorkspace(self.base / "put-only")
        fixture = option_fixture()
        fixture["quotes"] = [fixture["quotes"][1]]
        puts_only.ingest(fixture)
        put_data = puts_only.run(option_request(positions=[])).to_dict()["data"]
        self.assertEqual(put_data["surface"]["state"], "UNAVAILABLE")
        self.assertEqual(put_data["surface"]["diagnostics"]["state"], "UNAVAILABLE")
        self.assertEqual(put_data["metrics"]["valid_models"], 1)

    def test_invalid_admission_no_artifact_and_identity_conflicts(self):
        mutations = [lambda d: d.update(spot=float("nan")), lambda d: d.update(currency="EUR"),
            lambda d: d.update(acquired_at="2026-11-01T15:00Z"),
            lambda d: d.update(spot_knowledge_at=LATER), lambda d: d.update(spot_event_at=LATER),
            lambda d: d["quotes"][0].update(style="AMERICAN"),
            lambda d: d["quotes"][0].update(multiplier=True),
            lambda d: d["quotes"][0].update(expiry=AT),
            lambda d: d["quotes"][0].update(bid=10, ask=1),
            lambda d: d["quotes"][0].update(volume=-1),
            lambda d: d.update(quotes=d["quotes"]+[copy.deepcopy(d["quotes"][0])])]
        for mutation in mutations:
            fixture = option_fixture()
            mutation(fixture)
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                self.workspace.ingest(fixture)
        self.assertFalse(self.base.exists())
        self.workspace.ingest(option_fixture())
        conflict = option_fixture()
        conflict["quotes"][0]["ask"] += 1
        with self.assertRaises(ValueError):
            self.workspace.ingest(conflict)
        conflict = option_fixture(acquired_at=LATER, knowledge_at=LATER)
        conflict["quotes"][0]["multiplier"] = 10
        with self.assertRaises(ValueError):
            self.workspace.ingest(conflict)
        self.assertEqual(self.workspace.run(option_request()).to_dict()["state"], "AVAILABLE")

    def test_research_request_and_artifact_tampering_rejected(self):
        self.workspace.ingest(option_fixture())
        for changes in ({"max_age_seconds": None}, {"rate": True}, {"rate": 3}, {"spot_shocks": [-1]},
                        {"positions": [{"contract_id": "missing", "quantity": 1}]},
                        {"positions": [{"contract_id": "SYN100-6-100-CALL", "quantity": .5}]},
                        {"volatility_shocks": [0, 0]}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.workspace.run(option_request(**changes))
        result = self.workspace.run(option_request()).to_dict()
        identity = result["data"]["run_id"]
        with sqlite3.connect(self.base/"options-research/options.sqlite") as connection:
            connection.execute("UPDATE runs SET snapshot_id=? WHERE run_id=?", ("0"*64, identity))
        with self.assertRaises(ValueError):
            self.workspace.load(identity)


if __name__ == "__main__":
    unittest.main()
