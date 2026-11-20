"""Frozen annotation identities, whole-symbol syntax, durable replay and API bounds."""
from __future__ import annotations

import asyncio
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx

from gig.financial_intelligence import latent, nlp, nlp_v1, nlp_v2
from gig.financial_intelligence.store import FinancialCorpus, digest
from gig.service.app import create_app
from gig.service.serialization import dumps

T0 = "2026-11-13T21:00:00.000000001Z"
GOLDEN_TEXT = "Café NYSE:BRK.B and NASDAQ:AAPL. Treasury 10-year yield 4.250 percent. Revenue USD 1.20 million and $7. No default may occur; 1e999%."


def authored(text):
    return {"source_id": "profiles", "title": "GIG-authored profile fixture",
            "license": "GIG-AUTHORED", "text": text}


class ProfileTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)/"state"
        self.corpus = FinancialCorpus(self.root, clock=lambda: T0)

    def admit(self, text=GOLDEN_TEXT):
        receipt = self.corpus.admit_authored(authored(text))
        return self.corpus.document(receipt["document_id"])["passages"]

    def count_snapshots(self):
        with sqlite3.connect(self.corpus.path) as db:
            return db.execute("SELECT COUNT(*) FROM snapshots").fetchone()[0]

    def test_golden_preprofile_v1_annotation_bytes_and_saved_identities(self):
        self.admit()
        cases = [({}, "5a64bda426e7853b3941ae078c91791c3271823db8292f26a31bd5c8abbf73bc"),
                 ({"extract_facts": True}, "ba99f82a5ebffc4273a45fe44b9ce16117270b4cdcddcd5a74f7c3f6da86e4cd")]
        for options, expected_id in cases:
            default = self.corpus.search("Revenue", T0, **options)
            explicit = self.corpus.search("Revenue", T0, nlp_profile="v1", **options)
            self.assertEqual(dumps(default), dumps(explicit))
            self.assertEqual(default["snapshot_id"], expected_id)
            self.assertNotIn("nlp_profile", default)
            if "financial_nlp" in default:
                self.assertEqual(digest(default["financial_nlp"]), "0bf44f89534b948c6aaa4deb2f9499471d1555082a5a88dd9b904f509d275fec")
            self.assertEqual(FinancialCorpus(self.root).replay(expected_id), default)

    def test_historical_hybrid_fixture_replays_original_identity_without_fit(self):
        # Captured from the actual pre-profile source 5de784c, NumPy2.3.5.
        # Retain that historical model/version rather than pretending a fresh
        # SVD fit on every supported NumPy version must have identical bytes.
        fixture = json.loads((Path(__file__).parent/"fixtures/financial_intelligence/preprofile-v1-hybrid.json").read_text())
        self.assertEqual(fixture["source_commit"], "5de784c003f6eef2e0d505b76b4dc4db4a7d7dbb")
        self.admit(fixture["source_text"])
        snapshot, model = fixture["snapshot"], fixture["model"]
        expected_id = "66365bd40d61a5c7ba659312af0165f3a37a1eb80a01e2aeec6e90c379e59aec"
        self.assertEqual(snapshot["snapshot_id"], expected_id)
        body = {k: v for k, v in snapshot.items() if k != "snapshot_id"}
        self.assertEqual(digest(body), expected_id)
        self.assertEqual(digest(model), snapshot["model_id"])
        with sqlite3.connect(self.corpus.path) as db:
            db.execute("INSERT INTO latent_models VALUES(?,?)", (snapshot["model_id"], dumps(model)))
            db.execute("INSERT INTO snapshots VALUES(?,?)", (expected_id, dumps(body)))
        with patch.object(latent, "fit", side_effect=AssertionError("historical replay must not fit")):
            self.assertEqual(FinancialCorpus(self.root).replay(expected_id), snapshot)
        self.assertEqual(snapshot["model"]["numpy_version"], "2.3.5")

    def test_current_hybrid_fit_preserves_v1_output_and_dependency_provenance(self):
        import numpy as np
        self.admit()
        options = {"retrieval_mode": "hybrid", "extract_facts": True, "latent_dimensions": 2}
        default = self.corpus.search("Revenue", T0, **options)
        explicit = self.corpus.search("Revenue", T0, nlp_profile="v1", **options)
        self.assertEqual(dumps(default), dumps(explicit))
        self.assertNotIn("nlp_profile", default)
        self.assertEqual(default["model"]["numpy_version"], np.__version__)
        self.assertEqual(digest(default["financial_nlp"]), "0bf44f89534b948c6aaa4deb2f9499471d1555082a5a88dd9b904f509d275fec")
        with patch.object(latent, "fit", side_effect=AssertionError("replay must not fit")):
            self.assertEqual(FinancialCorpus(self.root).replay(default["snapshot_id"]), default)

    def test_legacy_replay_explicitly_selects_v1_despite_future_default(self):
        self.admit()
        result = self.corpus.search("Revenue", T0, extract_facts=True)
        # Simulate a future dispatcher whose unqualified extraction means v2.
        def future_extract(passages, *, profile="v2"):
            self.assertEqual(profile, "v1")
            return nlp_v1.extract(passages)
        with patch.object(nlp, "extract", side_effect=future_extract), patch.object(nlp_v2, "extract", side_effect=AssertionError("latest parser used")):
            self.assertEqual(FinancialCorpus(self.root).replay(result["snapshot_id"]), result)

    def test_v2_saved_profile_replays_without_refit_after_default_change(self):
        self.admit()
        result = self.corpus.search("Revenue", T0, retrieval_mode="hybrid", latent_dimensions=2,
                                    extract_facts=True, nlp_profile="v2")
        self.assertEqual(result["nlp_profile"], "v2")
        self.assertEqual(result["financial_nlp"]["implementation"], nlp_v2.IMPLEMENTATION)
        def future_extract(passages, *, profile="v1"):
            self.assertEqual(profile, "v2")
            return nlp_v2.extract(passages)
        with patch.object(nlp, "extract", side_effect=future_extract), patch.object(latent, "fit", side_effect=AssertionError("refit")):
            self.assertEqual(FinancialCorpus(self.root).replay(result["snapshot_id"]), result)
        self.assertEqual(self.count_snapshots(), 1)

    def test_unicode_quotes_exact_source_bytes_share_classes_and_unknown_authority(self):
        text = 'Café “NYSE:BRK.B” «NASDAQ:ZZZZZ.Z» and (NYSE: A). Treasury may default; revenue $7.'
        passages = self.admit(text)
        result = nlp.extract(passages, profile="v2")
        symbols = [m for m in result["entity_mentions"] if "symbol_mention" in m]
        self.assertEqual([m["mention"] for m in symbols], ["NYSE:BRK.B", "NASDAQ:ZZZZZ.Z", "NYSE: A"])
        self.assertEqual([m["share_class_mention"] for m in symbols], ["B", "Z", None])
        for mention in symbols:
            for state in ("instrument_identity_state", "issuer_linking_state", "exchange_membership_state", "existence_state"):
                self.assertEqual(mention[state], "UNKNOWN")
            self.assertEqual(mention["link_state"], "UNKNOWN_NO_ISSUER_AUTHORITY")
            evidence = mention["provenance"]
            self.assertEqual(text.encode()[evidence["source_start"]:evidence["source_end"]].decode(), mention["mention"])
            self.assertEqual(evidence["source_quote"], mention["mention"])
            self.assertEqual(evidence["evidence_kind"], "EXACT_SOURCE_BYTES")
        self.assertEqual(result["event_mentions"][0]["occurrence"], "UNKNOWN")
        self.assertEqual(result["sentiment_state"], "UNAVAILABLE")
        self.assertEqual(result["facts"][0]["currency_state"], "UNKNOWN")

    def test_complete_unsupported_candidates_never_become_partial_tickers(self):
        candidates = ["NYSE:BRK.BB", "NYSE:BRK.B/WS", "NYSE:AAPL-P", "NYSE:ABCDEFG",
                      "NASDAQ:AAPL.WS", "NYSE:AAPL_1", "NYSE:AAPL:B", "NYSE:AAPL+X",
                      "NYSE:ＡAPL", "NYSE:AΑPL", "NYSE:AAPL\u200bX", "NYSE:AAPL\u0301",
                      "NYSE:"+"A"*100, "NYSE:AAPL.."]
        passages = self.admit("Candidates "+"; ".join(candidates)+"; safe NYSE:BRK.B.")
        result = nlp.extract(passages, profile="v2")
        self.assertEqual([m["mention"] for m in result["entity_mentions"]], ["NYSE:BRK.B"])
        self.assertEqual(len(result["unsupported_mentions"]), len(candidates))
        self.assertEqual([m["provenance"]["surface_text"] for m in result["unsupported_mentions"]],
                         candidates[:-1]+["NYSE:AAPL."])
        self.assertIn("QUALIFIED_SYMBOL_BUDGET_EXCEEDED", {m["reason"] for m in result["unsupported_mentions"]})

    def test_namespace_lookalikes_embedded_tokens_and_other_namespaces_not_linked(self):
        passages = self.admit("Candidates NYSΕ:AAPL NАSDAQ:AAPL OTC:AAPL XNYSE:AAPL ＸNYSE:AAPL\nUnqualified AAPL BRK.B and safe NASDAQ:AAPL")
        result = nlp.extract(passages, profile="v2")
        self.assertEqual([m["mention"] for m in result["entity_mentions"]], ["NASDAQ:AAPL"])

    def test_v2_retains_frozen_numeric_currency_tenor_and_event_rules(self):
        text = "Revenue 1e999999% and $1e999999; 1.234,56 percent; NaN percent; USD Infinity; "+"9"*100+" bps; 1/2%; $1_000; $1abc; 1 / 2%; $1 / 2; $1⁄2; −7%. Safe 2.50 percent; $1,000; USD 7; US$8 million. Treasury 10-year yield 4.250 percent. No default may occur."
        passages = self.admit(text)
        v1, v2 = nlp.extract(passages), nlp.extract(passages, profile="v2")
        for category in ("facts", "tenor_mentions", "event_mentions", "unsupported_mentions", "sentiment_state", "issuer_linking_state"):
            self.assertEqual(dumps(v1[category]), dumps(v2[category]))
        self.assertEqual(len(v2["unsupported_mentions"]), 12)
        self.assertEqual({f["unit"] for f in v2["facts"]}, {"PERCENT", "USD", "UNSPECIFIED_DOLLAR"})

    def test_strict_profile_schema_rejected_before_persistence_even_empty_corpus(self):
        for profile in (None, True, [], {}, 1, "V2", "v3", "v2 ", "latest", "gig.financial-mention-rules.v2"):
            with self.subTest(profile=profile), self.assertRaises(ValueError):
                self.corpus.search("Revenue", T0, extract_facts=True, nlp_profile=profile)
        with self.assertRaises(ValueError):
            self.corpus.search("Revenue", T0, nlp_profile="v2")
        self.assertFalse(self.corpus.path.exists())
        self.admit()
        for options in ({"nlp_profile": "v2"}, {"nlp_profile": "v2", "extract_facts": "true"},
                        {"nlp_profile": "v2", "extract_facts": True, "limit": 21},
                        {"nlp_profile": "v2", "extract_facts": True, "query": "x"*513}):
            with self.assertRaises(ValueError):
                self.corpus.search(options.get("query", "Revenue"), T0, **{k: v for k, v in options.items() if k != "query"})
        self.assertEqual(self.count_snapshots(), 0)

    def test_both_profiles_bound_extraction_resources(self):
        passages = self.admit()
        for profile in ("v1", "v2"):
            for value in (None, passages*21, [{**passages[0], "text": "A"*4097}]):
                with self.subTest(profile=profile), self.assertRaises(ValueError):
                    nlp.extract(value, profile=profile)

    def test_unknown_stored_profile_fails_closed_without_rewriting_snapshot(self):
        self.admit()
        result = self.corpus.search("Revenue", T0, extract_facts=True, nlp_profile="v2")
        body = {k: v for k, v in result.items() if k != "snapshot_id"}
        body["nlp_profile"] = "v3"
        identity = digest(body)
        with sqlite3.connect(self.corpus.path) as db:
            db.execute("INSERT INTO snapshots VALUES(?,?)", (identity, dumps(body)))
        with self.assertRaises(ValueError):
            FinancialCorpus(self.root).replay(identity)
        self.assertEqual(self.count_snapshots(), 2)


class ProfileHTTPTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        async def heartbeat():
            while True:
                await asyncio.sleep(.01)
        self.heartbeat = asyncio.create_task(heartbeat())

    async def asyncTearDown(self):
        self.heartbeat.cancel()
        await asyncio.gather(self.heartbeat, return_exceptions=True)

    async def test_profile_api_restart_and_strict_invalid_requests(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            endpoint = "/api/v1/workspace/financial-intelligence"
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=create_app(state_dir=root)), base_url="http://test") as client:
                admitted = await client.post(endpoint, json={"operation": "admit-authored", **authored(GOLDEN_TEXT)})
                self.assertEqual(admitted.status_code, 200, admitted.text)
                at = admitted.json()["provenance"]["as_of"]
                payload = {"operation": "answer", "query": "Revenue", "as_of": at, "extract_facts": True}
                default = await client.post(endpoint, json=payload)
                explicit = await client.post(endpoint, json={**payload, "nlp_profile": "v1"})
                self.assertEqual(default.json(), explicit.json())
                self.assertNotIn("nlp_profile", default.json()["data"])
                response = await client.post(endpoint, json={**payload, "nlp_profile": "v2"})
                self.assertEqual(response.status_code, 200, response.text)
                self.assertEqual(response.json()["data"]["nlp_profile"], "v2")
                for bad in ({"nlp_profile": None}, {"nlp_profile": []}, {"nlp_profile": True}, {"nlp_profile": "v3"},
                            {"nlp_profile": "v2", "extract_facts": False}, {"nlp_profile": "v2", "extract_facts": "true"},
                            {"nlp_profile": "v2", "weights_url": "https://example.invalid/model"}):
                    rejected = await client.post(endpoint, json={**payload, **bad})
                    self.assertEqual(rejected.status_code, 422, rejected.text)
                rejected = await client.post(endpoint, json={"operation": "replay", "snapshot_id": response.json()["data"]["snapshot_id"], "nlp_profile": "v1"})
                self.assertEqual(rejected.status_code, 422, rejected.text)
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=create_app(state_dir=root)), base_url="http://test") as client:
                for original in (default, response):
                    replay = await client.post(endpoint, json={"operation": "replay", "snapshot_id": original.json()["data"]["snapshot_id"]})
                    self.assertEqual(replay.json(), original.json())
