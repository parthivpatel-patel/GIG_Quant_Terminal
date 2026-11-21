"""Read-only byte provenance audits: immutable query binding, never truth."""
from __future__ import annotations

import asyncio
import hashlib
import sqlite3
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

import httpx

from gig.financial_intelligence.citations import verify_citation
from gig.financial_intelligence.reference import treasury_document
from gig.financial_intelligence.store import FinancialCorpus, digest
from gig.service.app import create_app
from gig.service.financial_workspace import FinancialIntelligenceWorkspace
from test_financial_intelligence import T0, T1, authored


class CitationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)/"state"
        self.time = T0
        self.corpus = FinancialCorpus(self.root, clock=lambda: self.time)

    def saved(self, *, mode="bm25"):
        self.text = "Café revenue USD 7 million. NaN remains original.\n<script>window.citationInjected=1</script>"
        self.corpus.admit_authored(authored(self.text))
        return self.corpus.search("revenue", self.time, extract_facts=True, nlp_profile="v2", retrieval_mode=mode,
                                  **({"latent_dimensions": 2} if mode != "bm25" else {}))

    def test_unicode_original_bytes_hashes_scope_and_read_only_connections(self):
        result = self.saved()
        original_connect = self.corpus._connect
        def read_only(*, write=False):
            self.assertFalse(write)
            return original_connect(write=write)
        with patch.object(self.corpus, "_connect", side_effect=read_only):
            proof = verify_citation(self.corpus, result["snapshot_id"], result["rows"][0]["passage_id"])
        quote = self.text.encode()[proof["source_start"]:proof["source_end"]]
        self.assertEqual(proof["source_excerpt"].encode(), quote)
        self.assertEqual(proof["quoted_bytes_sha256"], hashlib.sha256(quote).hexdigest())
        self.assertEqual(proof["source_bytes_sha256"], hashlib.sha256(self.text.encode()).hexdigest())
        self.assertEqual(proof["verification_id"], digest({k: v for k, v in proof.items() if k != "verification_id"}))
        self.assertTrue(proof["direct_text_byte_match"])
        self.assertEqual(proof["content_truth_state"], "UNKNOWN")
        self.assertEqual(proof["issuer_identity_state"], "UNKNOWN")
        self.assertEqual(proof["freshness_state"], "NOT_ASSESSED")
        self.assertEqual(proof["signature_state"], "UNSIGNED_LOCAL_HASH_CHECK")
        self.assertFalse(proof["certified"])
        self.assertFalse(proof["new_state_written"])
        self.assertFalse(proof["trading_approved"])
        self.assertEqual(proof["publication_state"], "UNKNOWN")
        self.assertIsNone(proof["publication_time"])

    def test_foreign_passage_binding_and_bad_identities_are_rejected_without_state(self):
        result = self.saved()
        own = result["rows"][0]["passage_id"]
        foreign = self.corpus.document(result["rows"][0]["document_id"])["passages"][1]["passage_id"]
        with self.assertRaisesRegex(ValueError, "selected saved query"):
            verify_citation(self.corpus, result["snapshot_id"], foreign)
        empty = FinancialCorpus(self.root/"missing")
        for bad in [None, True, "../source", "A"*64, "x"*65]:
            with self.assertRaises(ValueError):
                verify_citation(empty, bad, own)
            with self.assertRaises(ValueError):
                verify_citation(empty, result["snapshot_id"], bad)
        with self.assertRaises(FileNotFoundError):
            verify_citation(empty, "0"*64, "0"*64)
        self.assertFalse(empty.directory.exists())

    def test_later_revisions_restart_and_caller_mutation_cannot_rewrite_proof(self):
        result = self.saved()
        snapshot, passage = result["snapshot_id"], result["rows"][0]["passage_id"]
        proof = verify_citation(self.corpus, snapshot, passage)
        self.time = T1
        self.corpus.admit_authored(authored("Revenue USD 999 million; corrected authored revision."))
        fresh = self.corpus.search("revenue", T1)
        self.assertNotEqual(fresh["rows"][0]["document_id"], proof["document_id"])
        restarted = FinancialCorpus(self.root)
        self.assertEqual(verify_citation(restarted, snapshot, passage), proof)
        proof["source_excerpt"] = "caller mutation"
        proof["content_truth_state"] = "FAKE"
        again = verify_citation(restarted, snapshot, passage)
        self.assertIn("NaN", again["source_excerpt"])
        self.assertEqual(again["content_truth_state"], "UNKNOWN")

    def test_hybrid_saved_model_is_verified_without_refitting_or_writing(self):
        result = self.saved(mode="hybrid")
        original_connect = self.corpus._connect
        def read_only(*, write=False):
            self.assertFalse(write)
            return original_connect(write=write)
        with patch("gig.financial_intelligence.latent.fit", side_effect=AssertionError("no refit")), patch.object(self.corpus, "_connect", side_effect=read_only):
            proof = verify_citation(self.corpus, result["snapshot_id"], result["rows"][0]["passage_id"])
        self.assertEqual(proof["retrieval_model_id"], result["model_id"])
        self.assertEqual(proof["annotation_profile"], "v2")

    def test_original_source_tampering_fails_closed(self):
        result = self.saved()
        with sqlite3.connect(self.corpus.path) as db:
            db.execute("UPDATE documents SET raw=?", (b"tampered source",))
        with self.assertRaisesRegex(ValueError, "integrity"):
            verify_citation(self.corpus, result["snapshot_id"], result["rows"][0]["passage_id"])

    def test_normalized_xml_rendering_is_not_claimed_as_direct_text_bytes(self):
        self.time = "2026-11-20T00:00:00Z"
        self.corpus.admit(*treasury_document())
        result = self.corpus.search("BC_10YEAR", self.time, limit=1)
        proof = verify_citation(self.corpus, result["snapshot_id"], result["rows"][0]["passage_id"])
        self.assertEqual(proof["classification"], "HISTORICAL")
        self.assertFalse(proof["direct_text_byte_match"])
        self.assertNotEqual(proof["normalized_passage"], proof["source_excerpt"])
        self.assertEqual(proof["source_license"], "CC0-1.0")
        self.assertEqual(proof["publication_state"], "UNKNOWN")

    def test_concurrent_readers_return_identical_proofs_without_new_snapshots(self):
        result = self.saved()
        payload = result["snapshot_id"], result["rows"][0]["passage_id"]
        with ThreadPoolExecutor(max_workers=4) as pool:
            reports = list(pool.map(lambda _: verify_citation(FinancialCorpus(self.root), *payload), range(8)))
        self.assertTrue(all(r == reports[0] for r in reports))
        with sqlite3.connect(self.corpus.path) as db:
            self.assertEqual(db.execute("SELECT COUNT(*) FROM snapshots").fetchone()[0], 1)


class CitationHTTPTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        async def heartbeat():
            while True:
                await asyncio.sleep(.01)
        self.heartbeat = asyncio.create_task(heartbeat())

    async def asyncTearDown(self):
        self.heartbeat.cancel()
        await asyncio.gather(self.heartbeat, return_exceptions=True)

    async def test_explicit_closed_http_audit_contract_and_restart(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            workspace = FinancialIntelligenceWorkspace(root)
            workspace.corpus.clock = lambda: T0
            workspace.run({"operation": "admit-authored", **authored("Café revenue USD 7 million. NaN.")})
            saved = workspace.run({"operation": "answer", "query": "revenue", "as_of": T0}).to_dict()["data"]
            payload = {"operation": "verify-citation", "snapshot_id": saved["snapshot_id"], "passage_id": saved["rows"][0]["passage_id"]}
            endpoint = "/api/v1/workspace/financial-intelligence"
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=create_app(state_dir=root)), base_url="http://test") as client:
                response = await client.post(endpoint, json=payload)
                self.assertEqual(response.status_code, 200, response.text)
                self.assertEqual(response.json()["state"], "AVAILABLE")
                self.assertFalse(response.json()["data"]["new_state_written"])
                for extra in [{"source_url": "https://example.invalid"}, {"certified": True}, {"source_excerpt": "forged"}, {"nlp_profile": "v2"}]:
                    rejected = await client.post(endpoint, json={**payload, **extra})
                    self.assertEqual(rejected.status_code, 422, rejected.text)
                unknown = await client.post(endpoint, json={**payload, "snapshot_id": "0"*64})
                self.assertEqual(unknown.json()["state"], "UNAVAILABLE")
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=create_app(state_dir=root)), base_url="http://test") as client:
                self.assertEqual((await client.post(endpoint, json=payload)).json(), response.json())
                self.assertFalse((root/"execution-engine").exists())
