"""Temporal cited retrieval, financial-source integrity and atomic durability."""
from __future__ import annotations

import asyncio
import math
import sqlite3
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

import httpx

from gig.financial_intelligence.reference import SHA256, treasury_document
from gig.financial_intelligence.retrieval import rank
from gig.financial_intelligence.store import FinancialCorpus
from gig.service.app import create_app
from gig.service.financial_workspace import FinancialIntelligenceWorkspace

T0 = "2026-11-13T21:00:00.000000001Z"
T1 = "2026-11-13T21:00:00.000000002Z"


def authored(text="Revenue rose10percent. NaN is original text.", **extra):
    return {"source_id": "authored-report", "title": "Authored financial protocol fixture",
            "text": text, "license": "GIG-AUTHORED", **extra}


class BM25Tests(unittest.TestCase):
    def test_independent_bm25_reference_and_deterministic_ties(self):
        passages = [{"text": "yield yield bonds", "document_id": "a", "ordinal": 0},
                    {"text": "yield cash", "document_id": "b", "ordinal": 0},
                    {"text": "cash flows", "document_id": "c", "ordinal": 0}]
        result, _ = rank(passages, "yield", 3)
        idf = math.log(1+(3-2+.5)/(2+.5))
        expected = idf*(2*2.2)/(2+1.2*(.25+.75*3/(7/3)))
        self.assertAlmostEqual(result[0]["score"], expected, places=14)
        self.assertEqual(result[0]["document_id"], "a")
        tied = [{"text": "cash", "document_id": "b", "ordinal": 0}, {"text": "cash", "document_id": "a", "ordinal": 0}]
        self.assertEqual([p["document_id"] for p in rank(tied, "cash", 2)[0]], ["a", "b"])

    def test_financial_terms_expansion_is_explicit_not_semantic(self):
        p = [{"text": "BC_10YEAR yield 3.79 percent on2023-01-03", "document_id": "a", "ordinal": 0}]
        self.assertEqual(rank(p, "ten-year", 1)[0], [])
        self.assertEqual(len(rank(p, "ten-year", 1, expand=True)[0]), 1)
        for query, limit, expand in [(" ", 5, False), ("a"*513, 5, False), ("a", True, False), ("a", 5, 1)]:
            with self.assertRaises(ValueError):
                rank(p, query, limit, expand=expand)


class FinancialCorpusTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)/"state"
        self.time = T0
        self.corpus = FinancialCorpus(self.root, clock=lambda: self.time)

    def count(self, table):
        with sqlite3.connect(self.corpus.path) as db:
            return db.execute("SELECT COUNT(*) FROM "+table).fetchone()[0]

    def test_missing_reads_and_rejected_inputs_create_no_state(self):
        self.assertIsNone(self.corpus.search("cash", T0))
        with self.assertRaises(FileNotFoundError):
            self.corpus.replay("0"*64)
        for request in (authored(license="UNVERIFIED"), authored(source_id="../../escape"), authored(text="x"*32769), authored(text="")):
            with self.assertRaises(ValueError):
                self.corpus.admit_authored(request)
        self.assertFalse(self.root.exists())

    def test_exact_source_text_citation_and_extract_only_answer(self):
        text = "Revenue grew. Original NaN substring remains.\n<script>alert('not executed')</script> café"
        metadata = self.corpus.admit_authored(authored(text))
        result = self.corpus.search("NaN", T0, answer=True)
        quote = result["rows"][0]
        self.assertIn("NaN", result["answer"])
        self.assertEqual(result["answer"], quote["text"])
        self.assertEqual(quote["source_excerpt"], text.encode()[quote["source_start"]:quote["source_end"]].decode())
        self.assertEqual(result["semantic_state"], "UNAVAILABLE")
        self.assertEqual(result["generative_state"], "UNAVAILABLE")
        self.assertEqual(result["classification"], "SYNTHETIC")
        self.assertFalse(result["trading_approved"])
        self.assertEqual(metadata["publication_state"], "UNKNOWN")
        self.assertEqual(self.corpus.search("nonexistent", T0, answer=True)["answer_state"], "UNAVAILABLE")

    def test_revision_knowledge_and_future_corpus_never_change_old_scores(self):
        self.corpus.admit_authored(authored("Cash profit profit growth"))
        before = self.corpus.search("profit", T0, answer=True)
        self.time = T1
        self.corpus.admit_authored(authored("Cash loss loss"))
        # One-nanosecond separation is preserved, not silently truncated.
        self.assertEqual(self.corpus.search("profit", T0, answer=True), before)
        after = self.corpus.search("profit", T1, answer=True)
        self.assertEqual(after["rows"], [])
        reopened = FinancialCorpus(self.root)
        self.assertEqual(reopened.replay(before["snapshot_id"]), before)
        self.assertNotEqual(before["corpus_identity"], after["corpus_identity"])

    def test_new_document_idf_and_restart_replay_are_point_in_time(self):
        self.corpus.admit_authored(authored("cash cash revenue"))
        before = self.corpus.search("cash", T0)
        self.time = T1
        self.corpus.admit_authored(authored("cash "*100, source_id="future-report"))
        self.assertEqual(self.corpus.search("cash", T0), before)
        self.assertEqual(FinancialCorpus(self.root).replay(before["snapshot_id"]), before)
        self.assertEqual(len(self.corpus.search("cash", T1)["corpus_document_ids"]), 2)

    def test_idempotence_concurrent_duplicates_and_backward_clock(self):
        with ThreadPoolExecutor(max_workers=4) as pool:
            results = list(pool.map(lambda _: self.corpus.admit_authored(authored()), range(8)))
        self.assertTrue(all(r == results[0] for r in results))
        self.assertEqual(self.count("documents"), 1)
        self.assertEqual(self.count("passages"), 1)
        self.time = "2026-11-13T20:00:00Z"
        with self.assertRaises(ValueError):
            self.corpus.admit_authored(authored("new revision"))
        self.assertEqual(self.count("documents"), 1)

    def test_atomic_failure_does_not_publish_a_document(self):
        with patch.object(self.corpus, "_insert_passages", side_effect=RuntimeError("injected")), self.assertRaises(RuntimeError):
            self.corpus.admit_authored(authored())
        self.assertEqual(self.count("documents"), 0)
        self.assertEqual(self.count("passages"), 0)
        self.assertIsNone(self.corpus.search("revenue", T0))
        self.corpus.admit_authored(authored())
        self.assertEqual(self.count("documents"), 1)

    def test_tampered_passage_document_and_snapshot_fail_closed(self):
        metadata = self.corpus.admit_authored(authored())
        original = self.corpus.search("Revenue", T0)
        with sqlite3.connect(self.corpus.path) as db:
            db.execute("UPDATE passages SET body='{}'")
        with self.assertRaises(ValueError):
            self.corpus.replay(original["snapshot_id"])
        with sqlite3.connect(self.corpus.path) as db:
            db.execute("UPDATE documents SET raw=?", (b"changed bytes",))
        with self.assertRaises(ValueError):
            self.corpus.document(metadata["document_id"])
        with sqlite3.connect(self.corpus.path) as db:
            db.execute("UPDATE snapshots SET body='{}'")
        with self.assertRaises(ValueError):
            self.corpus.replay(original["snapshot_id"])

    def test_reference_is_permitted_original_bytes_with_unknown_publication(self):
        raw, metadata, passages = treasury_document()
        self.assertEqual(metadata["raw_sha256"], SHA256)
        self.assertEqual(metadata["license"], "CC0-1.0")
        self.assertEqual(metadata["publication_state"], "UNKNOWN")
        self.assertEqual(len(passages), 250)
        receipt = self.corpus.admit(raw, metadata, passages)
        self.assertEqual(receipt, self.corpus.admit(raw, metadata, passages))
        self.assertIsNone(self.corpus.search("yield", "2023-12-31T00:00:00Z"))
        result = self.corpus.search("10-year Treasury yield 2023-01-03", T0, answer=True, expand=True)
        top = result["rows"][0]
        self.assertEqual(top["observation_date"], "2023-01-03")
        self.assertIn("BC_10YEAR yield 3.79 percent", result["answer"])
        self.assertIn("<d:BC_10YEAR", top["source_excerpt"])
        self.assertEqual(top["source_excerpt"].encode(), raw[top["source_start"]:top["source_end"]])

    def test_symlink_and_unknown_http_fields_cannot_choose_files_or_urls(self):
        destination = Path(self.temporary.name)/"destination"
        destination.mkdir()
        self.root.symlink_to(destination, target_is_directory=True)
        with self.assertRaises(ValueError):
            FinancialCorpus(self.root)
        directory = Path(self.temporary.name)/"fresh"
        workspace = FinancialIntelligenceWorkspace(directory)
        with self.assertRaises(ValueError):
            workspace.run({"operation": "ingest-reference", "url": "http://localhost/private"})
        self.assertFalse(directory.exists())


class FinancialHTTPTests(unittest.IsolatedAsyncioTestCase):
    async def test_reference_search_citations_exact_replay_and_origin_boundary(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            async def clock():
                while True:
                    await asyncio.sleep(.01)
            heartbeat = asyncio.create_task(clock())
            try:
                async with httpx.AsyncClient(transport=httpx.ASGITransport(app=create_app(state_dir=base)), base_url="http://test") as client:
                    endpoint = "/api/v1/workspace/financial-intelligence"
                    rejected = await client.post(endpoint, json={"operation": "ingest-reference"}, headers={"Origin": "https://hostile.invalid"})
                    self.assertEqual(rejected.status_code, 403)
                    response = await client.post(endpoint, json={"operation": "ingest-reference"})
                    self.assertEqual(response.status_code, 200, response.text)
                    at = response.json()["provenance"]["as_of"]
                    request = {"operation": "answer", "as_of": at, "query": "10-year Treasury yield 2023-01-03", "expand": True}
                    answer = await client.post(endpoint, json=request)
                    self.assertEqual(answer.status_code, 200, answer.text)
                    self.assertEqual(answer.json()["data"]["rows"][0]["observation_date"], "2023-01-03")
                async with httpx.AsyncClient(transport=httpx.ASGITransport(app=create_app(state_dir=base)), base_url="http://test") as client:
                    replay = await client.post(endpoint, json={"operation": "replay", "snapshot_id": answer.json()["data"]["snapshot_id"]})
                    self.assertEqual(replay.json(), answer.json())
                    bad = await client.post(endpoint, json={"operation": "search", "query": "yield", "as_of": at, "limit": True})
                    self.assertEqual(bad.status_code, 422)
                    self.assertFalse((base/"execution-engine").exists())
            finally:
                heartbeat.cancel()
