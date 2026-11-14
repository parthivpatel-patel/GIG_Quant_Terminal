"""Independent numerical references, temporal fitting, replay and cited rule mentions."""
from __future__ import annotations

import asyncio
import math
import sqlite3
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest.mock import patch

import httpx
import numpy as np

from gig.financial_intelligence import latent
from gig.financial_intelligence.reference import treasury_document
from gig.financial_intelligence.retrieval import rank, tokens
from gig.financial_intelligence.store import FinancialCorpus
from gig.service.app import create_app
from gig.service.serialization import dumps

T0 = "2026-11-13T21:00:00.000000001Z"
T1 = "2026-11-13T21:00:00.000000002Z"
T2 = "2026-11-13T21:00:00.000000003Z"
TOPICS = [
    "Treasury bonds duration yield curve interest rate.",
    "Treasury duration interest rates policy tightening.",
    "Policy tightening lifts bond coupon financing costs.",
    "Revenue earnings profit operating margin sales.",
    "Sales profit margin income expenses earnings.",
    "Quarterly income expenses guidance revenue forecast.",
    "Liquidity cash reserves funding deposits.",
    "Deposits cash funding bank withdrawals.",
    "Bank withdrawals liquidity buffer shortfall.",
    "Credit default debt obligation issuer.",
    "Issuer debt downgrade spread credit.",
    "Downgrade spread default risk obligation.",
]


def authored(text, source_id="evaluation"):
    return {"source_id": source_id, "title": "GIG-authored synthetic retrieval evaluation",
            "license": "GIG-AUTHORED", "text": text}


class LatentTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)/"state"
        self.time = T0
        self.corpus = FinancialCorpus(self.root, clock=lambda: self.time)

    def topic_document(self):
        receipt = self.corpus.admit_authored(authored("\n".join(TOPICS)))
        return self.corpus.document(receipt["document_id"])["passages"]

    def test_independent_eigendecomposition_tfidf_cosine_and_rrf(self):
        passages = self.topic_document()
        response = self.corpus.search("coupon", T0, retrieval_mode="latent", latent_dimensions=4, limit=20)
        # Independent Gram eigendecomposition, not production SVD or saved vectors.
        words = sorted({w for p in passages for w in tokens(p["text"]) if any(c.isalpha() for c in w)})
        counts = [Counter(tokens(p["text"])) for p in passages]
        weights = [math.log((len(passages)+1)/(sum(w in c for c in counts)+1))+1 for w in words]
        matrix = np.array([[(math.log(c[w])+1)*weight if c[w] else 0 for w, weight in zip(words, weights, strict=True)] for c in counts])
        matrix /= np.sqrt((matrix**2).sum(axis=1))[:, None]
        eigenvalues, eigenvectors = np.linalg.eigh(matrix.T @ matrix)
        projection = eigenvectors[:, np.argsort(eigenvalues)[-4:]]
        document_vectors = matrix @ projection
        query_vector = np.array([weight if word == "coupon" else 0 for word, weight in zip(words, weights, strict=True)]) @ projection
        expected = document_vectors @ query_vector/(np.linalg.norm(document_vectors, axis=1)*np.linalg.norm(query_vector))
        for row in response["rows"]:
            self.assertAlmostEqual(row["score"], expected[row["ordinal"]], places=11)
        self.assertTrue(any("coupon" not in tokens(p["text"]) for p in response["rows"]))
        hybrid = self.corpus.search("coupon", T0, retrieval_mode="hybrid", latent_dimensions=4, limit=20)
        lexical, _ = rank(passages, "coupon", 20)
        def reference_ranks(rows):
            result, first = {}, 0
            while first < len(rows):
                group = [first]
                for position in range(first+1, len(rows)):
                    difference = abs(rows[position]["score"]-rows[first]["score"])
                    tolerance = max(1e-12, 1e-10*max(abs(rows[position]["score"]), abs(rows[first]["score"])))
                    if difference > tolerance:
                        break
                    group.append(position)
                for position in group:
                    result[rows[position]["ordinal"]] = sum(i+1 for i in group)/len(group)
                first = group[-1]+1
            return result
        bm25_rank = reference_ranks(lexical)
        dense_rank = reference_ranks(response["rows"])
        for row in hybrid["rows"]:
            score = sum(1/(60+ranks[row["ordinal"]]) for ranks in (bm25_rank, dense_rank) if row["ordinal"] in ranks)
            self.assertEqual(row["score"], score)
            self.assertEqual(row["bm25_rank"], bm25_rank.get(row["ordinal"]))
            self.assertEqual(row["latent_rank"], dense_rank.get(row["ordinal"]))

    def test_declared_authored_multitopic_evaluation_and_unseen_language_failure(self):
        self.topic_document()
        dense_hits, lexical_hits = 0, 0
        for query, relevant in [("coupon", {0, 1, 2}), ("sales", {3, 4, 5}),
                                ("reserves", {6, 7, 8}), ("downgrade", {9, 10, 11})]:
            dense = self.corpus.search(query, T0, retrieval_mode="latent", latent_dimensions=4, limit=3)
            lexical = self.corpus.search(query, T0, limit=3)
            dense_hits += sum(row["ordinal"] in relevant for row in dense["rows"])
            lexical_hits += sum(row["ordinal"] in relevant for row in lexical["rows"])
        self.assertEqual(dense_hits, 12)  # Hand-declared synthetic judgments, not a benchmark.
        self.assertEqual(lexical_hits, 6)
        unknown = self.corpus.search("insolvency", T0, retrieval_mode="latent", answer=True)
        self.assertEqual(unknown["rows"], [])
        self.assertEqual(unknown["latent_query_state"], "UNKNOWN_VOCABULARY")
        self.assertIsNone(unknown["answer"])
        self.assertEqual(unknown["answer_state"], "UNAVAILABLE")
        numeric = self.corpus.search("2023-01-03", T0, retrieval_mode="latent")
        self.assertEqual(numeric["rows"], [])
        self.assertIn("NO_PURE_NUMBERS_OR_DATES", numeric["model"]["feature_policy"])

    def test_future_revision_statistics_fit_and_saved_restart_replay(self):
        self.topic_document()
        before = self.corpus.search("coupon", T0, retrieval_mode="hybrid", latent_dimensions=4, extract_facts=True)
        self.time = T1
        self.corpus.admit_authored(authored("coupon coupon futureword\nfutureword debt", "future"))
        self.time = T2
        self.corpus.admit_authored(authored("replacement only futureword", "evaluation"))
        with patch.object(latent, "fit", side_effect=AssertionError("replay must use saved fit")):
            self.assertEqual(FinancialCorpus(self.root).replay(before["snapshot_id"]), before)
        self.assertEqual(self.corpus.search("coupon", T0, retrieval_mode="hybrid", latent_dimensions=4, extract_facts=True), before)
        after = self.corpus.search("coupon", T2, retrieval_mode="latent", latent_dimensions=4)
        self.assertNotEqual(after["model_id"], before["model_id"])
        with sqlite3.connect(self.corpus.path) as db:
            wire = db.execute("SELECT body FROM latent_models WHERE id=?", (before["model_id"],)).fetchone()[0]
        self.assertNotIn("futureword", wire)
        with self.assertRaises(ValueError):
            self.corpus.search("coupon", T0, retrieval_mode="latent", document_ids=after["corpus_document_ids"])

    def test_resource_limits_reject_without_truncation_bm25_still_works(self):
        self.corpus.admit_authored(authored("\n".join("bond cash line" for _ in range(257))))
        with self.assertRaises(ValueError):
            self.corpus.search("cash", T0, retrieval_mode="latent")
        self.assertTrue(self.corpus.search("cash", T0)["rows"])
        self.time = T1
        self.corpus.admit_authored(authored(" ".join("word"+str(i) for i in range(513))))
        with self.assertRaises(ValueError):
            self.corpus.search("word1", T1, retrieval_mode="hybrid")
        for settings in ({"retrieval_mode": []}, {"retrieval_mode": "neural"}, {"latent_dimensions": True},
                         {"retrieval_mode": "latent", "latent_dimensions": 17}, {"extract_facts": 1}):
            with self.assertRaises(ValueError):
                self.corpus.search("cash", T1, **settings)
        with sqlite3.connect(self.corpus.path) as db:
            self.assertEqual(db.execute("SELECT COUNT(*) FROM latent_models").fetchone()[0], 0)

    def test_atomic_model_and_snapshot_failure_and_tamper(self):
        self.topic_document()
        with sqlite3.connect(self.corpus.path) as db:
            db.execute("CREATE TRIGGER reject_snapshot BEFORE INSERT ON snapshots BEGIN SELECT RAISE(ABORT, 'injected'); END")
        with self.assertRaises(sqlite3.IntegrityError):
            self.corpus.search("coupon", T0, retrieval_mode="latent")
        with sqlite3.connect(self.corpus.path) as db:
            self.assertEqual(db.execute("SELECT COUNT(*) FROM latent_models").fetchone()[0], 0)
            self.assertEqual(db.execute("SELECT COUNT(*) FROM snapshots").fetchone()[0], 0)
            db.execute("DROP TRIGGER reject_snapshot")
        result = self.corpus.search("coupon", T0, retrieval_mode="latent")
        with sqlite3.connect(self.corpus.path) as db:
            db.execute("UPDATE latent_models SET body='{}'")
        with self.assertRaises(ValueError):
            FinancialCorpus(self.root).replay(result["snapshot_id"])

    def test_preexisting_bm25_snapshot_schema_and_identity_unchanged(self):
        self.topic_document()
        result = self.corpus.search("sales", T0)
        self.assertEqual(result["implementation"], "gig.lexical-bm25.v1")
        self.assertNotIn("retrieval_mode", result)
        self.assertNotIn("model_id", result)
        # Simulate a v1 database created before model storage existed.
        with sqlite3.connect(self.corpus.path) as db:
            db.execute("DROP TABLE latent_models")
        self.assertEqual(FinancialCorpus(self.root).replay(result["snapshot_id"]), result)

    def test_treasury_lsa_fit_and_numeric_lexical_hybrid_citations(self):
        raw, metadata, passages = treasury_document()
        self.corpus.admit(raw, metadata, passages)
        result = self.corpus.search("2023-01-03", T0, retrieval_mode="hybrid", extract_facts=True)
        self.assertEqual(result["rows"][0]["observation_date"], "2023-01-03")
        self.assertEqual(result["model"]["passage_count"], 250)
        self.assertLessEqual(result["model"]["vocabulary_count"], 512)
        self.assertEqual(result["latent_query_state"], "UNKNOWN_VOCABULARY")
        fact = next(f for f in result["financial_nlp"]["facts"] if f["tenor_months"] == "120")
        self.assertEqual(fact["decimal_value"], "3.79")
        evidence = fact["provenance"]
        self.assertEqual(raw[evidence["source_start"]:evidence["source_end"]].decode(), evidence["source_quote"])
        self.assertIn("<d:BC_10YEAR", evidence["source_quote"])
        self.assertEqual(FinancialCorpus(self.root).replay(result["snapshot_id"]), result)

    def test_hybrid_dense_ties_preserve_late_date_lexical_evidence(self):
        self.corpus.admit(*treasury_document())
        query = "10-year Treasury yield 2023-12-29"
        lexical = self.corpus.search(query, T0)
        hybrid = self.corpus.search(query, T0, retrieval_mode="hybrid", limit=20)
        self.assertEqual(lexical["rows"][0]["observation_date"], "2023-12-29")
        self.assertEqual(hybrid["rows"][0]["observation_date"], "2023-12-29")
        # All250indistinguishable dense scores share the mean rank125.5.
        self.assertEqual({r["latent_rank"] for r in hybrid["rows"]}, {125.5})
        self.assertEqual(hybrid["rows"][0]["bm25_rank"], 1)
        self.assertAlmostEqual(hybrid["rows"][0]["score"], 1/61+1/185.5, places=15)
        self.assertEqual(hybrid["fusion"]["tie_policy"]["method"], "FRACTIONAL_MEAN_RANK")
        self.assertEqual(FinancialCorpus(self.root).replay(hybrid["snapshot_id"]), hybrid)

    def test_floating_tie_groups_compare_anchor_without_chaining(self):
        rows = [{"score": 1.0}, {"score": 1.0-9e-11}, {"score": 1.0-18e-11}]
        self.assertEqual([r for _, r in latent._fractional_ranks(rows)], [1.5, 1.5, 3])

    def test_exact_decimal_unicode_byte_spans_entities_and_event_cues(self):
        text = "Café U.S. Treasury 10-year yield 4.250 percent. Revenue USD 1.20 million.\nFederal Reserve may raise rates by 25 bps if inflation rises.\nThe board did not declare a default; no downgrade occurred."
        self.corpus.admit_authored(authored(text))
        result = self.corpus.search("Treasury Federal board", T0, extract_facts=True)
        nlp = result["financial_nlp"]
        self.assertEqual({(f["decimal_value"], f["unit"]) for f in nlp["facts"]},
                         {("4.250", "PERCENT"), ("1200000.00", "USD"), ("25", "BASIS_POINTS")})
        self.assertIsNone(next(f for f in nlp["facts"] if f["unit"] == "USD")["tenor_months"])
        self.assertEqual(next(f for f in nlp["facts"] if f["unit"] == "PERCENT")["tenor_months"], "120")
        self.assertTrue(nlp["entity_mentions"])
        self.assertTrue(any(e["conditional_cue_present"] for e in nlp["event_mentions"]))
        self.assertTrue(any(e["negation_cue_present"] for e in nlp["event_mentions"]))
        for category in ("facts", "tenor_mentions", "entity_mentions", "event_mentions"):
            for mention in nlp[category]:
                provenance = mention["provenance"]
                self.assertEqual(text.encode()[provenance["source_start"]:provenance["source_end"]].decode(), provenance["source_quote"])
        self.assertTrue(all(e["occurrence"] == "UNKNOWN" for e in nlp["event_mentions"]))
        self.assertEqual(FinancialCorpus(self.root).replay(result["snapshot_id"]), result)

    def test_fraction_and_numeric_identifier_cannot_be_verified_subfragments(self):
        self.corpus.admit_authored(authored("Fraction 1/2%; unsupported cost $1_000 and $1abc; spaced 1 / 2%; spaced money $1 / 2; Unicode $1⁄2 and −7%. Safe $1,000 and 2% and 3 bps."))
        nlp = self.corpus.search("Fraction", T0, extract_facts=True)["financial_nlp"]
        self.assertEqual({(f["decimal_value"], f["unit"]) for f in nlp["facts"]},
                         {("1000", "UNSPECIFIED_DOLLAR"), ("2", "PERCENT"), ("3", "BASIS_POINTS")})
        self.assertEqual(len(nlp["unsupported_mentions"]), 6)

    def test_dollar_currency_identity_requires_explicit_source_syntax(self):
        text = "Café Treasury 10-year yield 4.25 percent. Revenue $100 million; revenue USD 100 million; revenue US$100 million; revenue US$ 1.20 billion."
        payload = authored(text)
        payload["title"] = "U.S. company USD report in English"
        self.corpus.admit_authored(payload)
        result = self.corpus.search("revenue", T0, extract_facts=True)
        facts = result["financial_nlp"]["facts"]
        bare = next(f for f in facts if f["currency_mention"] == "$")
        self.assertEqual(bare["unit"], "UNSPECIFIED_DOLLAR")
        self.assertEqual(bare["currency_state"], "UNKNOWN")
        self.assertEqual(bare["decimal_value"], "100000000")
        self.assertEqual(bare["quoted_decimal"], "100")
        self.assertEqual(bare["provenance"]["source_quote"], "$100 million")
        self.assertIsNone(bare["tenor_months"])
        explicit = [f for f in facts if f["currency_state"] == "EXPLICIT_SOURCE_SYNTAX"]
        self.assertEqual(len(explicit), 3)
        self.assertEqual({f["unit"] for f in explicit}, {"USD"})
        self.assertEqual([f["decimal_value"] for f in explicit], ["100000000", "100000000", "1200000000.00"])
        self.assertEqual(next(f for f in facts if f["unit"] == "PERCENT")["currency_state"], "NOT_APPLICABLE")
        for fact in facts:
            evidence = fact["provenance"]
            self.assertEqual(text.encode()[evidence["source_start"]:evidence["source_end"]].decode(), evidence["source_quote"])
            self.assertEqual(fact["truth_state"], "UNKNOWN")
        self.assertEqual(FinancialCorpus(self.root).replay(result["snapshot_id"]), result)

    def test_numeric_adversaries_never_parse_suffix_or_expand_exponents(self):
        text = "Revenue 1e999999% and $1e999999; 1.234,56 percent; NaN percent; USD Infinity; "+"9"*100+" bps; safe 2.50 percent."
        self.corpus.admit_authored(authored(text))
        nlp = self.corpus.search("Revenue", T0, extract_facts=True)["financial_nlp"]
        self.assertEqual([(f["decimal_value"], f["unit"]) for f in nlp["facts"]], [("2.50", "PERCENT")])
        self.assertEqual(len(nlp["unsupported_mentions"]), 6)
        self.assertEqual({m["reason"] for m in nlp["unsupported_mentions"]},
            {"EXPONENT_SYNTAX_UNSUPPORTED", "AMBIGUOUS_OR_UNSUPPORTED_NUMBER_SYNTAX", "NONFINITE_UNSUPPORTED", "DIGIT_BUDGET_EXCEEDED"})
        self.assertLess(len(dumps(nlp)), 15000)


class LatentHTTPTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        async def heartbeat():
            while True:
                await asyncio.sleep(.01)
        self.heartbeat = asyncio.create_task(heartbeat())

    async def asyncTearDown(self):
        self.heartbeat.cancel()
        await asyncio.gather(self.heartbeat, return_exceptions=True)

    async def test_hybrid_fact_http_restart_unknown_and_malformed_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            endpoint = "/api/v1/workspace/financial-intelligence"
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=create_app(state_dir=root)), base_url="http://test") as client:
                admitted = await client.post(endpoint, json={"operation": "admit-authored", **authored("Revenue USD 1.20 million.\nDebt default may occur if cash falls.")})
                self.assertEqual(admitted.status_code, 200, admitted.text)
                at = admitted.json()["provenance"]["as_of"]
                response = await client.post(endpoint, json={"operation": "answer", "query": "Revenue", "as_of": at,
                    "retrieval_mode": "hybrid", "latent_dimensions": 2, "extract_facts": True})
                self.assertEqual(response.status_code, 200, response.text)
                data = response.json()["data"]
                self.assertEqual(data["model"]["model_kind"], "CORPUS_LOCAL_LSA")
                self.assertIn("corpus-local LSA", data["title"])
                self.assertTrue(data["financial_nlp"]["facts"])
                self.assertFalse(data["trading_approved"])
                for bad in ({"retrieval_mode": []}, {"latent_dimensions": True}, {"extract_facts": "true"}, {"weights_url": "https://example.invalid/model"}):
                    rejected = await client.post(endpoint, json={"operation": "search", "query": "cash", "as_of": at, **bad})
                    self.assertEqual(rejected.status_code, 422, rejected.text)
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=create_app(state_dir=root)), base_url="http://test") as client:
                replayed = await client.post(endpoint, json={"operation": "replay", "snapshot_id": data["snapshot_id"]})
                self.assertEqual(replayed.json(), response.json())
                unknown = await client.post(endpoint, json={"operation": "answer", "query": "unseenword", "as_of": at, "retrieval_mode": "latent"})
                self.assertEqual(unknown.json()["data"]["answer_state"], "UNAVAILABLE")
                self.assertEqual(unknown.json()["data"]["retrieval_state"], "NO_MATCH")
            self.assertFalse((root/"execution-engine").exists())
