"""Source preservation, point-in-time attribution and offline NLP references."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np

from gig.nlp.workflow import (
    ArticleEvidence,
    LocalFinBERT,
    NewsStore,
    admit_article,
    analyze_articles,
    finbert_capability,
    sentiment_timeline,
)


def article(**changes):
    return {
        "source": "caller_research_notes",
        "article_id": "article-1",
        "url": "https://example.com/research/1",
        "license": "Caller owns this synthetic research example",
        "title": "Acme earnings beat expectations",
        "body": "Acme profit growth.\n  Quoted source: NaN + Infinity is not a price.\r\n",
        "published_at": "2026-01-02T12:00:00.123456789Z",
        "available_at": "2026-01-02T12:00:01Z",
        **changes,
    }


class NewsWorkflowTests(unittest.TestCase):
    def test_preserves_source_text_and_exact_hash(self):
        raw = article()
        evidence = admit_article(raw)
        row = evidence.as_dict()
        self.assertEqual(row["title"], raw["title"])
        self.assertEqual(row["body"], raw["body"])
        self.assertEqual(row["published_at"], "2026-01-02T12:00:00.123456789+00:00")
        self.assertEqual(hashlib.sha256(evidence.payload).hexdigest(), evidence.evidence_hash)
        self.assertEqual(row["source_permission"], "CALLER_DECLARED")

    def test_original_and_returned_mutation_cannot_change_evidence(self):
        raw = article()
        evidence = admit_article(raw)
        raw["body"] = "changed"
        returned = evidence.as_dict()
        returned["body"] = "changed again"
        self.assertIn("Quoted source", evidence.as_dict()["body"])

    def test_lexicon_attribution_and_dictionary_offsets(self):
        raw = article()
        result = analyze_articles([raw], as_of="2026-01-02T13:00:00Z", aliases={"Acme": "ACME"})
        row = result["articles"][0]
        self.assertEqual(row["sentiment"]["method"], "LEXICON")
        self.assertEqual(row["sentiment"]["score"], 1.0)
        self.assertEqual(row["linked_symbols"], ["ACME"])
        self.assertFalse(result["trading_approved"])
        combined = raw["title"] + "\n" + raw["body"]
        for match in row["entities"]:
            self.assertEqual(combined[match["start"] : match["end"]], match["matched_text"])
        for match in row["sentiment"]["positive_matches"]:
            self.assertEqual(combined[match["start"] : match["end"]], match["text"])
        self.assertIn("EARNINGS", [event["event"] for event in row["events"]])

    def test_entity_boundaries_prevent_substring_false_match(self):
        result = analyze_articles(
            [article(title="Acmeology", body="ACME announced")],
            as_of="2026-01-03T00:00:00Z",
            aliases={"Acme": "ACME"},
        )
        self.assertEqual(len(result["articles"][0]["entities"]), 1)

    def test_provider_symbols_remain_declared_evidence(self):
        result = analyze_articles([article(symbols=["FOO", "FOO"])], as_of="2026-01-03T00:00:00Z")
        row = result["articles"][0]
        self.assertEqual(row["symbols"], ["FOO"])
        self.assertEqual(row["linked_symbols"], ["FOO"])
        self.assertEqual(row["entities"], [])

    def test_exact_duplicates_preserve_sources_but_not_double_count(self):
        first = article()
        second = article(
            source="syndicated_notes", article_id="second", available_at="2026-01-02T12:00:02Z"
        )
        result = analyze_articles(
            [second, first], as_of="2026-01-03T00:00:00Z", aliases={"Acme": "ACME"}
        )
        self.assertEqual(len(result["articles"]), 2)
        self.assertEqual(result["unique_content_count"], 1)
        self.assertEqual(
            result["articles"][1]["duplicate_of"], result["articles"][0]["evidence_hash"]
        )
        timeline = sentiment_timeline(result, "ACME")
        self.assertEqual(timeline[0]["count"], 1)
        self.assertEqual(timeline[0]["date"], "2026-01-02")

    def test_revisions_preserved_but_only_latest_known_revision_aggregated(self):
        first = article(title="Acme earnings growth", body="profit growth")
        later = article(
            title="Acme loss warning", body="loss warning", available_at="2026-01-03T00:00:00Z"
        )
        earlier = analyze_articles([first, later], as_of="2026-01-02T13:00:00Z")
        self.assertEqual(earlier["mean_sentiment"], 1.0)
        result = analyze_articles([later, first], as_of="2026-01-04T00:00:00Z")
        self.assertEqual(len(result["articles"]), 2)
        self.assertEqual(result["unique_content_count"], 1)
        self.assertEqual(
            result["articles"][0]["superseded_by"], result["articles"][1]["evidence_hash"]
        )
        self.assertEqual(result["mean_sentiment"], -1.0)
        self.assertEqual(sentiment_timeline(result)[0]["date"], "2026-01-03")

    def test_exact_revision_replay_deduplicated_and_conflict_rejected(self):
        raw = article()
        result = analyze_articles([raw, raw], as_of="2026-01-04T00:00:00Z")
        self.assertEqual(len(result["articles"]), 1)
        self.assertEqual(result["replayed_articles"], 1)
        with self.assertRaisesRegex(ValueError, "conflicting article revision"):
            analyze_articles(
                [raw, article(body="conflicting correction")], as_of="2026-01-04T00:00:00Z"
            )

    def test_subnanosecond_source_timestamp_cannot_be_truncated(self):
        with self.assertRaisesRegex(ValueError, "precision"):
            admit_article(article(published_at="2026-01-02T12:00:00.1234567891Z"))

    def test_knowledge_time_avoids_publication_lookahead(self):
        result = analyze_articles(
            [article(available_at="2026-01-03T00:00:00Z")], as_of="2026-01-02T13:00:00Z"
        )
        self.assertEqual(result["state"], "UNAVAILABLE")
        self.assertIsNone(result["mean_sentiment"])
        self.assertEqual(result["future_articles_excluded"], 1)

    def test_unknown_event_and_no_entity_are_explicit(self):
        result = analyze_articles(
            [article(title="No financial keyword", body="A quiet day")],
            as_of="2026-01-03T00:00:00Z",
        )
        row = result["articles"][0]
        self.assertEqual(row["events"], [{"event": "UNCLASSIFIED", "matched_terms": []}])
        self.assertEqual(row["relevance"]["score"], 0)
        self.assertEqual(row["sentiment"]["score"], 0)

    def test_malformed_or_unlicensed_articles_rejected(self):
        for change in [
            {"license": ""},
            {"source": ""},
            {"title": "", "body": ""},
            {"body": "\ud800"},
            {"available_at": "2026-01-02T00:00:00Z"},
            {"published_at": "2026-01-02"},
            {"url": "javascript:alert(1)"},
            {"symbols": ["foo"]},
            {"symbols": "AAPL"},
        ]:
            with self.subTest(change=repr(change)), self.assertRaises(ValueError):
                admit_article(article(**change))

    def test_missing_and_unknown_fields_rejected(self):
        raw = article()
        del raw["license"]
        with self.assertRaises(ValueError):
            admit_article(raw)
        with self.assertRaises(ValueError):
            admit_article(article(api_key="secret"))

    def test_entity_dictionary_escapes_regex_aliases(self):
        raw = article(title="Acme (Class A) grows", body="")
        result = analyze_articles(
            [raw], as_of="2026-01-03T00:00:00Z", aliases={"Acme (Class A)": "ACME.A"}
        )
        self.assertEqual(result["articles"][0]["linked_symbols"], ["ACME.A"])

    def test_evidence_canonical_schema_cannot_be_forged_by_hash_alone(self):
        row = admit_article(article()).as_dict()
        row["content_hash"] = "wrong"
        payload = json.dumps(row).encode()
        evidence = ArticleEvidence(payload, hashlib.sha256(payload).hexdigest())
        with self.assertRaisesRegex(ValueError, "noncanonical"):
            analyze_articles([evidence], as_of="2026-01-03T00:00:00Z")

    def test_article_bound_enforced(self):
        with self.assertRaises(ValueError):
            analyze_articles((article() for _ in range(1001)), as_of="2026-01-03T00:00:00Z")
        with self.assertRaises(ValueError):
            admit_article(article(body="x" * 100001))
        with self.assertRaisesRegex(ValueError, "aggregate article text"):
            analyze_articles(
                (article(article_id=str(i), body="x" * 99900) for i in range(22)),
                as_of="2026-01-03T00:00:00Z",
            )

    def test_timeline_uses_known_at_date_and_symbol_filter(self):
        raw = article(available_at="2026-01-03T00:00:00Z", symbols=["ACME"])
        result = analyze_articles([raw], as_of="2026-01-04T00:00:00Z")
        self.assertEqual(sentiment_timeline(result, "ACME")[0]["date"], "2026-01-03")
        self.assertEqual(sentiment_timeline(result, "OTHER"), [])

    def test_local_finbert_unavailable_does_not_download_or_substitute(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "manifest"):
                LocalFinBERT(tmp)
            capability = finbert_capability(tmp)
            self.assertFalse(capability["downloads_enabled"])
            self.assertFalse(capability["load_verified"])
            self.assertNotEqual(capability["state"], "AVAILABLE")

    def test_local_finbert_manifest_identity_and_hash_required(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "finbert-provenance.json").write_text(
                json.dumps({"model_id": "other", "revision": "1", "files": {}})
            )
            with self.assertRaisesRegex(ValueError, "identity"):
                LocalFinBERT(tmp)
            (root / "config.json").write_text("{}")
            (root / "model.safetensors").write_bytes(b"not weights")
            (root / "finbert-provenance.json").write_text(
                json.dumps(
                    {
                        "model_id": "ProsusAI/finbert",
                        "revision": "1",
                        "files": {"config.json": "incorrect", "model.safetensors": "incorrect"},
                    }
                )
            )
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                LocalFinBERT(tmp)

    def test_legacy_finbert_helper_requires_explicit_local_checkpoint(self):
        from gig.nlp.sentiment import transformer_sentiment

        with patch.dict("sys.modules", {"transformers": None}):
            self.assertIsNone(transformer_sentiment("original source text"))
        with patch("gig.nlp.workflow.LocalFinBERT") as model:
            model.return_value.score.return_value = {"score": 0.25}
            self.assertEqual(
                transformer_sentiment("source\n NaN", local_directory="local-checkpoint"), 0.25
            )
            model.assert_called_once_with("local-checkpoint")
            model.return_value.score.assert_called_once_with("source\n NaN")
        with patch("gig.nlp.workflow.LocalFinBERT", side_effect=ImportError("missing local model")), self.assertRaises(ImportError):
            transformer_sentiment("text", local_directory="absent")

    def test_finbert_adapter_enforces_local_cpu_loading_and_declares_truncation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = {"config.json": b"{}", "model.safetensors": b"offline mock weights"}
            for name, content in files.items():
                (root / name).write_bytes(content)
            manifest = {
                "model_id": "ProsusAI/finbert",
                "revision": "mock-test-only",
                "files": {
                    name: hashlib.sha256(content).hexdigest() for name, content in files.items()
                },
            }
            (root / "finbert-provenance.json").write_text(json.dumps(manifest))
            tokenizer = MagicMock()
            tokenizer.encode.return_value = list(range(600))
            tokenizer.return_value = {"input_ids": "offline-test"}
            model = MagicMock()
            model.to.return_value = model
            model.eval.return_value = model
            model.config = SimpleNamespace(
                id2label={0: "positive", 1: "negative", 2: "neutral"}, model_type="bert"
            )
            model.return_value = SimpleNamespace(logits="offline mock logits")
            auto_model = MagicMock()
            auto_model.from_pretrained.return_value = model
            auto_tokenizer = MagicMock()
            auto_tokenizer.from_pretrained.return_value = tokenizer
            fake_torch = SimpleNamespace(
                inference_mode=nullcontext, softmax=lambda *a, **k: np.array([[0.6, 0.1, 0.3]])
            )
            fake_transformers = SimpleNamespace(
                AutoModelForSequenceClassification=auto_model, AutoTokenizer=auto_tokenizer
            )
            with patch.dict(
                "sys.modules", {"torch": fake_torch, "transformers": fake_transformers}
            ):
                adapter = LocalFinBERT(root)
                result = adapter.score("original\n source NaN")
            auto_model.from_pretrained.assert_called_once_with(
                str(root), local_files_only=True, use_safetensors=True, trust_remote_code=False
            )
            auto_tokenizer.from_pretrained.assert_called_once_with(
                str(root), local_files_only=True, trust_remote_code=False
            )
            model.to.assert_called_once_with("cpu")
            tokenizer.assert_called_once_with(
                "original\n source NaN", return_tensors="pt", truncation=True, max_length=512
            )
            self.assertEqual(result["score"], 0.5)
            self.assertTrue(result["model_input_truncated"])
            self.assertEqual(result["method"], "FINBERT_LOCAL")
            self.assertEqual(result["origin_evidence"], "CALLER_DECLARED_LOCAL")


class NewsStoreTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "news.sqlite"
        self.store = NewsStore(self.path)

    def test_restart_replay_and_revision_asof(self):
        self.assertEqual(self.store.ingest([article()]), {"inserted": 1, "replayed": 0})
        store = NewsStore(self.path)
        self.assertEqual(store.ingest([article()]), {"inserted": 0, "replayed": 1})
        store.ingest([article(title="Acme earnings miss", available_at="2026-01-03T12:00:00Z")])
        before = store.snapshot("2026-01-03T00:00:00Z")
        after = store.snapshot("2026-01-04T00:00:00Z")
        self.assertEqual(before["articles"][0]["title"], "Acme earnings beat expectations")
        self.assertEqual(after["articles"][0]["title"], "Acme earnings miss")
        self.assertEqual(len(after["articles"]), 1)

    def test_conflicting_revision_rolls_back_batch(self):
        self.store.ingest([article()])
        with self.assertRaisesRegex(ValueError, "conflicting"):
            self.store.ingest([article(article_id="new"), article(title="conflicting source")])
        result = self.store.snapshot("2026-01-04T00:00:00Z")
        self.assertEqual(len(result["articles"]), 1)

    def test_snapshot_hash_tampering_is_detected(self):
        self.store.ingest([article()])
        import sqlite3

        with sqlite3.connect(self.path) as connection:
            connection.execute("UPDATE articles SET payload=?", (b"{}",))
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            self.store.snapshot("2026-01-04T00:00:00Z")

    def test_metadata_cannot_diverge_from_immutable_evidence(self):
        self.store.ingest([article()])
        import sqlite3

        with sqlite3.connect(self.path) as connection:
            connection.execute("UPDATE articles SET identity='other'")
        with self.assertRaisesRegex(ValueError, "metadata"):
            self.store.snapshot("2026-01-04T00:00:00Z")


if __name__ == "__main__":
    unittest.main()
