"""Actual pinned CPU FinBERT; weights remain outside the source repository."""

from __future__ import annotations

import importlib.util
import json
import os
import sqlite3
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from gig.nlp.model_assets import LICENSE, PINNED_REVISION, prepare_finbert_assets
from gig.nlp.workflow import LocalFinBERT, _finbert_cpu
from gig.service.contracts import EnvelopeSchema
from gig.service.sentiment_workspace import SentimentWorkspace

ASSETS = Path(
    os.environ.get("GIG_FINBERT_TEST_DIR", "/tmp/gig-finbert-assets/ProsusAI-finbert-0bc2ac10")
)
ACTUAL = (ASSETS / "finbert-provenance.json").is_file() and all(
    importlib.util.find_spec(name) is not None for name in ("torch", "transformers")
)
AS_OF = "2026-06-01T12:00:00.123456789Z"


def request():
    return {
        "as_of": AS_OF,
        "aliases": {"Acme": "SYN001"},
        "articles": [
            {
                "source": "authored-synthetic-finbert-fixture",
                "article_id": "one",
                "url": "",
                "license": "Authored synthetic research fixture",
                "title": "Acme earnings",
                "body": "  Acme reported record profit growth. NaN Labs\n",
                "published_at": "2026-06-01T11:00:00Z",
                "available_at": "2026-06-01T11:01:00Z",
            }
        ],
    }


class AssetAdmissionTests(unittest.TestCase):
    def test_download_needs_explicit_noncommercial_acknowledgement(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "noncommercial"):
                prepare_finbert_assets(Path(directory) / "missing")
            self.assertFalse((Path(directory) / "missing").exists())

    def test_model_path_symlinks_reject_before_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "target"
            target.mkdir()
            link = Path(directory) / "link"
            link.symlink_to(target, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "symbolic"):
                LocalFinBERT(link)

    def test_no_configured_checkpoint_is_not_a_lexicon_substitution(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = SentimentWorkspace(Path(directory) / "state", None)
            with self.assertRaises(ImportError):
                workspace.analyze(request())
            self.assertFalse((Path(directory) / "state").exists())


@unittest.skipUnless(ACTUAL, "Actual pinned FinBERT safetensors/runtime not provisioned locally")
class ActualFinBERTTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch

        cls.torch = torch
        cls.model = LocalFinBERT(ASSETS)

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.base = Path(self.temporary.name) / "state"
        self.workspace = SentimentWorkspace(self.base, ASSETS)
        # Reuse the genuine loaded checkpoint; no mocked predictions or weights.
        self.workspace._model = self.model

    def tearDown(self):
        self.temporary.cleanup()

    def test_real_weights_pinned_revision_license_and_cpu_labels(self):
        self.assertEqual(self.model.revision, PINNED_REVISION)
        self.assertEqual(self.model.license, LICENSE)
        self.assertFalse(self.model.commercial_use_permitted)
        self.assertEqual(
            {parameter.device.type for parameter in self.model.model.parameters()}, {"cpu"}
        )
        self.assertFalse(self.torch.version.cuda)
        cases = [
            ("The company reported record profits and strong revenue growth.", "positive"),
            ("The company reported a massive loss and warned of bankruptcy.", "negative"),
            ("The annual meeting will be held on Tuesday.", "neutral"),
        ]
        observations = []
        for text, expected in cases:
            started = time.perf_counter()
            result = self.model.score(text)
            self.assertEqual(
                max(result["probabilities"], key=result["probabilities"].get), expected
            )
            self.assertAlmostEqual(sum(result["probabilities"].values()), 1, places=6)
            self.assertIn("ProsusAI", result["attribution"])
            self.assertFalse(result["trading_approved"])
            observations.append(
                {
                    "expected": expected,
                    "probabilities": result["probabilities"],
                    "elapsed_ms": (time.perf_counter() - started) * 1000,
                }
            )
        Path("/workspace/work/evidence/finbert-actual-inference.json").write_text(
            json.dumps(
                {
                    "revision": PINNED_REVISION,
                    "model_manifest_hash": self.model.identity,
                    "license": LICENSE,
                    "scope": "NONCOMMERCIAL_RESEARCH_ONLY",
                    "device": "cpu",
                    "threads": 1,
                    "observations": observations,
                    "commercial_use_permitted": False,
                    "economically_validated": False,
                },
                indent=2,
            )
            + "\n"
        )

    def test_wrapper_matches_direct_transformers_forward(self):
        text = "Operating profit increased while costs fell."
        with _finbert_cpu(self.torch), self.torch.inference_mode():
            encoded = self.model.tokenizer(
                text, return_tensors="pt", truncation=True, max_length=512
            )
            direct = self.torch.softmax(self.model.model(**encoded).logits, dim=-1)[0].tolist()
        scored = self.model.score(text)
        np.testing.assert_allclose(
            [scored["probabilities"][self.model.labels[i]] for i in range(3)],
            direct,
            rtol=0,
            atol=1e-8,
        )

    def test_thread_rng_and_deterministic_settings_restore(self):
        torch = self.torch
        threads = torch.get_num_threads()
        det = torch.are_deterministic_algorithms_enabled()
        warn = torch.is_deterministic_algorithms_warn_only_enabled()
        mkldnn = torch.backends.mkldnn.enabled
        try:
            torch.set_num_threads(2)
            torch.use_deterministic_algorithms(True, warn_only=True)
            rng = torch.get_rng_state().clone()
            first = self.model.score("The company reported a loss.")
            second = self.model.score("The company reported a loss.")
            self.assertEqual(first, second)
            self.assertEqual(torch.get_num_threads(), 2)
            self.assertTrue(torch.is_deterministic_algorithms_warn_only_enabled())
            self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        finally:
            torch.set_num_threads(threads)
            torch.use_deterministic_algorithms(det, warn_only=warn)
            torch.backends.mkldnn.enabled = mkldnn

    def test_long_input_declares_truncation_and_source_stays_exact(self):
        payload = request()
        payload["articles"][0]["body"] = "  " + "profit " * 700 + "NaN source\n"
        result = self.workspace.analyze(payload).to_dict()
        EnvelopeSchema.model_validate(result)
        row = result["data"]["rows"][0]
        self.assertEqual(row["body"], payload["articles"][0]["body"])
        self.assertTrue(row["sentiment"]["model_input_truncated"])
        self.assertGreater(row["sentiment"]["original_token_count"], 512)

    def test_actual_inference_persistence_replay_and_explicit_model_view(self):
        payload = request()
        first = self.workspace.analyze(payload)
        self.assertEqual(first.wire, self.workspace.analyze(payload).wire)
        payload["articles"][0]["body"] = "mutated after admission"
        reopened = SentimentWorkspace(self.base, None)
        news = reopened.section("news", AS_OF, 86400).to_dict()
        models = reopened.section("models", AS_OF, 86400).to_dict()
        self.assertEqual(news["data"]["rows"][0]["body"], request()["articles"][0]["body"])
        self.assertEqual(news["data"]["rows"][0]["sentiment"]["method"], "FINBERT_LOCAL")
        self.assertEqual(models["provenance"]["readiness"], "NONCOMMERCIAL_RESEARCH_ONLY")
        self.assertTrue(models["data"]["rows"][0]["load_verified"])
        self.assertFalse(models["data"]["rows"][0]["commercial_use_permitted"])
        self.assertIsNone(reopened.section("news", AS_OF, None).to_dict()["data"])
        self.assertEqual(
            reopened.section("news", "2026-06-03T12:00:00Z", 60).to_dict()["state"], "STALE"
        )

    def test_corrupt_artifact_and_backdated_ledger_fail_closed(self):
        self.workspace.analyze(request())
        database = self.base / "finbert-runs.sqlite"
        with sqlite3.connect(database) as connection:
            connection.execute("UPDATE runs SET cutoff='2026-05-01T00:00:00+00:00'")
        with self.assertRaisesRegex(ValueError, "cutoff"):
            self.workspace.section("news", AS_OF, 86400)
        with sqlite3.connect(database) as connection:
            connection.execute("UPDATE runs SET cutoff='2026-06-01T12:00:00.123456789+00:00'")
        next((self.base / "finbert-runs").glob("*.json")).write_text("{}")
        with self.assertRaisesRegex(ValueError, "hash"):
            self.workspace.section("models", AS_OF, 86400)

    def test_tampered_manifest_rejects_before_library_load(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "config.json").write_text("{}")
            (root / "model.safetensors").write_bytes(b"not real weights")
            (root / "finbert-provenance.json").write_text(
                json.dumps(
                    {
                        "model_id": "ProsusAI/finbert",
                        "revision": PINNED_REVISION,
                        "files": {"config.json": "0" * 64, "model.safetensors": "0" * 64},
                    }
                )
            )
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                LocalFinBERT(root)

    def test_inference_does_not_contact_external_endpoints(self):
        with (
            patch("urllib.request.urlopen", side_effect=AssertionError("no network")),
            patch("requests.sessions.Session.request", side_effect=AssertionError("no network")),
        ):
            result = self.model.score("Revenue was unchanged from the prior year.")
        self.assertEqual(result["device"], "cpu")


if __name__ == "__main__":
    unittest.main()
