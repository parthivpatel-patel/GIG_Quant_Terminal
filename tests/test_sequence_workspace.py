"""Real CPU sequence persistence and final contract boundaries, without brokers."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from gig.service.contracts import EnvelopeSchema
from gig.service.sequence_workspace import SequenceWorkspace
from test_sequence_model import fixture, inference_rows


class SequenceWorkspaceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.request = fixture()
        cls.request["epochs"] = 3

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.directory = Path(self.temporary.name) / "sequence"
        self.workspace = SequenceWorkspace(self.directory)

    def tearDown(self):
        self.temporary.cleanup()

    def decoded(self, contract):
        result = contract.to_dict()
        EnvelopeSchema.model_validate(result)
        return result

    def test_missing_reads_and_unknown_prediction_do_not_create_state(self):
        self.assertIsNone(self.workspace.section("models", "2026-02-01T00:00:00Z", 60))
        result = self.decoded(
            self.workspace.predict({"model_id": "0" * 64, "rows": inference_rows(self.request)})
        )
        self.assertEqual(result["state"], "UNAVAILABLE")
        self.assertFalse(self.directory.exists())

    def test_actual_training_registry_replay_reopen_and_prediction(self):
        first = self.decoded(self.workspace.train(self.request))
        self.assertEqual(first["state"], "AVAILABLE")
        self.assertEqual(first["provenance"]["classification"], "SYNTHETIC")
        self.assertNotIn("model_state", first["data"])
        self.assertFalse(first["data"]["trading_approved"])
        second = self.decoded(self.workspace.train(self.request))
        self.assertEqual(first, second)
        reopened = SequenceWorkspace(self.directory)
        observed = first["provenance"]["as_of"]
        for name in ("models", "signals", "research"):
            result = self.decoded(reopened.section(name, observed, 60))
            self.assertEqual(result["state"], "AVAILABLE")
            self.assertEqual(result["provenance"]["model_identity"], first["data"]["model_id"])
        models = self.decoded(reopened.section("models", observed, 60))
        self.assertTrue(models["data"]["rows"][0]["trained"])
        inference = self.decoded(
            reopened.predict(
                {"model_id": first["data"]["model_id"], "rows": inference_rows(self.request)}
            )
        )
        self.assertEqual(inference["provenance"]["classification"], "MODEL_ESTIMATE")
        self.assertEqual(len(inference["data"]["rows"]), 16)
        self.assertEqual(inference["data"]["metrics"]["excluded_insample_predictions"], 43)

    def test_historical_predictions_cannot_use_a_future_fitted_model(self):
        trained = self.decoded(self.workspace.train(self.request))
        result = self.decoded(
            self.workspace.predict(
                {"model_id": trained["data"]["model_id"], "rows": inference_rows(self.request)[:30]}
            )
        )
        self.assertEqual(result["state"], "UNAVAILABLE")
        self.assertIsNone(result["data"])

    def test_original_time_stale_unknown_and_future_cutoff(self):
        result = self.decoded(self.workspace.train(self.request))
        observed = pd.Timestamp(result["provenance"]["as_of"])
        stale = self.decoded(
            self.workspace.section("signals", observed + pd.Timedelta(seconds=61), 60)
        )
        self.assertEqual(stale["state"], "STALE")
        self.assertEqual(stale["provenance"]["as_of"], observed.isoformat())
        unknown = self.decoded(self.workspace.section("research", observed, None))
        self.assertEqual(unknown["state"], "UNKNOWN")
        self.assertIsNone(unknown["data"])
        self.assertIsNone(
            self.workspace.section("models", observed - pd.Timedelta(nanoseconds=1), 60)
        )

    def test_model_and_response_corruption_fail_closed(self):
        result = self.decoded(self.workspace.train(self.request))
        model = self.directory / "registry" / "models" / f"{result['data']['model_id']}.json"
        model.write_text("{}")
        with self.assertRaisesRegex(ValueError, "integrity"):
            self.workspace.section("models", result["provenance"]["as_of"], 60)

    def test_invalid_freshness_and_paths_never_admit_state(self):
        for maximum in (True, -1, float("inf"), 400 * 86400):
            with self.subTest(maximum=maximum), self.assertRaises(ValueError):
                self.workspace.section("models", "2026-02-01T00:00:00Z", maximum)
        with self.assertRaises(ValueError):
            self.workspace.predict({"model_id": "../escape", "rows": inference_rows(self.request)})
        request = dict(self.request, output_dir="../escape")
        with self.assertRaises(ValueError):
            self.workspace.train(request)
        self.assertFalse(self.directory.exists())

    def test_symlink_registry_is_rejected(self):
        target = Path(self.temporary.name) / "elsewhere"
        target.mkdir()
        self.directory.mkdir()
        (self.directory / "registry").symlink_to(target, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "symbolic"):
            self.workspace.train(self.request)
        self.assertEqual(list(target.iterdir()), [])

    def test_serialization_preflight_before_registry_publication(self):
        invalid_artifact = {
            "metrics": {},
            "predictions": [],
            "training_loss": [],
            "model_id": "x",
            "run_id": "y",
            "dataset_sha256": "z",
            "configuration": {},
            "bidirectional_scope": "HISTORICAL_WINDOW_ONLY",
            "source": "fixture",
            "classification": "SYNTHETIC",
            "as_of": "2026-01-01T00:00:00Z",
            "model_state": {"implementation": "test"},
        }
        # No invalid identity can become a published trained model or run.
        with (
            patch("gig.service.sequence_workspace.train_sequence", return_value=invalid_artifact),
            self.assertRaises(ValueError),
        ):
            self.workspace.train(self.request)
        self.assertFalse((self.directory / "registry").exists())


if __name__ == "__main__":
    unittest.main()
