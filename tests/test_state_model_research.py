"""Research integration: chronological split, model identity, durable evidence."""

import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from gig.research.state_models import ResearchRegistry, run_state_model_research


class StateModelResearchReferences(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.values = np.random.default_rng(51).normal(0, .01, 80)
        # Preserve nanosecond precision in source text and UTC normalization.
        cls.timestamps = [timestamp.isoformat() for timestamp in pd.date_range("2026-01-01T00:00:00.123456789Z", periods=80, freq="D")]
        cls.artifact = run_state_model_research(cls.values, cls.timestamps, training_end=60, source_id="SYNTHETIC:seed51")

    def test_holdout_independence_of_fitted_model(self):
        changed = self.values.copy()
        changed[70:] *= 5
        second = run_state_model_research(changed, self.timestamps, training_end=60, source_id="SYNTHETIC:seed51")
        self.assertEqual(self.artifact["model_id"], second["model_id"])
        self.assertNotEqual(self.artifact["run_id"], second["run_id"])
        self.assertEqual(self.artifact["models"], second["models"])
        for key in ["hmm_filtered_probabilities", "kalman_filtered_mean", "garch_conditional_variances"]:
            np.testing.assert_array_equal(self.artifact["holdout"][key][:10], second["holdout"][key][:10])

    def test_serialization_provenance_and_safety(self):
        artifact = json.loads(json.dumps(self.artifact, allow_nan=False))
        self.assertEqual(artifact["trading_approval"], "NOT_APPROVED")
        self.assertEqual(artifact["operational_readiness"], "RESEARCH_ONLY")
        self.assertEqual(artifact["provenance"]["source_timestamps"], self.timestamps)
        self.assertIn("123456789", artifact["provenance"]["utc_timestamps"][0])
        self.assertEqual(len(artifact["holdout"]["hmm_filtered_probabilities"]), 20)
        self.assertNotIn("smoothed_probabilities", artifact["holdout"])

    def test_registry_restart_replay_mutation_and_corruption_detection(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "research.sqlite"
            registry = ResearchRegistry(path)
            identity = registry.save(self.artifact)
            self.assertEqual(registry.save(self.artifact), identity)
            second = ResearchRegistry(path)
            self.assertEqual(second.load(identity), self.artifact)
            loaded = second.load(identity)
            loaded["models"]["garch"]["alpha"] = .99
            self.assertEqual(registry.load(identity), self.artifact)
            with self.assertRaises(ValueError):
                registry.save(loaded)
            with sqlite3.connect(path) as connection:
                connection.execute("UPDATE state_model_runs SET artifact_json=? WHERE run_id=?", (json.dumps(loaded), identity))
            with self.assertRaises(ValueError):
                registry.load(identity)

    def test_invalid_temporal_authority_rejected(self):
        for timestamps in [self.timestamps[::-1], ["2026-01-01"] * 80, [self.timestamps[0]] * 80]:
            with self.subTest(timestamps=timestamps[:2]), self.assertRaises(ValueError):
                run_state_model_research(self.values, timestamps, training_end=60, source_id="SYNTHETIC")
        for boundary in [0, 19, 80, True]:
            with self.subTest(boundary=boundary), self.assertRaises(ValueError):
                run_state_model_research(self.values, self.timestamps, training_end=boundary, source_id="SYNTHETIC")

    def test_subnanosecond_timestamp_is_rejected_before_model_training(self):
        timestamps = self.timestamps.copy()
        timestamps[0] = "2026-01-01T00:00:00.1234567891Z"
        with self.assertRaises(ValueError):
            run_state_model_research(self.values, timestamps, training_end=60, source_id="SYNTHETIC")


if __name__ == "__main__":
    unittest.main()
