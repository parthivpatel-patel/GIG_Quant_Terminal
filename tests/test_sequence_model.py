"""Real CPU BiLSTM training, chronological leakage, and JSON state references."""

from __future__ import annotations

import importlib.util
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from gig.ml.sequence import SequenceRegistry, predict_sequence, train_sequence


def fixture():
    rng = np.random.default_rng(173)
    times = pd.date_range("2026-01-01T00:00:00.000000123Z", periods=64, freq="h")
    features = rng.normal(size=(64, 2))
    rows = [
        {
            "timestamp": stamp.isoformat(),
            "available_at": stamp.isoformat(),
            "label_end": (stamp + pd.Timedelta(hours=2)).isoformat(),
            "features": {"momentum": float(features[i, 0]), "reversal": float(features[i, 1])},
            "target": float(0.5 * features[i, 0] - 0.3 * features[i, 1]),
        }
        for i, stamp in enumerate(times)
    ]
    return {
        "rows": rows,
        "source": "authored-sequence-fixture",
        "classification": "SYNTHETIC",
        "sequence_length": 6,
        "train_size": 48,
        "hidden_size": 4,
        "epochs": 16,
        "learning_rate": 0.02,
        "seed": 17,
    }


def inference_rows(payload):
    return [
        {key: row[key] for key in ("timestamp", "features", "available_at")}
        for row in payload["rows"]
    ]


class SequenceAdmissionTests(unittest.TestCase):
    def test_missing_actual_torch_never_substitutes(self):
        with (
            patch("gig.ml.sequence._torch", side_effect=ImportError("actual Torch missing")),
            self.assertRaisesRegex(ImportError, "Torch"),
        ):
            train_sequence(fixture())

    def test_temporal_bounds_and_nonfinite_admission(self):
        cases = []
        request = fixture()
        request["rows"][0]["available_at"] = request["rows"][3]["timestamp"]
        cases.append(request)
        request = fixture()
        request["rows"][1]["timestamp"] = request["rows"][0]["timestamp"]
        cases.append(request)
        request = fixture()
        request["rows"][0]["label_end"] = "2020-01-01T00:00:00Z"
        cases.append(request)
        request = fixture()
        request["rows"][0]["features"]["momentum"] = float("nan")
        cases.append(request)
        request = fixture()
        request["rows"][0]["features"]["momentum"] = True
        cases.append(request)
        request = fixture()
        request["rows"][1]["features"] = {"other": 1}
        cases.append(request)
        request = fixture()
        request["epochs"] = True
        cases.append(request)
        request = fixture()
        request["sequence_length"] = 100
        cases.append(request)
        request = fixture()
        request["as_of"] = request["rows"][-1]["timestamp"]
        cases.append(request)
        request = fixture()
        request["rows"][0]["timestamp"] = "2026-01-01T00:00:00.1234567891Z"
        cases.append(request)
        request = fixture()
        request["model_path"] = "../escape"
        cases.append(request)
        for i, request in enumerate(cases):
            with self.subTest(case=i), self.assertRaises(ValueError):
                train_sequence(request)

    def test_declared_cpu_work_budget_rejects_before_training(self):
        request = fixture()
        request.update(hidden_size=32, epochs=100, sequence_length=32)
        with self.assertRaisesRegex(ValueError, "work budget"):
            train_sequence(request)

    def test_target_variance_overflow_rejects_before_torch_load(self):
        request = fixture()
        for i, row in enumerate(request["rows"]):
            row["target"] = 1e200 if i % 2 else -1e200
        with (
            patch("gig.ml.sequence._torch", side_effect=AssertionError("Torch loaded")),
            self.assertRaisesRegex(ValueError, "normalization overflow"),
        ):
            train_sequence(request)

    def test_feature_variance_overflow_rejects_before_torch_load(self):
        request = fixture()
        for i, row in enumerate(request["rows"]):
            row["features"]["momentum"] = 1e200 if i % 2 else -1e200
        with (
            patch("gig.ml.sequence._torch", side_effect=AssertionError("Torch loaded")),
            self.assertRaisesRegex(ValueError, "normalization overflow"),
        ):
            train_sequence(request)


@unittest.skipUnless(
    importlib.util.find_spec("torch") is not None, "Actual CPU PyTorch dependency unavailable"
)
class ActualSequenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.payload = fixture()
        cls.result = train_sequence(cls.payload)

    def test_true_bidirectional_architecture_and_cpu_execution(self):
        result = self.result
        self.assertEqual(result["model_state"]["architecture"], "BIDIRECTIONAL_LSTM_HISTORY_WINDOW")
        names = result["model_state"]["parameters"]
        self.assertIn("lstm.weight_ih_l0_reverse", names)
        self.assertEqual(result["configuration"]["device"], "cpu")
        self.assertEqual(result["measurements"]["threads"], 1)
        self.assertFalse(result["trading_approved"])
        self.assertFalse(result["economically_validated"])
        self.assertEqual(result["bidirectional_scope"], "HISTORICAL_WINDOW_ONLY")

    def test_actual_loss_optimization_and_finite_holdout(self):
        self.assertLess(self.result["training_loss"][-1], self.result["training_loss"][0])
        self.assertTrue(np.isfinite([r["prediction"] for r in self.result["predictions"]]).all())
        self.assertTrue(np.isfinite(self.result["metrics"]["holdout_mse"]))

    def test_label_purge_and_train_only_feature_normalization(self):
        state = self.result["model_state"]
        self.assertEqual(self.result["metrics"]["purged_windows"], 2)
        self.assertLess(
            pd.Timestamp(state["max_training_label_end"]), pd.Timestamp(state["training_cutoff"])
        )
        # Last admitted endpoint is row45; all training windows use rows0..45.
        values = np.array(
            [[r["features"][k] for k in state["features"]] for r in self.payload["rows"][:46]]
        )
        np.testing.assert_allclose(state["mean"], values.mean(axis=0), rtol=0, atol=1e-15)
        np.testing.assert_allclose(state["scale"], values.std(axis=0), rtol=0, atol=1e-15)

    def test_deterministic_retraining_preserves_model_and_run_identity(self):
        another = train_sequence(self.payload)
        self.assertEqual(another["model_id"], self.result["model_id"])
        self.assertEqual(another["run_id"], self.result["run_id"])
        self.assertEqual(another["predictions"], self.result["predictions"])

    def test_future_holdout_labels_cannot_change_model_or_predictions(self):
        changed = deepcopy(self.payload)
        for row in changed["rows"][48:]:
            row["target"] += 100
        future = train_sequence(changed)
        self.assertEqual(future["model_id"], self.result["model_id"])
        self.assertNotEqual(future["run_id"], self.result["run_id"])
        self.assertEqual(
            [r["prediction"] for r in future["predictions"]],
            [r["prediction"] for r in self.result["predictions"]],
        )

    def test_future_feature_mutation_preserves_prior_predictions(self):
        changed = deepcopy(self.payload)
        changed["rows"][-1]["features"]["momentum"] = 100
        future = train_sequence(changed)
        self.assertEqual(future["model_id"], self.result["model_id"])
        self.assertEqual(future["predictions"][:-1], self.result["predictions"][:-1])
        self.assertNotEqual(
            future["predictions"][-1]["prediction"], self.result["predictions"][-1]["prediction"]
        )

    def test_source_identity_and_input_mutation_boundaries(self):
        payload = fixture()
        before = deepcopy(payload)
        train_sequence(payload)
        self.assertEqual(payload, before)
        payload["source"] = "different-caller"
        self.assertNotEqual(train_sequence(payload)["model_id"], self.result["model_id"])

    def test_json_registry_replay_reopen_and_real_reload_prediction_parity(self):
        with tempfile.TemporaryDirectory() as directory:
            registry = SequenceRegistry(directory)
            identity = registry.save(self.result)
            self.assertEqual(registry.save(train_sequence(self.payload)), identity)
            reopened = SequenceRegistry(directory)
            loaded = reopened.load(identity)
            self.assertEqual(loaded["model_id"], self.result["model_id"])
            predictions = reopened.predict(loaded["model_id"], inference_rows(self.payload))
            # Reload processes samples in a larger batch; CPU results remain
            # numerically equal to the independently stored holdout predictions.
            np.testing.assert_allclose(
                [r["prediction"] for r in predictions[-16:]],
                [r["prediction"] for r in self.result["predictions"]],
                rtol=0,
                atol=1e-7,
            )
            self.assertEqual(predictions[-1]["timestamp"], self.payload["rows"][-1]["timestamp"])
            self.assertEqual(list(Path(directory).rglob("*.pt")), [])

    def test_registry_corruption_and_forged_identity_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            registry = SequenceRegistry(directory)
            identity = registry.save(self.result)
            model_path = Path(directory) / "models" / f"{self.result['model_id']}.json"
            model_path.write_text("{}")
            with self.assertRaisesRegex(ValueError, "integrity"):
                registry.load(identity)
            with self.assertRaises(ValueError):
                registry.load("../escape")
        forged = deepcopy(self.result)
        forged["model_state"]["target_mean"] += 1
        with (
            tempfile.TemporaryDirectory() as directory,
            self.assertRaisesRegex(ValueError, "identity"),
        ):
            SequenceRegistry(directory).save(forged)

    def test_inference_schema_time_and_parameter_bounds(self):
        rows = inference_rows(self.payload)
        with self.assertRaises(ValueError):
            predict_sequence(self.result["model_state"], rows[:2])
        changed = deepcopy(rows)
        changed[0]["features"] = {"bad": 0}
        with self.assertRaises(ValueError):
            predict_sequence(self.result["model_state"], changed)
        changed = deepcopy(rows)
        changed[-1]["available_at"] = "2030-01-01T00:00:00Z"
        with self.assertRaises(ValueError):
            predict_sequence(self.result["model_state"], changed)
        state = deepcopy(self.result["model_state"])
        state["parameters"]["lstm.weight_ih_l0"]["shape"] = [1000000000]
        with self.assertRaisesRegex(ValueError, "shape"):
            predict_sequence(state, rows)

    def test_rng_and_process_torch_execution_settings_are_restored(self):
        import torch

        original_rng = torch.random.get_rng_state().clone()
        settings = (
            torch.get_num_threads(),
            torch.are_deterministic_algorithms_enabled(),
            torch.backends.mkldnn.enabled,
        )
        train_sequence(self.payload)
        self.assertTrue(torch.equal(original_rng, torch.random.get_rng_state()))
        self.assertEqual(
            settings,
            (
                torch.get_num_threads(),
                torch.are_deterministic_algorithms_enabled(),
                torch.backends.mkldnn.enabled,
            ),
        )


if __name__ == "__main__":
    unittest.main()
