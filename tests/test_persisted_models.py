"""Actual CPU adapters on admitted persisted observations and immutable replay."""

from __future__ import annotations

import math
import sqlite3
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from gig.research.model_pipeline import KINDS, configuration, fit_dataset, identity
from gig.service.contracts import EnvelopeSchema
from gig.service.dataset_workspace import PersistedDatasetWorkspace
from gig.service.persisted_research import PersistedResearchWorkspace
from test_persisted_data import (
    dataset_request,
    fixture_bars,
    fixture_times,
    ingest_fixture_bars,
    seed_persisted_fixture,
)


class PersistedModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.base = Path(cls.temporary.name) / "workspace"
        request = seed_persisted_fixture(cls.base)
        cls.dataset_workspace = PersistedDatasetWorkspace(cls.base)
        cls.dataset = cls.dataset_workspace.build(request).to_dict()["data"]
        cls.workspace = PersistedResearchWorkspace(cls.base)
        cls.results = {}
        for model in KINDS:
            payload = {"dataset_id": cls.dataset["dataset_id"], "model": model,
                       "min_train": 30, "test_size": 12, "embargo": 1, "seed": 17}
            if model == "bilstm":
                payload.update(epochs=2, hidden_size=3, sequence_length=4)
            result = cls.workspace.run(payload).to_dict()
            EnvelopeSchema.model_validate(result)
            cls.results[model] = (payload, result)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def artifact(self, name):
        result = self.results[name][1]
        self.assertEqual(result["state"], "AVAILABLE")
        self.assertEqual(result["provenance"]["classification"], "MODEL_ESTIMATE")
        artifact = self.workspace.load(result["data"]["run_id"])
        self.assertFalse(artifact["trading_approved"])
        self.assertFalse(artifact["economically_validated"])
        self.assertTrue(artifact["rows"])
        self.assertTrue(all(math.isfinite(row["prediction"]) for row in artifact["rows"]))
        self.assertEqual(artifact["input_classification"], "SYNTHETIC")
        return artifact

    def test_actual_ridge_has_training_only_provenance_and_residual_diagnostics(self):
        artifact = self.artifact("ridge")
        state = self.workspace._model(artifact["model_ids"][0])
        self.assertEqual(state["estimator"]["model"], "ridge")
        self.assertEqual(artifact["rows"][0]["uncertainty"]["state"], "NOT_OOS_CALIBRATED")

    def test_actual_lightgbm_uses_date_grouped_lambdarank(self):
        artifact = self.artifact("lightgbm")
        state = self.workspace._model(artifact["model_ids"][0])
        self.assertEqual(state["estimator"]["objective"], "lambdarank")
        self.assertTrue(all(group >= 2 for group in state["estimator"]["group_sizes"]))
        self.assertIn("Tree=", state["estimator"]["booster_text"])
        self.assertIsNone(artifact["metrics"]["mse"])
        self.assertEqual(artifact["prediction_kind"], "CROSS_SECTIONAL_RANK_SCORE")

    def test_actual_bilstm_has_cpu_json_state(self):
        import torch

        self.assertFalse(torch.version.cuda)
        artifact = self.artifact("bilstm")
        state = self.workspace._model(artifact["model_ids"][0])
        self.assertEqual(state["estimator"]["architecture"], "BIDIRECTIONAL_LSTM_HISTORY_WINDOW")
        self.assertEqual(state["estimator"]["configuration"]["device"], "cpu")
        self.assertEqual(artifact["prediction_kind"], "FORWARD_RETURN_ESTIMATE")
        self.assertTrue(all("training_loss" in f for f in artifact["folds"]))

    def test_actual_hmm_retains_probability_and_convergence_truth(self):
        artifact = self.artifact("hmm")
        self.assertFalse(artifact["usable_for_portfolio"])
        for row in artifact["rows"]:
            self.assertAlmostEqual(sum(row["uncertainty"]["filtered_probabilities"]), 1, places=10)
            self.assertGreaterEqual(row["uncertainty"]["entropy"], 0)
            self.assertIn(row["fit_state"], {"CONVERGED", "NOT_CONVERGED"})

    def test_actual_garch_variance_cannot_be_promoted_to_alpha(self):
        artifact = self.artifact("garch")
        self.assertFalse(artifact["usable_for_portfolio"])
        self.assertEqual(artifact["prediction_kind"], "NEXT_OBSERVATION_VARIANCE")
        self.assertTrue(all(row["prediction"] > 0 for row in artifact["rows"]))
        self.assertIsNone(artifact["metrics"]["mse"])

    def test_actual_kalman_exposes_parameter_method_and_state_uncertainty(self):
        artifact = self.artifact("kalman")
        state = self.workspace._model(artifact["model_ids"][0])
        self.assertEqual(state["estimator"]["parameter_method"], "TRAINING_VARIANCE_HEURISTIC_NOT_MLE")
        self.assertTrue(all(row["uncertainty"]["value"] > 0 for row in artifact["rows"]))

    def test_all_model_folds_purge_label_knowledge_before_test(self):
        for name in KINDS:
            for fold in self.artifact(name)["folds"]:
                with self.subTest(model=name, fold=fold["model_id"]):
                    self.assertLess(pd.Timestamp(fold["max_training_label_end"]), pd.Timestamp(fold["test_start"]))

    def test_exact_retraining_reopen_and_saved_inference_for_all_models(self):
        reopened = PersistedResearchWorkspace(self.base)
        for name, (request, original) in self.results.items():
            with self.subTest(model=name):
                self.assertEqual(self.workspace.run(request).to_dict(), original)
                artifact = reopened.load(original["data"]["run_id"])
                predicted = reopened.predict({"run_id": artifact["run_id"], "dataset_id": self.dataset["dataset_id"]}).to_dict()
                EnvelopeSchema.model_validate(predicted)
                self.assertEqual(predicted["state"], "AVAILABLE")
                self.assertFalse(predicted["data"]["metrics"]["training_performed"])
                expected = {(row["instrument_id"], row["timestamp"], row["model_id"]): row["prediction"] for row in artifact["rows"]}
                compared = 0
                for row in predicted["data"]["rows"]:
                    key = (row["instrument_id"], row["timestamp"], row["model_id"])
                    if key in expected:
                        self.assertAlmostEqual(row["prediction"], expected[key], places=10)
                        compared += 1
                    else:
                        self.assertGreater(pd.Timestamp(row["timestamp"]),
                                           max(pd.Timestamp(r["timestamp"]) for r in artifact["rows"]))
                self.assertGreater(compared, 0)

    def test_future_inputs_do_not_change_first_fold_model_or_predictions(self):
        original = deepcopy(self.dataset)
        changed = deepcopy(original)
        cutoff = fixture_times()[55]
        for row in changed["rows"]:
            if pd.Timestamp(row["timestamp"]) >= cutoff:
                row["features"]["momentum"] += .25
                row["target"] += .10
        changed["dataset_id"] = identity({k: v for k, v in changed.items() if k != "dataset_id"})
        for name in ("ridge", "lightgbm", "bilstm", "hmm", "garch", "kalman"):
            with self.subTest(model=name):
                config = configuration(self.results[name][0])
                before, _ = fit_dataset(original, config)
                after, _ = fit_dataset(changed, config)
                a = [r for r in before["rows"] if r["fold"] == 0]
                b = [r for r in after["rows"] if r["fold"] == 0]
                self.assertEqual(a, b)

    def test_missing_reads_invalid_requests_and_paths_create_no_state(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary) / "missing"
            workspace = PersistedResearchWorkspace(base)
            self.assertEqual(workspace.run({"dataset_id": "0" * 64, "model": "ridge"}).to_dict()["state"], "UNAVAILABLE")
            self.assertEqual(workspace.predict({"dataset_id": "0" * 64, "run_id": "0" * 64}).to_dict()["state"], "UNAVAILABLE")
            for payload in [{"dataset_id": "../escape", "model": "ridge"},
                            {"dataset_id": "0" * 64, "model": "ridge", "epochs": 2},
                            {"dataset_id": "0" * 64, "model": "ridge", "seed": True}]:
                with self.assertRaises(ValueError):
                    workspace.run(payload)
            self.assertFalse(base.exists())

    def test_late_outcome_revision_never_rewrites_earlier_fitted_decisions(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            seed_persisted_fixture(base)
            workspace = PersistedDatasetWorkspace(base)
            before = workspace.build(dataset_request()).to_dict()["data"]
            correction = deepcopy(fixture_bars()[40])
            correction.update(available_at=fixture_times()[60].isoformat(),
                              open=150., high=150., low=150., close=150.)
            ingest_fixture_bars(base, [correction])
            after = workspace.build(dataset_request()).to_dict()["data"]
            self.assertNotEqual(before["dataset_id"], after["dataset_id"])
            for name, (request, _) in self.results.items():
                with self.subTest(model=name):
                    left, _ = fit_dataset(before, configuration(request))
                    right, _ = fit_dataset(after, configuration(request))
                    first_left = [row for row in left["rows"] if row["fold"] == 0]
                    first_right = [row for row in right["rows"] if row["fold"] == 0]
                    self.assertEqual(first_left, first_right)

    def test_freshness_and_historical_model_unavailability_are_explicit(self):
        artifact = self.artifact("ridge")
        at = pd.Timestamp(artifact["as_of"])
        reopened = PersistedResearchWorkspace(self.base)
        self.assertEqual(reopened.section("signals", at, 60).to_dict()["state"], "AVAILABLE")
        self.assertEqual(reopened.section("models", at + pd.Timedelta(seconds=61), 60).to_dict()["state"], "STALE")
        self.assertEqual(reopened.section("research", at, None).to_dict()["state"], "UNKNOWN")
        self.assertIsNone(reopened.section("research", at - pd.Timedelta(nanoseconds=1), 60))

    def test_final_serialization_precedes_model_publication(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            request = seed_persisted_fixture(base)
            dataset = PersistedDatasetWorkspace(base).build(request).to_dict()["data"]
            workspace = PersistedResearchWorkspace(base)
            with patch.object(workspace, "_contract", side_effect=ValueError("boundary")), self.assertRaises(ValueError):
                workspace.run({"dataset_id": dataset["dataset_id"], "model": "ridge"})
            self.assertFalse((base / "persisted-research").exists())

    def test_corrupt_model_and_index_fail_after_restart(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            request = seed_persisted_fixture(base)
            dataset = PersistedDatasetWorkspace(base).build(request).to_dict()["data"]
            workspace = PersistedResearchWorkspace(base)
            response = workspace.run({"dataset_id": dataset["dataset_id"], "model": "ridge"}).to_dict()
            run_id = response["data"]["run_id"]
            with sqlite3.connect(base / "persisted-research/research.sqlite") as connection:
                connection.execute("UPDATE runs SET as_of='2020-01-01T00:00:00Z'")
            with self.assertRaisesRegex(ValueError, "integrity"):
                workspace.section("research", dataset["as_of"], 60)
            model_id = response["data"]["model_ids"][0]
            (base / "persisted-research/models" / (model_id + ".json")).write_text("{}")
            with self.assertRaisesRegex(ValueError, "integrity"):
                workspace.load(run_id)


if __name__ == "__main__":
    unittest.main()
