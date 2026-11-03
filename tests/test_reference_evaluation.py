"""Independent scalar loss references, paired coverage and immutable replay."""
from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from gig.research.reference_evaluation import configuration, evaluate_reference, paired_interval
from gig.research.reference_research import digest
from gig.service.contracts import EnvelopeSchema
from gig.service.observed_workspace import ObservedWorkspace
from test_observed_reference import ACQUIRED, reference_fixture

AVAILABLE = "2026-11-03T20:00:00.123456789Z"


def scalar_report(prediction=.5, missing=False):
    state = {"model": "ridge", "fit_state": "CONVERGED", "training_targets": [.2, .2]}
    identity = digest(state)
    body = {"snapshot_id": "a"*64, "as_of": ACQUIRED, "actual_acquired_at": ACQUIRED,
            "input_classification": "SYNTHETIC", "model": "ridge", "model_ids": [identity],
            "prediction_kind": "DERIVED_REFERENCE_RETURN_ESTIMATE", "research_mode": "RETROSPECTIVE_OBSERVATION_ORDER_RESEARCH",
            "attribution": "AUTHORED_SCALAR_FIXTURE", "rows": [
                {"instrument_id": "AUTHORED", "observation_date": f"2025-01-{i+1:02d}", "target_derived_return": .1,
                 "prediction": None if missing and i == 0 else prediction, "model_id": identity} for i in range(10)]}
    report = {**body, "report_id": digest(body)}
    return report, {identity: state}, configuration({"report_id": report["report_id"], "block_length": 2})


class ReferenceEvaluationTests(unittest.TestCase):
    def test_hand_computed_losses_and_constant_paired_interval(self):
        report, models, config = scalar_report()
        result = evaluate_reference(report, models, config, AVAILABLE)
        # (.5-.1)^2=.16 versus .1^2=.01 and (.1-.2)^2=.01.
        self.assertAlmostEqual(result["metrics"]["matched_date_equal_weight_model_mse"], .16)
        for baseline in result["comparisons"].values():
            self.assertAlmostEqual(baseline["matched_date_equal_weight_baseline_mse"], .01)
            self.assertAlmostEqual(baseline["mean_model_minus_baseline_loss"], .15)
            np.testing.assert_allclose(baseline["percentile_95_interval"], [.15, .15])
            self.assertFalse(baseline["multiplicity_adjusted"])
        self.assertEqual(result["interpretation"], "MODEL_WORSE_THAN_BOTH_BASELINES_ON_MATCHED_DATES")
        self.assertFalse(result["economically_validated"])
        self.assertEqual(result["portfolio_pnl_state"], "UNAVAILABLE")
        self.assertEqual(result, evaluate_reference(report, models, config, AVAILABLE))

    def test_perfect_predictions_are_predictive_evidence_not_trading_approval(self):
        report, models, config = scalar_report(prediction=.1)
        result = evaluate_reference(report, models, config, AVAILABLE)
        self.assertEqual(result["metrics"]["matched_date_equal_weight_model_mse"], 0)
        self.assertFalse(result["trading_approved"])
        self.assertFalse(result["funded_allocation_eligible"])
        self.assertFalse(result["historical_point_in_time_validated"])

    def test_missing_fits_are_excluded_from_both_sides_and_disclosed(self):
        report, models, config = scalar_report(missing=True)
        result = evaluate_reference(report, models, config, AVAILABLE)
        self.assertEqual(result["coverage_state"], "PARTIAL")
        self.assertEqual(result["metrics"]["paired_rows"], 9)
        self.assertEqual(result["metrics"]["paired_dates"], 9)
        self.assertEqual(result["metrics"]["excluded_rows"], 1)
        self.assertNotIn("2025-01-01", result["observation_dates"])

    def test_date_blocks_keep_cross_currency_loss_differences_together(self):
        result = paired_interval([2.]*7, replications=100, block_length=3, seed=1)
        self.assertEqual(result["percentile_95_interval"], [2., 2.])
        with self.assertRaises(ValueError):
            paired_interval([1., 2.], replications=100, block_length=3, seed=1)
        with self.assertRaises(ValueError):
            paired_interval([1., 2., 3., 4., np.nan], replications=100, block_length=3, seed=1)

    def test_unsupported_prediction_units_have_no_invented_return_score(self):
        for kind in ("CROSS_REFERENCE_RANK_SCORE", "CONDITIONAL_VARIANCE_PER_REFERENCE_OBSERVATION"):
            report, models, _ = scalar_report()
            report["prediction_kind"] = kind
            report["report_id"] = digest({k: v for k, v in report.items() if k != "report_id"})
            result = evaluate_reference(report, models, configuration({"report_id": report["report_id"]}), AVAILABLE)
            self.assertEqual(result["evaluation_state"], "UNAVAILABLE")
            self.assertIsNone(result["metrics"])
            self.assertIsNone(result["comparisons"])

    def test_invalid_settings_or_modified_parents_cannot_be_scored(self):
        report, models, config = scalar_report()
        for key, value in (("bootstrap_replications", True), ("block_length", 0), ("seed", -1), ("extra", 1)):
            with self.assertRaises(ValueError):
                configuration({"report_id": report["report_id"], key: value})
        for value in ("2025-01-01T00:00:00Z", "2026-11-03T20:00:00"):
            with self.assertRaises(ValueError):
                evaluate_reference(report, models, config, value)
        changed = copy.deepcopy(report)
        changed["rows"][0]["prediction"] = 99
        with self.assertRaisesRegex(ValueError, "parent identity"):
            evaluate_reference(changed, models, config, AVAILABLE)
        changed_models = copy.deepcopy(models)
        next(iter(changed_models.values()))["training_targets"] = [999]
        with self.assertRaisesRegex(ValueError, "model integrity"):
            evaluate_reference(report, changed_models, config, AVAILABLE)

    def test_real_saved_ridge_restart_replay_and_failure_injection(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)/"state"
            workspace = ObservedWorkspace(base)
            snapshot = workspace.run({"operation": "admit", **reference_fixture()}).to_dict()["data"]
            report = workspace.run({"operation": "research", "snapshot_id": snapshot["snapshot_id"], "model": "ridge",
                                    "training_observations": 30, "test_observations": 10}).to_dict()["data"]
            request = {"operation": "evaluate", "report_id": report["report_id"], "bootstrap_replications": 100}
            with patch.object(workspace, "_index", side_effect=RuntimeError("injected publication failure")), self.assertRaises(RuntimeError):
                workspace.run(request)
            self.assertEqual(workspace.section("research", AVAILABLE, 86400).to_dict()["data"]["report_id"], report["report_id"])
            result = workspace.run(request).to_dict()
            EnvelopeSchema.model_validate(result)
            self.assertEqual(result["data"]["evaluation_state"], "AVAILABLE")
            self.assertEqual(result["data"]["input_classification"], "SYNTHETIC")
            replay = ObservedWorkspace(base).run({"operation": "evaluation-replay", "evaluation_id": result["data"]["evaluation_id"]}).to_dict()
            self.assertEqual(result, replay)
            # Evaluation availability uses its actual creation clock, not the old source date.
            earlier = workspace.section("research", report["as_of"], 86400).to_dict()["data"]
            self.assertNotIn("evaluation_id", earlier)
            path = workspace._path("evaluations/"+result["data"]["evaluation_id"]+".json")
            path.write_text("{}")
            with self.assertRaisesRegex(ValueError, "integrity"):
                workspace.load_evaluation(result["data"]["evaluation_id"])

    def test_missing_evaluation_has_no_mutation_or_synthetic_fallback(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)/"state"
            workspace = ObservedWorkspace(base)
            for request in ({"operation": "evaluate", "report_id": "0"*64},
                            {"operation": "evaluation-replay", "evaluation_id": "0"*64}):
                self.assertEqual(workspace.run(request).to_dict()["state"], "UNAVAILABLE")
            self.assertFalse(base.exists())
