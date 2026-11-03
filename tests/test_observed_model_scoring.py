"""Independent ranking/variance references and original-artifact replay."""
from __future__ import annotations

import copy
import json
import math
import unittest
from pathlib import Path

from gig.research.reference_evaluation import (
    VERSION,
    VERSION_V1,
    configuration,
    evaluate_reference,
    rank_scores,
    variance_scores,
)
from gig.research.reference_research import digest
from test_reference_evaluation import AVAILABLE, scalar_report


def model_report(model, *, missing=False):
    rank = model == "lightgbm"
    state = {"model": model, "fit_state": "CONVERGED"}
    if rank:
        state["ranking_baseline"] = {"mean_return_by_instrument": {"A": .1, "B": .3, "C": .2}}
    else:
        state.update(training_targets=[0., 2.], parameters={"mean": 1.})
    identity = digest(state)
    report, _, _ = scalar_report()
    report.update(model=model, model_ids=[identity], prediction_kind="CROSS_REFERENCE_RANK_SCORE" if rank else "CONDITIONAL_VARIANCE_PER_REFERENCE_OBSERVATION")
    report["rows"] = []
    for day in range(10):
        for index, instrument in enumerate(("A", "B", "C") if rank else ("A",)):
            report["rows"].append({"instrument_id": instrument, "observation_date": f"2025-01-{day+1:02d}",
                "target_derived_return": (index+1)*.1 if rank else 3., "model_id": identity,
                "prediction": None if missing and day == 0 and index == 0 else float(index+1) if rank else 4.})
    report["report_id"] = digest({k: v for k, v in report.items() if k != "report_id"})
    return report, {identity: state}, configuration({"report_id": report["report_id"], "block_length": 2})


class AppropriateModelScoreTests(unittest.TestCase):
    def test_independent_rank_perfect_reverse_and_expected_ties(self):
        perfect = rank_scores([.1, .2, .3], [1, 2, 3])
        self.assertAlmostEqual(perfect["ndcg"], 1.)
        self.assertAlmostEqual(perfect["spearman_ic"], 1.)
        reverse = rank_scores([.1, .2, .3], [3, 2, 1])
        self.assertAlmostEqual(reverse["spearman_ic"], -1.)
        self.assertAlmostEqual(reverse["ndcg"], (1/math.log2(3)+2/2)/(2+1/math.log2(3)))
        tied = rank_scores([.1, .2, .3], [0, 0, 0])
        self.assertAlmostEqual(tied["ndcg"], (1+1/math.log2(3)+.5)/(2+1/math.log2(3)))
        self.assertIsNone(tied["spearman_ic"])
        self.assertEqual(tied["ic_state"], "UNKNOWN")

    def test_constant_targets_are_unknown_not_a_zero_information_coefficient(self):
        result = rank_scores([1., 1., 1.], [3., 2., 1.])
        self.assertIsNone(result["ndcg"])
        self.assertIsNone(result["spearman_ic"])
        self.assertEqual(result["ndcg_state"], "UNKNOWN")

    def test_ranking_preserves_large_integer_order_and_rejects_coercion(self):
        score = rank_scores([2**53, 2**53+1, 2**53+2], [1, 2, 3])
        self.assertEqual(score["ndcg"], 1.)
        self.assertEqual(score["spearman_ic"], 1.)
        for values in ([True, False], ["1", "2"], [float("nan"), 1.]):
            with self.subTest(values=values), self.assertRaises(ValueError):
                rank_scores(values, [1, 2])

    def test_saved_training_only_rank_baseline_and_replay(self):
        report, models, config = model_report("lightgbm")
        result = evaluate_reference(report, models, config, AVAILABLE)
        self.assertEqual(result["schema_version"], VERSION)
        self.assertEqual(result["scoring_family"], "CROSS_SECTIONAL_RANKING")
        self.assertAlmostEqual(result["metrics"]["mean_ndcg"], 1.)
        expected = (1+2/math.log2(3))/(2+1/math.log2(3))
        self.assertAlmostEqual(result["date_scores"][0]["training_mean_return_rank"]["ndcg"], expected)
        self.assertEqual(result["comparisons"]["training_mean_return_rank"]["state"], "AVAILABLE")
        self.assertNotIn("matched_date_equal_weight_model_mse", result["metrics"])
        self.assertEqual(result, evaluate_reference(report, models, config, AVAILABLE, schema_version=result["schema_version"]))
        self.assertFalse(result["economically_validated"])

    def test_missing_rank_prediction_excludes_whole_cross_section(self):
        report, models, config = model_report("lightgbm", missing=True)
        result = evaluate_reference(report, models, config, AVAILABLE)
        self.assertEqual(result["coverage_state"], "PARTIAL")
        self.assertEqual(result["metrics"]["scored_dates"], 9)
        self.assertEqual(result["comparisons"]["uniform_tied_scores"]["paired_dates"], 9)
        self.assertEqual(len(result["excluded_dates"]), 1)

    def test_constant_target_cross_section_does_not_claim_complete_score_coverage(self):
        report, models, _ = model_report("lightgbm")
        for row in report["rows"]:
            if row["observation_date"] == "2025-01-01":
                row["target_derived_return"] = .1
        report["report_id"] = digest({k: v for k, v in report.items() if k != "report_id"})
        result = evaluate_reference(report, models, configuration({"report_id": report["report_id"]}), AVAILABLE)
        self.assertEqual(result["coverage_state"], "PARTIAL")
        self.assertEqual(result["metrics"]["scored_dates"], 9)
        self.assertEqual(result["metrics"]["undefined_target_dates"], 1)
        self.assertIsNone(result["date_scores"][0]["model"]["ndcg"])

    def test_missing_training_currency_cannot_silently_reduce_rank_universe(self):
        report, models, _ = model_report("lightgbm")
        report["rows"] = [r for r in report["rows"] if r["instrument_id"] != "C"]
        report["report_id"] = digest({k: v for k, v in report.items() if k != "report_id"})
        result = evaluate_reference(report, models, configuration({"report_id": report["report_id"]}), AVAILABLE)
        self.assertEqual(result["evaluation_state"], "UNAVAILABLE")
        self.assertIsNone(result["metrics"])

    def test_independent_gaussian_variance_score_uses_fitted_mean(self):
        score = variance_scores(3., 1., 4.)
        self.assertAlmostEqual(score["qlike"], math.log(4)+1)
        self.assertAlmostEqual(score["gaussian_nll"], .5*(math.log(2*math.pi)+math.log(4)+1))
        self.assertEqual(score["variance_proxy_squared_error"], 0.)
        self.assertEqual(score["innovation_squared_proxy"], 4.)
        self.assertEqual(score["standardized_innovation"], 1.)

    def test_variance_scoring_calibration_and_training_baseline(self):
        report, models, config = model_report("garch")
        result = evaluate_reference(report, models, config, AVAILABLE)
        self.assertEqual(result["scoring_family"], "CONDITIONAL_VARIANCE")
        self.assertAlmostEqual(result["metrics"]["mean_qlike"], math.log(4)+1)
        # Training targets 0,2 about frozen fitted mean 1 imply variance 1.
        self.assertEqual(result["score_rows"][0]["training_second_moment_baseline"], 1.)
        comparison = result["comparisons"]["training_second_moment_variance"]
        self.assertAlmostEqual(comparison["mean_model_minus_baseline_loss"], math.log(4)-3)
        self.assertEqual(result["calibration"]["standardized_innovation_second_moment"], 1.)
        self.assertEqual(result["calibration"]["gaussian_95_interval_coverage"], 1.)
        self.assertEqual(result["calibration"]["calibrated_confidence_state"], "UNKNOWN")
        self.assertNotIn("matched_date_equal_weight_model_mse", result["metrics"])

    def test_nonpositive_nonfinite_variance_never_receives_an_epsilon_score(self):
        for value in (0., -1., float("nan"), float("inf"), True):
            with self.subTest(value=value), self.assertRaises(ValueError):
                variance_scores(.1, 0., value)

    def test_degenerate_training_variance_keeps_model_scores_and_unknown_baseline(self):
        report, models, _ = model_report("garch")
        state = copy.deepcopy(next(iter(models.values())))
        state["training_targets"] = [1., 1.]
        identity = digest(state)
        report["model_ids"] = [identity]
        for row in report["rows"]:
            row["model_id"] = identity
        report["report_id"] = digest({k: v for k, v in report.items() if k != "report_id"})
        result = evaluate_reference(report, {identity: state}, configuration({"report_id": report["report_id"]}), AVAILABLE)
        self.assertEqual(result["evaluation_state"], "AVAILABLE")
        self.assertAlmostEqual(result["metrics"]["mean_qlike"], math.log(4)+1)
        self.assertEqual(result["comparisons"]["training_second_moment_variance"]["state"], "UNKNOWN")
        self.assertEqual(result["comparisons"]["training_second_moment_variance"]["paired_rows"], 0)
        self.assertTrue(all(row["baseline_state"] == "UNKNOWN" and row["baseline_qlike"] is None for row in result["score_rows"]))

    def test_hmm_partial_fold_and_stop_diagnostics_do_not_erase_negative_findings(self):
        report, models, _ = scalar_report()
        original = next(iter(models.values()))
        state = {**original, "model": "hmm", "stop_reason": "CONVERGED"}
        bad = {**state, "fit_state": "NONCONVERGED", "stop_reason": "MAX_ITERATIONS"}
        good_id, bad_id = digest(state), digest(bad)
        report.update(model="hmm", model_ids=[good_id, bad_id])
        for index, row in enumerate(report["rows"]):
            row["model_id"] = good_id if index < 5 else bad_id
            if index >= 5:
                row["prediction"] = None
        report["report_id"] = digest({k: v for k, v in report.items() if k != "report_id"})
        result = evaluate_reference(report, {good_id: state, bad_id: bad}, configuration({"report_id": report["report_id"], "block_length": 2}), AVAILABLE)
        self.assertEqual(result["fit_diagnostics"]["nonconverged_models"], 1)
        self.assertEqual(result["fit_diagnostics"]["excluded_rows"], 5)
        self.assertEqual(result["coverage_state"], "PARTIAL")
        self.assertEqual(result["interpretation"], "MODEL_WORSE_THAN_BOTH_BASELINES_ON_MATCHED_DATES")

    def test_duplicate_cross_section_or_future_training_boundary_rejects(self):
        report, models, _ = model_report("lightgbm")
        report["rows"].append(copy.deepcopy(report["rows"][0]))
        report["report_id"] = digest({k: v for k, v in report.items() if k != "report_id"})
        with self.assertRaisesRegex(ValueError, "duplicate"):
            evaluate_reference(report, models, configuration({"report_id": report["report_id"]}), AVAILABLE)
        with self.assertRaisesRegex(ValueError, "unsupported"):
            evaluate_reference(report, models, configuration({"report_id": report["report_id"]}), AVAILABLE, schema_version="unknown-version")


class ExistingAcquiredArtifactTests(unittest.TestCase):
    def _workspace(self):
        root = Path("/workspace/work/observed-reference-runtime/observed-reference")
        if not root.is_dir():
            self.skipTest("actual ECB acquisition evidence not available in this environment")
        return root

    def test_pre_sprint8_v1_actual_evaluation_replays_byte_identically(self):
        root = self._workspace()
        identities = (
            "42bc6bad1aad11c2a9dfe5bef1483e1b453de047e40aea9991cacf9a48f986b9",
            "2885c1178ef1316675e525360ecce67eecbc1baa4b1a1634744dac2b49a42365",
            "6744efc5423abd1ddff98f6465ff88c066b119902eb84e7a4804dfed9a4faa50",
            "ed31b4d1797480d8253947cb255a220aa0d437581ee00f548f0b658dff980a2a",
            "1ad70e19719e3cc7861a603e68528c61f7cb3b459a620e8a22c3ff86c3a6fda4",
            "420918609a98f60d72a29ab7116f60986dc7bf4aea9aec6b05ecd4f4fb4d3b1b",
        )
        for identity in identities:
            with self.subTest(identity=identity):
                wire = (root/"evaluations"/(identity+".json")).read_bytes()
                stored = json.loads(wire)
                report = json.loads((root/"reports"/(stored["report_id"]+".json")).read_bytes())
                report["report_id"] = stored["report_id"]
                models = {key: json.loads((root/"models"/(key+".json")).read_bytes()) for key in report["model_ids"]}
                result = evaluate_reference(report, models, configuration(stored["request"]), stored["as_of"], schema_version=VERSION_V1)
                self.assertEqual(result, stored)
                self.assertEqual(digest(result), identity)

    def test_old_lightgbm_baseline_reconstruction_matches_original_training_hash(self):
        root = self._workspace()
        identity = "8c06b437d22e9ab754251f3eec3715514bf145caa9ee2282c538bc7b2c83cea5"
        report = json.loads((root/"reports"/(identity+".json")).read_bytes())
        report["report_id"] = identity
        models = {key: json.loads((root/"models"/(key+".json")).read_bytes()) for key in report["model_ids"]}
        snapshot_id = report["snapshot_id"]
        snapshot = json.loads((root/"snapshots"/(snapshot_id+".json")).read_bytes())
        snapshot["snapshot_id"] = snapshot_id
        config = configuration({"report_id": identity})
        missing = evaluate_reference(report, models, config, AVAILABLE)
        self.assertEqual(missing["comparisons"]["training_mean_return_rank"]["state"], "UNKNOWN")
        self.assertEqual(missing["coverage_state"], "UNKNOWN")
        result = evaluate_reference(report, models, config, AVAILABLE, snapshot=snapshot)
        self.assertEqual(result["comparisons"]["training_mean_return_rank"]["state"], "AVAILABLE")
        self.assertEqual(result["metrics"]["scored_dates"], 80)
        self.assertTrue(all(r["training_baseline_evidence"] == "RECONSTRUCTED_FROM_HASH_VERIFIED_ORIGINAL_TRAINING_ROWS" for r in result["date_scores"]))
