"""Observed Treasury follow-up: numerical references and temporal adversaries."""

from __future__ import annotations

import copy
import hashlib
import json
import unittest
from decimal import Decimal
from unittest.mock import patch

from scipy.stats import binomtest

from gig.research_org.treasury_experiment import _load, run_treasury
from gig.research_org.treasury_followup import (
    HYPOTHESIS_ID,
    WINDOWS,
    _holm,
    challenge,
    evaluate_series,
    run_followup,
)


class FollowupTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.meta, cls.series = _load()
        cls.digest, cls.report = run_followup()

    def test_known_observed_result_matches_independent_integer_reference(self) -> None:
        metrics = self.report["metrics"]
        hits = baseline_hits = pairs = 0
        for start, end in WINDOWS:
            # Integer basis points supply a separate numeric representation.
            values = [int(Decimal(row["yield_percent"]) * 100) for row in self.series]
            training = []
            for index in range(2, start - 1):
                if values[index - 1] != values[index - 2] and values[index] != values[index - 1]:
                    training.append(values[index] > values[index - 1])
            baseline = sum(training) >= len(training) - sum(training)
            for index in range(start, end):
                if values[index - 1] == values[index - 2] or values[index] == values[index - 1]:
                    continue
                predicted_up = values[index - 1] < values[index - 2]
                actual_up = values[index] > values[index - 1]
                hits += predicted_up == actual_up
                baseline_hits += baseline == actual_up
                pairs += 1
        self.assertEqual((hits, pairs, baseline_hits), (61, 115, 58))
        self.assertEqual((metrics["hits"], metrics["pairs"], metrics["baseline_hits"]), (hits, pairs, baseline_hits))
        self.assertAlmostEqual(metrics["hit_rate"], 61 / 115)
        self.assertAlmostEqual(metrics["binomial_p_value"], 0.5760121204363976)
        self.assertAlmostEqual(metrics["hit_rate_interval_95"][0], 0.43512946573133937)
        self.assertAlmostEqual(metrics["hit_rate_interval_95"][1], 0.6241346647487588)
        self.assertEqual([row["hits"] for row in metrics["folds"]], [19, 21, 21])
        self.assertEqual([row["baseline_hits"] for row in metrics["folds"]], [23, 19, 16])
        self.assertEqual([row["excluded_zero_pairs"] for row in metrics["folds"]], [2, 4, 4])

    def test_parent_negative_result_is_preserved(self) -> None:
        _, parent = run_treasury()
        self.assertEqual((parent["hits"], parent["pairs"]), (105, 228))
        self.assertEqual((self.report["parent"]["hits"], self.report["parent"]["pairs"]), (105, 228))
        self.assertLess(parent["hit_rate"], 0.5)
        self.assertEqual(parent["verdict"], "INCONCLUSIVE")

    def test_deterministic_replay_and_content_hash(self) -> None:
        digest, replay = run_followup()
        self.assertEqual((digest, replay), (self.digest, self.report))
        body = copy.deepcopy(replay)
        expected = body.pop("result_hash")
        self.assertEqual(hashlib.sha256(json.dumps(body, sort_keys=True, allow_nan=False).encode()).hexdigest(), expected)
        self.assertEqual(len(expected), 64)
        self.assertEqual(replay["hypothesis_id"], HYPOTHESIS_ID)
        self.assertEqual(replay["dataset"]["sha256"], self.meta["sha256"])
        json.dumps(replay, allow_nan=False)

    def test_future_revisions_cannot_change_earlier_predictions_or_baselines(self) -> None:
        changed = copy.deepcopy(self.series)
        for index in range(180, len(changed)):
            changed[index]["yield_percent"] = str(Decimal(5) + Decimal(index % 7) / 100)
        original = self.report["metrics"]["prediction_rows"]
        perturbed = evaluate_series(changed)["prediction_rows"]
        def before(rows):
            return [row for row in rows if row["observation_index"] < 180]
        self.assertEqual(before(original), before(perturbed))
        # A later change affects later scores, rather than being silently ignored.
        self.assertNotEqual(original, perturbed)

    def test_purge_keeps_training_targets_before_validation_features(self) -> None:
        for row in self.report["metrics"]["prediction_rows"]:
            self.assertLess(row["last_training_target_date"], row["feature_date"])
            self.assertLess(row["feature_date"], row["observation_date"])
        folds = self.report["metrics"]["folds"]
        self.assertEqual([row["training_pairs"] for row in folds], [112, 151, 188])
        self.assertEqual([row["training_last_target_date"] for row in folds], ["2023-06-29", "2023-08-28", "2023-10-26"])

    def test_skeptic_recomputes_and_rejects_tampered_predictions_and_counts(self) -> None:
        reference = self.report["skeptic"]
        self.assertEqual(reference["verdict"], "INCONCLUSIVE")
        self.assertTrue(all(reference["checks"].values()))
        for field in ("hits", "pairs", "baseline_hits"):
            changed = copy.deepcopy(self.report["metrics"])
            changed[field] += 1
            result = challenge(self.series, changed)
            self.assertEqual(result["verdict"], "REJECTED_NUMERICAL_OR_TEMPORAL_INTEGRITY")
        changed = copy.deepcopy(self.report["metrics"])
        changed["prediction_rows"][0]["prediction"] *= -1
        self.assertFalse(challenge(self.series, changed)["checks"]["independent_prediction_reference"])
        changed = copy.deepcopy(self.report["metrics"])
        changed["hit_rate"] = 1.0
        self.assertFalse(challenge(self.series, changed)["checks"]["independent_accuracy_reference"])

    def test_unsupported_publication_and_economic_claims_stay_unavailable(self) -> None:
        self.assertEqual(self.report["classification"], "HISTORICAL")
        self.assertEqual(self.report["publication_time"], "UNKNOWN")
        self.assertEqual(self.report["revision_knowledge_time"], "UNKNOWN")
        self.assertTrue(self.report["post_selection_reuse"])
        self.assertFalse(self.report["independent_confirmation"])
        self.assertEqual(self.report["economic_claim"], "NONE")
        self.assertEqual(self.report["allocation_eligibility"], "UNAVAILABLE")
        self.assertEqual(self.report["orders_submitted"], 0)
        self.assertEqual(self.report["models_promoted"], 0)
        for field in ("total_return", "sharpe_after_cost", "max_drawdown"):
            self.assertIsNone(self.report[field])

    def test_multiplicity_corrects_declared_local_family(self) -> None:
        self.assertEqual(_holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])
        family = self.report["multiple_testing"]
        self.assertEqual(family["family_size"], 3)
        self.assertAlmostEqual(family["raw_p_values"][0], binomtest(105, 228, p=0.5).pvalue)
        self.assertTrue(all(adjusted >= raw for raw, adjusted in zip(family["raw_p_values"], family["adjusted_p_values"], strict=True)))
        self.assertTrue(all(0 <= value <= 1 for value in family["adjusted_p_values"]))

    def test_invalid_or_unavailable_datasets_are_rejected(self) -> None:
        cases = []
        cases.append(self.series[:-1])
        for key, value in (("event_date", "2023-02-30"), ("tenor", "BC_2YEAR"),
                           ("yield_percent", "NaN"), ("yield_percent", "Infinity"),
                           ("yield_percent", "invalid"), ("yield_percent", "1001")):
            changed = copy.deepcopy(self.series)
            changed[4][key] = value
            cases.append(changed)
        duplicate = copy.deepcopy(self.series)
        duplicate[1]["event_date"] = duplicate[0]["event_date"]
        cases.append(duplicate)
        missing = copy.deepcopy(self.series)
        del missing[3]["yield_percent"]
        cases.append(missing)
        constant = copy.deepcopy(self.series)
        for row in constant:
            row["yield_percent"] = "4.0"
        cases.append(constant)
        for rows in cases:
            with self.subTest(first=rows[0]), self.assertRaises(ValueError):
                evaluate_series(rows)
        with patch("gig.research_org.treasury_followup._load", side_effect=FileNotFoundError), self.assertRaises(FileNotFoundError):
            run_followup()

    def test_corrupt_source_bytes_fail_before_research(self) -> None:
        with patch("gig.research_org.treasury_experiment.FIXTURE") as fixture:
            fixture.read_bytes.return_value = b"corrupt Treasury source"
            with self.assertRaisesRegex(RuntimeError, "hash mismatch"):
                run_followup()


if __name__ == "__main__":
    unittest.main()
