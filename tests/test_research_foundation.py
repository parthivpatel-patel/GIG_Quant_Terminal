"""Offline adversarial references for point-in-time admission and research."""

from __future__ import annotations

import hashlib
import json
import math
import sqlite3
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from gig.artifacts import write_immutable
from gig.data.validated import DataSnapshot, PointInTimeStore, admit_bars
from gig.research.workflow import ResearchWorkflow, drift_report, model_capabilities


def bar(**overrides):
    row = {
        "instrument_id": "issuer:permanent-123",
        "timestamp": "2026-01-02T16:00:00-05:00",
        "available_at": "2026-01-02T21:00:01Z",
        "open": 100.0,
        "high": 103.0,
        "low": 98.0,
        "close": 101.0,
        "volume": 2**53 + 1,
    }
    return {**row, **overrides}


def admit(rows):
    return admit_bars(
        rows,
        source="synthetic_reference",
        classification="SYNTHETIC",
        corporate_action_policy="RAW",
    )


class DataAdmissionTests(unittest.TestCase):
    def test_exact_volume_and_nanosecond_precision(self):
        snap = admit([bar(timestamp="2026-01-02T21:00:00.123456789Z")])
        row = snap.rows[0]
        self.assertEqual(row["volume"], "9007199254740993")
        self.assertEqual(row["timestamp"], "2026-01-02T21:00:00.123456789+00:00")
        self.assertEqual(hashlib.sha256(snap.payload).hexdigest(), snap.sha256)

    def test_equivalent_timezones_have_identical_identity(self):
        first = admit([bar()])
        second = admit([bar(timestamp="2026-01-02T21:00:00Z")])
        self.assertEqual(first.sha256, second.sha256)

    def test_subnanosecond_timestamp_is_rejected_without_truncation(self):
        with self.assertRaisesRegex(ValueError, "precision"):
            admit([bar(timestamp="2026-01-02T21:00:00.1234567891Z")])

    def test_mutation_after_admission_cannot_change_snapshot(self):
        row = bar()
        snap = admit([row])
        digest = snap.sha256
        row["close"] = 88
        returned = snap.rows
        returned[0]["close"] = 77
        self.assertEqual(snap.rows[0]["close"], 101)
        self.assertEqual(snap.sha256, digest)

    def test_nonfinite_prices_rejected(self):
        for value in [float("nan"), float("inf"), -float("inf")]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                admit([bar(close=value)])

    def test_invalid_prices_rejected(self):
        for value in [0, -1, True, "101", 10**1000]:
            with self.subTest(value=type(value)), self.assertRaises(ValueError):
                admit([bar(close=value)])

    def test_invalid_ohlc_rejected(self):
        with self.assertRaisesRegex(ValueError, "high"):
            admit([bar(high=100)])
        with self.assertRaisesRegex(ValueError, "low"):
            admit([bar(low=102)])

    def test_naive_and_missing_timestamps_rejected(self):
        for value in ["2026-01-02", None, pd.NaT, "not a timestamp"]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                admit([bar(timestamp=value)])

    def test_available_before_closing_observation_rejected(self):
        with self.assertRaisesRegex(ValueError, "before"):
            admit([bar(available_at="2026-01-02T20:59:59Z")])

    def test_duplicate_revision_rejected(self):
        with self.assertRaisesRegex(ValueError, "duplicate"):
            admit([bar(), bar()])

    def test_instrument_identity_must_be_explicit_string(self):
        for value in [2**60, "", " " * 3]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                admit([bar(instrument_id=value)])

    def test_volume_must_be_nonnegative_bounded_integer(self):
        for value in [-1, 1.5, True, 2**63, "3"]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                admit([bar(volume=value)])

    def test_unknown_source_adjustment_and_classification_rejected(self):
        for kwargs in [
            {"source": "", "classification": "SYNTHETIC", "corporate_action_policy": "RAW"},
            {"source": "x", "classification": "LIVE", "corporate_action_policy": "RAW"},
            {"source": "x", "classification": "SYNTHETIC", "corporate_action_policy": "unknown"},
        ]:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                admit_bars([bar()], **kwargs)

    def test_missing_and_extra_fields_rejected(self):
        raw = bar()
        del raw["volume"]
        with self.assertRaises(ValueError):
            admit([raw])
        with self.assertRaises(ValueError):
            admit([bar(text="preserve raw provider evidence separately")])

    def test_order_independent_hash(self):
        a, b = bar(), bar(instrument_id="issuer:456")
        self.assertEqual(admit([a, b]).sha256, admit([b, a]).sha256)

    def test_hash_verified_file_roundtrip_and_tamper(self):
        with tempfile.TemporaryDirectory() as tmp:
            snap = admit([bar()])
            path = snap.save(tmp)
            self.assertEqual(DataSnapshot.load(path), snap)
            self.assertEqual(snap.save(tmp), path)
            path.write_bytes(b"{}")
            with self.assertRaises(ValueError):
                DataSnapshot.load(path)
            with self.assertRaises(ValueError):
                snap.save(tmp)


class PointInTimeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "bars.sqlite"
        self.store = PointInTimeStore(self.path)

    def test_restart_idempotent_replay(self):
        snap = admit([bar()])
        self.assertEqual(self.store.ingest(snap), {"inserted": 1, "replayed": 0})
        restarted = PointInTimeStore(self.path)
        self.assertEqual(restarted.ingest(snap), {"inserted": 0, "replayed": 1})

    def test_future_revision_never_leaks_into_past_snapshot(self):
        self.store.ingest(admit([bar(), bar(close=102.0, available_at="2026-01-03T00:00:00Z")]))
        earlier = self.store.snapshot("2026-01-02T22:00:00Z")
        later = self.store.snapshot("2026-01-03T00:00:00Z")
        self.assertEqual(earlier["rows"][0]["close"], 101.0)
        self.assertEqual(later["rows"][0]["close"], 102.0)
        self.assertEqual(len(later["rows"]), 1)

    def test_conflicting_revision_rolls_back_whole_batch(self):
        self.store.ingest(admit([bar()]))
        with self.assertRaisesRegex(ValueError, "conflicting"):
            self.store.ingest(admit([bar(instrument_id="new"), bar(close=102.0)]))
        snapshot = self.store.snapshot("2026-01-04T00:00:00Z")
        self.assertEqual(len(snapshot["rows"]), 1)
        self.assertEqual(snapshot["rows"][0]["close"], 101.0)

    def test_no_rows_before_knowledge_time(self):
        self.store.ingest(admit([bar()]))
        self.assertEqual(self.store.snapshot("2026-01-02T21:00:00Z")["state"], "UNAVAILABLE")

    def test_explicit_fresh_unknown_stale_states(self):
        self.store.ingest(admit([bar()]))
        self.assertEqual(self.store.snapshot("2026-01-02T21:00:01Z")["state"], "UNKNOWN")
        self.assertEqual(
            self.store.snapshot("2026-01-02T21:00:01Z", max_age_seconds=5)["state"], "AVAILABLE"
        )
        self.assertEqual(
            self.store.snapshot("2026-01-02T21:00:10Z", max_age_seconds=5)["state"], "STALE"
        )
        with self.assertRaises(ValueError):
            self.store.snapshot("2026-01-02T21:00:10Z", max_age_seconds=-1)

    def test_freshness_uses_latest_each_instrument_not_oldest_bar(self):
        self.store.ingest(
            admit(
                [bar(), bar(timestamp="2026-01-03T21:00:00Z", available_at="2026-01-03T21:00:01Z")]
            )
        )
        snapshot = self.store.snapshot("2026-01-03T21:00:02Z", max_age_seconds=5)
        self.assertEqual(snapshot["state"], "AVAILABLE")
        self.assertEqual(snapshot["latest_instrument_max_age_seconds"], 2)
        self.assertGreater(snapshot["oldest_observation_age_seconds"], 86400)

    def test_subnanosecond_ordering_is_not_text_lexical(self):
        self.store.ingest(
            admit(
                [
                    bar(timestamp="2026-01-02T21:00:00Z", available_at="2026-01-02T21:00:01Z"),
                    bar(close=102.0, available_at="2026-01-02T21:00:01.000000001Z"),
                ]
            )
        )
        early = self.store.snapshot("2026-01-02T21:00:01Z")
        late = self.store.snapshot("2026-01-02T21:00:01.000000001Z")
        self.assertEqual(early["rows"][0]["close"], 101.0)
        self.assertEqual(late["rows"][0]["close"], 102.0)

    def test_source_classification_and_adjustment_do_not_mix(self):
        self.store.ingest(admit([bar()]))
        other = admit_bars(
            [bar()],
            source="historical_reference",
            classification="HISTORICAL",
            corporate_action_policy="SPLIT_ADJUSTED",
        )
        self.store.ingest(other)
        self.assertEqual(len(self.store.snapshot("2026-01-04T00:00:00Z")["rows"]), 2)
        filtered = self.store.snapshot("2026-01-04T00:00:00Z", source="synthetic_reference")
        self.assertEqual(len(filtered["rows"]), 1)
        self.assertEqual(filtered["rows"][0]["classification"], "SYNTHETIC")

    def test_freshness_does_not_merge_different_adjustment_series(self):
        self.store.ingest(admit([bar()]))
        fresh = bar(timestamp="2026-01-03T21:00:00Z", available_at="2026-01-03T21:00:01Z")
        self.store.ingest(
            admit_bars(
                [fresh],
                source="synthetic_reference",
                classification="SYNTHETIC",
                corporate_action_policy="SPLIT_ADJUSTED",
            )
        )
        snapshot = self.store.snapshot("2026-01-03T21:00:02Z", max_age_seconds=5)
        self.assertEqual(snapshot["state"], "STALE")
        self.assertGreater(snapshot["latest_instrument_max_age_seconds"], 86400)

    def test_tampered_revision_is_rejected_against_original_snapshot(self):
        self.store.ingest(admit([bar()]))
        with sqlite3.connect(self.path) as connection:
            raw = json.loads(connection.execute("SELECT row_json FROM bar_revisions").fetchone()[0])
            raw["close"] = 102
            connection.execute("UPDATE bar_revisions SET row_json=?", (json.dumps(raw),))
        with self.assertRaisesRegex(ValueError, "source snapshot"):
            self.store.snapshot("2026-01-04T00:00:00Z")

    def test_snapshot_anchor_tamper_or_absence_fails_closed(self):
        self.store.ingest(admit([bar()]))
        with sqlite3.connect(self.path) as connection:
            connection.execute("UPDATE bar_snapshot_evidence SET payload=?", (b"{}",))
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            self.store.snapshot("2026-01-04T00:00:00Z")
        with sqlite3.connect(self.path) as connection:
            connection.execute("DELETE FROM bar_snapshot_evidence")
        with self.assertRaisesRegex(ValueError, "evidence absent"):
            self.store.snapshot("2026-01-04T00:00:00Z")

    def test_row_identity_columns_cannot_diverge_from_source(self):
        self.store.ingest(admit([bar()]))
        with sqlite3.connect(self.path) as connection:
            connection.execute("UPDATE bar_revisions SET instrument='other'")
        with self.assertRaisesRegex(ValueError, "source snapshot"):
            self.store.snapshot("2026-01-04T00:00:00Z")

    def test_boolean_freshness_policy_is_not_numeric_authority(self):
        with self.assertRaises(ValueError):
            self.store.snapshot("2026-01-04T00:00:00Z", max_age_seconds=True)


def dataset(n=120):
    index = pd.date_range("2026-01-01", periods=n, freq="D", tz="UTC")
    rng = np.random.default_rng(7)
    x = pd.DataFrame({"factor_a": rng.normal(size=n), "factor_b": rng.normal(size=n)}, index=index)
    y = 2 * x.factor_a - 0.4 * x.factor_b + 0.3
    ends = pd.Series(index + pd.Timedelta(days=4), index=index)
    available = pd.Series(index, index=index)
    return x, y, ends, available


class ResearchWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.workflow = ResearchWorkflow(self.temp.name)
        self.x, self.y, self.ends, self.available = dataset()

    def run_research(self, **kwargs):
        return self.workflow.run(
            self.x,
            self.y,
            label_end=self.ends,
            feature_available_at=self.available,
            min_train=40,
            test_size=20,
            embargo=1,
            **kwargs,
        )

    def test_reproducible_hash_and_no_future_label_leakage(self):
        first, second = self.run_research(), self.run_research()
        self.assertEqual(first, second)
        self.assertEqual(first["metrics"]["oos_count"], 60)
        self.assertFalse(first["trading_approved"])
        for fold in first["folds"]:
            self.assertLess(
                pd.Timestamp(fold["max_training_label_end"]), pd.Timestamp(fold["test_start"])
            )
            self.assertEqual(fold["purged_count"], 3)
            self.assertEqual(fold["uncertainty_state"], "NOT_OOS_CALIBRATED")
        self.assertLess(first["metrics"]["mse"], 0.02)

    def test_scaler_fitted_on_train_rows_only(self):
        self.x.iloc[41:] += 1000
        result = self.run_research()
        first = result["folds"][0]
        count = first["train_count"]
        np.testing.assert_allclose(first["train_only_mean"], self.x.iloc[:count].mean().to_numpy())
        self.assertLess(abs(first["train_only_mean"][0]), 1.0)

    def test_registry_predictions_match_oos_and_tamper_rejected(self):
        result = self.run_research()
        fold = result["folds"][0]
        test = self.x.loc[fold["test_start"] : fold["test_end"]]
        prediction = self.workflow.registry.predict(fold["model_id"], test)
        expected = [row["prediction"] for row in result["predictions"] if row["fold"] == 0]
        np.testing.assert_allclose(prediction, expected, rtol=0, atol=1e-14)
        path = self.workflow.registry.directory / f"{fold['model_id']}.json"
        path.write_text("{}")
        with self.assertRaisesRegex(ValueError, "hash"):
            self.workflow.registry.load(fold["model_id"])

    def test_ridge_matches_independent_regularized_normal_equations(self):
        result = self.run_research()
        fold = result["folds"][0]
        state = self.workflow.registry.load(fold["model_id"])
        n = fold["train_count"]
        x = self.x.iloc[:n].to_numpy()
        y = self.y.iloc[:n].to_numpy()
        z = (x - x.mean(0)) / x.std(0)
        expected = np.linalg.inv(z.T @ z + np.eye(2)) @ z.T @ (y - y.mean())
        np.testing.assert_allclose(state["coefficients"], expected, atol=1e-12)
        self.assertEqual(state["intercept"], float(y.mean()))

    def test_future_feature_availability_rejected(self):
        self.available.iloc[50] += pd.Timedelta(seconds=1)
        with self.assertRaisesRegex(ValueError, "look-ahead"):
            self.run_research()

    def test_label_ends_before_observation_rejected(self):
        self.ends.iloc[0] -= pd.Timedelta(days=5)
        with self.assertRaisesRegex(ValueError, "precede"):
            self.run_research()

    def test_unsorted_duplicate_and_naive_features_rejected(self):
        for replacement in [
            self.x.iloc[::-1],
            pd.concat([self.x.iloc[:1], self.x]),
            self.x.tz_localize(None),
        ]:
            with self.subTest(index=replacement.index), self.assertRaises(ValueError):
                self.workflow.run(replacement, self.y, label_end=self.ends)

    def test_nonfinite_features_and_labels_rejected(self):
        self.x.iloc[0, 0] = math.inf
        with self.assertRaisesRegex(ValueError, "finite"):
            self.run_research()
        self.x.iloc[0, 0] = 1
        self.y.iloc[0] = math.nan
        with self.assertRaisesRegex(ValueError, "finite"):
            self.run_research()

    def test_unavailable_lightgbm_never_substitutes(self):
        with patch("gig.research.workflow._exists", return_value=False):
            with self.assertRaisesRegex(ImportError, "no model substitution"):
                self.run_research(model="lightgbm")
            self.assertEqual(model_capabilities()["lightgbm"]["state"], "UNAVAILABLE")
            self.assertTrue(model_capabilities()["finbert"]["implemented"])
            self.assertFalse(model_capabilities()["finbert"]["load_verified"])

    def test_invalid_config_bounds_and_unknown_model_rejected(self):
        for kwargs in [
            {"min_train": 0},
            {"test_size": 0},
            {"embargo": -1},
            {"model": "auto"},
            {"ridge_alpha": 0},
            {"seed": -1},
        ]:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.workflow.run(self.x, self.y, label_end=self.ends, **kwargs)

    def test_no_fold_and_all_labels_purged_rejected(self):
        with self.assertRaisesRegex(ValueError, "no complete"):
            self.workflow.run(self.x, self.y, label_end=self.ends, min_train=200)
        ends = pd.Series(self.x.index[-1] + pd.Timedelta(days=1), index=self.x.index)
        with self.assertRaisesRegex(ValueError, "too few"):
            self.workflow.run(self.x, self.y, label_end=ends, min_train=40, test_size=20)

    def test_constant_features_and_targets_remain_finite_unknown_r2(self):
        x = self.x * 0
        y = self.y * 0 + 1
        result = self.workflow.run(x, y, label_end=self.ends, min_train=40, test_size=20)
        self.assertIsNone(result["metrics"]["r2"])
        self.assertEqual(result["metrics"]["mse"], 0.0)
        self.assertTrue(np.isfinite([r["prediction"] for r in result["predictions"]]).all())

    def test_inference_requires_same_feature_order(self):
        result = self.run_research()
        model_id = result["folds"][0]["model_id"]
        with self.assertRaisesRegex(ValueError, "feature names/order"):
            self.workflow.registry.predict(model_id, self.x[["factor_b", "factor_a"]])
        with self.assertRaises(ValueError):
            self.workflow.registry.load("../arbitrary")

    def test_artifact_hash_matches_persisted_bytes(self):
        result = self.run_research()
        path = Path(self.temp.name) / f"{result['run_id']}.json"
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), result["run_id"])
        persisted = json.loads(path.read_bytes())
        self.assertEqual(persisted["dataset_hash"], result["dataset_hash"])


class DriftTests(unittest.TestCase):
    def test_huge_finite_samples_do_not_overflow_drift_diagnostics(self):
        values = np.linspace(-1, 1, 100) * 1e308
        with np.errstate(all="raise"):
            report = drift_report(values, values)
        self.assertEqual(report["psi"], 0.0)
        self.assertEqual(report["standardized_mean_shift"], 0.0)

    def test_identity_psi_zero_and_shift_positive(self):
        x = np.linspace(-1, 1, 100)
        self.assertEqual(drift_report(x, x)["psi"], 0.0)
        shifted = drift_report(x, x + 3)
        self.assertGreater(shifted["psi"], 1)
        self.assertGreater(shifted["standardized_mean_shift"], 3)

    def test_constant_reference_mean_shift_unknown(self):
        result = drift_report(np.ones(20), np.ones(20) * 2)
        self.assertIsNone(result["standardized_mean_shift"])
        self.assertEqual(result["state"], "UNKNOWN")
        self.assertTrue(math.isfinite(result["psi"]))

    def test_bad_drift_inputs_rejected(self):
        for a, b in [([1], [2]), ([math.nan] * 10, list(range(10))), ([[1] * 10], list(range(10)))]:
            with self.subTest(a=a), self.assertRaises(ValueError):
                drift_report(a, b)
        with self.assertRaises(ValueError):
            drift_report(range(10), range(10), bins=0)


class AtomicArtifactTests(unittest.TestCase):
    def test_concurrent_identical_writers_replay_complete_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "artifact.json"
            payload = b'{"source":"exact research evidence"}'
            with ThreadPoolExecutor(max_workers=4) as executor:
                list(executor.map(lambda _: write_immutable(path, payload), range(16)))
            self.assertEqual(path.read_bytes(), payload)
            self.assertEqual(list(Path(tmp).glob(".admit-*")), [])

    def test_publish_failure_never_leaves_partial_target(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "artifact.json"
            with patch("gig.artifacts.os.link", side_effect=OSError("injected publish failure")), self.assertRaises(OSError):
                write_immutable(path, b"complete payload")
            self.assertFalse(path.exists())
            self.assertEqual(list(Path(tmp).glob(".admit-*")), [])

    def test_conflict_preserves_original_and_rejects_symlink(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "artifact.json"
            write_immutable(path, b"original")
            with self.assertRaises(ValueError):
                write_immutable(path, b"conflicting")
            self.assertEqual(path.read_bytes(), b"original")
            link = Path(tmp) / "link.json"
            link.symlink_to(path)
            with self.assertRaises(ValueError):
                write_immutable(link, b"original")


if __name__ == "__main__":
    unittest.main()
