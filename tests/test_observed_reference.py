"""Reference provenance, real CPU adapters and date-only research boundaries."""
from __future__ import annotations

import csv
import io
import sqlite3
import tempfile
import unittest
from datetime import date, timedelta
from pathlib import Path

import numpy as np

from gig.data.public_reference import PROVIDER, admit_fixture, parse_csv
from gig.ml.state_space import kalman_filter
from gig.research.reference_research import _features, _fit, _predict, configuration
from gig.service.contracts import EnvelopeSchema
from gig.service.observed_workspace import ObservedWorkspace

ACQUIRED = "2026-11-03T13:00:00.123456789Z"


def reference_fixture(periods=100):
    generator = np.random.default_rng(71)
    prices = np.exp(np.cumsum(generator.normal(0, .006, (periods, 3)), axis=0))*[1.1, .85, 150]
    text = io.StringIO()
    writer = csv.writer(text)
    writer.writerow(["Date", "USD", "GBP", "JPY"])
    for i, row in enumerate(prices):
        writer.writerow([(date(2025, 1, 1)+timedelta(days=i)).isoformat(), *row])
    return {"provider": PROVIDER, "currencies": ["USD", "GBP", "JPY"], "start": "2025-01-01",
            "end": (date(2025, 1, 1)+timedelta(days=periods-1)).isoformat(), "csv_text": text.getvalue(), "acquired_at": ACQUIRED}


class ObservedReferenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.base = Path(self.temp.name)/"state"
        self.workspace = ObservedWorkspace(self.base)

    def tearDown(self):
        self.temp.cleanup()

    def admit(self, payload=None):
        result = self.workspace.run({"operation": "admit", **(payload or reference_fixture())}).to_dict()
        EnvelopeSchema.model_validate(result)
        return result["data"]

    def test_date_only_source_keeps_actual_acquisition_missing_volume_and_scope(self):
        result = self.admit()
        self.assertEqual(result["classification"], "SYNTHETIC")
        self.assertFalse(result["point_in_time_backtest_eligible"])
        self.assertFalse(result["funded_allocation_eligible"])
        row = result["rows"][0]
        self.assertEqual(row["observation_date"], "2025-01-01")
        self.assertIn("123456789", row["acquired_at"])
        self.assertIsNone(row["observation_time"])
        self.assertIsNone(row["volume"])
        self.assertNotIn("open", row)
        self.assertNotIn("timestamp", row)
        self.assertEqual(row["knowledge_at"], row["acquired_at"])
        self.assertEqual(result["diagnostics"]["imputed_values"], 0)
        self.assertEqual(self.workspace.section("markets", "2025-03-01T00:00:00Z", 60), None)
        self.assertEqual(self.workspace.section("markets", ACQUIRED, None).to_dict()["state"], "UNKNOWN")

    def test_all_six_real_cpu_models_persist_and_replay_without_trading_claim(self):
        snapshot = self.admit()
        for model in ("ridge", "lightgbm", "hmm", "garch", "kalman", "bilstm"):
            with self.subTest(model=model):
                result = self.workspace.run({"operation": "research", "snapshot_id": snapshot["snapshot_id"],
                    "model": model, "training_observations": 30, "test_observations": 10, "embargo": 1, "seed": 17}).to_dict()
                EnvelopeSchema.model_validate(result)
                self.assertEqual(result["state"], "AVAILABLE")
                report = result["data"]
                self.assertFalse(report["historical_point_in_time_validated"])
                self.assertFalse(report["funded_allocation_eligible"])
                self.assertEqual(report["capacity_state"], "UNKNOWN")
                self.assertEqual(report["portfolio_pnl_state"], "UNAVAILABLE")
                self.assertTrue(report["model_ids"])
                self.assertEqual(report["actual_acquired_at"], snapshot["acquired_at"])
                for fold in report["folds"]:
                    self.assertLess(fold["training_end_observation_date"], fold["first_test_observation_date"])
                replay = ObservedWorkspace(self.base).run({"operation": "replay", "report_id": report["report_id"]}).to_dict()
                self.assertEqual(result, replay)
                detached = self.workspace.load(report["report_id"])
                detached["rows"][0]["prediction"] = 999
                self.assertNotEqual(self.workspace.load(report["report_id"])["rows"][0]["prediction"], 999)

    def test_future_observation_corrections_cannot_change_earlier_fitted_numbers(self):
        original = reference_fixture()
        first = self.admit(original)
        config = {"operation": "research", "snapshot_id": first["snapshot_id"], "model": "ridge", "training_observations": 30, "test_observations": 10}
        old = self.workspace.run(config).to_dict()["data"]
        lines = original["csv_text"].splitlines()
        tail = lines[-1].split(",")
        tail[1] = str(float(tail[1])*2)
        lines[-1] = ",".join(tail)
        corrected = {**original, "csv_text": "\n".join(lines), "acquired_at": "2026-11-03T13:10:00Z"}
        second = self.admit(corrected)
        new = self.workspace.run({**config, "snapshot_id": second["snapshot_id"]}).to_dict()["data"]
        self.assertEqual([r["prediction"] for r in old["rows"][:30]], [r["prediction"] for r in new["rows"][:30]])
        self.assertEqual(self.workspace.load(old["report_id"])["rows"], old["rows"])
        self.assertNotEqual(first["snapshot_id"], second["snapshot_id"])

    def test_missing_quote_is_preserved_and_model_rejects_without_imputation(self):
        payload = reference_fixture()
        lines = payload["csv_text"].splitlines()
        values = lines[20].split(",")
        values[1] = "N/A"
        lines[20] = ",".join(values)
        payload["csv_text"] = "\n".join(lines)
        snapshot = self.admit(payload)
        self.assertEqual(snapshot["diagnostics"]["missing_quotes"], 1)
        with self.assertRaisesRegex(ValueError, "missing"):
            self.workspace.run({"operation": "research", "snapshot_id": snapshot["snapshot_id"], "model": "ridge"})
        self.assertFalse((self.base/"observed-reference/reports").exists())

    def test_invalid_bounds_duplicates_nonfinite_and_future_acquisition_reject(self):
        original = reference_fixture()
        cases = [{**original, "currencies": ["USD", "USD"]}, {**original, "currencies": ["XXX"]},
                 {**original, "acquired_at": "2024-01-01T00:00:00Z"},
                 {**original, "start": "2024-01-01"}, {**original, "csv_text": original["csv_text"]+original["csv_text"].splitlines()[1]+"\n"}]
        for payload in cases:
            with self.subTest(case=list(payload.keys())), self.assertRaises(ValueError):
                admit_fixture(payload)
        invalid = original["csv_text"].replace(original["csv_text"].splitlines()[1].split(",")[1], "NaN", 1)
        with self.assertRaises(ValueError):
            parse_csv(invalid, original, ACQUIRED, observed=False)
        self.assertFalse(self.base.exists())

    def test_integrity_checks_raw_models_and_index(self):
        snapshot = self.admit()
        result = self.workspace.run({"operation": "research", "snapshot_id": snapshot["snapshot_id"], "model": "ridge", "training_observations": 30, "test_observations": 10}).to_dict()["data"]
        with sqlite3.connect(self.base/"observed-reference/reference.sqlite") as connection:
            connection.execute("UPDATE entries SET as_of='2020-01-01T00:00:00Z' WHERE kind='report'")
        with self.assertRaisesRegex(ValueError, "index"):
            self.workspace.load(result["report_id"])
        raw = self.base/"observed-reference/raw"/(snapshot["raw_sha256"]+".bin")
        raw.write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "source evidence"):
            self.workspace.snapshot(snapshot["snapshot_id"])

    def test_missing_read_has_no_state_or_synthetic_fallback(self):
        self.assertIsNone(self.workspace.section("markets", ACQUIRED, 60))
        result = self.workspace.run({"operation": "replay", "report_id": "0"*64}).to_dict()
        self.assertEqual(result["state"], "UNAVAILABLE")
        self.assertFalse(self.base.exists())

    def test_embargo_conditions_filter_on_known_intervening_returns_without_refitting(self):
        snapshot = self.admit()
        rows = _features(snapshot)["ECB:EXR:USD:PER_EUR"]
        config = configuration({"snapshot_id": snapshot["snapshot_id"], "model": "kalman", "training_observations": 30, "test_observations": 10, "embargo": 3})
        state = _fit("kalman", rows[:30], config)
        trained_variance = state["variance"]
        state["conditioning_returns"] = [row["features"][0] for row in rows[31:33]]
        prediction = _predict(state, rows[33:34])[0]
        history = [row["target"] for row in rows[:30]]+[row["features"][0] for row in rows[31:34]]
        expected = kalman_filter(np.asarray(history)[:, None], [[1.]], [[1.]], [[trained_variance*.01]], [[trained_variance]], [history[0]], [[trained_variance]])
        self.assertAlmostEqual(prediction, float(expected.filtered_mean[-1, 0]), places=15)
        self.assertEqual(state["variance"], trained_variance)
        full = self.workspace.run({"operation": "research", **config}).to_dict()["data"]
        first_model = self.workspace._path("models/"+full["folds"][0]["model_id"]+".json")
        from gig.service.serialization import loads
        saved = loads(first_model.read_bytes())
        self.assertEqual(saved["conditioning_observation_dates"], [row["observation_date"] for row in _features(snapshot)[saved["instrument_id"]][31:33]])
