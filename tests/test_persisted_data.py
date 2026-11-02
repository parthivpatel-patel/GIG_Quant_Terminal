"""Persisted causal features, universe/action knowledge and immutable replay."""

from __future__ import annotations

import math
import sqlite3
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from gig.data.actions import ActionStore, admit_actions
from gig.data.ingestion import UniverseStore
from gig.data.validated import PointInTimeStore, admit_bars
from gig.research.persisted_data import build_document, digest
from gig.service.contracts import EnvelopeSchema
from gig.service.dataset_workspace import PersistedDatasetWorkspace

SOURCE = "authored-persisted-research-fixture-v1"
UNIVERSE = "authored-pit-universe-v1"
INSTRUMENTS = [
    {"instrument_id": "US:AAA", "symbol": "AAA", "calendar": "XNYS", "currency": "USD"},
    {"instrument_id": "US:BBB", "symbol": "BBB", "calendar": "XNYS", "currency": "USD"},
]


def fixture_times(periods=80):
    return pd.date_range("2026-01-02T20:00:00.000000123Z", periods=periods, freq="D")


def dataset_request(*, periods=80, **changes):
    stamps = fixture_times(periods)
    return {"source": SOURCE, "classification": "SYNTHETIC", "currency": "USD", "universe_id": UNIVERSE,
            "start": stamps[0].isoformat(), "end": stamps[-1].isoformat(),
            "as_of": (stamps[-1] + pd.Timedelta(days=1)).isoformat(),
            "decision_delay_seconds": 0, "horizon": 1, "lookback": 5, "adjustment": "RAW", **changes}


def fixture_bars(periods=80):
    rows = []
    for index, stamp in enumerate(fixture_times(periods)):
        for instrument, identity in enumerate(INSTRUMENTS):
            close = (100 + instrument * 30) * math.exp(.001 * index + .018 * math.sin(index / (3. + instrument))
                                                     + .007 * math.cos(index * 1.73 + instrument))
            rows.append({"instrument_id": identity["instrument_id"], "timestamp": stamp.isoformat(),
                         "available_at": stamp.isoformat(), "open": close, "high": close,
                         "low": close, "close": close, "volume": 1000 + 7 * index + instrument})
    return rows


def ingest_fixture_bars(base_dir: Path, rows):
    return PointInTimeStore(base_dir / "bars.sqlite").ingest(admit_bars(
        rows, source=SOURCE, classification="SYNTHETIC", corporate_action_policy="RAW", currency="USD"))


def seed_persisted_fixture(base_dir: Path, *, periods=80, rows=None, members=None):
    """Two authored assets with explicit acquisition clocks; no public data claim."""
    base_dir = Path(base_dir)
    ingest_fixture_bars(base_dir, fixture_bars(periods) if rows is None else rows)
    known = fixture_times(periods)[0] - pd.Timedelta(days=1)
    UniverseStore(base_dir / "universe.sqlite").ingest({
        "universe_id": UNIVERSE, "source": "authored-universe-fixture",
        "effective_at": known.isoformat(), "available_at": known.isoformat(),
        "members": INSTRUMENTS if members is None else members})
    return dataset_request(periods=periods)


class PersistedDataTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.base = Path(self.temporary.name) / "workspace"
        self.workspace = PersistedDatasetWorkspace(self.base)

    def tearDown(self):
        self.temporary.cleanup()

    def decoded(self, contract):
        value = contract.to_dict()
        EnvelopeSchema.model_validate(value)
        return value

    def build(self, request=None):
        return self.decoded(self.workspace.build(request or dataset_request()))

    def test_missing_reads_and_admission_do_not_create_state(self):
        self.assertIsNone(self.workspace.section("research", dataset_request()["as_of"], 60))
        self.assertEqual(self.build()["state"], "UNAVAILABLE")
        with self.assertRaises(FileNotFoundError):
            self.workspace.load("0" * 64)
        self.assertFalse(self.base.exists())

    def test_persisted_rows_build_replay_reload_and_defensive_copy(self):
        request = seed_persisted_fixture(self.base)
        first = self.build(request)
        self.assertEqual(first["state"], "AVAILABLE")
        self.assertEqual(first["provenance"]["classification"], "SYNTHETIC")
        identity = first["data"]["dataset_id"]
        self.assertEqual(first, self.build(request))
        saved = PersistedDatasetWorkspace(self.base).load(identity)
        self.assertEqual(digest({k: v for k, v in saved.items() if k != "dataset_id"}), identity)
        self.assertEqual(len(saved["rows"]), 148)
        self.assertEqual(len(saved["panel_rows"]), 160)
        saved["rows"][0]["features"]["momentum"] = 12345
        self.assertNotEqual(self.workspace.load(identity)["rows"][0]["features"]["momentum"], 12345)

    def test_feature_and_label_knowledge_are_distinct_and_ns_preserved(self):
        seed_persisted_fixture(self.base)
        data = self.build()["data"]
        for row in data["rows"]:
            self.assertLessEqual(pd.Timestamp(row["available_at"]), pd.Timestamp(row["timestamp"]))
            self.assertGreater(pd.Timestamp(row["label_end"]), pd.Timestamp(row["timestamp"]))
            self.assertIn(".000000123", row["timestamp"])
        self.assertEqual(data["provenance"]["stored_corporate_action_policy"], "RAW")
        self.assertEqual(data["diagnostics"]["missing_values_imputed"], 0)
        self.assertFalse(data["trading_approved"])

    def test_late_revision_never_rewrites_past_features_but_labels_record_knowledge(self):
        seed_persisted_fixture(self.base)
        before = build_document(self.base, dataset_request())
        correction = deepcopy(fixture_bars()[40])
        correction.update(available_at=fixture_times()[60].isoformat(), open=150., high=150., low=150., close=150.)
        ingest_fixture_bars(self.base, [correction])
        after = build_document(self.base, dataset_request())
        def at(data, position):
            return next(r for r in data["rows"] if r["instrument_id"] == "US:AAA"
                        and r["timestamp"] == fixture_times()[position].isoformat())
        self.assertEqual(at(before, 25)["features"], at(after, 25)["features"])
        self.assertEqual(at(before, 25)["feature_source_sha256"], at(after, 25)["feature_source_sha256"])
        self.assertNotEqual(at(before, 19)["target"], at(after, 19)["target"])
        self.assertEqual(at(after, 19)["label_end"], fixture_times()[60].isoformat())
        vintages = at(after, 19)["label_vintages"]
        self.assertEqual(vintages[0]["target"], at(before, 19)["target"])
        self.assertEqual(vintages[-1]["target"], at(after, 19)["target"])
        self.assertEqual(vintages[0]["knowledge_at"], fixture_times()[20].isoformat())

    def test_late_discovered_grid_event_does_not_rewrite_historical_features(self):
        original = [r for r in fixture_bars() if r["timestamp"] != fixture_times()[20].isoformat()]
        seed_persisted_fixture(self.base, rows=original)
        before = build_document(self.base, dataset_request())
        late = [dict(r, available_at=fixture_times()[60].isoformat()) for r in fixture_bars()
                if r["timestamp"] == fixture_times()[20].isoformat()]
        ingest_fixture_bars(self.base, late)
        after = build_document(self.base, dataset_request())
        def features(data):
            return {r["instrument_id"]: r["features"] for r in data["rows"]
                    if r["timestamp"] == fixture_times()[25].isoformat()}
        self.assertEqual(features(before), features(after))

    def test_page_acquired_after_historical_decisions_is_not_backdated(self):
        late = [dict(r, available_at=dataset_request()["as_of"]) for r in fixture_bars()]
        seed_persisted_fixture(self.base, rows=late)
        self.assertEqual(self.build()["state"], "UNAVAILABLE")
        self.assertFalse((self.base / "research-datasets").exists())

    def test_universe_membership_has_effective_and_knowledge_time(self):
        seed_persisted_fixture(self.base)
        at = fixture_times()[40].isoformat()
        UniverseStore(self.base / "universe.sqlite").ingest({
            "universe_id": UNIVERSE, "source": "authored-universe-fixture", "effective_at": at,
            "available_at": fixture_times()[39].isoformat(), "members": [INSTRUMENTS[0]]})
        rows = self.build()["data"]["rows"]
        self.assertTrue(any(r["instrument_id"] == "US:BBB" for r in rows))
        self.assertTrue(all(pd.Timestamp(r["timestamp"]) < pd.Timestamp(at) for r in rows if r["instrument_id"] == "US:BBB"))

    def test_missing_observation_windows_are_diagnosed_without_imputation(self):
        missing = [r for r in fixture_bars() if not (r["instrument_id"] == "US:AAA"
                   and r["timestamp"] == fixture_times()[20].isoformat())]
        seed_persisted_fixture(self.base, rows=missing)
        result = self.build()["data"]
        self.assertEqual(result["diagnostics"]["missing_or_late_current_bars"], 1)
        self.assertGreater(result["diagnostics"]["incomplete_feature_windows"], 0)
        self.assertEqual(result["diagnostics"]["missing_values_imputed"], 0)

    def test_source_classification_or_identity_cannot_silently_fall_back(self):
        seed_persisted_fixture(self.base)
        for change in ({"source": "unavailable-provider"}, {"classification": "HISTORICAL"}):
            self.assertEqual(self.build(dataset_request(**change))["state"], "UNAVAILABLE")

    def test_source_index_and_snapshot_evidence_tampering_fail_closed(self):
        seed_persisted_fixture(self.base)
        with sqlite3.connect(self.base / "bars.sqlite") as connection:
            connection.execute("UPDATE bar_revisions SET event='2020-01-01T00:00:00Z' WHERE rowid=1")
        with self.assertRaisesRegex(ValueError, "index"):
            self.build()
        self.assertFalse((self.base / "research-datasets").exists())

    def test_dataset_and_index_corruption_are_rejected_after_restart(self):
        seed_persisted_fixture(self.base)
        admitted = self.build()
        identity = admitted["data"]["dataset_id"]
        with sqlite3.connect(self.base / "research-datasets/datasets.sqlite") as connection:
            connection.execute("UPDATE datasets SET as_of='2020-01-01T00:00:00Z'")
        with self.assertRaisesRegex(ValueError, "index"):
            PersistedDatasetWorkspace(self.base).load(identity)

    def test_missing_adjustment_archive_is_not_assumed_empty(self):
        seed_persisted_fixture(self.base)
        self.assertEqual(self.build(dataset_request(adjustment="SPLIT_ADJUSTED"))["state"], "UNAVAILABLE")
        self.assertFalse((self.base / "actions.sqlite").exists())

    def test_declared_split_adjustment_is_causal_and_preserves_raw_marks(self):
        bars = fixture_bars()
        for row in bars:
            index = fixture_times().get_loc(pd.Timestamp(row["timestamp"]))
            if index >= 40:
                for name in ("open", "high", "low", "close"):
                    row[name] /= 2
        seed_persisted_fixture(self.base, rows=bars)
        event = {"action_id": "split-one", "instrument_id": "US:AAA", "source": "authored-actions",
                 "kind": "SPLIT", "effective_at": fixture_times()[40].isoformat(),
                 "available_at": fixture_times()[39].isoformat(), "ratio_new": 2, "ratio_old": 1}
        event2 = {**event, "action_id": "split-two", "instrument_id": "US:BBB"}
        ActionStore(self.base / "actions.sqlite").ingest(admit_actions([event, event2]))
        data = self.build(dataset_request(adjustment="SPLIT_ADJUSTED"))["data"]
        split_row = next(r for r in data["rows"] if r["instrument_id"] == "US:AAA"
                         and r["timestamp"] == fixture_times()[41].isoformat())
        self.assertLess(abs(split_row["features"]["momentum"]), .1)
        mark = next(r for r in data["panel_rows"] if r["instrument_id"] == "US:AAA"
                    and r["timestamp"] == fixture_times()[40].isoformat())
        self.assertEqual(mark["close"], bars[80]["close"])
        self.assertLess(abs(mark["return_1"]), .1)
        old = data["rows"][0]
        self.assertEqual(old["action_sha256"], digest([]))

    def test_serialization_preflight_precedes_dataset_publication(self):
        seed_persisted_fixture(self.base)
        with patch.object(self.workspace, "_contract", side_effect=ValueError("boundary")), self.assertRaises(ValueError):
            self.build()
        self.assertFalse((self.base / "research-datasets").exists())

    def test_replay_freshness_and_future_cutoff_are_explicit(self):
        seed_persisted_fixture(self.base)
        admitted = self.build()
        at = pd.Timestamp(admitted["provenance"]["as_of"])
        self.assertEqual(self.decoded(self.workspace.section("research", at, 60))["state"], "AVAILABLE")
        self.assertEqual(self.decoded(self.workspace.section("markets", at+pd.Timedelta(seconds=61), 60))["state"], "STALE")
        unknown = self.decoded(self.workspace.section("research", at, None))
        self.assertEqual(unknown["state"], "UNKNOWN")
        self.assertIsNone(unknown["data"])
        self.assertIsNone(self.workspace.section("research", at-pd.Timedelta(nanoseconds=1), 60))

    def test_invalid_fields_paths_numbers_and_timestamp_precision_reject(self):
        cases = [{"output_dir": "../escape"}, {"lookback": True}, {"horizon": 0},
                 {"decision_delay_seconds": -1}, {"start": "2026-01-01"},
                 {"as_of": "2026-04-01T00:00:00.1234567891Z"}, {"currency": "US💵"}]
        for change in cases:
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.build(dataset_request(**change))
        for identity in ("../escape", "a"*63, 123):
            with self.assertRaises(ValueError):
                self.workspace.load(identity)
        self.assertFalse(self.base.exists())

    def test_symlink_state_is_not_followed(self):
        target = Path(self.temporary.name) / "real"
        target.mkdir()
        self.base.symlink_to(target, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "symbolic"):
            self.build()
        self.assertEqual(list(target.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
