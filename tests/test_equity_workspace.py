"""Persisted weekly equity evidence is distinct from PIT/trading admission."""
from __future__ import annotations

import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from gig.service.contracts import EnvelopeSchema
from gig.service.equity_workspace import ObservedEquityWorkspace
from test_public_equity import ACQUIRED, REQUEST, fixture


class EquityWorkspaceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.base = Path(self.temporary.name)/"state"
        self.workspace = ObservedEquityWorkspace(self.base)

    def tearDown(self):
        self.temporary.cleanup()

    def request(self):
        return {"operation": "admit", **REQUEST, "csv_text": fixture(), "acquired_at": ACQUIRED}

    def test_snapshot_restart_replay_and_acquisition_cutoff(self):
        result = self.workspace.run(self.request()).to_dict()
        EnvelopeSchema.model_validate(result)
        source = result["data"]
        self.assertEqual(result["state"], "AVAILABLE")
        self.assertEqual(source["classification"], "SYNTHETIC")
        self.assertEqual(source["rows"][0]["week_ending_date"], "2011-01-07")
        self.assertNotIn("timestamp", source["rows"][0])
        self.assertFalse(source["point_in_time_backtest_eligible"])
        self.assertFalse(source["funded_allocation_eligible"])
        restarted = ObservedEquityWorkspace(self.base)
        self.assertEqual(restarted.run({"operation": "replay", "snapshot_id": source["snapshot_id"]}).to_dict(), result)
        self.assertIsNone(restarted.section("markets", "2011-01-21T23:59:59Z", 60))
        for section in ("markets", "research", "operations"):
            selected = restarted.section(section, ACQUIRED, 60).to_dict()
            self.assertEqual(selected["data"]["snapshot_id"], source["snapshot_id"])
        self.assertEqual(restarted.section("markets", ACQUIRED, None).to_dict()["state"], "UNKNOWN")
        self.assertEqual(restarted.section("markets", "2026-11-03T19:00:02Z", 1).to_dict()["state"], "STALE")

    def test_duplicate_concurrent_admission_and_detached_reads(self):
        with ThreadPoolExecutor(max_workers=4) as pool:
            results = list(pool.map(lambda _: ObservedEquityWorkspace(self.base).run(self.request()).to_dict(), range(8)))
        self.assertTrue(all(value == results[0] for value in results))
        identity = results[0]["data"]["snapshot_id"]
        altered = self.workspace.snapshot(identity)
        altered["rows"][0]["close"] = "99999"
        self.assertNotEqual(self.workspace.snapshot(identity)["rows"][0]["close"], "99999")

    def test_source_integrity_and_missing_reads(self):
        self.assertIsNone(self.workspace.section("markets", ACQUIRED, 60))
        self.assertEqual(self.workspace.run({"operation": "replay", "snapshot_id": "0"*64}).to_dict()["state"], "UNAVAILABLE")
        self.assertFalse(self.base.exists())
        source = self.workspace.run(self.request()).to_dict()["data"]
        self.workspace._path("raw/"+source["raw_sha256"]+".bin").write_bytes(b"tampered source")
        with self.assertRaisesRegex(ValueError, "source evidence"):
            self.workspace.snapshot(source["snapshot_id"])

    def test_equity_source_cannot_enter_reference_model_or_choose_paths(self):
        for payload in ({"operation": "research"}, {"operation": "replay", "snapshot_id": "../escape"},
                        {**self.request(), "classification": "HISTORICAL"}):
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                self.workspace.run(payload)
        self.assertFalse(self.base.exists())
