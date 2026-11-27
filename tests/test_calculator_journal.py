"""Saved calculator runs replay from the pinned inputs and reject a changed record."""
from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path

import httpx

from gig.service.app import create_app
from gig.service.calculator_journal import MAX_RECORDS, CalculatorJournal
from gig.service.quant_analytics import ANALYSES


class JournalTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)/"state"
        self.journal = CalculatorJournal(self.root)

    def test_every_registered_calculator_replays_its_example(self):
        self.assertGreaterEqual(len(ANALYSES), 32)
        for kind, analysis in ANALYSES.items():
            with self.subTest(kind):
                saved = self.journal.save(kind, analysis.example)
                again = self.journal.save(kind, analysis.example)
                self.assertEqual(again["record_id"], saved["record_id"])
                replayed = CalculatorJournal(self.root).replay(saved["record_id"])
                self.assertEqual(replayed["replay_state"], "REPRODUCED_FROM_PINNED_INPUTS")
                self.assertEqual(replayed["result"], saved["result"])
                self.assertFalse(replayed["trading_approved"])

    def test_a_changed_record_or_a_missing_one_is_refused(self):
        saved = self.journal.save("track-record-length", {"sharpe": .1, "benchmark": 0, "skew": -.5, "kurtosis": 5, "confidence": .95})
        with sqlite3.connect(self.journal.path) as db:
            db.execute("UPDATE records SET body=replace(body,'0.1','0.2') WHERE id=?", (saved["record_id"],))
        with self.assertRaisesRegex(ValueError, "integrity"):
            self.journal.replay(saved["record_id"])
        fresh = CalculatorJournal(self.root/"empty")
        with self.assertRaises(FileNotFoundError):
            fresh.replay(saved["record_id"])
        self.assertFalse(fresh.path.exists())
        with self.assertRaises(ValueError):
            self.journal.save("not-a-calculator", {})
        with self.assertRaises(ValueError):
            self.journal.replay("not-a-hash")

    def test_the_journal_stops_at_its_stated_capacity(self):
        for index in range(MAX_RECORDS):
            self.journal.save("track-record-length", {"sharpe": .05+index/10000})
        with self.assertRaisesRegex(ValueError, "capacity"):
            self.journal.save("track-record-length", {"sharpe": .9})
        self.assertEqual(sqlite3.connect(self.journal.path).execute("SELECT COUNT(*) FROM records").fetchone()[0], MAX_RECORDS)


class HTTPTests(unittest.IsolatedAsyncioTestCase):
    async def test_save_and_replay_over_the_local_workspace(self):
        with tempfile.TemporaryDirectory() as directory:
            transport = httpx.ASGITransport(app=create_app(state_dir=Path(directory)))
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                endpoint = "/api/v1/workspace/calculator-journal"
                saved = await client.post(endpoint, json={"operation": "save", "kind": "roll-spread",
                                                          "inputs": ANALYSES["roll-spread"].example})
                self.assertEqual(saved.status_code, 200, saved.text)
                body = saved.json()
                self.assertEqual(body["provenance"]["classification"], "MODEL_ESTIMATE")
                self.assertEqual(body["provenance"]["readiness"], "RESEARCH_ONLY")
                self.assertEqual(body["data"]["result"]["spread_state"], "IDENTIFIED")
                replay = await client.post(endpoint, json={"operation": "replay", "record_id": body["data"]["record_id"]})
                self.assertEqual(replay.json()["data"]["replay_state"], "REPRODUCED_FROM_PINNED_INPUTS")
                missing = await client.post(endpoint, json={"operation": "replay", "record_id": "0"*64})
                self.assertEqual(missing.json()["state"], "UNAVAILABLE")
                for bad in ({"operation": "save", "kind": "roll-spread"},
                            {"operation": "save", "kind": "roll-spread", "inputs": {"prices": [1, 2]}, "broker": "live"},
                            {"operation": "replay"},
                            {"operation": "trade", "kind": "roll-spread", "inputs": {}}):
                    self.assertEqual((await client.post(endpoint, json=bad)).status_code, 422)
                hostile = await client.post(endpoint, json={"operation": "replay", "record_id": body["data"]["record_id"]},
                                            headers={"Origin": "https://hostile.invalid"})
                self.assertEqual(hostile.status_code, 403)


if __name__ == "__main__":
    unittest.main()
