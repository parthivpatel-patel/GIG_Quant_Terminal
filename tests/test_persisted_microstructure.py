"""Original-decision event provenance and independent microstructure references."""
from __future__ import annotations

import copy
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from gig.service.contracts import EnvelopeSchema
from gig.service.microstructure_workspace import MicrostructureWorkspace

BASE = pd.Timestamp("2026-11-02T15:00:00.123456789Z")


def stamp(seconds=0):
    return (BASE+pd.Timedelta(seconds=seconds)).isoformat()


def microstructure_fixture():
    """Authored best quotes and prints; neither vendor evidence nor own fills."""
    events = []
    for i, mid in enumerate((100, 101, 100, 103, 104)):
        events.append({"event_id": "q"+str(i), "kind": "QUOTE", "event_at": stamp(i),
            "acquired_at": stamp(i), "knowledge_at": stamp(i), "sequence": 0,
            "bid": mid-1, "ask": mid+1, "bid_size": 10+i, "ask_size": 20-i})
    for i, (price, size) in enumerate(((101, 1), (100, 2), (101, 3), (102, 1))):
        events.append({"event_id": "t"+str(i), "kind": "TRADE", "event_at": stamp(i+.5),
            "acquired_at": stamp(i+.5), "knowledge_at": stamp(i+.5), "sequence": 1,
            "price": price, "size": size})
    return {"source": "authored-sprint6-events", "classification": "SYNTHETIC",
        "feed": "AUTHORED_PRINT_AND_QUOTE_FIXTURE", "instrument_id": "SYN101", "currency": "USD", "events": events}


def microstructure_request(**changes):
    return {"source": "authored-sprint6-events", "classification": "SYNTHETIC",
        "feed": "AUTHORED_PRINT_AND_QUOTE_FIXTURE", "instrument_id": "SYN101",
        "start": stamp(), "end": stamp(4), "as_of": stamp(4), "max_quote_age_ns": 5_000_000_000,
        "lag_ns": 0, "bucket_volume": 2, "window_buckets": 2,
        "participation_rate": .1, "proposed_quantity": 1, **changes}


class PersistedMicrostructureTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.base = Path(self.temporary.name)/"state"
        self.workspace = MicrostructureWorkspace(self.base)

    def tearDown(self):
        self.temporary.cleanup()

    def seed(self):
        return self.workspace.ingest(microstructure_fixture()).to_dict()

    def test_scalar_ofi_lambda_vpin_capacity_and_source_reference(self):
        self.seed()
        result = self.workspace.run(microstructure_request()).to_dict()
        EnvelopeSchema.model_validate(result)
        self.assertEqual(result["state"], "AVAILABLE")
        data = result["data"]
        self.assertEqual(data["input_classification"], "SYNTHETIC")
        self.assertEqual([r["estimated_sign"] for r in data["rows"]], [1, -1, 1, -1])
        self.assertEqual([r["best_level_ofi"] for r in data["ofi_rows"]], [None, 31., -29., 31., 31.])
        self.assertEqual(data["metrics"]["best_level_ofi_sum"]["value"], 64.)
        x, y = [1., -2., 3., -1.], [1., -1., 3., 1.]
        xm, ym = sum(x)/4, sum(y)/4
        slope = sum((a-xm)*(b-ym) for a, b in zip(x, y, strict=True))/sum((a-xm)**2 for a in x)
        self.assertAlmostEqual(data["kyle_lambda"]["coefficient"], slope, places=14)
        self.assertEqual(data["kyle_lambda"]["sample_count"], 4)
        self.assertEqual(data["vpin"]["imbalances"], [0., 0., 1.])
        self.assertEqual(data["vpin"]["values"], [0., .5])
        self.assertEqual(data["vpin"]["incomplete_volume"], 1.)
        self.assertEqual(data["capacity"]["observed_print_volume"], 7.)
        self.assertAlmostEqual(data["capacity"]["modeled_quantity_budget"], .7)
        self.assertFalse(data["capacity"]["within_observed_volume_budget"])
        self.assertEqual(data["capacity"]["executable_capacity"]["state"], "UNKNOWN")
        self.assertEqual(data["execution_costs"]["state"], "UNAVAILABLE")
        self.assertFalse(data["trading_approved"])
        self.assertFalse(data["economically_validated"])
        self.assertIn("123456789", data["observed_at"])

    def test_equal_clock_sequence_and_future_quote_knowledge_cannot_look_ahead(self):
        fixture = microstructure_fixture()
        future = copy.deepcopy(fixture["events"][0])
        future.update(event_id="future-knowledge", event_at=stamp(.4), acquired_at=stamp(.6), knowledge_at=stamp(.6), bid=200, ask=202)
        same_time = copy.deepcopy(future)
        same_time.update(event_id="future-sequence", event_at=stamp(.5), acquired_at=stamp(.5), knowledge_at=stamp(.5), sequence=2)
        fixture["events"] += [future, same_time]
        self.workspace.ingest(fixture)
        data = self.workspace.run(microstructure_request()).to_dict()["data"]
        first = data["rows"][0]
        self.assertEqual(first["estimated_sign"], 1)
        self.assertEqual(first["quote_age_ns"], 500_000_000)
        self.assertEqual(first["classification_method"], "QUOTE_ESTIMATE")
        self.assertEqual(self.workspace.run(microstructure_request(max_quote_age_ns=499_999_999)).to_dict()["data"]["rows"][0]["quote_match"], "STALE_QUOTE")

    def test_unknown_sign_and_late_prior_tick_are_not_zero_estimates(self):
        fixture = microstructure_fixture()
        fixture["events"] = [r for r in fixture["events"] if r["kind"] == "TRADE"]
        earlier = copy.deepcopy(fixture["events"][0])
        earlier.update(event_id="delayed-prior", event_at=stamp(.1), acquired_at=stamp(.6), knowledge_at=stamp(.6), price=500)
        fixture["events"].append(earlier)
        self.workspace.ingest(fixture)
        data = self.workspace.run(microstructure_request()).to_dict()["data"]
        first = next(r for r in data["rows"] if r["event_id"] == "t0")
        self.assertEqual(first["estimated_sign"], 0)
        self.assertEqual(first["sign_state"], "UNKNOWN")
        self.assertEqual(data["vpin"]["state"], "UNKNOWN")
        self.assertGreater(data["vpin"]["excluded_unknown_volume"], 0)
        self.assertEqual(data["kyle_lambda"]["state"], "UNAVAILABLE")
        self.assertIsNone(data["kyle_lambda"]["coefficient"])
        self.assertEqual(data["metrics"]["best_level_ofi_sum"]["state"], "UNAVAILABLE")
        self.assertIsNone(data["metrics"]["best_level_ofi_sum"]["value"])

    def test_late_corrections_preserve_original_features_and_restart_replay(self):
        self.seed()
        original = self.workspace.run(microstructure_request()).to_dict()
        correction = microstructure_fixture()
        correction["events"] = [correction["events"][0]]
        correction["events"][0].update(acquired_at=stamp(6), knowledge_at=stamp(6), bid=98, ask=100, bid_size=100)
        self.workspace.ingest(correction)
        reopened = MicrostructureWorkspace(self.base)
        self.assertEqual(reopened.run(microstructure_request()).to_dict(), original)
        self.assertEqual(reopened.replay({"report_id": original["data"]["report_id"]}).to_dict(), original)
        later = reopened.run(microstructure_request(as_of=stamp(6))).to_dict()["data"]
        self.assertEqual(later["rows"], original["data"]["rows"])
        self.assertEqual(later["ofi_rows"], original["data"]["ofi_rows"])
        self.assertNotEqual(later["report_id"], original["data"]["report_id"])
        detached = reopened.load(original["data"]["report_id"])
        detached["rows"][0]["price"] = -1
        self.assertGreater(reopened.load(original["data"]["report_id"])["rows"][0]["price"], 0)

    def test_immutable_duplicates_conflicts_source_terms_and_index_tamper(self):
        first = self.seed()
        self.assertEqual(first, self.seed())
        conflicting = microstructure_fixture()
        conflicting["events"][0]["bid_size"] = 100
        with self.assertRaisesRegex(ValueError, "conflicting"):
            self.workspace.ingest(conflicting)
        changed_identity = microstructure_fixture()
        changed_identity["events"][0].update(event_at=stamp(.01), acquired_at=stamp(.01), knowledge_at=stamp(.01))
        with self.assertRaisesRegex(ValueError, "identity"):
            self.workspace.ingest(changed_identity)
        data = self.workspace.run(microstructure_request()).to_dict()["data"]
        with sqlite3.connect(self.base/"microstructure-research/events.sqlite") as connection:
            connection.execute("UPDATE reports SET as_of=?", (stamp(8),))
        with self.assertRaisesRegex(ValueError, "index"):
            self.workspace.replay({"report_id": data["report_id"]})

    def test_admission_failure_rolls_back_publication_and_retry_recovers(self):
        with patch("gig.service.microstructure_workspace.write_immutable", side_effect=OSError("injected disk failure")), self.assertRaises(OSError):
            self.seed()
        self.assertEqual(self.workspace.run(microstructure_request()).to_dict()["state"], "UNAVAILABLE")
        self.assertIsNone(self.workspace.section("execution", stamp(4), 60))
        self.seed()
        self.assertEqual(self.workspace.run(microstructure_request()).to_dict()["state"], "AVAILABLE")

    def test_missing_future_nanosecond_freshness_and_invalid_request_states(self):
        self.assertEqual(self.workspace.run(microstructure_request()).to_dict()["state"], "UNAVAILABLE")
        self.assertEqual(self.workspace.replay({"report_id": "0"*64}).to_dict()["state"], "UNAVAILABLE")
        self.assertFalse(self.base.exists())
        self.seed()
        self.workspace.run(microstructure_request())
        self.assertEqual(self.workspace.section("research", stamp(4), 0).to_dict()["state"], "AVAILABLE")
        nanosecond_later = (pd.Timestamp(stamp(4))+pd.Timedelta(nanoseconds=1)).isoformat()
        self.assertEqual(self.workspace.section("research", nanosecond_later, 0).to_dict()["state"], "STALE")
        self.assertEqual(self.workspace.section("research", stamp(4), None).to_dict()["state"], "UNKNOWN")
        self.assertIsNone(self.workspace.section("research", stamp(3), 0))
        for value in (True, float("nan"), -1):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.workspace.run(microstructure_request(participation_rate=value))
        invalid = microstructure_fixture()
        invalid["events"][0]["event_at"] = "2026-11-02T15:00:00.1234567890Z"
        with self.assertRaises(ValueError):
            self.workspace.ingest(invalid)


if __name__ == "__main__":
    unittest.main()
