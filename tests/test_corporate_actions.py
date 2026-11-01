"""Deterministic corporate-action arithmetic and point-in-time evidence tests."""
import sqlite3
import tempfile
import unittest
from fractions import Fraction
from pathlib import Path

import pandas as pd

from gig.data.actions import ActionSnapshot, ActionStore, adjust_bars, admit_actions
from gig.research.workflow import ResearchWorkflow
from gig.service.contracts import Contract

T0 = "2026-06-01T20:00:00.000000123Z"
T1 = "2026-06-02T13:30:00.000000123Z"
T2 = "2026-06-02T20:00:00.000000123Z"
T3 = "2026-06-03T20:00:00.000000123Z"


def bar(stamp, close, volume=101, instrument="ID:AAA", available=None):
    return {"instrument_id": instrument, "timestamp": stamp, "available_at": available or stamp,
            "open": close, "high": close, "low": close, "close": close, "volume": volume}


def split(**changes):
    return {"action_id": "split-1", "instrument_id": "ID:AAA", "source": "caller-actions",
            "kind": "SPLIT", "effective_at": T1, "available_at": T0,
            "ratio_new": 2, "ratio_old": 1, **changes}


def dividend(**changes):
    return {"action_id": "cash-1", "instrument_id": "ID:AAA", "source": "caller-actions",
            "kind": "CASH_DIVIDEND", "effective_at": T1, "available_at": T0,
            "cash_per_share": "2.00", "currency": "USD", "reference_timestamp": T0,
            "reference_close": "100.00", **changes}


def adjust(rows, actions, *, cutoff=T3, mode="TOTAL_RETURN_ADJUSTED"):
    return adjust_bars(rows, actions, as_of=cutoff, mode=mode,
                       source="caller-bars", classification="SYNTHETIC", currency="USD")


def rational(value):
    return Fraction(int(value["numerator"]), int(value["denominator"]))


class AdjustmentReferences(unittest.TestCase):
    def test_two_for_one_split_price_and_exact_huge_volume(self):
        volume = 2**53 + 17
        result = adjust([bar(T0, 100, volume), bar(T2, 50)], [split()])
        before, after = result["rows"]
        self.assertEqual(before["close"], 100)
        self.assertEqual(before["volume"], str(volume))
        self.assertEqual(before["adjusted_close"], 50)
        self.assertEqual(rational(before["adjusted_volume_exact"]), 2*volume)
        self.assertEqual(after["adjusted_close"], 50)
        self.assertEqual(rational(after["price_factor_exact"]), 1)

    def test_three_for_two_split_retains_noninteger_volume(self):
        result = adjust([bar(T0, 120, 101), bar(T2, 80)], [split(ratio_new=3, ratio_old=2)])
        self.assertEqual(result["rows"][0]["adjusted_close"], 80)
        self.assertEqual(rational(result["rows"][0]["adjusted_volume_exact"]), Fraction(303, 2))

    def test_cash_dividend_flat_total_return_reference(self):
        result = adjust([bar(T0, 100), bar(T2, 98)], [dividend()])
        self.assertEqual(result["rows"][0]["adjusted_close"], 98)
        self.assertEqual(rational(result["rows"][0]["price_factor_exact"]), Fraction(49, 50))
        self.assertEqual(rational(result["rows"][0]["adjusted_volume_exact"]), 101)
        self.assertEqual(result["rows"][1]["adjusted_close"] / result["rows"][0]["adjusted_close"] - 1, 0)
        self.assertEqual((2+98)/100-1, 0)

    def test_cash_wealth_return_is_exact_when_ex_price_moves(self):
        result = adjust([bar(T0, 100), bar(T2, 99)], [dividend()])
        previous = rational(result["rows"][0]["adjusted_close_exact"])
        current = rational(result["rows"][1]["adjusted_close_exact"])
        self.assertEqual(current/previous-1, Fraction(1, 100))
        self.assertEqual(current/previous, Fraction(99+2, 100))
        self.assertEqual(result["rows"][0]["available_at"], pd.Timestamp(T0).isoformat())
        self.assertEqual(result["rows"][0]["adjusted_available_at"], pd.Timestamp(T2).isoformat())
        self.assertTrue(result["rows"][0]["uses_later_observations"])

    def test_combined_wealth_return_with_nonflat_ex_close(self):
        event = split(kind="SPLIT_AND_DIVIDEND", cash_per_pre_action_share="2", currency="USD",
                      reference_timestamp=T0, reference_close="100")
        result = adjust([bar(T0, 100), bar(T2, 48)], [event])
        previous = rational(result["rows"][0]["adjusted_close_exact"])
        self.assertEqual(Fraction(48)/previous-1, Fraction(2*48+2, 100)-1)

    def test_missing_ex_close_or_intervening_action_does_not_guess_wealth(self):
        with self.assertRaisesRegex(ValueError, "post-effective"):
            adjust([bar(T0, 100)], [dividend()])
        later_split = split(action_id="later", effective_at="2026-06-02T15:00:00Z")
        with self.assertRaisesRegex(ValueError, "spans another action"):
            adjust([bar(T0, 100), bar(T2, 49)], [dividend(), later_split])

    def test_derived_availability_prevents_historical_training_lookahead(self):
        index = pd.date_range("2026-01-01T20:00:00Z", periods=25, freq="D")
        rows = [bar(stamp.isoformat(), 100 if i < 20 else 50) for i, stamp in enumerate(index)]
        event = split(effective_at="2026-01-21T13:30:00Z", available_at=index[19].isoformat())
        result = adjust(rows, [event], cutoff=index[-1].isoformat())
        available = pd.Series([pd.Timestamp(row["adjusted_available_at"]) for row in result["rows"]], index=index)
        features = pd.DataFrame({"adjusted_close": [row["adjusted_close"] for row in result["rows"]]}, index=index)
        with tempfile.TemporaryDirectory() as directory, self.assertRaisesRegex(ValueError, "look-ahead"):
            ResearchWorkflow(directory).run(features, pd.Series(0., index=index),
                                             label_end=pd.Series(index+pd.Timedelta(hours=1), index=index),
                                             feature_available_at=available, min_train=10, test_size=5)

    def test_combined_units_are_explicit_pre_action_cash(self):
        event = split(kind="SPLIT_AND_DIVIDEND", cash_per_pre_action_share="2", currency="USD",
                      reference_timestamp=T0, reference_close="100")
        result = adjust([bar(T0, 100), bar(T2, 49)], [event])
        self.assertEqual(result["rows"][0]["adjusted_close"], 49)
        self.assertEqual(rational(result["rows"][0]["price_factor_exact"]), Fraction(49, 100))
        self.assertEqual((2*49+2)/100-1, 0)

    def test_sequential_actions_compound_and_volume_only_splits(self):
        cash = dividend(effective_at="2026-06-03T13:30:00Z", available_at=T2,
                        reference_timestamp=T2, reference_close="50", cash_per_share="1")
        result = adjust([bar(T0, 100), bar(T2, 50), bar(T3, 49)], [split(), cash])
        self.assertEqual([row["adjusted_close"] for row in result["rows"]], [49, 49, 49])
        self.assertEqual(rational(result["rows"][0]["adjusted_volume_exact"]), 202)
        self.assertEqual(rational(result["rows"][1]["adjusted_volume_exact"]), 101)

    def test_split_mode_explicitly_excludes_cash(self):
        result = adjust([bar(T0, 100), bar(T2, 98)], [dividend()], mode="SPLIT_ADJUSTED")
        self.assertEqual(result["rows"][0]["adjusted_close"], 100)

    def test_future_event_or_unknown_announcement_never_changes_earlier_identity(self):
        rows = [bar(T0, 100), bar(T2, 50)]
        future = split(effective_at="2027-06-01T13:30:00Z")
        unknown = split(available_at=T3)
        baseline = adjust(rows, [], cutoff=T2)
        for event in (future, unknown):
            actual = adjust(rows, [event], cutoff=T2)
            self.assertEqual(actual["transformation_sha256"], baseline["transformation_sha256"])
            self.assertEqual(actual["rows"], baseline["rows"])

    def test_late_price_revision_preserves_earlier_transform(self):
        rows = [bar(T0, 100), bar(T2, 50)]
        correction = bar(T0, 110, available=T3)
        before = adjust(rows, [split()], cutoff=T2)
        with_future = adjust([*rows, correction], [split()], cutoff=T2)
        after = adjust([*rows, correction], [split()], cutoff=T3)
        self.assertEqual(before["transformation_sha256"], with_future["transformation_sha256"])
        self.assertEqual(after["rows"][0]["adjusted_close"], 55)

    def test_equivalent_timezones_and_nanoseconds_preserve_identity(self):
        event = split(effective_at="2026-06-02T09:30:00.000000123-04:00")
        baseline = adjust([bar(T0, 100), bar(T2, 50)], [split()])
        actual = adjust([bar(T0, 100), bar(T2, 50)], [event])
        self.assertEqual(actual["transformation_sha256"], baseline["transformation_sha256"])
        self.assertIn(".000000123", actual["actions"][0]["effective_at"])

    def test_mutation_after_admission_cannot_change_snapshot_or_result(self):
        records = [split()]
        snapshot = admit_actions(records)
        immutable = snapshot.payload
        records[0]["ratio_new"] = 99
        returned = snapshot.as_dict()
        returned["rows"][0]["ratio_new"] = 42
        self.assertEqual(snapshot.payload, immutable)
        rows = [bar(T0, 100), bar(T2, 50)]
        result = adjust(rows, [split()])
        rows[0]["close"] = 1
        self.assertEqual(result["rows"][0]["close"], 100)

    def test_reference_must_match_actual_latest_source_close(self):
        rows = [bar(T0, 100), bar(T2, 98)]
        for event in (dividend(reference_close="99"),
                      dividend(reference_timestamp="2026-05-30T20:00:00Z"),
                      dividend(currency="EUR"), dividend(cash_per_share="100")):
            with self.subTest(event=event), self.assertRaises(ValueError):
                adjust(rows, [event])
        with self.assertRaisesRegex(ValueError, "prior source"):
            adjust([bar(T2, 98)], [dividend()])

    def test_simultaneous_uncombined_events_and_mixed_providers_reject(self):
        rows = [bar(T0, 100), bar(T2, 50), bar(T3, 49)]
        with self.assertRaisesRegex(ValueError, "simultaneous"):
            adjust(rows, [split(), dividend()])
        other = split(action_id="later", effective_at="2026-06-03T13:30:00Z", source="other-provider")
        with self.assertRaisesRegex(ValueError, "mixed"):
            adjust(rows, [split(), other])

    def test_unknown_instrument_and_invalid_actions_reject(self):
        with self.assertRaisesRegex(ValueError, "instrument"):
            adjust([bar(T0, 100)], [split(instrument_id="OTHER")])
        for changes in ({"ratio_new": True}, {"ratio_old": 0}, {"kind": "SPINOFF"},
                        {"effective_at": "2026-06-02"}, {"effective_at": "2026-06-02T13:30:00.1234567890Z"}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                admit_actions([split(**changes)])
        for cash in ("NaN", "Infinity", "-1", 1.0, "1e101"):
            with self.subTest(cash=cash), self.assertRaises(ValueError):
                admit_actions([dividend(cash_per_share=cash)])

    def test_empty_visible_bars_and_bounds_are_explicit(self):
        self.assertEqual(adjust([bar(T3, 100)], [], cutoff=T0)["state"], "UNAVAILABLE")
        with self.assertRaisesRegex(ValueError, "500"):
            adjust([bar(T0, 100)]*501, [])
        with self.assertRaisesRegex(ValueError, "128"):
            admit_actions([split(action_id=str(i)) for i in range(129)])
        with self.assertRaises(ValueError):
            adjust_bars([bar(T0, 100)], [], as_of=T3, mode="RAW", source="x", classification="SYNTHETIC", currency="USD")

    def test_declared_max_rows_actions_and_source_fields_cross_real_wire(self):
        start = pd.Timestamp(T0)
        identity = "I"*256
        rows = [bar((start+pd.Timedelta(nanoseconds=i)).isoformat(), 100, 2**53+17,
                    instrument=identity) for i in range(500)]
        events = [split(action_id=str(i)+"A"*250, instrument_id=identity, source="S"*512,
                        effective_at=(start-pd.Timedelta(seconds=1, nanoseconds=i)).isoformat())
                  for i in range(128)]
        result = adjust_bars(rows, events, as_of=T3, mode="SPLIT_ADJUSTED",
                             source="B"*512, classification="SYNTHETIC", currency="USD")
        admitted = Contract.admit("AVAILABLE", result, source="corporate-action-analysis",
                                   classification="MODEL_ESTIMATE", scope="analytics.corporate-actions")
        self.assertEqual(len(admitted.to_dict()["data"]["rows"]), 500)
        self.assertEqual(len(admitted.to_dict()["data"]["actions"]), 128)
        self.assertLess(len(admitted.wire), 1_048_576)

    def test_decimal_source_spelling_preserved_and_nondecimal_strings_rejected(self):
        snapshot = admit_actions([dividend(cash_per_share="2.00e+0")])
        self.assertEqual(snapshot.as_dict()["rows"][0]["cash_per_share"], "2.00e+0")
        for value in ("1_0", " 2 ", "", "-0"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                admit_actions([dividend(cash_per_share=value)])


class ActionStoreReferences(unittest.TestCase):
    def test_restart_replay_and_knowledge_time_revision(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"actions.sqlite"
            store = ActionStore(path)
            self.assertEqual(store.ingest(admit_actions([split()])), {"inserted": 1, "replayed": 0})
            self.assertEqual(store.ingest(admit_actions([split()])), {"inserted": 0, "replayed": 1})
            store.ingest(admit_actions([split(ratio_new=3, available_at=T3)]))
            self.assertEqual(ActionStore(path).snapshot(T2).as_dict()["rows"][0]["ratio_new"], 2)
            self.assertEqual(ActionStore(path).snapshot(T3).as_dict()["rows"][0]["ratio_new"], 3)

    def test_conflicting_batch_rolls_back_and_identity_cannot_reassign(self):
        with tempfile.TemporaryDirectory() as directory:
            store = ActionStore(Path(directory)/"actions.sqlite")
            store.ingest(admit_actions([split()]))
            batch = admit_actions([split(action_id="new"), split(ratio_new=3)])
            with self.assertRaisesRegex(ValueError, "conflicting"):
                store.ingest(batch)
            self.assertEqual(len(store.snapshot(T3).as_dict()["rows"]), 1)
            with self.assertRaisesRegex(ValueError, "identity"):
                store.ingest(admit_actions([split(available_at=T3, instrument_id="OTHER")]))

    def test_evidence_tampering_and_forged_snapshot_reject(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"actions.sqlite"
            store = ActionStore(path)
            store.ingest(admit_actions([split()]))
            with sqlite3.connect(path) as connection:
                connection.execute("UPDATE action_revisions SET payload=?", (b"{}",))
            with self.assertRaisesRegex(ValueError, "corrupt"):
                store.snapshot(T3)
        with self.assertRaises(ValueError):
            ActionSnapshot(b"{}", "0"*64)


if __name__ == "__main__":
    unittest.main()
