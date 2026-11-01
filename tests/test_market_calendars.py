"""Actual calendar-rule references; caller/provider timestamps remain separate."""
from __future__ import annotations

import importlib.util
import unittest
from unittest.mock import patch

from gig.data.calendars import ExchangeCalendar, calendar_analytics


class CalendarDependencyTests(unittest.TestCase):
    def test_missing_dependency_has_no_calendar_fallback(self):
        with patch("gig.data.calendars.importlib.import_module", side_effect=ImportError("missing")), \
                self.assertRaisesRegex(ImportError, "optional calendar"):
            ExchangeCalendar("XNYS")


@unittest.skipUnless(importlib.util.find_spec("exchange_calendars") is not None,
                     "Optional exchange_calendars dependency is not installed")
class MarketCalendarTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.calendar = ExchangeCalendar("XNYS")

    def test_thanksgiving_holiday_and_early_close_reference(self):
        result = self.calendar.schedule("2026-11-23", "2026-11-30")
        self.assertEqual([row["session"] for row in result["rows"]],
                         ["2026-11-23", "2026-11-24", "2026-11-25", "2026-11-27", "2026-11-30"])
        early = result["rows"][3]
        self.assertEqual(early["open"], "2026-11-27T14:30:00+00:00")
        self.assertEqual(early["close"], "2026-11-27T18:00:00+00:00")
        self.assertEqual(early["trading_seconds"], 3.5 * 3600)
        self.assertTrue(early["early_close"])
        self.assertFalse(result["rows"][-1]["early_close"])

    def test_juneteenth_holiday_and_weekend(self):
        result = self.calendar.schedule("2026-06-18", "2026-06-22")
        self.assertEqual([row["session"] for row in result["rows"]], ["2026-06-18", "2026-06-22"])

    def test_spring_and_autumn_dst_changes_preserve_local_market_hours(self):
        spring = self.calendar.schedule("2026-03-06", "2026-03-09")["rows"]
        self.assertEqual(spring[0]["open"], "2026-03-06T14:30:00+00:00")
        self.assertEqual(spring[1]["open"], "2026-03-09T13:30:00+00:00")
        self.assertEqual(spring[1]["close"], "2026-03-09T20:00:00+00:00")
        autumn = self.calendar.schedule("2026-10-30", "2026-11-02")["rows"]
        self.assertEqual(autumn[0]["open"], "2026-10-30T13:30:00+00:00")
        self.assertEqual(autumn[1]["open"], "2026-11-02T14:30:00+00:00")

    def test_schedule_empty_weekend_is_an_explicit_calendar_fact(self):
        result = self.calendar.schedule("2026-11-28", "2026-11-29")
        self.assertEqual(result["state"], "AVAILABLE")
        self.assertEqual(result["metrics"]["session_count"], 0)
        self.assertEqual(result["rows"], [])

    def test_session_open_and_close_nanosecond_boundaries(self):
        self.assertEqual(self.calendar.status("2026-11-27T14:29:59.999999999Z")["scheduled_state"], "CLOSED")
        self.assertEqual(self.calendar.status("2026-11-27T14:30:00Z")["scheduled_state"], "OPEN")
        self.assertEqual(self.calendar.status("2026-11-27T17:59:59.999999999Z")["scheduled_state"], "OPEN")
        at_close = self.calendar.status("2026-11-27T18:00:00Z")
        self.assertEqual(at_close["scheduled_state"], "CLOSED")
        self.assertEqual(at_close["last_completed_session"], "2026-11-27")

    def test_timezone_equivalent_status_has_identical_evidence(self):
        utc = self.calendar.status("2026-11-27T17:00:00.000000123Z")
        local = self.calendar.status("2026-11-27T12:00:00.000000123-05:00")
        self.assertEqual(utc, local)
        self.assertEqual(utc["as_of"], "2026-11-27T17:00:00.000000123+00:00")

    def test_before_open_and_weekend_last_completed_session(self):
        preopen = self.calendar.status("2026-11-30T14:00:00Z")
        self.assertEqual(preopen["last_completed_close"], "2026-11-27T18:00:00+00:00")
        weekend = self.calendar.status("2026-11-29T20:00:00Z")
        self.assertEqual(weekend["last_completed_session"], "2026-11-27")

    def test_daily_bars_match_real_session_closes_and_report_gaps(self):
        result = self.calendar.validate_daily_closes([
            "2026-11-25T21:00:00Z", "2026-11-27T13:00:00-05:00", "2026-11-30T21:00:00Z"],
            require_contiguous=True)
        self.assertTrue(result["contiguous"])
        self.assertEqual(result["session_labels"], ["2026-11-25", "2026-11-27", "2026-11-30"])
        missing = self.calendar.validate_daily_closes(["2026-11-25T21:00:00Z", "2026-11-30T21:00:00Z"])
        self.assertEqual(missing["missing_sessions"], ["2026-11-27"])
        with self.assertRaisesRegex(ValueError, "omit"):
            self.calendar.validate_daily_closes(["2026-11-25T21:00:00Z", "2026-11-30T21:00:00Z"],
                                                require_contiguous=True)

    def test_daily_bar_unknown_timestamp_holiday_and_duplicate_instants_reject(self):
        for values in (["2026-11-27T21:00:00Z"], ["2026-11-26T21:00:00Z"],
                       ["2026-11-27T18:00:00Z", "2026-11-27T13:00:00-05:00"],
                       ["2026-11-30T21:00:00Z", "2026-11-27T18:00:00Z"]):
            with self.subTest(values=values), self.assertRaises(ValueError):
                self.calendar.validate_daily_closes(values)

    def test_weekend_freshness_is_session_based_and_does_not_invent_bars(self):
        result = self.calendar.freshness("2026-11-27T18:00:00Z", "2026-11-29T20:00:00Z")
        self.assertEqual(result["state"], "AVAILABLE")
        self.assertEqual(result["missed_session_count"], 0)
        self.assertEqual(result["observed_close"], "2026-11-27T18:00:00+00:00")
        self.assertFalse(result["provenance"]["trading_approved"])

    def test_freshness_late_close_grace_and_missed_session_limit(self):
        before_deadline = self.calendar.freshness("2026-11-27T18:00:00Z", "2026-11-30T21:00:29.999999999Z",
                                                 grace_seconds=30)
        self.assertEqual(before_deadline["state"], "AVAILABLE")
        at_deadline = self.calendar.freshness("2026-11-27T18:00:00Z", "2026-11-30T21:00:30Z", grace_seconds=30)
        self.assertEqual(at_deadline["state"], "STALE")
        self.assertEqual(at_deadline["missing_sessions"], ["2026-11-30"])
        tolerated = self.calendar.freshness("2026-11-27T18:00:00Z", "2026-11-30T21:00:30Z",
                                           max_missed_sessions=1)
        self.assertEqual(tolerated["state"], "AVAILABLE")

    def test_freshness_early_arriving_current_close_is_valid(self):
        result = self.calendar.freshness("2026-11-30T21:00:00Z", "2026-11-30T21:00:01Z", grace_seconds=30)
        self.assertEqual(result["state"], "AVAILABLE")
        self.assertEqual(result["missed_session_count"], 0)

    def test_bad_freshness_parameters_fail_closed(self):
        for options in ({"max_missed_sessions": True}, {"max_missed_sessions": -1},
                        {"max_missed_sessions": 31}, {"grace_seconds": True},
                        {"grace_seconds": float("nan")}, {"grace_seconds": float("inf")},
                        {"grace_seconds": -1}, {"grace_seconds": 3601}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                self.calendar.freshness("2026-11-27T18:00:00Z", "2026-11-29T20:00:00Z", **options)
        with self.assertRaisesRegex(ValueError, "future"):
            self.calendar.freshness("2026-11-30T21:00:00Z", "2026-11-29T20:00:00Z")

    def test_unknown_calendar_and_invalid_date_ranges_are_not_substituted(self):
        for name in ("MISSING", "", 3):
            with self.subTest(name=name), self.assertRaises(ValueError):
                ExchangeCalendar(name)
        for left, right in (("2026-01-01", "2025-12-31"), ("2019-12-31", "2020-01-02"),
                            ("2026-01-01", "2027-01-02"), ("2026-01-01T00:00:00Z", "2026-01-02")):
            with self.subTest(left=left, right=right), self.assertRaises(ValueError):
                self.calendar.schedule(left, right)

    def test_timestamp_precision_timezone_and_coverage_guards(self):
        for stamp in ("2026-01-01T00:00:00", "2026-01-01T00:00:00.1234567891Z",
                      "2019-01-01T00:00:00Z", "2031-01-01T00:00:00Z"):
            with self.subTest(stamp=stamp), self.assertRaises(ValueError):
                self.calendar.status(stamp)
        for start, end in (("2026-01-01", "2026-01-03"), ("2010-01-01", "2030-01-01")):
            with self.subTest(start=start), self.assertRaises(ValueError):
                ExchangeCalendar("XNYS", coverage_start=start, coverage_end=end)

    def test_schedule_mutation_cannot_modify_snapshot_or_identity(self):
        first = self.calendar.schedule("2026-11-23", "2026-11-30")
        digest = first["provenance"]["calendar_sha256"]
        first["rows"][0]["close"] = "corrupted"
        first["provenance"]["calendar"] = "MISSING"
        second = self.calendar.schedule("2026-11-23", "2026-11-30")
        self.assertEqual(second["rows"][0]["close"], "2026-11-23T21:00:00+00:00")
        self.assertEqual(second["provenance"]["calendar_sha256"], digest)
        self.assertEqual(second["provenance"]["calendar"], "XNYS")

    def test_alias_preserves_requested_and_canonical_identity(self):
        alias = ExchangeCalendar("NYSE")
        self.assertEqual(alias.provenance()["requested_calendar"], "NYSE")
        self.assertEqual(alias.provenance()["calendar"], "XNYS")
        self.assertEqual(alias.identity, self.calendar.identity)

    def test_hong_kong_break_interval_and_trading_duration(self):
        calendar = ExchangeCalendar("XHKG")
        row = calendar.schedule("2026-11-23", "2026-11-23")["rows"][0]
        self.assertEqual(row["break_start"], "2026-11-23T04:00:00+00:00")
        self.assertEqual(row["break_end"], "2026-11-23T05:00:00+00:00")
        self.assertEqual(row["session_span_seconds"] - row["trading_seconds"], 3600)
        self.assertEqual(calendar.status("2026-11-23T04:30:00Z")["scheduled_state"], "BREAK")
        self.assertEqual(calendar.status("2026-11-23T05:00:00Z")["scheduled_state"], "OPEN")

    def test_http_adapter_payload_combines_defined_diagnostics(self):
        result = calendar_analytics({"calendar": "XNYS", "start": "2026-11-23", "end": "2026-11-30",
                                     "as_of": "2026-11-29T20:00:00Z", "last_bar_close": "2026-11-27T18:00:00Z",
                                     "daily_closes": ["2026-11-25T21:00:00Z", "2026-11-27T18:00:00Z"],
                                     "require_contiguous": True})
        self.assertEqual(result["metrics"]["session_count"], 5)
        self.assertEqual(result["scheduled_market_status"]["scheduled_state"], "CLOSED")
        self.assertEqual(result["freshness"]["state"], "AVAILABLE")
        self.assertTrue(result["daily_close_validation"]["contiguous"])

    def test_http_adapter_rejects_unused_and_unknown_fields(self):
        base = {"calendar": "XNYS", "start": "2026-11-23", "end": "2026-11-30"}
        for extra in ({"unknown": True}, {"grace_seconds": 30}, {"require_contiguous": True},
                      {"last_bar_close": "2026-11-27T18:00:00Z"}):
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                calendar_analytics({**base, **extra})


if __name__ == "__main__":
    unittest.main()
