"""Weekly source identities and dates never become guessed intraday/PIT bars."""
from __future__ import annotations

import csv
import io
import unittest
import zipfile
from decimal import Decimal
from unittest.mock import patch

from gig.data.public_equity import (
    COLUMNS,
    PROVIDER,
    QUARANTINED,
    _UCIRedirect,
    admit_equity_fixture,
    fetch_public_equity,
    parse_equity_csv,
    request_window,
)

ACQUIRED = "2026-11-03T19:00:00.123456789Z"
REQUEST = {"provider": PROVIDER, "symbols": ["AA"], "start": "2011-01-07", "end": "2011-01-21"}


def fixture():
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=COLUMNS)
    writer.writeheader()
    for day in (7, 14, 21):
        writer.writerow({"quarter": "1", "stock": "AA", "date": f"1/{day}/2011", "open": "$10.10", "high": "$12.00",
                         "low": "$9.50", "close": "$11.25", "volume": "10000", "percent_change_price": "11.3861",
                         "percent_change_volume_over_last_wk": "", "previous_weeks_volume": "", "next_weeks_open": "$99.00",
                         "next_weeks_close": "$101.00", "percent_change_next_weeks_price": "2.0202",
                         "days_to_next_dividend": "3", "percent_return_next_dividend": "9"})
    return stream.getvalue()


class EquityEvidenceTests(unittest.TestCase):
    def test_exact_weekly_values_provenance_and_unavailable_identity(self):
        result = parse_equity_csv(fixture(), REQUEST, ACQUIRED)
        row = result["rows"][0]
        self.assertEqual(row["source_date"], "1/7/2011")
        self.assertEqual(row["week_ending_date"], "2011-01-07")
        self.assertEqual(row["instrument_id"], "DJIA_MEMBER_LABEL:AA")
        self.assertEqual(row["close"], "11.25")
        self.assertEqual(Decimal(row["close"])-Decimal(row["open"]), Decimal("1.15"))
        self.assertEqual(row["volume"], 10000)
        self.assertIn("123456789", row["knowledge_at"])
        self.assertEqual(row["knowledge_at"], row["acquired_at"])
        self.assertIsNone(row["currency"])
        self.assertIsNone(row["observation_time"])
        self.assertNotIn("timestamp", row)
        self.assertFalse(result["point_in_time_backtest_eligible"])
        self.assertFalse(result["funded_allocation_eligible"])
        self.assertEqual(result["historical_universe_membership_state"], "UNKNOWN")
        self.assertEqual(result["corporate_actions_state"], "UNAVAILABLE")
        self.assertEqual(result["diagnostics"]["source_auxiliary_missing_cells"], 6)

    def test_future_labels_and_dividend_predictors_never_appear_in_normalized_features(self):
        original = parse_equity_csv(fixture(), REQUEST, ACQUIRED)
        changed = parse_equity_csv(fixture().replace("$101.00", "$999999.00").replace(",3,9", ",50000,99999"), REQUEST, ACQUIRED)
        self.assertEqual(original["rows"], changed["rows"])
        for row in original["rows"]:
            self.assertFalse(set(row)&set(QUARANTINED))
        self.assertEqual(set(original["quarantined_future_columns"]), set(QUARANTINED))

    def test_fixture_cannot_claim_observed_source_or_license(self):
        document, raw = admit_equity_fixture({**REQUEST, "acquired_at": ACQUIRED, "csv_text": fixture()})
        self.assertEqual(document["classification"], "SYNTHETIC")
        self.assertIsNone(document["source_url"])
        self.assertIsNone(document["dataset_doi"])
        self.assertIsNone(document["license_url"])
        self.assertEqual(document["raw_sha256"], document["csv_sha256"])
        self.assertEqual(raw.decode(), fixture())
        with self.assertRaises(ValueError):
            admit_equity_fixture({**REQUEST, "acquired_at": ACQUIRED, "csv_text": fixture(), "classification": "HISTORICAL"})

    def test_invalid_quotes_missing_volume_duplicates_and_dates_reject(self):
        for text in (fixture().replace("$10.10", "$NaN", 1), fixture().replace("10000", "", 1),
                     fixture().replace("$12.00", "$1.00", 1), fixture()+fixture().splitlines()[1]+"\n",
                     fixture().replace("1/7/2011", "1/7/2012", 1), fixture().replace("10000", "1e4", 1)):
            with self.subTest(text=text[:50]), self.assertRaises(ValueError):
                parse_equity_csv(text, REQUEST, ACQUIRED)
        with self.assertRaises(ValueError):
            parse_equity_csv(fixture(), REQUEST, "2010-01-01T00:00:00Z")
        with self.assertRaises(ValueError):
            parse_equity_csv(fixture(), REQUEST, "2026-11-03T19:00:00")

    def test_explicit_observed_zero_volume_does_not_mean_unknown(self):
        result = parse_equity_csv(fixture().replace("10000", "0", 1), REQUEST, ACQUIRED)
        self.assertEqual(result["rows"][0]["volume"], 0)

    def test_requests_reject_credentials_arbitrary_urls_and_current_ticker_aliases_before_fetch(self):
        for payload in ({**REQUEST, "provider": "https://example.invalid"}, {**REQUEST, "url": "https://example.invalid"},
                        {**REQUEST, "symbols": ["AA", "AA"]}, {**REQUEST, "symbols": ["GOOG"]},
                        {**REQUEST, "api_key": "credential-must-not-be-used"}, {**REQUEST, "end": "2026-01-01"}):
            with self.subTest(payload=payload), patch("gig.data.public_equity.urllib.request.build_opener") as opener:
                with self.assertRaises(ValueError):
                    fetch_public_equity(payload)
                opener.assert_not_called()
        with self.assertRaises(ValueError):
            request_window({**REQUEST, "symbols": ["AA", "AXP", "BA", "BAC", "CAT", "KO"]})

    def test_redirects_cannot_leave_fixed_official_host(self):
        redirect = _UCIRedirect()
        for url in ("https://example.invalid/source", "http://archive.ics.uci.edu/source", "https://user:password@archive.ics.uci.edu/source"):
            with self.assertRaises(ValueError):
                redirect.redirect_request(None, None, 302, "", {}, url)

    def test_missing_requested_security_is_not_silently_dropped(self):
        with self.assertRaises(ValueError):
            parse_equity_csv(fixture(), {**REQUEST, "symbols": ["AA", "KO"]}, ACQUIRED)

    def test_archive_license_and_decompression_bounds(self):
        class Response:
            status = 200

            def __init__(self, wire, url):
                self.wire, self.url = wire, url

            def __enter__(self):
                return self

            def __exit__(self, *_):
                return False

            def read(self, _):
                return self.wire

            def geturl(self):
                return self.url

        def source(extra=False):
            stream = io.BytesIO()
            with zipfile.ZipFile(stream, "w") as archive:
                archive.writestr("dow_jones_index.data", fixture())
                archive.writestr("dow_jones_index.names", "Authored test source")
                if extra:
                    archive.writestr("../escape", "not allowed")
            return stream.getvalue()

        for page, raw in ((b"no verified license", source()), (b"creativecommons.org/licenses/by/4.0/legalcode CC BY 4.0", source(True))):
            with patch("gig.data.public_equity.urllib.request.build_opener") as opener:
                opener.return_value.open.side_effect = [Response(page, "https://archive.ics.uci.edu/dataset/312/dow+jones+index"),
                                                        Response(raw, "https://archive.ics.uci.edu/static/public/312/dow+jones+index.zip")]
                with self.assertRaises(ValueError):
                    fetch_public_equity(REQUEST)
