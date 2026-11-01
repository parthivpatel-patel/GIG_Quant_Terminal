"""Offline provider protocol, durable recovery, provenance and PIT references."""

from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd

from gig.data.ingestion import (
    AlpacaIEXAdapter,
    IngestionStore,
    Instrument,
    Page,
    ProviderWorkspace,
    UniverseStore,
    fetch_treasury_year,
    parse_treasury,
    provider_capabilities,
)
from gig.data.validated import PointInTimeStore
from gig.service.contracts import EnvelopeSchema

NOW = "2026-01-02T14:40:00.000000123+00:00"
INSTRUMENTS = [Instrument("US:AAA", "AAA", "XNYS", "USD")]


def config():
    return {
        "instruments": [
            {"instrument_id": "US:AAA", "symbol": "AAA", "calendar": "XNYS", "currency": "USD"}
        ],
        "start": "2026-01-02T14:30:00Z",
        "end": "2026-01-02T14:35:00Z",
        "timeframe": "1Min",
    }


def page(minute=30, token=None, *, close=100):
    body = {
        "bars": {
            "AAA": [
                {
                    "t": f"2026-01-02T14:{minute:02d}:00Z",
                    "o": close,
                    "h": close + 1,
                    "l": close - 1,
                    "c": close,
                    "v": 9007199254740993,
                }
            ]
        },
        "next_page_token": token,
    }
    return Page(json.dumps(body).encode(), NOW)


def treasury_xml(rate="4.25", other="4.50", event="2025-01-02"):
    return (
        f'<feed xmlns:m="http://schemas.microsoft.com/ado/2007/08/dataservices/metadata" '
        f'xmlns:d="http://schemas.microsoft.com/ado/2007/08/dataservices"><entry><content><m:properties>'
        f"<d:NEW_DATE>{event}T00:00:00</d:NEW_DATE><d:BC_10YEAR>{rate}</d:BC_10YEAR>"
        f"<d:BC_30YEAR>{other}</d:BC_30YEAR></m:properties></content></entry></feed>"
    ).encode()


class ProviderProtocolTests(unittest.TestCase):
    def test_capabilities_never_confuse_iex_sip_or_rates_with_equity(self):
        equity, rates = provider_capabilities()
        self.assertEqual(equity["feed"], "IEX")
        self.assertFalse(equity["consolidated_sip"])
        self.assertEqual(equity["connection_state"], "BLOCKED_PENDING_APPROVAL")
        self.assertEqual(rates["asset_class"], "INTEREST_RATE")
        self.assertEqual(rates["event_time_precision"], "DATE_ONLY")
        self.assertTrue(rates["license"].startswith("CC0-1.0"))
        self.assertIn("BC_1_5MONTH", rates["supported_tenors"])

    def test_strict_instrument_identity_and_approved_request_shape(self):
        for values in (
            ("US:X", "aaa", "XNYS", "USD"),
            ("US:X", "AAA/EVIL", "XNYS", "USD"),
            ("US:X", "AAA", "AUTO", "USD"),
            ("US:X", "AAA", "XNYS", "EUR"),
        ):
            with self.subTest(values=values), self.assertRaises(ValueError):
                Instrument(*values)
        url = AlpacaIEXAdapter.request_url(
            INSTRUMENTS, config()["start"], config()["end"], "1Min", "next & token"
        )
        self.assertTrue(url.startswith("https://data.alpaca.markets/v2/stocks/bars?"))
        self.assertIn("feed=iex", url)
        self.assertIn("page_token=next+%26+token", url)
        self.assertNotIn("key", url.lower())

    def test_minute_bars_preserve_exact_volume_and_acquisition_ns(self):
        snapshot, token = AlpacaIEXAdapter.parse(
            page(), INSTRUMENTS, start=config()["start"], end=config()["end"], timeframe="1Min"
        )
        self.assertIsNone(token)
        self.assertEqual(snapshot.rows[0]["timestamp"], "2026-01-02T14:31:00+00:00")
        self.assertEqual(snapshot.rows[0]["available_at"], NOW)
        self.assertEqual(snapshot.rows[0]["volume"], "9007199254740993")
        self.assertEqual(snapshot.as_dict()["classification"], "SYNTHETIC")

    def test_daily_session_label_uses_real_early_close(self):
        raw = {
            "bars": {
                "AAA": [
                    {"t": "2026-11-27T05:00:00Z", "o": 100, "h": 101, "l": 99, "c": 100, "v": 10}
                ]
            }
        }
        result, _ = AlpacaIEXAdapter.parse(
            Page(json.dumps(raw).encode(), "2026-11-27T19:00:00Z"),
            INSTRUMENTS,
            start="2026-11-27T05:00:00Z",
            end="2026-11-27T20:00:00Z",
            timeframe="1Day",
        )
        self.assertEqual(result.rows[0]["timestamp"], "2026-11-27T18:00:00+00:00")

    def test_duplicate_undeclared_future_and_invalid_bars_fail(self):
        original = json.loads(page().body)
        cases = []
        duplicate = deepcopy(original)
        duplicate["bars"]["AAA"] *= 2
        cases.append(duplicate)
        unknown = deepcopy(original)
        unknown["bars"]["BBB"] = unknown["bars"].pop("AAA")
        cases.append(unknown)
        badprice = deepcopy(original)
        badprice["bars"]["AAA"][0]["h"] = 1
        cases.append(badprice)
        badvolume = deepcopy(original)
        badvolume["bars"]["AAA"][0]["v"] = True
        cases.append(badvolume)
        for raw in cases:
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                AlpacaIEXAdapter.parse(
                    Page(json.dumps(raw).encode(), NOW),
                    INSTRUMENTS,
                    start=config()["start"],
                    end=config()["end"],
                    timeframe="1Min",
                )
        with self.assertRaises(ValueError):
            AlpacaIEXAdapter.parse(
                Page(page().body, "2026-01-02T14:30:30Z"),
                INSTRUMENTS,
                start=config()["start"],
                end=config()["end"],
                timeframe="1Min",
            )


class DurableIngestionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.base = Path(self.temp.name) / "state"
        self.store = IngestionStore(self.base)
        self.store.start("job", config())

    def tearDown(self):
        self.temp.cleanup()

    def test_paginated_restart_and_complete_replay_sends_nothing(self):
        first = self.store.step("job", lambda cursor: page(token="next"), now=NOW)
        self.assertEqual((first["state"], first["pages"], first["rows"]), ("READY", 1, 1))
        second = IngestionStore(self.base).step(
            "job",
            lambda cursor: page(31) if cursor == "next" else self.fail("wrong cursor"),
            now=NOW,
        )
        self.assertEqual((second["state"], second["pages"], second["rows"]), ("COMPLETED", 2, 2))
        with patch("builtins.print"):
            done = self.store.step(
                "job", lambda cursor: self.fail("must not refetch completed job"), now=NOW
            )
        self.assertEqual(done["pages"], 2)
        bars = PointInTimeStore(self.base / "bars.sqlite").snapshot(NOW, max_age_seconds=3600)
        self.assertEqual(len(bars["rows"]), 2)

    def test_crash_after_admission_replays_exact_receipt_without_transport(self):
        def crash():
            raise RuntimeError("injected crash")

        with self.assertRaisesRegex(RuntimeError, "injected"):
            self.store.step("job", lambda cursor: page(), now=NOW, after_admission=crash)
        self.assertEqual(self.store.status("job")["state"], "ADMITTING")
        result = IngestionStore(self.base).step(
            "job", lambda cursor: self.fail("pending bytes must be replayed"), now=NOW
        )
        self.assertEqual(result["state"], "COMPLETED")
        self.assertEqual(result["last_page_admission"], {"inserted": 0, "replayed": 1})
        self.assertEqual(
            len(
                PointInTimeStore(self.base / "bars.sqlite").snapshot(NOW, max_age_seconds=3600)[
                    "rows"
                ]
            ),
            1,
        )

    def test_retry_after_backoff_then_recovery(self):
        response = self.store.step(
            "job", lambda cursor: Page(b"", NOW, status=429, retry_after_seconds=10), now=NOW
        )
        self.assertEqual(response["state"], "WAITING")
        early = self.store.step(
            "job",
            lambda cursor: self.fail("rate limited"),
            now=(pd.Timestamp(NOW) + pd.Timedelta(seconds=9)).isoformat(),
        )
        self.assertEqual(early["attempts"], 1)
        done = self.store.step(
            "job",
            lambda cursor: page(),
            now=(pd.Timestamp(NOW) + pd.Timedelta(seconds=10)).isoformat(),
        )
        self.assertEqual(done["state"], "COMPLETED")
        self.assertEqual(done["attempts"], 0)

    def test_disconnection_retry_is_bounded_and_diagnostics_not_retained(self):
        def disconnected(cursor):
            raise RuntimeError("FAKE_PROVIDER_SECRET")

        at = pd.Timestamp(NOW)
        for _ in range(4):
            result = self.store.step("job", disconnected, now=at.isoformat())
            at += pd.Timedelta(seconds=61)
        self.assertEqual(result["state"], "ERROR")
        self.assertNotIn("FAKE_PROVIDER_SECRET", json.dumps(result))

    def test_intent_conflict_and_same_intent_replay(self):
        self.store.start("job", config())
        changed = config()
        changed["timeframe"] = "1Day"
        with self.assertRaisesRegex(ValueError, "conflicts"):
            self.store.start("job", changed)

    def test_cursor_cycle_rejects_before_second_admission(self):
        self.store.step("job", lambda cursor: page(token="next"), now=NOW)
        with self.assertRaisesRegex(ValueError, "cyclic"):
            self.store.step("job", lambda cursor: page(31, "next"), now=NOW)
        self.assertIsNone(self.store.status("job")["pending"])
        self.assertEqual(self.store.status("job")["pages"], 1)

    def test_pending_receipt_tampering_fails(self):
        with self.assertRaises(RuntimeError):
            self.store.step(
                "job",
                lambda cursor: page(),
                now=NOW,
                after_admission=lambda: (_ for _ in ()).throw(RuntimeError()),
            )
        with sqlite3.connect(self.base / "ingestion.sqlite") as con:
            con.execute("UPDATE ingestion_pages SET payload='{}'")
        with self.assertRaisesRegex(ValueError, "integrity"):
            self.store.step("job", lambda cursor: page(), now=NOW)


class ProviderWorkspaceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.base = Path(self.temp.name) / "state"
        self.workspace = ProviderWorkspace(self.base)

    def tearDown(self):
        self.temp.cleanup()

    def test_missing_reads_have_no_state_side_effects(self):
        self.assertEqual(self.workspace.status().to_dict()["state"], "AVAILABLE")
        self.assertEqual(
            self.workspace.rates(NOW, max_age_seconds=60).to_dict()["state"], "UNAVAILABLE"
        )
        self.assertEqual(
            self.workspace.universe_snapshot("u", NOW).to_dict()["state"], "UNAVAILABLE"
        )
        self.assertFalse(self.base.exists())

    def test_live_alpaca_always_blocked_without_transport_or_credentials(self):
        with patch("urllib.request.build_opener", side_effect=AssertionError("network")):
            result = self.workspace.ingest({"provider": "alpaca_iex", "mode": "LIVE"}).to_dict()
        self.assertEqual(result["state"], "UNAVAILABLE")
        self.assertEqual(result["error"]["code"], "BROKER_DISABLED")
        self.assertFalse(self.base.exists())

    def test_fixture_http_shape_admits_synthetic_bars_and_replays(self):
        request = {
            **config(),
            "provider": "alpaca_iex",
            "mode": "OFFLINE_FIXTURE",
            "job_id": "fixture-job",
            "now": NOW,
            "pages": [{"body": json.loads(page().body), "observed_at": NOW}],
        }
        original = deepcopy(request)
        result = self.workspace.ingest(request).to_dict()
        EnvelopeSchema.model_validate(result)
        self.assertEqual(result["state"], "AVAILABLE")
        self.assertEqual(result["provenance"]["classification"], "SYNTHETIC")
        self.assertEqual(request, original)
        self.assertEqual(self.workspace.ingest(request).to_dict()["data"]["metrics"]["pages"], 1)

    def test_retry_and_rejection_are_not_healthy_empty_success(self):
        for status, expected in ((429, "UNKNOWN"), (403, "ERROR")):
            request = {
                **config(),
                "provider": "alpaca_iex",
                "mode": "OFFLINE_FIXTURE",
                "job_id": "job" + str(status),
                "now": NOW,
                "pages": [{"body": {}, "observed_at": NOW, "status": status}],
            }
            result = self.workspace.ingest(request).to_dict()
            self.assertEqual(result["state"], expected)
            self.assertIsNone(result["data"])

    def test_empty_completed_fixture_is_unavailable(self):
        request = {
            **config(),
            "provider": "alpaca_iex",
            "mode": "OFFLINE_FIXTURE",
            "job_id": "empty",
            "now": NOW,
            "pages": [{"body": {"bars": {}, "next_page_token": None}, "observed_at": NOW}],
        }
        self.assertEqual(self.workspace.ingest(request).to_dict()["state"], "UNAVAILABLE")


class TreasuryEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.base = Path(self.temp.name) / "state"
        self.workspace = ProviderWorkspace(self.base)

    def tearDown(self):
        self.temp.cleanup()

    def test_public_transport_is_fixed_bounded_and_uses_actual_acquisition_time(self):
        response = MagicMock()
        response.__enter__.return_value = response
        response.read.return_value = treasury_xml()
        opener = MagicMock()
        opener.open.return_value = response
        before = pd.Timestamp.now(tz="UTC")
        with patch("urllib.request.build_opener", return_value=opener):
            body, observed = fetch_treasury_year(2025)
        request = opener.open.call_args.args[0]
        self.assertEqual(
            request.full_url,
            "https://home.treasury.gov/resource-center/data-chart-center/interest-rates/pages/xml?data=daily_treasury_yield_curve&field_tdr_date_value=2025",
        )
        self.assertEqual(request.get_method(), "GET")
        self.assertEqual(opener.open.call_args.kwargs, {"timeout": 20})
        response.read.assert_called_once_with(4_194_305)
        self.assertEqual(body, treasury_xml())
        self.assertGreaterEqual(pd.Timestamp(observed), before)

    def test_public_transport_failure_has_no_workspace_side_effects(self):
        with (
            patch("urllib.request.build_opener", side_effect=OSError("unavailable")),
            self.assertRaises(OSError),
        ):
            self.workspace.ingest(
                {"provider": "us_treasury", "mode": "PUBLIC_OBSERVED", "year": 2025}
            )
        self.assertFalse(self.base.exists())

    def test_client_cannot_supply_public_url_or_backdate_acquisition(self):
        for extra in ({"observed_at": NOW}, {"url": "http://localhost/private"}):
            with (
                patch("urllib.request.build_opener", side_effect=AssertionError("network")),
                self.assertRaises(ValueError),
            ):
                self.workspace.ingest(
                    {
                        "provider": "us_treasury",
                        "mode": "PUBLIC_OBSERVED",
                        "year": 2025,
                        **extra,
                    }
                )
        self.assertFalse(self.base.exists())

    def test_date_precision_and_exact_decimal_without_fake_publication_time(self):
        rows = parse_treasury(treasury_xml(), NOW)
        self.assertEqual(rows[0]["event_date"], "2025-01-02")
        self.assertEqual(rows[0]["yield_percent"], "4.25")
        self.assertEqual(rows[0]["available_at"], NOW)
        self.assertEqual(rows[0]["publication_time"], "UNKNOWN")

    def test_six_week_tenor_in_current_official_schema_is_not_dropped(self):
        body = treasury_xml().replace(
            b"<d:BC_10YEAR>", b"<d:BC_1_5MONTH>3.50</d:BC_1_5MONTH><d:BC_10YEAR>"
        )
        rows = parse_treasury(body, NOW)
        self.assertEqual(len(rows), 3)
        self.assertEqual(
            next(row["yield_percent"] for row in rows if row["tenor"] == "BC_1_5MONTH"), "3.50"
        )
        self.assertNotIn("timestamp", rows[0])

    def test_direct_bytes_are_fixture_not_public_observation(self):
        result = self.workspace.ingest_treasury_bytes(
            treasury_xml(), observed_at=NOW, year=2025
        ).to_dict()
        self.assertEqual(result["provenance"]["classification"], "SYNTHETIC")
        before = (pd.Timestamp(NOW) - pd.Timedelta(nanoseconds=1)).isoformat()
        self.assertEqual(
            self.workspace.rates(before, max_age_seconds=3600).to_dict()["state"], "UNAVAILABLE"
        )
        rates = self.workspace.rates(NOW, max_age_seconds=3600).to_dict()
        self.assertEqual(rates["provenance"]["classification"], "SYNTHETIC")

    def test_registered_public_fetch_path_records_observed_origin(self):
        with patch(
            "gig.data.ingestion.fetch_treasury_year", return_value=(treasury_xml(), NOW)
        ) as fetch:
            result = self.workspace.ingest(
                {"provider": "us_treasury", "mode": "PUBLIC_OBSERVED", "year": 2025}
            ).to_dict()
        fetch.assert_called_once_with(2025)
        self.assertEqual(result["provenance"]["classification"], "HISTORICAL")
        self.assertFalse(result["data"]["metrics"]["equity_bars_fabricated"])

    def test_restart_freshness_keeps_original_acquisition_and_unknown_policy(self):
        self.workspace.ingest_treasury_bytes(treasury_xml(), observed_at=NOW, year=2025)
        reopened = ProviderWorkspace(self.base)
        stale = reopened.rates(
            (pd.Timestamp(NOW) + pd.Timedelta(seconds=61)).isoformat(), max_age_seconds=60
        ).to_dict()
        self.assertEqual(stale["state"], "STALE")
        self.assertEqual(stale["provenance"]["as_of"], NOW)
        unknown = reopened.rates(NOW).to_dict()
        self.assertEqual(unknown["state"], "UNKNOWN")
        self.assertIsNone(unknown["data"])

    def test_revision_knowledge_time_and_conflicts(self):
        self.workspace.ingest_treasury_bytes(treasury_xml(), observed_at=NOW, year=2025)
        later = (pd.Timestamp(NOW) + pd.Timedelta(seconds=30)).isoformat()
        self.workspace.ingest_treasury_bytes(treasury_xml("4.30"), observed_at=later, year=2025)
        self.assertEqual(
            self.workspace.rates(NOW, max_age_seconds=60).to_dict()["data"]["rows"][0][
                "yield_percent"
            ],
            "4.25",
        )
        self.assertEqual(
            self.workspace.rates(later, max_age_seconds=60).to_dict()["data"]["rows"][0][
                "yield_percent"
            ],
            "4.30",
        )
        with self.assertRaisesRegex(ValueError, "conflicting"):
            self.workspace.ingest_treasury_bytes(treasury_xml("4.40"), observed_at=NOW, year=2025)

    def test_metadata_and_source_corruption_are_detected(self):
        self.workspace.ingest_treasury_bytes(treasury_xml(), observed_at=NOW, year=2025)
        path = next((self.base / "provider_receipts").glob("*.json"))
        path.write_text("{}")
        with self.assertRaisesRegex(ValueError, "integrity"):
            self.workspace.rates(NOW, max_age_seconds=60)

    def test_mixed_fixture_and_observed_origins_never_merge(self):
        self.workspace.ingest_treasury_bytes(treasury_xml(), observed_at=NOW, year=2025)
        later = (pd.Timestamp(NOW) + pd.Timedelta(seconds=1)).isoformat()
        self.workspace.ingest_treasury_bytes(
            treasury_xml(), observed_at=later, year=2025, public_observed=True
        )
        self.assertEqual(
            self.workspace.rates(later, max_age_seconds=60).to_dict()["state"], "UNKNOWN"
        )

    def test_malformed_duplicate_nonfinite_future_and_external_entity_xml_fail(self):
        values = [
            b"not XML",
            b'<!DOCTYPE x [<!ENTITY a SYSTEM "file:///etc/passwd">]><x/>',
            treasury_xml("NaN"),
            treasury_xml(event="2030-01-01"),
            treasury_xml().replace(b"<d:BC_10YEAR>", b"<d:NEW_DATE>x</d:NEW_DATE><d:BC_10YEAR>"),
        ]
        for body in values:
            with self.subTest(body=body[:30]), self.assertRaises(ValueError):
                parse_treasury(body, NOW)

    def test_bounded_response_and_invalid_freshness(self):
        self.workspace.ingest_treasury_bytes(treasury_xml(), observed_at=NOW, year=2025)
        result = self.workspace.rates(NOW, limit=1, max_age_seconds=60).to_dict()
        self.assertEqual(result["data"]["rows_total"], 2)
        self.assertFalse(result["data"]["rows_complete"])
        for age in (True, float("nan"), -1, float("inf")):
            with self.subTest(age=age), self.assertRaises(ValueError):
                self.workspace.rates(NOW, max_age_seconds=age)


class UniverseKnowledgeTests(unittest.TestCase):
    def test_atomic_identity_replay_and_point_in_time_membership(self):
        with tempfile.TemporaryDirectory() as directory:
            store = UniverseStore(Path(directory) / "universe.sqlite")
            payload = {
                "universe_id": "research-u",
                "source": "caller-universe",
                "effective_at": "2025-01-01T00:00:00Z",
                "available_at": "2025-01-02T00:00:00Z",
                "members": config()["instruments"],
            }
            digest = store.ingest(payload)
            self.assertEqual(store.ingest(payload), digest)
            self.assertEqual(
                store.snapshot("research-u", "2025-01-01T23:59:59Z")["state"], "UNAVAILABLE"
            )
            self.assertEqual(
                store.snapshot("research-u", NOW)["members"][0]["instrument_id"], "US:AAA"
            )
            corrected = deepcopy(payload)
            corrected["available_at"] = "2025-01-03T00:00:00Z"
            corrected["members"] = []
            store.ingest(corrected)
            self.assertEqual(
                len(store.snapshot("research-u", "2025-01-02T12:00:00Z")["members"]), 1
            )
            self.assertEqual(store.snapshot("research-u", NOW)["members"], [])
            broken = deepcopy(payload)
            broken["members"] = []
            with self.assertRaisesRegex(ValueError, "conflicting"):
                store.ingest(broken)

    def test_corrupt_universe_index_is_not_admitted_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "u.sqlite"
            store = UniverseStore(path)
            payload = {
                "universe_id": "u",
                "source": "declared",
                "effective_at": NOW,
                "available_at": NOW,
                "members": [],
            }
            store.ingest(payload)
            with sqlite3.connect(path) as con:
                con.execute("UPDATE universe_revisions SET payload='{}'")
            with self.assertRaisesRegex(ValueError, "integrity"):
                store.snapshot("u", NOW)


if __name__ == "__main__":
    unittest.main()
