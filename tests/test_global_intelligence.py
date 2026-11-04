"""Global Intelligence: provider normalization, revisions, outages, safety policy, research links."""
from __future__ import annotations

import asyncio
import json
import math
import tempfile
import unittest
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx

from gig.intel import defense, research
from gig.intel.events import normalize_eonet, normalize_usgs
from gig.intel.fetch import refresh
from gig.intel.reference import research_links
from gig.intel.store import IntelStore
from gig.service.app import create_app
from gig.service.contracts import EnvelopeSchema

NOW = datetime(2026, 11, 4, 3, 0, tzinfo=UTC)


def usgs(mag=4.6, status="automatic", updated=1793760000000):
    return {"type": "FeatureCollection", "metadata": {"generated": 1793762280000}, "features": [
        {"type": "Feature", "id": "authored1",
         "properties": {"mag": mag, "time": 1793750000000, "updated": updated, "status": status,
                        "title": "Authored fixture M4.6", "url": "https://earthquake.usgs.gov/earthquakes/eventpage/authored1",
                        "magType": "mb"},
         "geometry": {"type": "Point", "coordinates": [56.0, 26.0, 10.0]}},
        {"type": "Feature", "id": "nogeom", "properties": {"time": 1793750000000}, "geometry": None}]}


def eonet():
    return {"events": [{"id": "EONET_1", "title": "Authored storm", "link": "https://eonet.gsfc.nasa.gov/api/v3/events/EONET_1",
                        "categories": [{"id": "severeStorms"}], "sources": [{"id": "JTWC", "url": "https://example.invalid/jtwc"}],
                        "geometry": [{"date": "2026-11-01T00:00:00Z", "type": "Point", "coordinates": [86.2, 11.4], "magnitudeValue": 35, "magnitudeUnit": "kts"},
                                     {"date": "2026-11-02T00:00:00Z", "type": "Point", "coordinates": [81.5, 13.9], "magnitudeValue": 60, "magnitudeUnit": "kts"},
                                     {"date": "2026-11-09T00:00:00Z", "type": "Point", "coordinates": [70.0, 20.0]}]},
                       {"id": "EONET_2", "title": "Authored fire", "categories": [{"id": "wildfires"}],
                        "geometry": [{"date": "2026-11-01T00:00:00Z", "type": "Polygon",
                                      "coordinates": [[[-120, 38], [-121, 38], [-121, 39], [-120, 39], [-120, 38]]]}]}]}


def defense_record(**changes):
    record = {"record_id": "navy-exercise-1", "branch": "NAVY", "kind": "EXERCISE_ANNOUNCEMENT",
              "title": "Authored fixture: announced multinational naval exercise",
              "summary": "Authored synthetic research fixture describing a publicly announced exercise.",
              "region": "Arabian Sea", "country_codes": ["IN", "US"], "latitude": 18.37, "longitude": 64.81,
              "published_at": "2026-10-20T08:00:00Z", "observed_start": "2026-10-25T00:00:00Z",
              "observed_end": "2026-10-30T00:00:00Z", "acquired_at": "2026-11-03T00:00:00Z",
              "sources": [{"publisher": "Authored ministry fixture", "publisher_type": "OFFICIAL_GOVERNMENT",
                           "url": "https://example.invalid/announcement"}],
              "verification": "CONFIRMED_OFFICIAL", "classification": "SYNTHETIC"}
    record.update(changes)
    return record


def opener_for(payload, status=200):
    def opener(url, timeout, max_bytes):
        return status, json.dumps(payload).encode()
    return opener


class NormalizationTests(unittest.TestCase):
    def test_usgs_keeps_clocks_and_unknown_precision(self):
        events = normalize_usgs(usgs(), acquired_at="2026-11-04T03:00:00Z")
        self.assertEqual(len(events), 1)
        event = events[0]
        self.assertEqual(event["event_id"], "USGS:authored1")
        self.assertIsNone(event["geo_precision_km"])
        self.assertEqual(event["verification"], "PROVIDER_AUTOMATIC_SUBJECT_TO_REVISION")
        self.assertLess(event["observed_at"], event["published_at"])
        self.assertGreater(event["reporting_delay_seconds"], 0)

    def test_eonet_never_uses_geometry_after_acquisition(self):
        events = {e["event_id"]: e for e in normalize_eonet(eonet(), acquired_at="2026-11-04T03:00:00Z")}
        storm = events["EONET:EONET_1"]
        self.assertEqual((storm["latitude"], storm["longitude"]), (13.9, 81.5))
        self.assertEqual(storm["track_points"], 2)
        self.assertIsNone(storm["published_at"])
        fire = events["EONET:EONET_2"]
        self.assertEqual(fire["category"], "WILDFIRE")
        self.assertGreater(fire["geo_precision_km"], 50)

    def test_invalid_payloads_reject(self):
        for bad in ({"type": "Feature"}, {"type": "FeatureCollection", "features": "x"}):
            with self.assertRaises(ValueError):
                normalize_usgs(bad, acquired_at="2026-11-04T03:00:00Z")
        payload = usgs()
        payload["features"][0]["geometry"]["coordinates"] = [200, 26]
        with self.assertRaises(ValueError):
            normalize_usgs(payload, acquired_at="2026-11-04T03:00:00Z")


class StoreTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = IntelStore(Path(self.tmp.name))

    def tearDown(self):
        self.tmp.cleanup()

    def test_dedup_revisions_restart_and_replay(self):
        first = refresh(self.store, "USGS_EARTHQUAKES", now=NOW, opener=opener_for(usgs()))
        self.assertEqual((first["state"], first["new_revisions"]), ("ACQUIRED", 1))
        again = refresh(self.store, "USGS_EARTHQUAKES", now=NOW + timedelta(minutes=10), opener=opener_for(usgs()))
        self.assertEqual(again["new_revisions"], 0)
        revised = refresh(self.store, "USGS_EARTHQUAKES", now=NOW + timedelta(minutes=20),
                          opener=opener_for(usgs(mag=4.8, status="reviewed", updated=1793762000000)))
        self.assertEqual(revised["new_revisions"], 1)
        restarted = IntelStore(Path(self.tmp.name))
        current = restarted.events(as_of=NOW + timedelta(hours=1))
        self.assertEqual(current[0]["magnitude"], 4.8)
        self.assertEqual(current[0]["revision_count"], 2)
        historical = restarted.events(as_of=NOW + timedelta(minutes=5))
        self.assertEqual(historical[0]["magnitude"], 4.6)
        self.assertEqual(restarted.events(as_of=NOW - timedelta(minutes=1)), [])

    def test_rate_limit_backoff_and_outage_semantics(self):
        refresh(self.store, "USGS_EARTHQUAKES", now=NOW, opener=opener_for(usgs()))
        deferred = refresh(self.store, "USGS_EARTHQUAKES", now=NOW + timedelta(seconds=10), opener=opener_for(usgs()))
        self.assertEqual(deferred["state"], "DEFERRED")
        failed = refresh(self.store, "USGS_EARTHQUAKES", now=NOW + timedelta(minutes=6), opener=opener_for({}, status=503))
        self.assertEqual(failed["state"], "FAILED")
        allowed = self.store.next_allowed("USGS_EARTHQUAKES")
        self.assertEqual(allowed, NOW + timedelta(minutes=6) + timedelta(seconds=600))
        now = NOW + timedelta(minutes=7)
        state = self.store.semantics("USGS_EARTHQUAKES", now=now, as_of=now)
        self.assertEqual(state["semantics"], "CACHED")
        self.assertEqual(state["consecutive_failures"], 1)
        self.assertEqual(len(self.store.events(as_of=now)), 1)

        def broken(url, timeout, max_bytes):
            raise TimeoutError("timed out")
        self.assertEqual(refresh(self.store, "NASA_EONET", now=NOW, opener=broken)["error_code"], "TIMEOUT_OR_NETWORK")
        self.assertEqual(self.store.semantics("NASA_EONET", now=now, as_of=now)["semantics"], "UNAVAILABLE")
        self.assertEqual(self.store.events(as_of=now, layers={"geopolitical"}), [])

    def test_fresh_and_historical_semantics_never_live(self):
        refresh(self.store, "USGS_EARTHQUAKES", now=NOW, opener=opener_for(usgs()))
        self.assertEqual(self.store.semantics("USGS_EARTHQUAKES", now=NOW, as_of=NOW)["semantics"], "DELAYED")
        later = NOW + timedelta(days=2)
        self.assertEqual(self.store.semantics("USGS_EARTHQUAKES", now=later, as_of=later)["semantics"], "CACHED")
        self.assertEqual(self.store.semantics("USGS_EARTHQUAKES", now=later, as_of=NOW)["semantics"], "HISTORICAL")

    def test_unregistered_source_rejected(self):
        with self.assertRaises(ValueError):
            refresh(self.store, "OPENSKY", now=NOW)


class DefensePolicyTests(unittest.TestCase):
    def test_coarsened_cited_delayed_record(self):
        record = defense.admit_record(defense_record())
        self.assertEqual((record["latitude"], record["longitude"]), (18.0, 65.0))
        self.assertEqual(record["interpretation"], "NOT_EVIDENCE_OF_IMPENDING_CONFLICT")
        self.assertTrue(record["observation_time_known"])
        themes = {t["theme"] for t in record["research_links"]["themes"]}
        self.assertTrue({"defense", "tanker_shipping", "crude_oil"} <= themes)

    def test_operational_and_unsupported_inputs_reject(self):
        bad = [defense_record(published_at="2026-11-02T12:00:00Z"),
               defense_record(heading_degrees=90), defense_record(hull_number="X-1"),
               defense_record(sources=[]), defense_record(branch="SPACE"),
               defense_record(kind="AIRSPACE_RESTRICTION_NOTICE"),
               defense_record(sources=[{"publisher": "Blog", "publisher_type": "OTHER", "url": "https://example.invalid"}]),
               defense_record(verification="MULTI_SOURCE_REPORTED"),
               defense_record(sources=[{"publisher": "Ministry", "publisher_type": "OFFICIAL_GOVERNMENT", "url": "http://insecure.invalid"}]),
               defense_record(observed_start="2026-11-05T00:00:00Z", observed_end=None),
               defense_record(country_codes=["usa"]), defense_record(latitude=math.nan)]
        for record in bad:
            with self.subTest(record=record), self.assertRaises((ValueError, TypeError)):
                defense.admit_record({k: v for k, v in record.items() if v is not None})

    def test_aggregate_is_counts_only(self):
        records = [defense.admit_record(defense_record()),
                   defense.admit_record(defense_record(record_id="air-1", branch="AIR_FORCE", kind="AIRSPACE_RESTRICTION_NOTICE",
                                                       region="Eastern Mediterranean", verification="UNVERIFIED_REPORT"))]
        aggregate = defense.aggregate(records)
        self.assertEqual(sum(r["count"] for r in aggregate["weekly"]), 2)
        self.assertNotIn("latitude", json.dumps(aggregate["weekly"]))


class ResearchTests(unittest.TestCase):
    def series(self, jump=0.0):
        start = datetime(2026, 1, 1, tzinfo=UTC)
        price, rows = 100.0, []
        for i in range(120):
            price *= math.exp(0.001 * ((i * 7) % 5 - 2) + (jump if i == 90 else 0))
            rows.append({"date": (start + timedelta(days=i)).isoformat(), "close": price})
        return rows

    def test_event_study_uses_only_pre_event_estimation(self):
        base = {"instrument": "SYN-ENERGY", "classification": "SYNTHETIC", "source": "authored", "event_at": "2026-04-01T00:00:00Z",
                "estimation_window": 60, "pre": 1, "post": 5}
        plain = research.event_study({**base, "prices": self.series()})
        shocked = research.event_study({**base, "prices": self.series(jump=0.05)})
        self.assertEqual(plain["parameters"], shocked["parameters"])
        self.assertAlmostEqual(shocked["car"] - plain["car"], 0.05, places=9)
        self.assertEqual(shocked["interpretation"], "DESCRIPTIVE_SINGLE_EVENT_NOT_CAUSAL")
        market = research.event_study({**base, "prices": self.series(jump=0.05), "market_prices": self.series()})
        self.assertEqual(market["model"], "MARKET_MODEL_OLS")
        with self.assertRaises(ValueError):
            research.event_study({**base, "prices": self.series(), "event_at": "2026-01-10T00:00:00Z"})

    def test_exposure_and_scenario(self):
        result = research.theme_exposure({"nav": 1000, "holdings": [
            {"instrument": "A", "market_value": 300, "themes": ["crude_oil", "tanker_shipping"]},
            {"instrument": "B", "market_value": 200, "themes": ["aviation"]},
            {"instrument": "C", "market_value": 100, "themes": []}],
            "shocks": {"crude_oil": 0.1, "tanker_shipping": -0.2, "aviation": -0.15}})
        values = {r["theme"]: r["market_value"] for r in result["themes"]}
        self.assertEqual(values["crude_oil"], 300)
        self.assertAlmostEqual(result["scenario"]["pnl"], 300 * -0.2 + 200 * -0.15)
        self.assertEqual(result["untagged_market_value"], 100)
        with self.assertRaises(ValueError):
            research.theme_exposure({"holdings": [{"instrument": "S", "market_value": -1, "themes": []}]})

    def test_links_are_hypotheses(self):
        links = research_links("SEVERE_STORM", 26.0, 56.0)
        self.assertEqual(links["link_type"], "RESEARCH_HYPOTHESIS_NOT_CAUSAL")
        self.assertEqual(links["nearby_chokepoints"][0]["chokepoint"], "Strait of Hormuz")


class WorkspaceRefreshTests(unittest.TestCase):
    def test_deferred_and_failed_outcomes_are_valid_envelopes(self):
        from gig.service.intel_workspace import IntelWorkspace
        with tempfile.TemporaryDirectory() as tmp:
            workspace = IntelWorkspace(Path(tmp), opener=opener_for(usgs()))
            first = json.loads(workspace.refresh({"source_id": "USGS_EARTHQUAKES"}).wire)
            self.assertEqual((first["data"]["state"], first["provenance"]["classification"]), ("ACQUIRED", "DELAYED"))
            again = json.loads(workspace.refresh({"source_id": "USGS_EARTHQUAKES"}).wire)
            self.assertEqual((again["data"]["state"], again["provenance"]["classification"]), ("DEFERRED", "SNAPSHOT"))
            workspace.opener = opener_for({}, status=503)
            failed = json.loads(workspace.refresh({"source_id": "NASA_EONET"}).wire)
            self.assertEqual((failed["data"]["state"], failed["provenance"]["classification"]), ("FAILED", "SNAPSHOT"))


class IntelHTTPTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp = tempfile.TemporaryDirectory()

        async def clock():
            while True:
                await asyncio.sleep(.01)
        self.clock = asyncio.create_task(clock())
        self.client = httpx.AsyncClient(transport=httpx.ASGITransport(app=create_app(state_dir=Path(self.tmp.name))),
                                        base_url="http://test")

    async def asyncTearDown(self):
        await self.client.aclose()
        self.clock.cancel()
        self.tmp.cleanup()

    async def test_overview_admit_demo_and_boundaries(self):
        empty = (await self.client.get("/api/v1/intel/overview")).json()
        EnvelopeSchema.model_validate(empty)
        self.assertEqual(empty["data"]["event_count"], 0)
        self.assertTrue(all(h["semantics"] == "UNAVAILABLE" for h in empty["data"]["source_health"]))
        response = await self.client.post("/api/v1/intel/admit", json={"records": [defense_record()]})
        self.assertEqual(response.status_code, 200, response.text)
        response = await self.client.post("/api/v1/intel/demonstration", json={})
        self.assertEqual(response.json()["data"]["admitted"], 6)
        overview = (await self.client.get("/api/v1/intel/overview?days=60")).json()["data"]
        self.assertEqual(overview["counts_by_provider"], {"ADMITTED_REPORTS": 1, "DEMONSTRATION": 6})
        self.assertEqual(overview["data_semantics"]["DEMONSTRATION"], "DEMONSTRATION")
        self.assertEqual(overview["data_semantics"]["ADMITTED_REPORTS"], "DEMONSTRATION")
        self.assertEqual(sum(r["count"] for r in overview["defense_aggregate"]["weekly"]), 1)
        filtered = (await self.client.get("/api/v1/intel/overview?days=60&layers=defense")).json()["data"]
        self.assertEqual(filtered["event_count"], 1)
        self.assertEqual((await self.client.get("/api/v1/intel/overview?layers=military")).status_code, 422)
        cross = await self.client.post("/api/v1/intel/admit", json={"records": [defense_record()]},
                                       headers={"Origin": "http://evil.invalid"})
        self.assertEqual(cross.status_code, 403)
        unsafe = await self.client.post("/api/v1/intel/admit", json={"records": [defense_record(callsign="X")]})
        self.assertEqual(unsafe.status_code, 422)
        refused = await self.client.post("/api/v1/intel/refresh", json={"source_id": "OPENSKY"})
        self.assertEqual(refused.status_code, 422)
        saved = await self.client.post("/api/v1/intel/investigation", json={"name": "Hormuz shipping", "body": {
            "event_ids": ["ADMITTED:navy-exercise-1"], "themes": ["crude_oil"], "as_of": "2026-11-04T00:00:00Z"}})
        self.assertEqual(saved.status_code, 200, saved.text)
        overview = (await self.client.get("/api/v1/intel/overview")).json()["data"]
        self.assertEqual(overview["investigations"][0]["name"], "Hormuz shipping")
