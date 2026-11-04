"""Sprint 9 market-data lineage: four clocks, identity and fail-closed PIT eligibility."""
from __future__ import annotations

import asyncio
import copy
import unittest

import httpx

from gig.data.provenance import assess_eligibility, documented_sources, lineage_record
from gig.service.app import create_app
from gig.service.contracts import EnvelopeSchema

SHA = "a" * 64


def complete(**changes):
    record = {
        "record_id": "vendor:bars:v1", "source_identity": "authored-fixture-vendor",
        "dataset": "daily bars", "dataset_version": "v1", "classification": "HISTORICAL",
        "source_url": "https://example.invalid/bars", "license": {
            "identifier": "fixture", "url": None, "research_use": True, "commercial_use": False,
            "attribution": None},
        "access_permission": "VERIFIED_PUBLIC_TERMS",
        "observation_start": "2026-01-02T21:00:00Z", "observation_end": "2026-01-30T21:00:00Z",
        "publication_time": "2026-01-30T22:00:00Z", "acquisition_time": "2026-02-01T00:00:00Z",
        "ingestion_time": "2026-02-01T00:05:00Z",
        "historical_availability": {"state": "VERIFIED", "evidence": "vendor vintage archive"},
        "revision_history": [{"revision_id": "r1", "available_at": "2026-01-30T22:00:00Z", "content_sha256": SHA}],
        "corporate_action_basis": "RAW", "security_identity": {"scheme": "FIGI", "value": "BBG000000000"},
        "currency": "USD", "unit": "price per share", "quality": {"passed": ["ohlc-order"], "failed": []},
        "content_sha256": SHA,
    }
    record.update(changes)
    return record


class ProvenanceTests(unittest.TestCase):
    def assess(self, records, record_id="vendor:bars:v1", as_of="2026-03-01T00:00:00Z",
               purpose="point_in_time_research"):
        return assess_eligibility(records, record_id=record_id, as_of=as_of, purpose=purpose)

    def test_complete_lineage_is_eligible_and_hash_is_stable(self):
        first, second = lineage_record(complete()), lineage_record(complete())
        self.assertEqual(first["lineage_sha256"], second["lineage_sha256"])
        result = self.assess([complete()])
        self.assertEqual(result["eligibility"], "ELIGIBLE")
        self.assertTrue(result["point_in_time_claimed"])

    def test_missing_evidence_is_unknown_not_eligible(self):
        for change, reason in (
            ({"publication_time": None}, "PUBLICATION_TIME_UNKNOWN"),
            ({"revision_history": None}, "REVISION_HISTORY_UNKNOWN"),
            ({"corporate_action_basis": "UNKNOWN"}, "CORPORATE_ACTION_BASIS_UNKNOWN"),
            ({"security_identity": {"scheme": "SOURCE_LABEL", "value": "AA"}}, "PERMANENT_SECURITY_IDENTITY_UNAVAILABLE"),
            ({"currency": None}, "CURRENCY_UNKNOWN"),
            ({"historical_availability": {"state": "UNKNOWN", "evidence": "none"}}, "HISTORICAL_AVAILABILITY_UNVERIFIED"),
        ):
            with self.subTest(reason=reason):
                result = self.assess([complete(**change)])
                self.assertEqual(result["eligibility"], "UNKNOWN")
                self.assertIn(reason, result["records"][0]["unknown_reasons"])
                self.assertFalse(result["point_in_time_claimed"])

    def test_future_clocks_are_ineligible(self):
        result = self.assess([complete()], as_of="2026-01-31T00:00:00Z")
        self.assertEqual(result["eligibility"], "INELIGIBLE")
        self.assertIn("NOT_YET_ACQUIRED_AT_CUTOFF", result["records"][0]["ineligible_reasons"])
        late = complete(revision_history=[{"revision_id": "r1", "available_at": "2026-02-15T00:00:00Z",
                                           "content_sha256": SHA}])
        result = self.assess([late], as_of="2026-02-10T00:00:00Z")
        self.assertIn("FIRST_VINTAGE_AFTER_CUTOFF", result["records"][0]["ineligible_reasons"])

    def test_funded_allocation_rejects_synthetic(self):
        result = self.assess([complete(classification="SYNTHETIC")], purpose="funded_allocation")
        self.assertEqual(result["eligibility"], "INELIGIBLE")
        self.assertEqual(self.assess([complete(classification="SYNTHETIC")])["eligibility"], "ELIGIBLE")

    def test_derived_record_inherits_weakest_parent(self):
        parent = lineage_record(complete(publication_time=None))
        child = complete(record_id="derived", parents=[parent["lineage_sha256"]], transformation="log returns")
        result = self.assess([parent, child], record_id="derived")
        self.assertEqual(result["eligibility"], "UNKNOWN")
        derived = next(r for r in result["records"] if r["record_id"] == "derived")
        self.assertIn("INHERITED_FROM_PARENT", derived["unknown_reasons"])
        orphan = complete(record_id="orphan", parents=["b" * 64], transformation="t")
        self.assertIn("PARENT_LINEAGE_MISSING",
                      self.assess([orphan], record_id="orphan")["records"][0]["unknown_reasons"])

    def test_documented_sources_are_never_pit_eligible(self):
        records = documented_sources()
        for record in records:
            with self.subTest(record=record["record_id"]):
                cutoff = "2026-11-04T00:00:00Z"
                result = self.assess(records, record_id=record["record_id"], as_of=cutoff)
                self.assertEqual(result["eligibility"], "UNKNOWN")
                early = self.assess(records, record_id=record["record_id"], as_of="2026-03-01T00:00:00Z")
                self.assertEqual(early["eligibility"], "INELIGIBLE")
                inspection = self.assess(records, record_id=record["record_id"], as_of=cutoff,
                                         purpose="archive_inspection")
                self.assertEqual(inspection["eligibility"], "ELIGIBLE")
                self.assertFalse(inspection["point_in_time_claimed"])

    def test_invalid_records_reject(self):
        tampered = lineage_record(complete())
        tampered["dataset_version"] = "v2"
        bad = [
            complete(classification="LIVE"), complete(ingestion_time="2026-01-31T00:00:00Z"),
            complete(observation_start="2026-02-01T00:00:00Z"), complete(publication_time="2025-12-01T00:00:00Z"),
            complete(license={"identifier": "x"}), complete(parents=["c" * 64]),
            complete(acquisition_time="2026-02-01T00:00:00"), tampered,
            complete(revision_history=[{"revision_id": "b", "available_at": "2026-02-01T00:00:00Z", "content_sha256": SHA},
                                       {"revision_id": "a", "available_at": "2026-01-31T00:00:00Z", "content_sha256": SHA}]),
        ]
        for record in bad:
            with self.subTest(record=record), self.assertRaises((ValueError, TypeError)):
                lineage_record(record)
        with self.assertRaises(ValueError):
            self.assess([complete(), complete()])
        with self.assertRaises(ValueError):
            self.assess([complete()], purpose="trading")

    def test_inputs_not_mutated(self):
        record = complete()
        original = copy.deepcopy(record)
        self.assess([record])
        self.assertEqual(record, original)


class ProvenanceHTTPTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        async def clock():
            while True:
                await asyncio.sleep(.01)
        self.clock = asyncio.create_task(clock())
        self.client = httpx.AsyncClient(transport=httpx.ASGITransport(app=create_app()), base_url="http://test")

    async def asyncTearDown(self):
        await self.client.aclose()
        self.clock.cancel()

    async def test_documented_and_supplied_records(self):
        response = await self.client.post("/api/v1/analytics/provenance", json={
            "record_id": "UCI_DOW_JONES_WEEKLY_2011", "as_of": "2026-11-03T12:00:00Z",
            "purpose": "point_in_time_research"})
        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        EnvelopeSchema.model_validate(body)
        self.assertEqual(body["data"]["eligibility"], "UNKNOWN")
        response = await self.client.post("/api/v1/analytics/provenance", json={
            "records": [complete()], "record_id": "vendor:bars:v1", "as_of": "2026-03-01T00:00:00Z",
            "purpose": "funded_allocation"})
        self.assertEqual(response.json()["data"]["eligibility"], "ELIGIBLE")
        response = await self.client.post("/api/v1/analytics/provenance", json={"record_id": "x"})
        self.assertEqual(response.status_code, 422)
