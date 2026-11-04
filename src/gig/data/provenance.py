"""Market-data lineage records and fail-closed point-in-time eligibility.

A lineage record states what is known about one dataset version: source,
license, the four clocks (observation, publication, acquisition, ingestion),
historical availability, revisions, corporate-action basis, security identity,
currency/unit and quality checks. Missing evidence is UNKNOWN, never inferred.

Eligibility is evaluated for a purpose and a cutoff. Point-in-time research
requires every clock and identity needed to rule out look-ahead; funded
allocation additionally requires a non-synthetic, research-permitted source.
Derived datasets inherit the weakest eligibility of their parents.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any

from gig.timestamps import aware_timestamp

VERSION = "market-data-lineage-v1"
PURPOSES = ("archive_inspection", "point_in_time_research", "funded_allocation")
PERMANENT_IDENTITY_SCHEMES = frozenset({"FIGI", "ISIN", "CUSIP", "PERMNO", "ECB_SERIES_KEY"})
IDENTITY_SCHEMES = PERMANENT_IDENTITY_SCHEMES | {"SOURCE_LABEL", "NONE"}
CORPORATE_ACTION_BASES = frozenset({"RAW", "SPLIT_ADJUSTED", "TOTAL_RETURN_ADJUSTED",
                                    "NOT_APPLICABLE", "UNKNOWN"})
CLASSIFICATIONS = frozenset({"REAL_TIME", "DELAYED", "SNAPSHOT", "HISTORICAL", "SYNTHETIC"})
PERMISSIONS = frozenset({"VERIFIED_PUBLIC_TERMS", "CALLER_ASSERTED", "UNVERIFIED"})
AVAILABILITY = frozenset({"VERIFIED", "UNKNOWN"})
_ORDER = {"ELIGIBLE": 2, "UNKNOWN": 1, "INELIGIBLE": 0}
_REQUIRED = {"record_id", "source_identity", "dataset", "dataset_version", "classification",
             "source_url", "license", "access_permission", "observation_start", "observation_end",
             "publication_time", "acquisition_time", "ingestion_time", "historical_availability",
             "revision_history", "corporate_action_basis", "security_identity", "currency", "unit",
             "quality", "content_sha256"}
_OPTIONAL = {"parents", "transformation", "notes", "version", "lineage_sha256"}


def _text(value: Any, name: str, *, nullable: bool = False) -> str | None:
    if value is None and nullable:
        return None
    if not isinstance(value, str) or not value.strip() or len(value) > 512:
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _clock(value: Any, name: str, *, nullable: bool = False) -> str | None:
    if value is None and nullable:
        return None
    if value is None:
        raise ValueError(f"{name} is required")
    return aware_timestamp(value).isoformat()


def _sha(value: Any, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def lineage_record(raw: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and canonicalize a lineage record; returns it with ``lineage_sha256``."""
    if not isinstance(raw, Mapping) or not set(raw) >= _REQUIRED or set(raw) - _REQUIRED - _OPTIONAL:
        raise ValueError("lineage record fields do not match the contract")
    if raw["classification"] not in CLASSIFICATIONS:
        raise ValueError("unsupported data classification")
    if raw["access_permission"] not in PERMISSIONS:
        raise ValueError("unsupported access permission state")
    if raw["corporate_action_basis"] not in CORPORATE_ACTION_BASES:
        raise ValueError("unsupported corporate-action basis")
    license_ = raw["license"]
    if not isinstance(license_, Mapping) or set(license_) != {"identifier", "url", "research_use", "commercial_use", "attribution"}:
        raise ValueError("license must state identifier, url, research_use, commercial_use and attribution")
    for key in ("research_use", "commercial_use"):
        if license_[key] not in (True, False, None):
            raise ValueError(f"license.{key} must be true, false or null (unknown)")
    availability = raw["historical_availability"]
    if not isinstance(availability, Mapping) or set(availability) != {"state", "evidence"} or availability["state"] not in AVAILABILITY:
        raise ValueError("historical_availability must state VERIFIED or UNKNOWN with evidence")
    identity = raw["security_identity"]
    if not isinstance(identity, Mapping) or set(identity) != {"scheme", "value"} or identity["scheme"] not in IDENTITY_SCHEMES:
        raise ValueError("security_identity must state a supported scheme and value")
    quality = raw["quality"]
    if not isinstance(quality, Mapping) or set(quality) != {"passed", "failed"}:
        raise ValueError("quality must list passed and failed checks")
    for key in ("passed", "failed"):
        if not isinstance(quality[key], list) or not all(isinstance(x, str) and x for x in quality[key]):
            raise ValueError("quality checks must be named strings")
    revisions = raw["revision_history"]
    if revisions is not None:
        if not isinstance(revisions, list) or len(revisions) > 1000:
            raise ValueError("revision_history must be null (unknown) or a bounded list")
        canonical_revisions = []
        for item in revisions:
            if not isinstance(item, Mapping) or set(item) != {"revision_id", "available_at", "content_sha256"}:
                raise ValueError("revision entries require revision_id, available_at and content_sha256")
            canonical_revisions.append({"revision_id": _text(item["revision_id"], "revision_id"),
                                        "available_at": _clock(item["available_at"], "available_at"),
                                        "content_sha256": _sha(item["content_sha256"], "revision sha")})
        times = [aware_timestamp(r["available_at"]) for r in canonical_revisions]
        if times != sorted(times):
            raise ValueError("revision history must be chronological")
        revisions = canonical_revisions
    start = _clock(raw["observation_start"], "observation_start")
    end = _clock(raw["observation_end"], "observation_end")
    acquired = _clock(raw["acquisition_time"], "acquisition_time")
    ingested = _clock(raw["ingestion_time"], "ingestion_time")
    published = _clock(raw["publication_time"], "publication_time", nullable=True)
    if aware_timestamp(start) > aware_timestamp(end):
        raise ValueError("observation window is inverted")
    if aware_timestamp(ingested) < aware_timestamp(acquired):
        raise ValueError("ingestion cannot precede acquisition")
    if published is not None and aware_timestamp(published) < aware_timestamp(start):
        raise ValueError("publication cannot precede the first observation")
    parents = raw.get("parents", [])
    if not isinstance(parents, list) or len(parents) > 64:
        raise ValueError("parents must be a bounded list of lineage hashes")
    record = {
        "version": VERSION,
        "record_id": _text(raw["record_id"], "record_id"),
        "source_identity": _text(raw["source_identity"], "source_identity"),
        "dataset": _text(raw["dataset"], "dataset"),
        "dataset_version": _text(raw["dataset_version"], "dataset_version"),
        "classification": raw["classification"],
        "source_url": _text(raw["source_url"], "source_url", nullable=True),
        "license": {"identifier": _text(license_["identifier"], "license.identifier", nullable=True),
                    "url": _text(license_["url"], "license.url", nullable=True),
                    "research_use": license_["research_use"],
                    "commercial_use": license_["commercial_use"],
                    "attribution": _text(license_["attribution"], "license.attribution", nullable=True)},
        "access_permission": raw["access_permission"],
        "observation_start": start, "observation_end": end,
        "publication_time": published, "acquisition_time": acquired, "ingestion_time": ingested,
        "historical_availability": {"state": availability["state"],
                                    "evidence": _text(availability["evidence"], "availability evidence")},
        "revision_history": revisions,
        "corporate_action_basis": raw["corporate_action_basis"],
        "security_identity": {"scheme": identity["scheme"],
                              "value": _text(identity["value"], "identity value", nullable=True)},
        "currency": _text(raw["currency"], "currency", nullable=True),
        "unit": _text(raw["unit"], "unit", nullable=True),
        "quality": {"passed": sorted(quality["passed"]), "failed": sorted(quality["failed"])},
        "content_sha256": _sha(raw["content_sha256"], "content_sha256"),
        "parents": sorted(_sha(p, "parent") for p in parents),
        "transformation": _text(raw.get("transformation"), "transformation", nullable=True),
        "notes": _text(raw.get("notes"), "notes", nullable=True),
    }
    if record["parents"] and record["transformation"] is None:
        raise ValueError("derived records must name their transformation")
    if raw.get("version", VERSION) != VERSION:
        raise ValueError("unsupported lineage record version")
    record["lineage_sha256"] = _canonical_hash(record)
    if raw.get("lineage_sha256", record["lineage_sha256"]) != record["lineage_sha256"]:
        raise ValueError("lineage_sha256 does not match the record content")
    return record


def _reasons(record: Mapping[str, Any], cutoff: Any, purpose: str) -> tuple[list[str], list[str]]:
    """Return (ineligible, unknown) reason codes for one record."""
    bad: list[str] = []
    unknown: list[str] = []
    if aware_timestamp(record["ingestion_time"]) > cutoff or aware_timestamp(record["acquisition_time"]) > cutoff:
        bad.append("NOT_YET_ACQUIRED_AT_CUTOFF")
    if aware_timestamp(record["observation_start"]) > cutoff:
        bad.append("OBSERVATIONS_AFTER_CUTOFF")
    if record["license"]["research_use"] is False:
        bad.append("LICENSE_PROHIBITS_RESEARCH")
    elif record["license"]["research_use"] is None:
        unknown.append("LICENSE_RESEARCH_USE_UNKNOWN")
    if record["access_permission"] == "UNVERIFIED":
        unknown.append("ACCESS_PERMISSION_UNVERIFIED")
    if record["quality"]["failed"]:
        bad.append("QUALITY_CHECKS_FAILED")
    if purpose == "archive_inspection":
        return bad, unknown
    if record["publication_time"] is None:
        unknown.append("PUBLICATION_TIME_UNKNOWN")
    elif aware_timestamp(record["publication_time"]) > cutoff:
        bad.append("PUBLISHED_AFTER_CUTOFF")
    if record["historical_availability"]["state"] != "VERIFIED":
        unknown.append("HISTORICAL_AVAILABILITY_UNVERIFIED")
    if record["revision_history"] is None:
        unknown.append("REVISION_HISTORY_UNKNOWN")
    elif any(aware_timestamp(r["available_at"]) > cutoff for r in record["revision_history"][:1]):
        bad.append("FIRST_VINTAGE_AFTER_CUTOFF")
    if record["corporate_action_basis"] == "UNKNOWN":
        unknown.append("CORPORATE_ACTION_BASIS_UNKNOWN")
    if record["security_identity"]["scheme"] not in PERMANENT_IDENTITY_SCHEMES:
        unknown.append("PERMANENT_SECURITY_IDENTITY_UNAVAILABLE")
    if record["currency"] is None:
        unknown.append("CURRENCY_UNKNOWN")
    if record["unit"] is None:
        unknown.append("UNIT_UNKNOWN")
    if purpose == "funded_allocation":
        if record["classification"] == "SYNTHETIC":
            bad.append("SYNTHETIC_NOT_FUNDABLE")
        if record["access_permission"] != "VERIFIED_PUBLIC_TERMS":
            unknown.append("ACCESS_PERMISSION_NOT_VERIFIED_FOR_FUNDED_USE")
    return bad, unknown


def assess_eligibility(records: Sequence[Mapping[str, Any]], *, record_id: str, as_of: Any,
                       purpose: str) -> dict[str, Any]:
    """Evaluate one record (and its full ancestry) for a purpose at a cutoff."""
    if purpose not in PURPOSES:
        raise ValueError("unsupported eligibility purpose")
    if not isinstance(records, Sequence) or isinstance(records, (str, bytes)) or len(records) > 256:
        raise ValueError("records must be a bounded list")
    cutoff = aware_timestamp(as_of)
    canonical = [lineage_record(r) for r in records]
    by_hash = {r["lineage_sha256"]: r for r in canonical}
    by_id: dict[str, dict[str, Any]] = {}
    for record in canonical:
        if record["record_id"] in by_id:
            raise ValueError("duplicate lineage record_id")
        by_id[record["record_id"]] = record
    if record_id not in by_id:
        raise ValueError("unknown record_id")
    visited: dict[str, dict[str, Any]] = {}

    def visit(record: dict[str, Any], path: tuple[str, ...]) -> str:
        key = record["lineage_sha256"]
        if key in path:
            raise ValueError("lineage cycle detected")
        if key in visited:
            return visited[key]["eligibility"]
        bad, unknown = _reasons(record, cutoff, purpose)
        missing = [p for p in record["parents"] if p not in by_hash]
        if missing:
            unknown.append("PARENT_LINEAGE_MISSING")
        parent_states = [visit(by_hash[p], (*path, key)) for p in record["parents"] if p in by_hash]
        own = "INELIGIBLE" if bad else "UNKNOWN" if unknown else "ELIGIBLE"
        state = min([own, *parent_states], key=_ORDER.__getitem__)
        if state != own:
            (unknown if state == "UNKNOWN" else bad).append("INHERITED_FROM_PARENT")
        visited[key] = {"record_id": record["record_id"], "lineage_sha256": key,
                        "eligibility": state, "ineligible_reasons": sorted(set(bad)),
                        "unknown_reasons": sorted(set(unknown))}
        return state

    root = by_id[record_id]
    state = visit(root, ())
    return {"version": VERSION, "purpose": purpose, "as_of": cutoff.isoformat(),
            "record_id": record_id, "lineage_sha256": root["lineage_sha256"],
            "eligibility": state, "point_in_time_claimed": state == "ELIGIBLE" and purpose != "archive_inspection",
            "records": sorted(visited.values(), key=lambda r: r["record_id"]),
            "policy": "Missing clocks, identity, availability or revision evidence yields UNKNOWN; "
                      "UNKNOWN is never promoted to ELIGIBLE."}


def documented_sources() -> list[dict[str, Any]]:
    """Lineage records for sources whose terms and gaps are documented in this repository.

    Values restate `docs/observed-data.md` and `docs/observed-equity-data.md`;
    nothing here is fetched. Unknown clocks/identities remain null/UNKNOWN.
    """
    zero = "0" * 64
    return [
        lineage_record({
            "record_id": "ECB_REFERENCE_FX:2025", "source_identity": "European Central Bank",
            "dataset": "Euro foreign exchange reference rates (eurofxref-hist.zip)",
            "dataset_version": "acquired 2026-11-03", "classification": "HISTORICAL",
            "source_url": "https://www.ecb.europa.eu/stats/eurofxref/eurofxref-hist.zip",
            "license": {"identifier": "ECB disclaimer: reproduction with citation",
                        "url": "https://www.ecb.europa.eu/services/using-our-site/disclaimer/html/index.en.html",
                        "research_use": True, "commercial_use": None,
                        "attribution": "Source: European Central Bank"},
            "access_permission": "VERIFIED_PUBLIC_TERMS",
            "observation_start": "2025-01-02T00:00:00+00:00", "observation_end": "2025-12-31T00:00:00+00:00",
            "publication_time": None, "acquisition_time": "2026-11-03T00:00:00+00:00",
            "ingestion_time": "2026-11-03T00:00:00+00:00",
            "historical_availability": {"state": "UNKNOWN",
                                        "evidence": "Source dates known; intraday publication and historical revision clocks unknown."},
            "revision_history": None, "corporate_action_basis": "NOT_APPLICABLE",
            "security_identity": {"scheme": "ECB_SERIES_KEY", "value": "ECB:EXR:<CCY>:PER_EUR"},
            "currency": "quote currency per EUR", "unit": "reference rate",
            "quality": {"passed": ["bounded-archive", "no-invented-ohlc"], "failed": []},
            "content_sha256": zero,
            "notes": "Information-purpose reference rates; not executable prices. Exact acquisition hash is in the saved verification record.",
        }),
        lineage_record({
            "record_id": "UCI_DOW_JONES_WEEKLY_2011", "source_identity": "UCI Machine Learning Repository",
            "dataset": "Dow Jones Index (weekly)", "dataset_version": "DOI 10.24432/C5788V",
            "classification": "HISTORICAL",
            "source_url": "https://archive.ics.uci.edu/dataset/312/dow+jones+index",
            "license": {"identifier": "CC-BY-4.0", "url": "https://creativecommons.org/licenses/by/4.0/",
                        "research_use": True, "commercial_use": True,
                        "attribution": "Michael Brown, Dow Jones Index [Dataset], UCI Machine Learning Repository, 2013"},
            "access_permission": "VERIFIED_PUBLIC_TERMS",
            "observation_start": "2011-01-07T00:00:00+00:00", "observation_end": "2011-06-24T00:00:00+00:00",
            "publication_time": None, "acquisition_time": "2026-11-03T00:00:00+00:00",
            "ingestion_time": "2026-11-03T00:00:00+00:00",
            "historical_availability": {"state": "UNKNOWN",
                                        "evidence": "Archive published 2013; weekly vintage clocks are not supplied."},
            "revision_history": None, "corporate_action_basis": "UNKNOWN",
            "security_identity": {"scheme": "SOURCE_LABEL", "value": "DJIA_MEMBER_LABEL:<label>"},
            "currency": None, "unit": "source-reported dollar-sign price",
            "quality": {"passed": ["future-label-columns-quarantined"], "failed": []},
            "content_sha256": "08ad8e30915f071b46f6f5bef2b1ccdfe70aacfa31aee116117048ad2adec0bf",
        }),
    ]
