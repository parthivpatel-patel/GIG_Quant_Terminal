"""Geopolitical and defense research records for financial-risk awareness.

Scope boundary: these are cited, delayed public announcements and aggregated
history. The schema deliberately has no unit identifiers, hull/tail numbers,
callsigns, headings, speeds or tracks, and rejects any extra field. Locations are
coarsened to a 1-degree grid; publication must be at least 24 hours old at
admission. An exercise or deployment is never treated as evidence of conflict.
"""
from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

from gig.intel.events import canonical_hash
from gig.intel.reference import research_links
from gig.timestamps import aware_timestamp

BRANCHES = ("ARMY", "AIR_FORCE", "NAVY", "GEOPOLITICAL")
KINDS = {
    "ARMY": {"EXERCISE_ANNOUNCEMENT", "DEPLOYMENT_ANNOUNCEMENT", "MOBILIZATION_ANNOUNCEMENT", "HISTORICAL_AGGREGATE"},
    "AIR_FORCE": {"EXERCISE_ANNOUNCEMENT", "AIRSPACE_RESTRICTION_NOTICE", "HISTORICAL_AGGREGATE", "CIVIL_AVIATION_DISRUPTION"},
    "NAVY": {"EXERCISE_ANNOUNCEMENT", "DEPLOYMENT_ANNOUNCEMENT", "MARITIME_SECURITY_NOTICE", "HISTORICAL_AGGREGATE", "SHIPPING_DISRUPTION"},
    "GEOPOLITICAL": {"DIPLOMATIC_EVENT", "SANCTIONS_ANNOUNCEMENT", "CONFLICT_REPORT", "TRADE_DISRUPTION"},
}
VERIFICATION = ("CONFIRMED_OFFICIAL", "MULTI_SOURCE_REPORTED", "UNVERIFIED_REPORT", "HISTORICAL_OBSERVATION", "ANALYTICAL_SCENARIO")
PUBLISHER_TYPES = ("OFFICIAL_GOVERNMENT", "INTERGOVERNMENTAL", "NEWS_MEDIA", "RESEARCH_INSTITUTION", "OTHER")
CLASSIFICATIONS = ("HISTORICAL", "SYNTHETIC")
MIN_DELAY_SECONDS = 24 * 3600
_FIELDS = {"record_id", "branch", "kind", "title", "summary", "region", "country_codes", "latitude", "longitude",
           "published_at", "observed_start", "observed_end", "acquired_at", "sources", "verification",
           "classification"}
_OPTIONAL = {"observed_start", "observed_end", "latitude", "longitude"}


def _text(value: Any, name: str, limit: int) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > limit:
        raise ValueError(f"{name} must be a nonempty string of at most {limit} characters")
    return value.strip()


def admit_record(raw: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(raw, Mapping) or not set(raw) >= (_FIELDS - _OPTIONAL) or set(raw) - _FIELDS:
        raise ValueError("defense research record fields do not match the contract")
    branch, kind = raw["branch"], raw["kind"]
    if branch not in BRANCHES or kind not in KINDS[branch]:
        raise ValueError("unsupported branch or record kind")
    if raw["verification"] not in VERIFICATION or raw["classification"] not in CLASSIFICATIONS:
        raise ValueError("unsupported verification or classification")
    sources = raw["sources"]
    if not isinstance(sources, list) or not 1 <= len(sources) <= 10:
        raise ValueError("at least one cited source is required")
    cited = []
    for source in sources:
        if not isinstance(source, Mapping) or set(source) != {"publisher", "publisher_type", "url"}:
            raise ValueError("sources require publisher, publisher_type and url")
        if source["publisher_type"] not in PUBLISHER_TYPES:
            raise ValueError("unsupported publisher type")
        url = _text(source["url"], "source url", 500)
        if not url.startswith("https://"):
            raise ValueError("source url must be https")
        cited.append({"publisher": _text(source["publisher"], "publisher", 200),
                      "publisher_type": source["publisher_type"], "url": url})
    publishers = {s["publisher"].lower() for s in cited}
    if raw["verification"] == "CONFIRMED_OFFICIAL" and not any(
            s["publisher_type"] in ("OFFICIAL_GOVERNMENT", "INTERGOVERNMENTAL") for s in cited):
        raise ValueError("CONFIRMED_OFFICIAL requires an official or intergovernmental source")
    if raw["verification"] == "MULTI_SOURCE_REPORTED" and len(publishers) < 2:
        raise ValueError("MULTI_SOURCE_REPORTED requires at least two independent publishers")
    published = aware_timestamp(raw["published_at"])
    acquired = aware_timestamp(raw["acquired_at"])
    if (acquired - published).total_seconds() < MIN_DELAY_SECONDS:
        raise ValueError("defense records must be at least 24 hours old at admission; no real-time tracking")
    start = aware_timestamp(raw["observed_start"]) if raw.get("observed_start") else None
    end = aware_timestamp(raw["observed_end"]) if raw.get("observed_end") else None
    if start and end and end < start:
        raise ValueError("observation window is inverted")
    if start and start > acquired:
        raise ValueError("observation cannot follow acquisition")
    codes = raw["country_codes"]
    if not isinstance(codes, list) or len(codes) > 20 or not all(isinstance(c, str) and len(c) == 2 and c.isalpha() and c.isupper() for c in codes):
        raise ValueError("country_codes must be ISO 3166-1 alpha-2 codes")
    lat = lon = None
    if raw.get("latitude") is not None or raw.get("longitude") is not None:
        values = (raw.get("latitude"), raw.get("longitude"))
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in values):
            raise ValueError("coordinates must be finite numbers")
        if not (-90 <= values[0] <= 90 and -180 <= values[1] <= 180):  # type: ignore[operator]
            raise ValueError("coordinates out of range")
        lat, lon = float(round(values[0])), float(round(values[1]))  # type: ignore[arg-type]
    observed = start or published
    record = {
        "event_id": f"ADMITTED:{_text(raw['record_id'], 'record_id', 128)}",
        "provider_event_id": raw["record_id"], "provider": "ADMITTED_REPORTS",
        "provider_name": "Analyst-admitted public report", "dataset": "Geopolitical and defense research records",
        "license": "Governed by each cited publisher", "attribution": "; ".join(s["publisher"] for s in cited),
        "category": branch, "kind": kind, "layer": "defense" if branch != "GEOPOLITICAL" else "geopolitical",
        "title": _text(raw["title"], "title", 300), "summary": _text(raw["summary"], "summary", 2000),
        "region": _text(raw["region"], "region", 120), "country_codes": sorted(set(codes)),
        "latitude": lat, "longitude": lon,
        "geo_precision_km": None if lat is None else 111.0,
        "geo_precision_note": "Coarsened to a 1-degree grid (about 111 km); region-level research only.",
        "observed_at": observed.isoformat(), "observed_end": None if end is None else end.isoformat(),
        "observation_time_known": start is not None,
        "published_at": published.isoformat(), "acquired_at": acquired.isoformat(),
        "verification": raw["verification"], "data_classification": raw["classification"],
        "citations": cited,
        "interpretation": "NOT_EVIDENCE_OF_IMPENDING_CONFLICT",
        "research_links": research_links(branch, lat, lon),
    }
    record["reporting_delay_seconds"] = (acquired - observed).total_seconds()
    record["revision_sha256"] = canonical_hash({k: v for k, v in record.items() if k not in ("acquired_at", "reporting_delay_seconds")})
    return record


def aggregate(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Weekly counts by branch and region; no positions, tracks or forecasts."""
    weekly: Counter[tuple[str, str, str]] = Counter()
    verification: Counter[tuple[str, str]] = Counter()
    for record in records:
        if record.get("provider") != "ADMITTED_REPORTS":
            continue
        iso = aware_timestamp(record["observed_at"]).isocalendar()
        weekly[(record["category"], record["region"], f"{iso[0]}-W{iso[1]:02d}")] += 1
        verification[(record["category"], record["verification"])] += 1
    return {"weekly": [{"branch": b, "region": r, "iso_week": w, "count": c} for (b, r, w), c in sorted(weekly.items())],
            "by_verification": [{"branch": b, "verification": v, "count": c} for (b, v), c in sorted(verification.items())],
            "policy": "Aggregated, delayed public information for financial-risk research; no tactical tracking or forecasts."}
