"""Normalize provider payloads into attributed, clock-separated event records.

Normalizers are pure: they never fetch, never fill missing clocks, and never
invent coordinates. Unknown publication time or location error stays null.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any

from gig.intel.sources import SOURCES
from gig.timestamps import aware_timestamp

MAX_EVENTS = 5000
EONET_CATEGORIES = {
    "earthquakes": "EARTHQUAKE", "wildfires": "WILDFIRE", "severeStorms": "SEVERE_STORM",
    "volcanoes": "VOLCANO", "floods": "FLOOD", "seaLakeIce": "SEA_LAKE_ICE", "drought": "DROUGHT",
    "landslides": "LANDSLIDE",
}


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False).encode()).hexdigest()


def _iso_ms(value: Any) -> str:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 32503680000000:
        raise ValueError("epoch milliseconds out of range")
    return datetime.fromtimestamp(value / 1000, tz=UTC).isoformat()


def _coordinate(lat: Any, lon: Any) -> tuple[float, float]:
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in (lat, lon)):
        raise ValueError("coordinates must be finite numbers")
    if not (-90 <= lat <= 90 and -180 <= lon <= 180):
        raise ValueError("coordinates out of range")
    return float(lat), float(lon)


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return None
    return float(value)


def _text(value: Any, limit: int = 300) -> str:
    if not isinstance(value, str):
        return ""
    return value.strip()[:limit]


def _base(source_id: str, acquired_at: str) -> dict[str, Any]:
    source = SOURCES[source_id]
    return {"provider": source_id, "provider_name": source["provider"], "dataset": source["dataset"],
            "license": source["license"], "attribution": source.get("attribution"),
            "acquired_at": aware_timestamp(acquired_at).isoformat()}


def _finish(record: dict[str, Any]) -> dict[str, Any]:
    observed = aware_timestamp(record["observed_at"])
    acquired = aware_timestamp(record["acquired_at"])
    if observed > acquired:
        raise ValueError("observation cannot follow acquisition")
    record["reporting_delay_seconds"] = (acquired - observed).total_seconds()
    record["revision_sha256"] = canonical_hash({k: v for k, v in record.items() if k != "acquired_at"
                                                and k != "reporting_delay_seconds"})
    return record


def normalize_usgs(payload: Mapping[str, Any], *, acquired_at: str) -> list[dict[str, Any]]:
    if not isinstance(payload, Mapping) or payload.get("type") != "FeatureCollection":
        raise ValueError("USGS payload must be a GeoJSON FeatureCollection")
    features = payload.get("features")
    if not isinstance(features, list) or len(features) > MAX_EVENTS:
        raise ValueError("USGS features must be a bounded list")
    out = []
    for feature in features:
        props, geometry = feature.get("properties") or {}, feature.get("geometry") or {}
        coordinates = geometry.get("coordinates")
        if geometry.get("type") != "Point" or not isinstance(coordinates, list) or len(coordinates) < 2:
            continue
        lon, lat = coordinates[0], coordinates[1]
        lat, lon = _coordinate(lat, lon)
        event_id = _text(feature.get("id"), 64)
        if not event_id:
            continue
        status = props.get("status")
        record = {
            **_base("USGS_EARTHQUAKES", acquired_at),
            "event_id": f"USGS:{event_id}", "provider_event_id": event_id, "category": "EARTHQUAKE",
            "layer": "hazards", "title": _text(props.get("title")), "latitude": lat, "longitude": lon,
            "depth_km": _number(coordinates[2]) if len(coordinates) > 2 else None,
            "geo_precision_km": None, "geo_precision_note": "Summary feed omits horizontal location error.",
            "observed_at": _iso_ms(props.get("time")),
            "published_at": None,
            "provider_updated_at": _iso_ms(props["updated"]) if isinstance(props.get("updated"), int) else None,
            "magnitude": _number(props.get("mag")),
            "magnitude_unit": _text(props.get("magType"), 16) or None,
            "verification": "PROVIDER_REVIEWED" if status == "reviewed" else "PROVIDER_AUTOMATIC_SUBJECT_TO_REVISION",
            "citations": [{"publisher": "U.S. Geological Survey", "url": _text(props.get("url"), 300)}],
        }
        out.append(_finish(record))
    return out


def _centroid(geometry: Mapping[str, Any]) -> tuple[float, float, float | None] | None:
    kind, coords = geometry.get("type"), geometry.get("coordinates")
    if kind == "Point" and isinstance(coords, list) and len(coords) >= 2:
        lat, lon = _coordinate(coords[1], coords[0])
        return lat, lon, None
    if kind == "Polygon" and isinstance(coords, list) and coords and isinstance(coords[0], list):
        ring = [_coordinate(p[1], p[0]) for p in coords[0] if isinstance(p, list) and len(p) >= 2]
        if not ring:
            return None
        lats, lons = [p[0] for p in ring], [p[1] for p in ring]
        lat, lon = (min(lats) + max(lats)) / 2, (min(lons) + max(lons)) / 2
        extent = max(haversine(lat, lon, la, lo) for la, lo in ring)
        return lat, lon, extent
    return None


def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    from gig.intel.reference import haversine_km
    return haversine_km(lat1, lon1, lat2, lon2)


def normalize_eonet(payload: Mapping[str, Any], *, acquired_at: str) -> list[dict[str, Any]]:
    events = payload.get("events") if isinstance(payload, Mapping) else None
    if not isinstance(events, list) or len(events) > MAX_EVENTS:
        raise ValueError("EONET events must be a bounded list")
    cutoff = aware_timestamp(acquired_at)
    out = []
    for event in events:
        geometries = [g for g in event.get("geometry") or [] if isinstance(g, Mapping) and g.get("date")]
        dated = sorted(((aware_timestamp(g["date"]), g) for g in geometries), key=lambda x: x[0])
        dated = [(t, g) for t, g in dated if t <= cutoff]
        if not dated:
            continue
        observed, geometry = dated[-1]
        located = _centroid(geometry)
        if located is None:
            continue
        lat, lon, extent = located
        categories = [c.get("id") for c in event.get("categories") or [] if isinstance(c, Mapping)]
        category = next((EONET_CATEGORIES[c] for c in categories if c in EONET_CATEGORIES), "OTHER_HAZARD")
        event_id = _text(event.get("id"), 64)
        if not event_id:
            continue
        citations = [{"publisher": "NASA EONET", "url": _text(event.get("link"), 300)}]
        citations += [{"publisher": _text(s.get("id"), 64), "url": _text(s.get("url"), 300)}
                      for s in event.get("sources") or [] if isinstance(s, Mapping)]
        record = {
            **_base("NASA_EONET", acquired_at),
            "event_id": f"EONET:{event_id}", "provider_event_id": event_id, "category": category,
            "layer": "hazards", "title": _text(event.get("title")), "latitude": lat, "longitude": lon,
            "depth_km": None, "geo_precision_km": None if extent is None else round(extent, 1),
            "geo_precision_note": "Latest dated geometry; polygon extent where supplied, otherwise unknown.",
            "observed_at": observed.isoformat(), "published_at": None,
            "magnitude": _number(geometry.get("magnitudeValue")),
            "magnitude_unit": _text(geometry.get("magnitudeUnit"), 16) or None,
            "track_points": len(dated),
            "verification": "PROVIDER_CURATED",
            "citations": citations[:10],
        }
        out.append(_finish(record))
    return out
