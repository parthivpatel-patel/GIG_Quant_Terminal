"""Public-source registry for GIG Global Intelligence.

Each entry states what is verified about a provider: fixed endpoint, license,
permission state, refresh bounds and data semantics. ``INTEGRATED`` means an
adapter exists and was exercised; it does not imply completeness or LIVE status.
Candidates that were investigated but not integrated keep their exact blocker.
"""
from __future__ import annotations

from typing import Any

INTEGRATED, CANDIDATE, BLOCKED, REFERENCE = "INTEGRATED", "CANDIDATE", "BLOCKED", "REFERENCE"

SOURCES: dict[str, dict[str, Any]] = {
    "USGS_EARTHQUAKES": {
        "status": INTEGRATED, "layer": "hazards", "provider": "U.S. Geological Survey",
        "dataset": "Earthquake Hazards Program GeoJSON summary feed, M2.5+ past 7 days",
        "url": "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/2.5_week.geojson",
        "documentation": "https://earthquake.usgs.gov/earthquakes/feed/v1.0/geojson.php",
        "license": "U.S. Government public domain (USGS information policy)",
        "license_url": "https://www.usgs.gov/information-policies-and-instructions/copyrights-and-credits",
        "commercial_use": True, "attribution": "U.S. Geological Survey",
        "min_refresh_seconds": 300, "stale_after_seconds": 3600, "max_bytes": 8_000_000,
        "semantics": "DELAYED",
        "semantics_note": "Events appear after detection; automatic solutions are revised. Not LIVE.",
        "coverage": "Global M2.5+ with regional completeness variation; summary omits location error.",
    },
    "NASA_EONET": {
        "status": INTEGRATED, "layer": "hazards", "provider": "NASA Earth Observatory Natural Event Tracker",
        "dataset": "EONET v3 open events, past 30 days",
        "url": "https://eonet.gsfc.nasa.gov/api/v3/events?status=open&days=30&limit=500",
        "documentation": "https://eonet.gsfc.nasa.gov/docs/v3",
        "license": "NASA data: no copyright restriction; upstream event sources keep their own terms",
        "license_url": "https://www.nasa.gov/nasa-brand-center/images-and-media/",
        "commercial_use": None, "attribution": "NASA EONET and the cited upstream sources",
        "min_refresh_seconds": 900, "stale_after_seconds": 6 * 3600, "max_bytes": 8_000_000,
        "semantics": "DELAYED",
        "semantics_note": "Curated metadata, typically hours to days after onset; no publication clock.",
        "coverage": "Curated selection of wildfires, storms, volcanoes, ice, floods; not exhaustive.",
    },
    "GDACS": {
        "status": CANDIDATE, "layer": "hazards", "provider": "Global Disaster Alert and Coordination System",
        "dataset": "GDACS event list API", "url": "https://www.gdacs.org/gdacsapi/api/events/geteventlist/SEARCH",
        "license": "Terms not yet reviewed in this repository", "commercial_use": None,
        "blocker": "Endpoint reachable (HTTP 200, 2026-11-04); reuse terms not yet verified.",
    },
    "NASA_FIRMS": {
        "status": BLOCKED, "layer": "hazards", "provider": "NASA FIRMS",
        "dataset": "Active fire / thermal anomaly detections",
        "blocker": "API requires a free MAP_KEY credential; none is configured. Thermal detections are not evidence of military activity.",
    },
    "AIS_MARITIME": {
        "status": BLOCKED, "layer": "maritime", "provider": "No permitted global AIS source configured",
        "dataset": "Commercial vessel positions",
        "blocker": "Free AIS streams require registration/API keys and restrict redistribution; terrestrial coverage is partial. No vessel positions are shown or simulated.",
    },
    "OPENSKY": {
        "status": BLOCKED, "layer": "aviation", "provider": "OpenSky Network",
        "dataset": "Civil aircraft state vectors",
        "blocker": "Connection timed out from this environment (2026-11-04); terms restrict use to non-commercial research. No aircraft positions are shown or simulated.",
    },
    "SENTINEL_LANDSAT": {
        "status": BLOCKED, "layer": "imagery", "provider": "Copernicus Sentinel / USGS Landsat",
        "dataset": "Periodic satellite imagery scenes",
        "blocker": "Scene search/download requires registered accounts (Copernicus Data Space, USGS EarthExplorer); the terminal does not load external tiles. Imagery is periodic, never live video.",
    },
    "GIG_REFERENCE_GEOMETRY": {
        "status": REFERENCE, "layer": "reference", "provider": "GIG authored reference",
        "dataset": "Approximate chokepoint and port locations",
        "license": "Authored by GIG; locations are approximate public geographic facts",
        "commercial_use": True, "semantics": "HISTORICAL",
        "coverage": "Selected locations only; no traffic volumes or vessel counts are asserted.",
    },
    "ADMITTED_REPORTS": {
        "status": INTEGRATED, "layer": "geopolitical", "provider": "Caller-admitted public reports",
        "dataset": "Cited geopolitical and defense announcements",
        "license": "Each record cites its publisher; reuse governed by that publisher",
        "commercial_use": None, "semantics": "HISTORICAL",
        "coverage": "Only what an analyst admits with citations; minimum 24-hour reporting delay enforced.",
    },
}

FETCHABLE = ("USGS_EARTHQUAKES", "NASA_EONET")


def registry() -> list[dict[str, Any]]:
    return [{"source_id": key, **value} for key, value in SOURCES.items()]
