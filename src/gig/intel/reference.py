"""Approximate reference geometry and documented research themes.

Coordinates are approximate public geographic locations (roughly 0.5 degrees).
No flow volumes, vessel counts or traffic shares are asserted here. Theme links
are research hypotheses for investigation, never causal claims.
"""
from __future__ import annotations

import math
from typing import Any

CHOKEPOINTS: tuple[dict[str, Any], ...] = (
    {"id": "hormuz", "name": "Strait of Hormuz", "lat": 26.6, "lon": 56.3, "themes": ["crude_oil", "lng", "tanker_shipping"]},
    {"id": "malacca", "name": "Strait of Malacca", "lat": 2.5, "lon": 101.3, "themes": ["crude_oil", "container_shipping", "lng"]},
    {"id": "suez", "name": "Suez Canal", "lat": 30.6, "lon": 32.3, "themes": ["crude_oil", "container_shipping", "tanker_shipping"]},
    {"id": "bab-el-mandeb", "name": "Bab el-Mandeb", "lat": 12.6, "lon": 43.4, "themes": ["crude_oil", "container_shipping", "tanker_shipping"]},
    {"id": "bosporus", "name": "Turkish Straits (Bosporus)", "lat": 41.1, "lon": 29.1, "themes": ["crude_oil", "grain", "tanker_shipping"]},
    {"id": "panama", "name": "Panama Canal", "lat": 9.1, "lon": -79.7, "themes": ["container_shipping", "lng", "grain"]},
    {"id": "gibraltar", "name": "Strait of Gibraltar", "lat": 35.95, "lon": -5.6, "themes": ["container_shipping", "tanker_shipping"]},
    {"id": "danish-straits", "name": "Danish Straits", "lat": 55.6, "lon": 11.0, "themes": ["crude_oil", "tanker_shipping"]},
    {"id": "good-hope", "name": "Cape of Good Hope", "lat": -34.4, "lon": 18.5, "themes": ["container_shipping", "tanker_shipping"]},
)

PORTS: tuple[dict[str, Any], ...] = (
    {"id": "shanghai", "name": "Shanghai", "lat": 31.2, "lon": 121.5},
    {"id": "singapore", "name": "Singapore", "lat": 1.26, "lon": 103.8},
    {"id": "rotterdam", "name": "Rotterdam", "lat": 51.95, "lon": 4.1},
    {"id": "busan", "name": "Busan", "lat": 35.1, "lon": 129.0},
    {"id": "jebel-ali", "name": "Jebel Ali", "lat": 25.0, "lon": 55.06},
    {"id": "los-angeles", "name": "Los Angeles / Long Beach", "lat": 33.74, "lon": -118.25},
    {"id": "houston", "name": "Houston", "lat": 29.73, "lon": -95.27},
    {"id": "ras-tanura", "name": "Ras Tanura", "lat": 26.64, "lon": 50.16},
    {"id": "santos", "name": "Santos", "lat": -23.98, "lon": -46.3},
    {"id": "antwerp", "name": "Antwerp-Bruges", "lat": 51.27, "lon": 4.33},
)

THEMES: dict[str, dict[str, Any]] = {
    "crude_oil": {"label": "Crude oil", "markets": ["Brent/WTI futures (research)", "Integrated oil equities", "Refiners"]},
    "lng": {"label": "Natural gas / LNG", "markets": ["Henry Hub / TTF (research)", "LNG shippers", "Gas utilities"]},
    "tanker_shipping": {"label": "Tanker shipping", "markets": ["Tanker equities", "Marine insurers"]},
    "container_shipping": {"label": "Container shipping & logistics", "markets": ["Container lines", "Freight forwarders", "Retail importers"]},
    "grain": {"label": "Grains", "markets": ["Wheat/corn futures (research)", "Agribusiness equities"]},
    "aviation": {"label": "Aviation & travel", "markets": ["Airlines", "Air freight", "Travel equities", "Jet fuel"]},
    "defense": {"label": "Defense sector", "markets": ["Defense equities"]},
    "insurance": {"label": "Insurance & reinsurance", "markets": ["P&C insurers", "Reinsurers", "Cat bonds"]},
    "utilities": {"label": "Utilities & power", "markets": ["Regional utilities", "Power prices"]},
    "regional_fx": {"label": "Regional currency & sovereign risk", "markets": ["Regional FX", "Sovereign spreads"]},
    "regional_equity": {"label": "Regional equities", "markets": ["Country equity indices"]},
    "metals": {"label": "Industrial metals", "markets": ["Copper/aluminium (research)", "Miners"]},
}

CATEGORY_THEMES: dict[str, list[str]] = {
    "EARTHQUAKE": ["regional_equity", "insurance", "regional_fx"],
    "WILDFIRE": ["utilities", "insurance"],
    "SEVERE_STORM": ["crude_oil", "lng", "insurance", "container_shipping"],
    "VOLCANO": ["aviation", "regional_equity"],
    "FLOOD": ["grain", "insurance", "container_shipping"],
    "SEA_LAKE_ICE": ["tanker_shipping"],
    "DROUGHT": ["grain", "utilities"],
    "LANDSLIDE": ["regional_equity"],
    "OTHER_HAZARD": ["regional_equity"],
    "ARMY": ["defense", "regional_fx", "regional_equity"],
    "AIR_FORCE": ["defense", "aviation"],
    "NAVY": ["defense", "tanker_shipping", "container_shipping", "crude_oil"],
    "GEOPOLITICAL": ["regional_fx", "regional_equity", "crude_oil"],
}


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * 6371.0088 * math.asin(min(1.0, math.sqrt(a)))


def research_links(category: str, lat: float | None, lon: float | None, *, radius_km: float = 500) -> dict[str, Any]:
    """Themes worth investigating for an event; explicitly not causal."""
    themes = list(CATEGORY_THEMES.get(category, ["regional_equity"]))
    nearby = []
    if lat is not None and lon is not None:
        for point in CHOKEPOINTS:
            distance = haversine_km(lat, lon, point["lat"], point["lon"])
            if distance <= radius_km:
                nearby.append({"chokepoint": point["name"], "distance_km": round(distance, 1)})
                themes.extend(point["themes"])
    ordered = list(dict.fromkeys(themes))
    return {"themes": [{"theme": t, **THEMES[t]} for t in ordered], "nearby_chokepoints": nearby,
            "radius_km": radius_km, "link_type": "RESEARCH_HYPOTHESIS_NOT_CAUSAL"}
