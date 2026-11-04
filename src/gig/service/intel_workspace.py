"""HTTP-facing Global Intelligence workspace built on the intel store.

Reads never contact providers. ``refresh`` is the only provider path and runs
only on an explicit request for a registry source. Demonstration fixtures are
authored, labeled DEMONSTRATION and never presented as observations.
"""
from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from gig.intel import defense, research
from gig.intel.events import _finish
from gig.intel.fetch import Opener, refresh
from gig.intel.reference import CHOKEPOINTS, PORTS, THEMES, research_links
from gig.intel.sources import FETCHABLE, SOURCES, registry
from gig.intel.store import IntelStore
from gig.service.contracts import Contract, State
from gig.timestamps import aware_timestamp

LAYERS = ("hazards", "geopolitical", "defense", "maritime", "aviation", "imagery", "reference")


def _envelope(data: dict[str, Any], *, scope: str, classification: str = "SNAPSHOT",
              source: str = "gig-global-intelligence", as_of: str | None = None) -> Contract:
    return Contract.admit(State.AVAILABLE, data, source=source, classification=classification,
                          as_of=as_of, scope=scope, evidence_state="RESEARCH_OUTPUT", readiness="RESEARCH_ONLY")


def _demonstration_events(now: datetime) -> list[dict[str, Any]]:
    """Authored, clearly fictional hazard records used when no provider data is admitted."""
    base = now - timedelta(days=2)
    rows = [("DEMO-1", "EARTHQUAKE", "Authored demonstration: M5.8 offshore event", 35.5, 140.5, 5.8, "mww"),
            ("DEMO-2", "SEVERE_STORM", "Authored demonstration: tropical storm near Gulf coast", 27.5, -91.0, 55, "kts"),
            ("DEMO-3", "WILDFIRE", "Authored demonstration: wildfire complex", 38.8, -121.2, None, None),
            ("DEMO-4", "VOLCANO", "Authored demonstration: volcanic ash advisory", 63.6, -19.6, None, None),
            ("DEMO-5", "FLOOD", "Authored demonstration: river flooding", 23.7, 90.4, None, None),
            ("DEMO-6", "SEVERE_STORM", "Authored demonstration: cyclone approaching Strait of Hormuz", 25.9, 57.5, 65, "kts")]
    out = []
    for i, (identity, category, title, lat, lon, magnitude, unit) in enumerate(rows):
        observed = base + timedelta(hours=6 * i)
        out.append(_finish({
            "provider": "DEMONSTRATION", "provider_name": "GIG authored demonstration fixture",
            "dataset": "Authored fictional hazard events", "license": "Authored by GIG", "attribution": None,
            "acquired_at": (observed + timedelta(hours=1)).isoformat(), "event_id": f"DEMO:{identity}",
            "provider_event_id": identity, "category": category, "layer": "hazards", "title": title,
            "latitude": lat, "longitude": lon, "depth_km": None, "geo_precision_km": None,
            "geo_precision_note": "Fictional location.", "observed_at": observed.isoformat(),
            "published_at": None, "magnitude": magnitude, "magnitude_unit": unit,
            "verification": "DEMONSTRATION_NOT_OBSERVED", "citations": []}))
    return out


class IntelWorkspace:
    def __init__(self, directory: Path, *, opener: Opener | None = None):
        self.store = IntelStore(directory)
        self.opener = opener

    def overview(self, *, as_of: datetime, layers: set[str] | None, categories: set[str] | None,
                 days: int, query: str | None) -> Contract:
        now = datetime.now(UTC)
        since = as_of - timedelta(days=days)
        events = self.store.events(as_of=as_of, layers=layers, categories=categories, since=since, query=query)
        for event in events:
            event.setdefault("research_links", research_links(event["category"], event.get("latitude"), event.get("longitude")))
        health = [self.store.semantics(s, now=now, as_of=as_of) for s in FETCHABLE]
        admitted = [e for e in events if e["provider"] == "ADMITTED_REPORTS"]
        demo = [e for e in events if e["provider"] == "DEMONSTRATION"]
        semantics = {h["source_id"]: h["semantics"] for h in health}
        if demo:
            semantics["DEMONSTRATION"] = "DEMONSTRATION"
        if admitted:
            kinds = {e.get("data_classification") for e in admitted}
            semantics["ADMITTED_REPORTS"] = ("HISTORICAL" if kinds == {"HISTORICAL"} else "DEMONSTRATION"
                                             if kinds == {"SYNTHETIC"} else "HISTORICAL_AND_DEMONSTRATION")
        data = {
            "title": "Global Intelligence", "as_of": as_of.isoformat(), "window_days": days,
            "events": events, "event_count": len(events),
            "counts_by_provider": {p: sum(e["provider"] == p for e in events) for p in sorted({e["provider"] for e in events})},
            "source_health": health, "data_semantics": semantics, "sources": registry(),
            "reference": {"chokepoints": list(CHOKEPOINTS), "ports": list(PORTS),
                          "note": "Approximate locations; no traffic volumes asserted."},
            "themes": THEMES, "risk_monitor": research.risk_monitor(events),
            "defense_aggregate": defense.aggregate(admitted),
            "layers": [{"layer": layer, "available": layer in {"hazards", "geopolitical", "defense", "reference"},
                        "blocker": next((s.get("blocker") for s in SOURCES.values()
                                         if s.get("layer") == layer and s.get("blocker")), None)}
                       for layer in LAYERS],
            "investigations": self.store.investigations(),
            "limitations": ["Coverage is limited to integrated sources and admitted reports; absence of an event is not evidence of absence.",
                            "No vessel, aircraft or military-unit positions are shown or simulated.",
                            "Theme links are research hypotheses, not causal or predictive claims."],
        }
        return _envelope(data, scope="intel.overview", as_of=as_of.isoformat())

    def refresh(self, payload: dict[str, Any]) -> Contract:
        if set(payload) != {"source_id"} or payload["source_id"] not in FETCHABLE:
            raise ValueError("refresh requires one integrated source_id")
        result = refresh(self.store, payload["source_id"], opener=self.opener)
        # A deferred or failed attempt carries no observations; the envelope
        # describes the request outcome only.
        return _envelope(result, scope="intel.refresh", source=payload["source_id"],
                         classification="DELAYED" if result["state"] == "ACQUIRED" else "SNAPSHOT")

    def admit(self, payload: dict[str, Any]) -> Contract:
        if set(payload) != {"records"} or not isinstance(payload["records"], list) or not 1 <= len(payload["records"]) <= 200:
            raise ValueError("admit requires 1 to 200 records")
        records = [defense.admit_record(r) for r in payload["records"]]
        if len({r["event_id"] for r in records}) != len(records):
            raise ValueError("duplicate record_id in request")
        added = self.store.append(records, classification="ADMITTED_REPORT", ingested_at=datetime.now(UTC).isoformat())
        return _envelope({"admitted": len(records), "new_revisions": added,
                          "event_ids": [r["event_id"] for r in records]}, scope="intel.admit", classification="HISTORICAL")

    def demonstration(self, payload: dict[str, Any]) -> Contract:
        if payload:
            raise ValueError("demonstration admission takes no parameters")
        records = _demonstration_events(datetime.now(UTC))
        added = self.store.append(records, classification="DEMONSTRATION", ingested_at=datetime.now(UTC).isoformat())
        return _envelope({"admitted": len(records), "new_revisions": added}, scope="intel.demonstration",
                         classification="SYNTHETIC")

    def event_study(self, payload: dict[str, Any]) -> Contract:
        result = research.event_study(payload)
        return _envelope(result, scope="intel.event-study", classification="MODEL_ESTIMATE"
                         if payload["classification"] == "HISTORICAL" else "SYNTHETIC")

    def exposure(self, payload: dict[str, Any]) -> Contract:
        return _envelope(research.theme_exposure(payload), scope="intel.exposure", classification="MODEL_ESTIMATE")

    def investigate(self, payload: dict[str, Any]) -> Contract:
        if set(payload) != {"name", "body"} or not isinstance(payload["name"], str) or not 1 <= len(payload["name"]) <= 120 \
                or not isinstance(payload["body"], dict):
            raise ValueError("investigation requires name and body")
        body = payload["body"]
        allowed = {"event_ids", "layers", "categories", "as_of", "notes", "themes"}
        if set(body) - allowed:
            raise ValueError("unsupported investigation fields")
        if "as_of" in body:
            body = {**body, "as_of": aware_timestamp(body["as_of"]).isoformat()}
        return _envelope(self.store.save_investigation(payload["name"], body), scope="intel.investigation")
