"""
=============================================================================
GIG — GEOPOLITICAL INTELLIGENCE ENGINE v2.0
=============================================================================
Institutional-grade real-time global situational awareness.

INTELLIGENCE ARCHITECTURE:
  ┌─────────────────────────────────────────────────────────────────────────┐
  │  LAYER 1 — KINETIC DOMAIN (Live Military & Naval Operations)            │
  ├─────────────────────────────────────────────────────────────────────────┤
  │                                                                         │
  │  ✈  ADS-B Exchange          adsbexchange.com/api                       │
  │     The ONLY global tracker with ZERO military/government filtering.    │
  │     Flightradar24, FlightAware & others suppress military at request.   │
  │     ADS-B Exchange publishes raw ADS-B/MLAT — everything included.      │
  │     Detects: C-17 Globemaster, C-130 Hercules, P-8 Poseidon,            │
  │              KC-135/KC-46 Tankers, E-8 JSTARS, RC-135 Rivet Joint,      │
  │              B-52/B-1/B-2 Bombers, E-4B Nightwatch, RQ-4/MQ-9 UAVs      │
  │     Endpoint: /v2/mil/ → returns global military fleet only             │
  │     Keys: rapidapi.com (adsbexchange-com1) or adsbexchange.com/data/    │
  │     Cost: RapidAPI $0.01/100 calls, or direct subscription              │
  │                                                                         │
  │  🚢 MarineTraffic API        services.marinetraffic.com                 │
  │     Industry-standard AIS vessel tracking — 400,000+ vessels.           │
  │     Naval filter: vessel_type=35 (Military Operations per ITU)          │
  │     Detects: Carriers (CVN/LHA), Destroyers (DDG), Cruisers (CG),       │
  │              Frigates (FFG), Amphibious assault (LPD/LHD),              │
  │              Submarines (surfaced), Fleet oilers (AOE/T-AO)             │
  │     Free fallback: AISHub community aggregator (register free)          │
  │     Cost: $50/mo (PS01) to enterprise; AISHub = free with reg.          │
  │                                                                         │
  ├─────────────────────────────────────────────────────────────────────────┤
  │  LAYER 2 — CONFLICT & INTELLIGENCE DOMAIN                               │
  ├─────────────────────────────────────────────────────────────────────────┤
  │                                                                         │
  │  ⚔  ACLED                    api.acleddata.com                          │
  │     Armed Conflict Location & Event Data Project.                       │
  │     Used by: UN OCHA, World Bank, US DoD, EU EEAS, ICRC, NATO.          │
  │     Coverage: 60+ countries. 1M+ events since 1997. Real-time.          │
  │     Types: Battles, Air strikes, Missile attacks, IEDs, Protests        │
  │     Free: register at developer.acleddata.com (250K calls/month)        │
  │                                                                         │
  │  📰 Event Registry           eventregistry.org                          │
  │     AI-powered geopolitical news intelligence.                          │
  │     30,000 articles/day from 200,000+ global sources.                   │
  │     Features: Named entity recognition, sentiment, geolocation,         │
  │              topic classification, story clustering, 15+ languages      │
  │     Free tier: 2,500 requests/day at eventregistry.org                  │
  │                                                                         │
  │  🚫 OpenSanctions            api.opensanctions.org                      │
  │     Aggregates 100+ sanctions lists: OFAC SDN, EU Consolidated,         │
  │     UN Security Council, UK HMT, FATF high-risk, Interpol Red.        │
  │     Updated daily. Free. No key required for basic searches.           │
  │                                                                         │
  ├─────────────────────────────────────────────────────────────────────────┤
  │  LAYER 3 — MACRO DOMAIN (Infrastructure & Trade)                       │
  ├─────────────────────────────────────────────────────────────────────────┤
  │                                                                         │
  │  🌍 USGS Earthquake Feed     earthquake.usgs.gov       [FREE, no key]  │
  │  🌪  NOAA/NWS Weather Alerts  api.weather.gov           [FREE, no key]  │
  │  🛰  NASA FIRMS Satellite     firms.modaps.eosdis.nasa.gov [FREE+key]   │
  │                                                                         │
  ├─────────────────────────────────────────────────────────────────────────┤
  │  LAYER 4 — SYNTHESIS ENGINE                                            │
  ├─────────────────────────────────────────────────────────────────────────┤
  │  Composite Geopolitical Risk Score (0–100)                             │
  │  Statistical confidence intervals per component                        │
  │  Sector impact vectors → Prioritized trading signals                   │
  │  Theater-level assessment (Taiwan, Ukraine, Persian Gulf, etc.)        │
  │  Real-time alerts on threshold breaches                                │
  └─────────────────────────────────────────────────────────────────────────┘

ENVIRONMENT SETUP:
  # Tier-1 (military aircraft — ADS-B Exchange, choose one):
  export RAPIDAPI_KEY="..."          # rapidapi.com → adsbexchange-com1
  export ADSBX_API_KEY="..."         # adsbexchange.com/data/ (direct)

  # Tier-1 (naval vessels — choose one):
  export MARINETRAFFIC_KEY="..."     # marinetraffic.com/en/p/api-services
  export AISHUB_USERNAME="..."       # aishub.net (free, register)

  # Tier-1 (conflict events):
  export ACLED_KEY="..."             # developer.acleddata.com (free)
  export ACLED_EMAIL="..."           # your registration email

  # Tier-1 (news intelligence):
  export EVENT_REGISTRY_KEY="..."    # eventregistry.org (free tier)

  # Optional:
  export ALPHA_VANTAGE_KEY="..."     # alphavantage.co (financial data)

CLI USAGE:
  python geopolitical_engine.py                          # full dashboard
  python geopolitical_engine.py aircraft                 # military aircraft
  python geopolitical_engine.py naval                    # naval vessels
  python geopolitical_engine.py theater taiwan           # theater assessment
  python geopolitical_engine.py theater persian_gulf
  python geopolitical_engine.py theater ukraine
  python geopolitical_engine.py sanctions "Gazprom"      # sanctions search
  python geopolitical_engine.py conflicts                # ACLED conflict feed

GIG — Institutional Quantitative Intelligence
=============================================================================
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("GEO_INTEL")


# =============================================================================
# §1  CONFIGURATION
# =============================================================================

@dataclass(frozen=False)
class APIConfig:
    """
    All API credentials and base URLs.
    Override via environment variables — never hard-code keys in source.
    """

    # ── Military Aircraft (ADS-B Exchange) ───────────────────────────────────
    # Docs:  adsbexchange.com/data/
    # RapidAPI (recommended): rapidapi.com/adsbx/api/adsbexchange-com1
    #   - /v2/mil/ endpoint returns ONLY military aircraft worldwide
    #   - Cost: $0.01 per 100 calls (practically free for quant use)
    ADSBX_API_KEY:      str = field(default_factory=lambda: os.getenv("ADSBX_API_KEY", ""))
    RAPIDAPI_KEY:       str = field(default_factory=lambda: os.getenv("RAPIDAPI_KEY", ""))
    ADSBX_BASE:         str = "https://adsbexchange.com/api/aircraft"
    ADSBX_RAPIDAPI:     str = "https://adsbexchange-com1.p.rapidapi.com/v2"
    ADSBX_GLOBE:        str = "https://globe.adsbexchange.com/re-api"

    # ── Naval Vessels (MarineTraffic + AISHub fallback) ────────────────────
    # Docs:  marinetraffic.com/en/p/api-services
    # AISHub: aishub.net (community AIS aggregator — free with registration)
    MARINETRAFFIC_KEY:  str = field(default_factory=lambda: os.getenv("MARINETRAFFIC_KEY", ""))
    AISHUB_USERNAME:    str = field(default_factory=lambda: os.getenv("AISHUB_USERNAME", ""))
    MT_BASE:            str = "https://services.marinetraffic.com/api"
    AISHUB_BASE:        str = "https://data.aishub.net/ws.php"

    # ── Conflict Events (ACLED) ────────────────────────────────────────────
    # Docs:  developer.acleddata.com
    # Free registration → 250,000 API calls/month
    ACLED_KEY:          str = field(default_factory=lambda: os.getenv("ACLED_KEY", ""))
    ACLED_EMAIL:        str = field(default_factory=lambda: os.getenv("ACLED_EMAIL", ""))
    ACLED_BASE:         str = "https://api.acleddata.com/acled/read"

    # ── News Intelligence (Event Registry) ────────────────────────────────
    # Docs:  eventregistry.org/documentation
    # Free tier: 2,500 req/day, 30K articles/day indexed
    EVENT_REGISTRY_KEY: str = field(default_factory=lambda: os.getenv("EVENT_REGISTRY_KEY", ""))
    ER_BASE:            str = "https://eventregistry.org/api/v1"

    # ── Financial Data (Alpha Vantage) ─────────────────────────────────────
    ALPHA_VANTAGE_KEY:  str = field(default_factory=lambda: os.getenv("ALPHA_VANTAGE_KEY", ""))

    # ── Free APIs — no registration required ──────────────────────────────
    USGS_BASE:          str = "https://earthquake.usgs.gov/fdsnws/event/1/query"
    NOAA_ALERTS:        str = "https://api.weather.gov/alerts/active"
    GDELT_BASE:         str = "https://api.gdeltproject.org/api/v2"
    OPENSANCTIONS_BASE: str = "https://api.opensanctions.org"
    NASA_FIRMS_BASE:    str = "https://firms.modaps.eosdis.nasa.gov/api"


CONFIG = APIConfig()


# =============================================================================
# §2  HTTP CLIENT — Circuit Breaker + Rate Limiter + Cache
# =============================================================================

class CircuitState(Enum):
    CLOSED    = "closed"      # Normal operation
    OPEN      = "open"        # Failing — reject fast
    HALF_OPEN = "half-open"   # Probing recovery


class CircuitBreaker:
    """
    Per-API circuit breaker. Prevents cascade failures when external APIs degrade.
    Pattern: CLOSED → (failures ≥ threshold) → OPEN → (timeout) → HALF_OPEN → CLOSED
    """

    def __init__(self, name: str, failure_threshold: int = 5,
                 recovery_timeout: float = 60.0):
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self._failures = 0
        self._state = CircuitState.CLOSED
        self._last_failure = 0.0
        self._lock = threading.Lock()

    @property
    def state(self) -> CircuitState:
        return self._state

    def can_proceed(self) -> bool:
        with self._lock:
            if self._state is CircuitState.CLOSED:
                return True
            if self._state is CircuitState.OPEN:
                if time.monotonic() - self._last_failure > self.recovery_timeout:
                    self._state = CircuitState.HALF_OPEN
                    logger.debug(f"[CB:{self.name}] Half-open — testing recovery")
                    return True
                return False
            return True  # HALF_OPEN: allow one probe

    def record_success(self) -> None:
        with self._lock:
            self._failures = 0
            if self._state is not CircuitState.CLOSED:
                logger.info(f"[CB:{self.name}] Closed — circuit recovered")
            self._state = CircuitState.CLOSED

    def record_failure(self) -> None:
        with self._lock:
            self._failures += 1
            self._last_failure = time.monotonic()
            if self._failures >= self.failure_threshold:
                self._state = CircuitState.OPEN
                logger.warning(
                    f"[CB:{self.name}] OPEN — tripped after {self._failures} failures")


_CACHE:    Dict[str, Dict[str, Any]] = {}
_BREAKERS: Dict[str, CircuitBreaker] = {}
_BRKR_LOCK = threading.Lock()

_PLATFORM_HEADERS = {
    "User-Agent":      "GIG-GeoIntel/2.0 (Institutional Quantitative Platform)",
    "Accept":          "application/json",
    "Accept-Encoding": "gzip, deflate",
}


def _breaker(source: str) -> CircuitBreaker:
    with _BRKR_LOCK:
        if source not in _BREAKERS:
            _BREAKERS[source] = CircuitBreaker(source)
        return _BREAKERS[source]


def _get(url: str,
         cache_key: str,
         ttl: int = 900,
         headers: Optional[Dict] = None,
         params: Optional[Dict] = None,
         payload: Optional[Dict] = None,
         method: str = "GET",
         source: str = "default",
         timeout: int = 20) -> Optional[Any]:
    """
    Unified HTTP request with:
      • TTL-based in-memory cache (stale-while-revalidate on error)
      • Per-source circuit breaker (5 failures → 60s open)
      • Exponential backoff retry (3 attempts: 1s, 2s, 4s)
      • Graceful stale-cache fallback when all retries fail
    """
    import requests  # lazy import — allows module-level use before install check

    now = time.monotonic()
    wall = time.time()

    # Serve fresh cache immediately
    entry = _CACHE.get(cache_key)
    if entry and now - entry["ts"] < ttl:
        return entry["data"]

    cb = _breaker(source)
    if not cb.can_proceed():
        return entry["data"] if entry else None

    merged_headers = {**_PLATFORM_HEADERS, **(headers or {})}

    for attempt in range(3):
        try:
            if method.upper() == "POST":
                r = requests.post(url, json=payload, headers=merged_headers,
                                  timeout=timeout)
            else:
                r = requests.get(url, params=params, headers=merged_headers,
                                 timeout=timeout)
            r.raise_for_status()
            data = r.json()
            _CACHE[cache_key] = {"data": data, "ts": now, "wall": wall}
            cb.record_success()
            return data

        except Exception as exc:
            wait = 2 ** attempt
            logger.debug(f"[{source}] attempt={attempt+1} failed ({exc}) — "
                         f"retry in {wait}s")
            if attempt < 2:
                time.sleep(wait)
            else:
                cb.record_failure()

    # Stale cache as last resort
    return entry["data"] if entry else None


# =============================================================================
# §3  MILITARY AIRCRAFT — ADS-B EXCHANGE
# =============================================================================

# ── ICAO 24-bit address ranges allocated to military forces ──────────────────
# These hex ranges are assigned by ITU/ICAO to specific nations' military.
# Military aircraft inside these ranges are NOT usually callsign-filed.
MILITARY_ICAO_RANGES: Dict[str, List[Tuple[int, int]]] = {
    "US Military":         [(0xAE0000, 0xAFFFFF)],  # USAF, USN, USMC, USA
    "UK Military":         [(0x43C000, 0x43FFFF)],  # RAF, Royal Navy Air
    "France Military":     [(0x3A0000, 0x3AFFFF)],  # Armée de l'Air
    "Germany Military":    [(0x3CC000, 0x3CFFFF)],  # Luftwaffe
    "Canada Military":     [(0xC00000, 0xC3FFFF)],  # RCAF
    "Australia Military":  [(0x7C0000, 0x7FFFFF)],  # RAAF
    "NATO AWACS":          [(0x4A0000, 0x4AFFFF)],  # NATO E-3 fleet
    "Russia Military":     [(0x100000, 0x1FFFFF)],  # VKS (approximate)
    "China Military":      [(0x780000, 0x7BFFFF)],  # PLAAF (approximate)
    "Israel Military":     [(0x738000, 0x73FFFF)],  # IAF
    "Japan Military":      [(0x840000, 0x87FFFF)],  # JASDF
}

# ── Military callsign prefix → mission description ───────────────────────────
# Source: FAA JO 7340.2, NATO STANAG, public ATCR databases
MILITARY_CALLSIGNS: Dict[str, str] = {
    # US Air Force — Mobility & Airlift
    "RCH":    "USAF C-17/C-5 Strategic Airlift (Air Mobility Command)",
    "REACH":  "USAF Strategic Airlift — long-range troop/equipment movement",
    "RIFLE":  "USAF Airlift",
    # US Air Force — Tankers
    "VAPOR":  "USAF KC-135/KC-46 Aerial Refueling Tanker",
    "SHELL":  "USAF KC-135 Tanker",
    "QUID":   "USAF Tanker",
    "DUKE":   "USAF KC-46 Tanker",
    # US Air Force — Bombers
    "IRON":   "USAF B-52 Stratofortress Strategic Bomber",
    "BONE":   "USAF B-1B Lancer Supersonic Bomber",
    "DEATH":  "USAF B-52 Bomber",
    # US Air Force — Fighters / Aggressors
    "JAKE":   "USAF Fighter / Aggressor",
    "STING":  "USAF Fighter (F-15/F-16/F-22)",
    "GHOST":  "USAF Special Operations / Fighter",
    "HAVOC":  "USAF Fighter",
    "COBRA":  "USAF Fighter",
    "GATOR":  "USAF Special Operations",
    # US Air Force — ISR / Command & Control
    "SNTRY":  "USAF E-3 Sentry AWACS (Airborne Warning & Control)",
    "DISCO":  "USAF E-8 JSTARS (Ground Surveillance / Battle Mgmt)",
    "ARIES":  "USAF RC-135 Rivet Joint SIGINT Collection",
    "OLIVE":  "USAF RC-135 Intelligence",
    "JANUS":  "USAF U-2/RQ-4 High-altitude Reconnaissance",
    # US Air Force — Nuclear & Special
    "PRIME":  "USAF E-4B Nightwatch (Airborne Nuclear Command Post)",
    "FLASH":  "USAF E-6B Mercury (Nuclear Sub Communications Relay)",
    "SAM":    "Special Air Mission — VIP/Presidential transport",
    "EXEC":   "US Government Executive Flight",
    # US Navy & Marines
    "NAVY":   "US Navy Aviation",
    "GRIFF":  "US Navy Fighter (F/A-18 Hornet / Super Hornet)",
    "TOPGUN": "US Navy Fighter Weapons School",
    "VMGR":   "USMC KC-130 Tanker (Marine Aerial Refueler/Transport)",
    "VRC":    "US Navy C-2 Greyhound (Carrier Onboard Delivery)",
    "VP":     "US Navy P-3 Orion / P-8 Poseidon Maritime Patrol",
    "VQ":     "US Navy Electronic Reconnaissance",
    "VAW":    "US Navy E-2C/D Hawkeye (Carrier-based AEW)",
    # US Army
    "ARMY":   "US Army Aviation",
    "MEDEVAC":"US Army Medical Evacuation",
    # NATO
    "NATO":   "NATO Command / E-3 AWACS",
    "MAGIC":  "NATO AWACS",
    # Special operations
    "EVAC":   "Medical/Non-Combatant Evacuation Operation",
    "NOBLE":  "Humanitarian / Special Mission",
}

# ── High-value intelligence (HVI) aircraft types ─────────────────────────────
HVI_AIRCRAFT: Dict[str, str] = {
    "C17":   "C-17 Globemaster III — Strategic inter-theater airlift (troops/armor)",
    "C5":    "C-5M Super Galaxy — Heavy airlift (outsized cargo, armor, helicopters)",
    "C130":  "C-130J Hercules — Tactical/special ops airlift",
    "C40":   "C-40 Clipper — VIP transport (flag officer / senior official movements)",
    "VC25":  "VC-25 Air Force One — Presidential transport",
    "E4":    "E-4B Nightwatch — Airborne National Command Post (nuclear doomsday plane)",
    "E6":    "E-6B Mercury — TACAMO (nuclear sub launch order relay)",
    "E3":    "E-3 Sentry — AWACS (airborne surveillance, battle command)",
    "E2":    "E-2D Hawkeye — Carrier-based AEW (fleet defense picket)",
    "E8":    "E-8C JSTARS — Ground Moving Target Indicator (battle management)",
    "KC135": "KC-135 Stratotanker — Aerial refueling (extends all strike aircraft range)",
    "KC46":  "KC-46A Pegasus — Next-generation aerial refueling tanker",
    "KC10":  "KC-10 Extender — Aerial refueling / cargo combo",
    "RC135": "RC-135 Rivet Joint — SIGINT collection (strategic ISR)",
    "U2":    "U-2 Dragon Lady — High-altitude reconnaissance (70,000 ft)",
    "RQ4":   "RQ-4 Global Hawk — High-altitude long-endurance UAV reconnaissance",
    "MQ9":   "MQ-9 Reaper — Armed UAV (ISR + strike, signals active operation)",
    "MQ1":   "MQ-1 Predator — Armed UAV (predecessor to Reaper)",
    "P8":    "P-8A Poseidon — Maritime patrol / ASW (tracks submarines)",
    "P3":    "P-3 Orion — Maritime patrol / ASW",
    "B52":   "B-52H Stratofortress — Nuclear-capable strategic bomber",
    "B1":    "B-1B Lancer — Supersonic conventional bomber",
    "B2":    "B-2 Spirit — Stealth strategic bomber (nuclear-capable)",
    "F35":   "F-35 Lightning II — 5th-gen stealth multi-role fighter",
    "F22":   "F-22 Raptor — Air superiority stealth fighter",
    "V22":   "V-22 Osprey — Tiltrotor (special ops assault / CSAR)",
    "CH47":  "CH-47 Chinook — Heavy-lift helicopter (large troop movements)",
}


class MilitaryAircraftTracker:
    """
    ADS-B Exchange — The only open global aircraft tracker without military filtering.

    Background:
      Commercial trackers (Flightradar24, FlightAware, Plane Finder) remove military
      and government aircraft from public feeds at government request.
      ADS-B Exchange publishes raw ADS-B and MLAT data as received — zero filtering.
      This makes it the authoritative open-source military aviation intelligence feed.

    API priority chain:
      1. RapidAPI /v2/mil/  (military-only endpoint, global, $0.01/100 calls)
      2. Direct API /json/   (full feed, requires adsbexchange.com subscription)
      3. Globe API           (rate-limited public feed, free but unstable)

    Registration:
      RapidAPI (recommended): rapidapi.com → search "adsbexchange-com1"
      Direct:                 adsbexchange.com/data/
    """

    # ── ICAO hex → nation classifier ─────────────────────────────────────────
    @staticmethod
    def _nation_from_hex(icao_hex: str) -> Optional[str]:
        try:
            n = int(icao_hex, 16)
            for nation, ranges in MILITARY_ICAO_RANGES.items():
                for lo, hi in ranges:
                    if lo <= n <= hi:
                        return nation
        except (ValueError, TypeError):
            pass
        return None

    # ── Classify a single ADS-B state vector ─────────────────────────────────
    @classmethod
    def _classify(cls, raw: Dict) -> Dict:
        icao    = (raw.get("hex") or raw.get("icao24") or "").strip().upper()
        cs      = (raw.get("flight") or raw.get("callsign") or "").strip().upper()
        ac_type = (raw.get("t") or raw.get("category") or "").strip().upper()

        # ICAO range → nation
        nation = cls._nation_from_hex(icao)

        # Callsign → mission
        mission = None
        for prefix, desc in MILITARY_CALLSIGNS.items():
            if cs.startswith(prefix):
                mission = desc
                break

        # Aircraft type → HVI description
        hvi_desc = None
        for code, desc in HVI_AIRCRAFT.items():
            if code in ac_type.replace("-", "").replace(" ", ""):
                hvi_desc = desc
                break

        is_military = (nation is not None) or (mission is not None)
        is_hvi      = hvi_desc is not None

        # Squawk 7700 (emergency) / 7600 (radio failure) / 7500 (hijack)
        squawk = str(raw.get("squawk") or "")
        emergency = (squawk in ("7700", "7600", "7500"))

        return {
            "icao24":       icao,
            "callsign":     cs,
            "aircraft_type":ac_type,
            "latitude":     raw.get("lat"),
            "longitude":    raw.get("lon"),
            "altitude_ft":  raw.get("alt_baro") or raw.get("altitude"),
            "speed_kts":    raw.get("gs") or raw.get("velocity"),
            "heading":      raw.get("track"),
            "vertical_rate":raw.get("baro_rate"),
            "on_ground":    raw.get("ground") or raw.get("on_ground") or False,
            "squawk":       squawk,
            "emergency":    emergency,
            "nation":       nation or "Unknown",
            "mission":      mission,
            "hvi_desc":     hvi_desc,
            "is_military":  is_military,
            "is_hvi":       is_hvi,
        }

    # ── Fetch global military aircraft ────────────────────────────────────────
    @classmethod
    def fetch_global(cls) -> Dict:
        """
        Fetch all military aircraft currently airborne worldwide.

        Returns structured intelligence including:
          - Total military count, HVI count
          - By-nation breakdown
          - Aircraft type breakdown
          - All high-value intelligence (HVI) aircraft with details
          - Automated alerts for anomalous patterns
        """
        raw_aircraft: List[Dict] = []
        source_used = "ADS-B Exchange (unavailable)"

        # ── Priority 1: RapidAPI /v2/mil/ ─────────────────────────────────
        if CONFIG.RAPIDAPI_KEY:
            data = _get(
                url=f"{CONFIG.ADSBX_RAPIDAPI}/mil/",
                cache_key="adsbx_mil_global",
                ttl=30,           # 30s — live military data
                headers={
                    "X-RapidAPI-Key":  CONFIG.RAPIDAPI_KEY,
                    "X-RapidAPI-Host": "adsbexchange-com1.p.rapidapi.com",
                },
                source="adsbx_rapidapi",
                timeout=8,
            )
            if data and "ac" in data:
                raw_aircraft = data["ac"] or []
                source_used = "ADS-B Exchange RapidAPI /v2/mil/ [military-only endpoint]"

        # ── Priority 2: Direct ADS-B Exchange API ────────────────────────
        elif CONFIG.ADSBX_API_KEY:
            data = _get(
                url=f"{CONFIG.ADSBX_BASE}/json/",
                cache_key="adsbx_direct_all",
                ttl=30,
                headers={"api-auth": CONFIG.ADSBX_API_KEY},
                source="adsbx_direct",
                timeout=10,
            )
            if data and "ac" in data:
                raw_aircraft = data["ac"] or []
                source_used = "ADS-B Exchange Direct API [full unfiltered feed]"

        # ── Priority 3: Globe API (free, rate-limited) ────────────────────
        else:
            data = _get(
                url=f"{CONFIG.ADSBX_GLOBE}/",
                cache_key="adsbx_globe",
                ttl=60,
                params={"all": ""},
                source="adsbx_globe",
                timeout=15,
            )
            if data:
                raw_aircraft = data.get("aircraft") or data.get("ac") or []
                source_used = "ADS-B Exchange Globe [rate-limited public feed]"

        # ── Classify all aircraft ─────────────────────────────────────────
        classified = [cls._classify(a) for a in raw_aircraft]
        military   = [a for a in classified if a["is_military"]]
        hvi        = [a for a in classified if a["is_hvi"]]
        emergency  = [a for a in classified if a["emergency"]]

        # By-nation breakdown
        by_nation: Dict[str, int] = defaultdict(int)
        for a in military:
            by_nation[a["nation"]] += 1

        # Aircraft type breakdown
        type_counts: Dict[str, int] = defaultdict(int)
        for a in hvi:
            short = (a["hvi_desc"] or "").split("—")[0].strip()
            type_counts[short] += 1

        # ── Pattern-based alerts ──────────────────────────────────────────
        alerts: List[str] = []

        tankers = [a for a in military
                   if a.get("mission") and "Tanker" in a["mission"]
                   or any(c in (a.get("aircraft_type") or "")
                          for c in ["KC13", "KC46", "KC10"])]
        if len(tankers) > 25:
            alerts.append(
                f"⚠  ELEVATED TANKER OPS — {len(tankers)} aerial refueling "
                f"aircraft airborne (extends strike range for operations)")

        bombers = [a for a in hvi
                   if a.get("hvi_desc") and "Bomber" in a["hvi_desc"]]
        if bombers:
            alerts.append(
                f"🔴 STRATEGIC BOMBER ACTIVITY — {len(bombers)} bomber(s) airborne: "
                + ", ".join(set(a.get("callsign", "UNKN") for a in bombers[:4])))

        doomsday = [a for a in hvi
                    if a.get("hvi_desc") and ("Nightwatch" in a["hvi_desc"]
                                              or "TACAMO" in a["hvi_desc"])]
        if doomsday:
            alerts.append(
                f"🔴 NUCLEAR C2 AIRCRAFT AIRBORNE — "
                f"{', '.join(a.get('callsign','?') for a in doomsday)} "
                f"(E-4B/E-6B — extreme escalation signal)")

        if len(hvi) > 15:
            alerts.append(
                f"⚠  {len(hvi)} HIGH-VALUE INTELLIGENCE aircraft airborne "
                f"simultaneously — above normal baseline")

        if emergency:
            alerts.append(
                f"🚨 EMERGENCY SQUAWK — {len(emergency)} aircraft: "
                + ", ".join(a.get("callsign", a["icao24"]) for a in emergency[:5]))

        return {
            "source":           source_used,
            "total_tracked":    len(classified),
            "military_count":   len(military),
            "hvi_count":        len(hvi),
            "emergency_count":  len(emergency),
            "by_nation":        dict(by_nation),
            "type_breakdown":   dict(type_counts),
            "high_value":       sorted(hvi, key=lambda a: a.get("altitude_ft") or 0, reverse=True)[:25],
            "military_assets":  military[:60],
            "alerts":           alerts,
            "timestamp":        datetime.now(timezone.utc).isoformat(),
        }

    # ── Fetch aircraft in a geographic region of interest ────────────────────
    @classmethod
    def fetch_region(cls, lat_min: float, lat_max: float,
                     lon_min: float, lon_max: float, label: str) -> Dict:
        """
        Fetch all aircraft (and classify military) in a geographic bounding box.
        Used for monitoring specific theaters of operation.
        """
        if not CONFIG.RAPIDAPI_KEY:
            return {
                "region": label, "aircraft": [], "military": [],
                "count": 0, "military_count": 0,
                "note": "Set RAPIDAPI_KEY for regional theater monitoring",
            }

        lat_c = (lat_min + lat_max) / 2
        lon_c = (lon_min + lon_max) / 2
        # Approximate bounding box to radius (rough — ADS-B Exchange uses nm radius)
        dist_nm = max(150, int(max(lat_max - lat_min, lon_max - lon_min) * 60))

        data = _get(
            url=f"{CONFIG.ADSBX_RAPIDAPI}/lat/{lat_c:.3f}/lon/{lon_c:.3f}/dist/{dist_nm}/",
            cache_key=f"adsbx_region_{label.replace(' ','_')}",
            ttl=45,
            headers={
                "X-RapidAPI-Key":  CONFIG.RAPIDAPI_KEY,
                "X-RapidAPI-Host": "adsbexchange-com1.p.rapidapi.com",
            },
            source="adsbx_rapidapi",
            timeout=10,
        )

        if not data:
            return {"region": label, "count": 0, "military_count": 0}

        classified = [cls._classify(a) for a in (data.get("ac") or [])]
        military   = [a for a in classified if a["is_military"]]
        hvi        = [a for a in classified if a["is_hvi"]]

        return {
            "region":          label,
            "center":          {"lat": lat_c, "lon": lon_c},
            "radius_nm":       dist_nm,
            "count":           len(classified),
            "military_count":  len(military),
            "hvi_count":       len(hvi),
            "military":        military,
            "all_aircraft":    classified,
            "timestamp":       datetime.now(timezone.utc).isoformat(),
        }


# =============================================================================
# §4  NAVAL VESSELS — MARINETRAFFIC + AISHUB
# =============================================================================

# ── Naval vessel name prefix → navy ──────────────────────────────────────────
NAVAL_PREFIXES: Dict[str, str] = {
    "USS":    "United States Navy",
    "USNS":   "US Naval Ship (Military Sealift Command)",
    "HMS":    "Royal Navy (UK)",
    "HMAS":   "Royal Australian Navy",
    "HMCS":   "Royal Canadian Navy",
    "HNLMS":  "Royal Netherlands Navy",
    "HNoMS":  "Royal Norwegian Navy",
    "HDMS":   "Royal Danish Navy",
    "FNS":    "Finnish Navy",
    "ESPS":   "Spanish Navy",
    "FS":     "French Navy (Marine Nationale)",
    "ITS":    "Italian Navy (Marina Militare)",
    "TCG":    "Turkish Naval Forces",
    "JS":     "Japan Maritime Self-Defense Force",
    "INS":    "Indian Navy",
    "CNS":    "Chinese People's Liberation Army Navy",
    "RFS":    "Russian Federation Navy",
    "VADM":   "NATO / Allied Flagship",
    "NRP":    "Portuguese Navy",
}

# AIS vessel type codes defined in ITU-R M.1371
NAVAL_AIS_TYPES: Dict[int, str] = {
    35: "Military Operations",
    51: "Search and Rescue Vessel",
    55: "Law Enforcement",
    57: "Spare — Law Enforcement",
    58: "Medical Transport",
}

# High-value naval vessel hull classification symbols
NAVAL_HVI_CODES = {
    "CVN": "Nuclear Aircraft Carrier",
    "CV":  "Conventional Aircraft Carrier",
    "LHA": "Amphibious Assault Ship (General Purpose)",
    "LHD": "Amphibious Assault Ship (Multi-Purpose)",
    "LPD": "Amphibious Transport Dock",
    "LSD": "Dock Landing Ship",
    "DDG": "Guided Missile Destroyer",
    "CG":  "Guided Missile Cruiser",
    "SSN": "Nuclear Attack Submarine (surfaced)",
    "SSBN":"Ballistic Missile Submarine (surfaced)",
    "SSGN":"Guided Missile Submarine (surfaced)",
    "FFG": "Guided Missile Frigate",
    "FF":  "Frigate",
    "LCS": "Littoral Combat Ship",
    "AOE": "Fast Combat Support Ship",
    "T-AO":"Fleet Replenishment Oiler",
}


class NavalVesselTracker:
    """
    MarineTraffic API — Industry-standard AIS vessel tracking (400K+ vessels).
    Fallback: AISHub community AIS aggregator (free, register at aishub.net).

    Naval vessels broadcast AIS with vessel_type=35 (ITU Military Operations).
    Additional identification: name prefix (USS/HMS/etc.) + MMSI range analysis.

    Market intelligence:
      - Carrier Strike Group deployment → defense sector, energy, safe havens
      - Amphibious ready groups → imminent military operation signal
      - Fleet oiler concentrations → sustained operations indicator
      - Submarine on surface → anomalous — normally submerged
    """

    @staticmethod
    def _parse_vessel(raw: Dict) -> Dict:
        name     = (raw.get("SHIPNAME") or raw.get("name") or "").upper().strip()
        vtype    = int(raw.get("SHIPTYPE") or raw.get("type") or 0)
        mmsi     = str(raw.get("MMSI") or raw.get("mmsi") or "")

        # Navy affiliation from name prefix
        navy = None
        for prefix, nation in NAVAL_PREFIXES.items():
            if name.startswith(prefix.upper()):
                navy = nation
                break

        # HVI hull classification
        hvi_class = None
        for code, desc in NAVAL_HVI_CODES.items():
            if code in name:
                hvi_class = desc
                break

        # MMSI military ranges (rough heuristic — not exhaustive)
        mil_mmsi = (
            mmsi.startswith("338")  # US Navy approximate range
            or mmsi.startswith("232")  # Royal Navy approximate
            or mmsi.startswith("227")  # French Navy approximate
            or vtype == 35
        )

        return {
            "mmsi":         mmsi,
            "name":         raw.get("SHIPNAME") or raw.get("name") or "Unknown",
            "ais_type":     vtype,
            "ais_category": NAVAL_AIS_TYPES.get(vtype, f"Type-{vtype}"),
            "latitude":     raw.get("LAT") or raw.get("latitude"),
            "longitude":    raw.get("LON") or raw.get("longitude"),
            "speed_kts":    raw.get("SPEED") or raw.get("speed"),
            "heading":      raw.get("HEADING") or raw.get("heading"),
            "destination":  raw.get("DESTINATION") or raw.get("destination") or "",
            "nav_status":   raw.get("STATUS") or raw.get("status") or "",
            "flag":         raw.get("FLAG") or raw.get("flag") or "",
            "navy":         navy,
            "hvi_class":    hvi_class,
            "is_naval":     navy is not None or mil_mmsi,
            "is_hvi":       hvi_class is not None,
            "last_pos":     raw.get("LAST_POS") or raw.get("timestamp") or "",
        }

    @classmethod
    def fetch_global(cls) -> Dict:
        """
        Fetch military/naval vessels globally.
        Priority: MarineTraffic (vessel_type=35) → AISHub fallback.
        """
        raw_vessels: List[Dict] = []
        source_used = "No AIS source configured"

        # ── Priority 1: MarineTraffic API ─────────────────────────────────
        if CONFIG.MARINETRAFFIC_KEY:
            data = _get(
                url=f"{CONFIG.MT_BASE}/getVesselsInArea/{CONFIG.MARINETRAFFIC_KEY}",
                cache_key="mt_military_global",
                ttl=120,
                params={"v": 8, "protocol": "jsono",
                        "msgtype": "extended", "vessel_type": 35},
                source="marinetraffic",
                timeout=15,
            )
            if data:
                raw_vessels = data if isinstance(data, list) else data.get("data", [])
                source_used = "MarineTraffic API [AIS vessel_type=35 Military]"

        # ── Priority 2: AISHub ────────────────────────────────────────────
        if not raw_vessels and CONFIG.AISHUB_USERNAME:
            data = _get(
                url=CONFIG.AISHUB_BASE,
                cache_key="aishub_military",
                ttl=120,
                params={
                    "username": CONFIG.AISHUB_USERNAME,
                    "format": 1, "output": "jsono",
                    "compress": 0, "shiptype": 35,
                },
                source="aishub",
                timeout=15,
            )
            if data and isinstance(data, list) and len(data) > 1:
                raw_vessels = data[1] if isinstance(data[1], list) else []
                source_used = "AISHub Community Aggregator [AIS type 35]"

        parsed  = [cls._parse_vessel(v) for v in raw_vessels]
        naval   = [v for v in parsed if v["is_naval"]]
        hvi     = [v for v in parsed if v["is_hvi"]]

        # By-navy count
        by_navy: Dict[str, int] = defaultdict(int)
        for v in naval:
            by_navy[v["navy"] or "Unknown"] += 1

        # By hull class
        by_class: Dict[str, int] = defaultdict(int)
        for v in hvi:
            by_class[v["hvi_class"] or "Other"] += 1

        # ── Tactical alerts ───────────────────────────────────────────────
        alerts: List[str] = []

        carriers = [v for v in naval if v.get("hvi_class") and
                    any(c in (v["hvi_class"] or "")
                        for c in ["Aircraft Carrier", "Amphibious Assault"])]
        if carriers:
            alerts.append(
                f"🚢 CARRIER / AMPHIB ACTIVITY — {len(carriers)} major surface "
                f"combatant(s) tracked: "
                + ", ".join(v["name"] for v in carriers[:4]))

        subs = [v for v in naval if v.get("hvi_class") and
                any(c in (v["hvi_class"] or "") for c in ["Submarine"])]
        if subs:
            alerts.append(
                f"🔴 SUBMARINE SURFACE CONTACT — {len(subs)} sub(s) on surface "
                f"(unusual — possible port transit or emergency)")

        replenishment = [v for v in naval if v.get("hvi_class") and
                         "Oiler" in (v["hvi_class"] or "")]
        if len(replenishment) > 5:
            alerts.append(
                f"⚠  FLEET LOGISTICS ELEVATED — {len(replenishment)} fleet oilers "
                f"underway (indicates sustained at-sea operations)")

        return {
            "source":        source_used,
            "total_tracked": len(parsed),
            "naval_count":   len(naval),
            "hvi_count":     len(hvi),
            "by_navy":       dict(by_navy),
            "by_class":      dict(by_class),
            "high_value":    hvi[:25],
            "naval_assets":  naval[:60],
            "alerts":        alerts,
            "timestamp":     datetime.now(timezone.utc).isoformat(),
        }

    # ── Strategic maritime chokepoint monitoring ──────────────────────────────
    CHOKEPOINTS = {
        "Taiwan Strait":        {"lat": 24.5,  "lon": 120.3, "radius_nm": 150,
                                 "significance": "China-Taiwan — semiconductor supply chain"},
        "Strait of Hormuz":     {"lat": 26.6,  "lon": 56.4,  "radius_nm": 80,
                                 "significance": "20% of global oil transit (XLE, USO)"},
        "South China Sea":      {"lat": 12.5,  "lon": 115.0, "radius_nm": 500,
                                 "significance": "$3.4T annual trade — major dispute zone"},
        "Black Sea":            {"lat": 43.0,  "lon": 34.0,  "radius_nm": 350,
                                 "significance": "Ukraine grain/energy — commodity signal"},
        "Bab-el-Mandeb":        {"lat": 12.6,  "lon": 43.5,  "radius_nm": 100,
                                 "significance": "Red Sea/Suez access — Houthi threat zone"},
        "Strait of Malacca":    {"lat": 3.0,   "lon": 101.0, "radius_nm": 200,
                                 "significance": "Asia-Europe shipping — piracy risk"},
        "Baltic Sea Approaches":{"lat": 58.0,  "lon": 19.0,  "radius_nm": 350,
                                 "significance": "NATO eastern flank — Nord Stream area"},
        "Bosphorus / TPAO":     {"lat": 41.0,  "lon": 29.0,  "radius_nm": 50,
                                 "significance": "Black Sea NATO access — Turkey control"},
    }

    @classmethod
    def get_chokepoint_status(cls) -> Dict:
        return {
            "chokepoints": cls.CHOKEPOINTS,
            "note": ("Live vessel counts require MarineTraffic or AISHub credentials. "
                     "Configure MARINETRAFFIC_KEY or AISHUB_USERNAME."),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }


# =============================================================================
# §5  CONFLICT EVENTS — ACLED
# =============================================================================

# ACLED event types → market impact metadata
ACLED_EVENT_META: Dict[str, Dict] = {
    "Battles":                     {"severity": 9, "sectors": ["defense", "energy", "currency", "gold"]},
    "Explosions/Remote violence":  {"severity": 8, "sectors": ["defense", "insurance", "energy"]},
    "Violence against civilians":  {"severity": 7, "sectors": ["currency", "bonds", "humanitarian_aid"]},
    "Protests":                    {"severity": 3, "sectors": ["currency", "policy", "equities"]},
    "Riots":                       {"severity": 6, "sectors": ["currency", "equities", "insurance"]},
    "Strategic developments":      {"severity": 4, "sectors": ["defense", "policy", "intelligence"]},
}

# Sub-event types flagged as critical intelligence
CRITICAL_SUBTYPES: Dict[str, str] = {
    "Air/drone strike":                      "🔴 AIR STRIKE — significant kinetic escalation",
    "Shelling/artillery/missile attack":     "🔴 ARTILLERY/MISSILE ATTACK — active combat",
    "Non-state actor overtakes territory":   "🔴 TERRITORY LOST — strategic reversal",
    "Government regains territory":          "🟡 COUNTER-OFFENSIVE — territory recovered",
    "Chemical weapon":                       "🔴 CBRN/CHEMICAL — potential WMD use (UN alert)",
    "Nuclear/radiological weapon":           "🔴 NUCLEAR — DEFCON escalation signal",
    "Suicide bomb/IED":                      "🟠 IED/SUICIDE BOMB — terrorism / insurgency",
    "Grenade":                               "🟠 EXPLOSIVE DEVICE — sub-state violence",
    "Remote explosive/landmine/IED":         "🟠 REMOTE EXPLOSIVE — territory denial",
}


class ACLEDConflictTracker:
    """
    Armed Conflict Location & Event Data Project (ACLED).

    The global gold standard for structured political violence data.
    Trusted by: UN, World Bank, US DoD, EU External Action Service,
                ICRC, Amnesty International, Human Rights Watch, NATO.

    Coverage: 60+ countries, all sub-Saharan Africa, MENA, South/SE Asia,
              Latin America, Eastern Europe, Caucasus.
    Historical depth: 1M+ events since 1997.
    Latency: real-time to ~1-week lag depending on region.

    Free API access: developer.acleddata.com (250K calls/month)
    Fallback: GDELT (lower quality but always available)
    """

    @classmethod
    def fetch(cls, days: int = 7, limit: int = 500,
              event_types: Optional[List[str]] = None) -> Dict:
        """
        Fetch recent armed conflict events from ACLED.

        Args:
            days:        Lookback window (1–30 recommended)
            limit:       Max events returned (ACLED max: 10,000/call)
            event_types: Filter e.g. ["Battles", "Explosions/Remote violence"]
        """
        if not (CONFIG.ACLED_KEY and CONFIG.ACLED_EMAIL):
            logger.warning(
                "ACLED credentials not configured. "
                "Register free at developer.acleddata.com → falling back to GDELT")
            return cls._gdelt_fallback(days)

        since = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")

        params: Dict[str, Any] = {
            "key":              CONFIG.ACLED_KEY,
            "email":            CONFIG.ACLED_EMAIL,
            "event_date":       since,
            "event_date_where": ">=",
            "limit":            limit,
            "order":            "timestamp|desc",
            "fields": (
                "iso|country|region|admin1|location|latitude|longitude|"
                "event_date|event_type|sub_event_type|"
                "actor1|inter1|actor2|inter2|"
                "fatalities|notes|source|timestamp"
            ),
        }
        if event_types:
            params["event_type"] = "|".join(event_types)

        data = _get(
            url=CONFIG.ACLED_BASE,
            cache_key=f"acled_{days}d_{limit}",
            ttl=600,        # 10-min TTL
            params=params,
            source="acled",
            timeout=30,
        )

        if not data or "data" not in data:
            logger.warning("ACLED returned no data — falling back to GDELT")
            return cls._gdelt_fallback(days)

        return cls._process(data["data"], days)

    @classmethod
    def _process(cls, events: List[Dict], days: int) -> Dict:
        parsed = []
        for e in events:
            etype    = e.get("event_type") or "Unknown"
            sub_type = e.get("sub_event_type") or ""
            meta     = ACLED_EVENT_META.get(etype, {"severity": 3, "sectors": []})
            flag     = CRITICAL_SUBTYPES.get(sub_type)

            parsed.append({
                "date":       e.get("event_date"),
                "country":    e.get("country"),
                "region":     e.get("region"),
                "location":   f"{e.get('admin1','')}, {e.get('location','')}".strip(", "),
                "latitude":   _safe_float(e.get("latitude")),
                "longitude":  _safe_float(e.get("longitude")),
                "event_type": etype,
                "sub_type":   sub_type,
                "actor1":     e.get("actor1") or "",
                "actor2":     e.get("actor2") or "",
                "fatalities": int(e.get("fatalities") or 0),
                "severity":   meta["severity"],
                "sectors":    meta["sectors"],
                "intel_flag": flag,
                "notes":      (e.get("notes") or "")[:220],
                "source":     e.get("source") or "",
            })

        # Country risk scores
        by_country: Dict[str, Dict] = defaultdict(
            lambda: {"events": 0, "fatalities": 0, "severity_sum": 0,
                     "types": defaultdict(int)})
        for e in parsed:
            c = e["country"] or "Unknown"
            by_country[c]["events"]      += 1
            by_country[c]["fatalities"]  += e["fatalities"]
            by_country[c]["severity_sum"]+= e["severity"]
            by_country[c]["types"][e["event_type"]] += 1

        country_risk = {}
        for country, st in by_country.items():
            n = st["events"]
            avg_sev = st["severity_sum"] / n if n else 0
            score   = min(100, n * 2 + st["fatalities"] * 0.4 + avg_sev * 4)
            dom_type = max(st["types"].items(), key=lambda x: x[1])[0] if st["types"] else ""
            country_risk[country] = {
                "risk_score":    round(score, 1),
                "events":        n,
                "fatalities":    st["fatalities"],
                "avg_severity":  round(avg_sev, 2),
                "dominant_type": dom_type,
            }

        hotspots = sorted(country_risk.items(), key=lambda x: -x[1]["risk_score"])[:10]
        critical  = [e for e in parsed if e.get("intel_flag") and "🔴" in e["intel_flag"]]
        total_fat = sum(e["fatalities"] for e in parsed)

        # Global intensity
        weights = {"Battles": 10, "Explosions/Remote violence": 8,
                   "Violence against civilians": 6, "Riots": 4,
                   "Protests": 2, "Strategic developments": 3}
        w_sum = sum(weights.get(e["event_type"], 3) for e in parsed)
        intensity_score = min(100, w_sum / max(len(parsed), 1) * 10 + total_fat * 0.04)
        intensity_level = ("CRITICAL" if intensity_score > 70 else
                           "HIGH"     if intensity_score > 50 else
                           "ELEVATED" if intensity_score > 30 else
                           "MODERATE" if intensity_score > 10 else "LOW")

        return {
            "source":             "ACLED — Armed Conflict Location & Event Data Project",
            "period_days":        days,
            "total_events":       len(parsed),
            "total_fatalities":   total_fat,
            "intensity_score":    round(intensity_score, 1),
            "intensity_level":    intensity_level,
            "critical_events":    critical[:12],
            "recent_events":      parsed[:35],
            "country_risk":       country_risk,
            "hotspots":           hotspots,
            "timestamp":          datetime.now(timezone.utc).isoformat(),
        }

    @classmethod
    def _gdelt_fallback(cls, days: int) -> Dict:
        """GDELT v2 fallback when ACLED credentials are unavailable."""
        hours = min(days * 24, 72)
        queries = [
            "military conflict attack airstrike",
            "war battle explosion artillery",
            "sanctions embargo ceasefire diplomacy",
        ]
        articles = []
        for q in queries:
            url = (f"{CONFIG.GDELT_BASE}/doc/doc?query="
                   f"{q.replace(' ', '%20')}&mode=artlist"
                   f"&maxrecords=15&format=json&timespan={hours}h")
            data = _get(url, f"gdelt_fb_{hash(q)&0xFFFF}", ttl=900, source="gdelt")
            for a in (data or {}).get("articles", [])[:8]:
                articles.append({
                    "date":       a.get("seendate"),
                    "country":    a.get("sourcecountry"),
                    "event_type": "GDELT Event",
                    "actor1":     a.get("domain"),
                    "fatalities": 0,
                    "severity":   max(1, min(10, int(abs(float(a.get("tone") or 0))))),
                    "sectors":    ["defense"],
                    "intel_flag": None,
                    "notes":      (a.get("title") or "")[:200],
                })
        articles.sort(key=lambda x: x.get("date") or "", reverse=True)
        return {
            "source":           "GDELT v2 [fallback — configure ACLED for structured data]",
            "period_days":      days,
            "total_events":     len(articles),
            "total_fatalities": 0,
            "recent_events":    articles[:30],
            "country_risk":     {},
            "hotspots":         [],
            "note": (
                "ACLED provides structured, fatality-tracked conflict data "
                "used by UN/DoD. Register free at developer.acleddata.com"
            ),
        }


# =============================================================================
# §6  GEOPOLITICAL NEWS — EVENT REGISTRY
# =============================================================================

# Topic taxonomy — each maps to market sector impact
GEO_TOPICS: Dict[str, Dict] = {
    "Military Conflict":       {"sectors": ["defense", "energy", "gold"],      "weight": 10},
    "Nuclear Weapons":         {"sectors": ["uranium", "defense", "bonds"],     "weight": 10},
    "Economic Sanctions":      {"sectors": ["currency", "trade", "em"],         "weight": 8},
    "Trade War":               {"sectors": ["manufacturing", "tech", "agri"],   "weight": 8},
    "Oil & Gas":               {"sectors": ["energy", "currency", "inflation"],  "weight": 9},
    "Cybersecurity Attack":    {"sectors": ["tech", "defense", "insurance"],    "weight": 7},
    "Coup / Political Crisis": {"sectors": ["currency", "equities", "bonds"],   "weight": 8},
    "Pandemic / Epidemic":     {"sectors": ["pharma", "travel", "consumer"],    "weight": 9},
    "Supply Chain Crisis":     {"sectors": ["tech", "auto", "manufacturing"],   "weight": 7},
    "Climate Disaster":        {"sectors": ["agri", "insurance", "utilities"],  "weight": 6},
    "Central Bank Policy":     {"sectors": ["bonds", "currency", "equities"],   "weight": 8},
    "Terrorism":               {"sectors": ["defense", "travel", "insurance"],  "weight": 7},
}


class EventRegistryIntelligence:
    """
    Event Registry — AI-powered global news intelligence.

    30,000+ articles/day crawled from 200,000+ sources across 15+ languages.
    Features: named entity recognition, sentiment analysis, geolocation tagging,
              topic classification, story deduplication, social scoring.

    Free tier: 2,500 requests/day — register at eventregistry.org
    Upgrade tiers available for institutional volume.

    Fallback: GDELT v2 (always available, free, lower quality).
    """

    @classmethod
    def fetch(cls, hours: int = 24) -> Dict:
        if not CONFIG.EVENT_REGISTRY_KEY:
            return cls._gdelt_fallback(hours)

        since = (datetime.now(timezone.utc) - timedelta(hours=hours)).strftime(
            "%Y-%m-%dT%H:%M:%S")

        payload = {
            "action":               "getArticles",
            "keyword":              (
                "military attack strike war invasion sanctions nuclear "
                "cyber coup terrorism missile airstrike"),
            "keywordOper":          "or",
            "dateStart":            since,
            "articlesPage":         1,
            "articlesCount":        100,
            "articlesSortBy":       "socialScore",
            "articlesSortByAsc":    False,
            "dataType":             ["news"],
            "lang":                 "eng",
            "resultType":           "articles",
            "apiKey":               CONFIG.EVENT_REGISTRY_KEY,
            "includeArticleSentiment":    True,
            "includeArticleLocation":     True,
            "includeArticleCategories":   True,
            "includeArticleEventUri":     True,
        }

        data = _get(
            url=f"{CONFIG.ER_BASE}/article/getArticles",
            cache_key=f"er_geo_{hours}h",
            ttl=900,
            payload=payload,
            method="POST",
            source="event_registry",
            timeout=20,
        )

        if not data:
            return cls._gdelt_fallback(hours)

        articles_raw = (data.get("articles") or {}).get("results") or []
        return cls._process(articles_raw, hours)

    @classmethod
    def _process(cls, articles: List[Dict], hours: int) -> Dict:
        processed     = []
        topic_counts: Dict[str, int] = defaultdict(int)
        sentiments    = []

        for art in articles:
            sent = float(art.get("sentiment") or 0)
            sentiments.append(sent)

            cats = [c.get("label", "") for c in (art.get("categories") or [])]
            matched: List[str] = []
            for topic, meta in GEO_TOPICS.items():
                if any(topic.lower() in cat.lower() for cat in cats):
                    matched.append(topic)
                    topic_counts[topic] += 1

            loc = art.get("location") or {}
            processed.append({
                "title":        (art.get("title") or "")[:160],
                "source":       (art.get("source") or {}).get("title") or "",
                "date":         art.get("dateTime") or "",
                "url":          art.get("url") or "",
                "sentiment":    round(sent, 3),
                "sentiment_lbl":("Negative" if sent < -0.1 else
                                 "Positive" if sent >  0.1 else "Neutral"),
                "topics":       matched,
                "country":      (loc.get("country") or {}).get("label") or "",
                "latitude":     loc.get("lat"),
                "longitude":    loc.get("long"),
                "social_score": art.get("socialScore") or 0,
            })

        avg_sent = float(np.mean(sentiments)) if sentiments else 0.0
        top_topics = sorted(topic_counts.items(), key=lambda x: -x[1])[:8]

        return {
            "source":          "Event Registry [AI-powered news intelligence]",
            "period_hours":    hours,
            "total_articles":  len(processed),
            "avg_sentiment":   round(avg_sent, 3),
            "market_tone":     ("Risk-off / Bearish" if avg_sent < -0.2 else
                                "Risk-on / Bullish"  if avg_sent >  0.2 else "Neutral"),
            "top_topics":      top_topics,
            "articles":        processed[:30],
            "timestamp":       datetime.now(timezone.utc).isoformat(),
        }

    @classmethod
    def _gdelt_fallback(cls, hours: int) -> Dict:
        all_articles: List[Dict] = []
        topic_counts: Dict[str, int] = defaultdict(int)
        for topic in list(GEO_TOPICS.keys())[:6]:
            kw  = topic.lower().replace(" ", "%20").replace("/", "%20")
            url = (f"{CONFIG.GDELT_BASE}/doc/doc?query={kw}"
                   f"&mode=artlist&maxrecords=10&format=json&timespan={hours}h")
            data = _get(url, f"gdelt_er_{kw[:20]}", ttl=900, source="gdelt")
            for art in (data or {}).get("articles", [])[:6]:
                sent = float(art.get("tone") or 0)
                all_articles.append({
                    "title":     (art.get("title") or "")[:160],
                    "source":    art.get("domain") or "",
                    "date":      art.get("seendate") or "",
                    "url":       art.get("url") or "",
                    "sentiment": round(sent, 2),
                    "topics":    [topic],
                    "country":   art.get("sourcecountry") or "",
                })
                topic_counts[topic] += 1

        all_articles.sort(key=lambda x: x.get("date") or "", reverse=True)
        avg_sent = float(np.mean([a["sentiment"] for a in all_articles])) if all_articles else 0.0

        return {
            "source":         "GDELT v2 [fallback — configure EVENT_REGISTRY_KEY for AI analysis]",
            "period_hours":   hours,
            "total_articles": len(all_articles),
            "avg_sentiment":  round(avg_sent, 2),
            "market_tone":    ("Risk-off" if avg_sent < -2 else
                               "Risk-on"  if avg_sent >  2 else "Neutral"),
            "top_topics":     sorted(topic_counts.items(), key=lambda x: -x[1])[:8],
            "articles":       all_articles[:30],
        }


# =============================================================================
# §7  SANCTIONS INTELLIGENCE — OPENSANCTIONS
# =============================================================================

class OpenSanctionsTracker:
    """
    OpenSanctions — Global sanctions and watchlist database.

    Aggregates 100+ sanctions/watchlist sources including:
      • OFAC SDN (US Treasury — most comprehensive list)
      • EU Consolidated Financial Sanctions List
      • UN Security Council Consolidated List
      • UK Office of Financial Sanctions Implementation (HMT)
      • FATF High-Risk Jurisdictions (grey/black list)
      • Interpol Red/Blue Notices
      • Various national lists (Australia, Canada, Switzerland, Japan, etc.)

    Updated daily. Free. No key required.
    Docs: api.opensanctions.org

    Market relevance:
      New OFAC designations → currency moves, sector selloffs
      Vessel designations → shipping disruption
      Energy company designations → crude/gas price impact
    """

    BASE = CONFIG.OPENSANCTIONS_BASE

    @classmethod
    def search(cls, query: str, schema: Optional[str] = None) -> Dict:
        """Search for a sanctioned entity by name."""
        params: Dict[str, Any] = {"q": query, "limit": 10}
        if schema:
            params["schema"] = schema

        data = _get(
            url=f"{cls.BASE}/search/default",
            cache_key=f"os_{hashlib.md5(query.encode()).hexdigest()[:10]}",
            ttl=3600,
            params=params,
            source="opensanctions",
        )

        results = []
        for r in (data or {}).get("results") or []:
            props = r.get("properties") or {}
            results.append({
                "name":        r.get("caption") or "",
                "schema":      r.get("schema") or "",
                "datasets":    r.get("datasets") or [],
                "score":       r.get("score") or 0,
                "topics":      r.get("topics") or [],
                "nationality": props.get("nationality") or [],
                "birth_date":  props.get("birthDate") or [],
                "id_numbers":  props.get("idNumber") or [],
                "first_seen":  r.get("first_seen") or "",
            })

        return {
            "query":   query,
            "total":   (data or {}).get("total", {}).get("value", 0) if data else 0,
            "results": results,
        }

    @classmethod
    def recent_designations(cls, days: int = 7) -> Dict:
        """
        Fetch recently added sanctions designations.
        New OFAC/EU/UN additions are potential market-moving events.
        """
        since = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        data = _get(
            url=f"{cls.BASE}/entities",
            cache_key=f"os_recent_{days}d",
            ttl=3600,
            params={"since": since, "topics": "sanction", "limit": 50},
            source="opensanctions",
        )

        designations = []
        energy_hits  = []
        vessel_hits  = []

        for r in (data or {}).get("results") or []:
            name   = r.get("caption") or ""
            schema = r.get("schema") or ""
            d = {
                "name":       name,
                "schema":     schema,
                "datasets":   r.get("datasets") or [],
                "topics":     r.get("topics") or [],
                "first_seen": r.get("first_seen") or "",
            }
            designations.append(d)

            if any(k in name.upper() for k in ["OIL", "GAS", "ENERGY", "PETRO", "LNG"]):
                energy_hits.append(name)
            if schema in ("Vessel", "Ship"):
                vessel_hits.append(name)

        signals = []
        if energy_hits:
            signals.append({
                "signal":    "ENERGY SANCTIONS",
                "entities":  energy_hits[:5],
                "impact":    "XLE, USO, BNO — energy sector exposure",
            })
        if vessel_hits:
            signals.append({
                "signal":    "VESSEL DESIGNATIONS",
                "entities":  vessel_hits[:5],
                "impact":    "BDRY, ZIM, SBLK — shipping disruption risk",
            })

        return {
            "source":         "OpenSanctions [OFAC + EU + UN + UK + 100 lists]",
            "period_days":    days,
            "count":          len(designations),
            "designations":   designations,
            "market_signals": signals,
            "timestamp":      datetime.now(timezone.utc).isoformat(),
        }


# =============================================================================
# §8  MACRO LAYER — USGS + NOAA + NASA FIRMS
# =============================================================================

class USGSSeismicMonitor:
    """
    USGS Earthquake Hazards Program — real-time global seismic feed.
    Gold standard. Free. No registration.
    Updated continuously (~1-minute latency for large events).

    Market impact:
      ≥M6.0 → insurance losses (PGR, ALL, TRV, CB), construction
      ≥M7.0 → critical infrastructure risk, re-insurance (MKL, RNR)
      Tsunami warning → ports, coastal supply chain disruption
    """

    @classmethod
    def fetch(cls, min_mag: float = 4.5, days: int = 7) -> Dict:
        start = (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%d")

        data = _get(
            url=CONFIG.USGS_BASE,
            cache_key=f"usgs_{min_mag}_{days}d",
            ttl=1800,
            params={
                "format": "geojson", "orderby": "magnitude",
                "starttime": start,
                "endtime": datetime.utcnow().strftime("%Y-%m-%d"),
                "minmagnitude": min_mag,
            },
            source="usgs",
            timeout=15,
        )

        if not data:
            return {"earthquakes": [], "count": 0, "max_magnitude": 0}

        quakes = []
        for feat in (data.get("features") or [])[:30]:
            p  = feat.get("properties") or {}
            c  = (feat.get("geometry") or {}).get("coordinates") or [0, 0, 0]
            mag = _safe_float(p.get("mag")) or 0.0
            quakes.append({
                "magnitude":  round(mag, 1),
                "place":      p.get("place") or "Unknown",
                "time":       datetime.utcfromtimestamp(
                              (p.get("time") or 0) / 1000).isoformat(),
                "depth_km":   round(_safe_float(c[2] if len(c) > 2 else 0), 1),
                "latitude":   round(_safe_float(c[1] if len(c) > 1 else 0), 4),
                "longitude":  round(_safe_float(c[0]), 4),
                "tsunami":    bool(p.get("tsunami")),
                "alert":      p.get("alert"),
                "sig":        int(p.get("sig") or 0),
            })

        max_mag = max((q["magnitude"] for q in quakes), default=0.0)
        tsunamis = [q for q in quakes if q["tsunami"]]

        impact = ("CRITICAL" if max_mag >= 8.0 else
                  "HIGH"     if max_mag >= 7.0 else
                  "MODERATE" if max_mag >= 5.5 else "LOW")

        alerts = []
        if tsunamis:
            alerts.append(
                f"🌊 TSUNAMI WARNING — {len(tsunamis)} event(s): "
                + ", ".join(q["place"] for q in tsunamis[:3]))
        if max_mag >= 7.0:
            top = quakes[0]
            alerts.append(
                f"🔴 M{top['magnitude']} — {top['place']} "
                f"({top['depth_km']}km depth)")

        return {
            "source":          "USGS Earthquake Hazards Program",
            "earthquakes":     quakes,
            "count":           len(quakes),
            "max_magnitude":   max_mag,
            "tsunami_warnings":tsunamis,
            "market_impact":   impact,
            "alerts":          alerts,
            "affected_sectors":(["insurance", "construction", "utilities",
                                  "real_estate"] if max_mag >= 5.5 else []),
        }


class NOAAWeatherMonitor:
    """
    NOAA/NWS Official Weather Alerts — US extreme weather. Free, no key.
    Updated in real-time. Covers: hurricanes, tornadoes, floods, blizzards.

    Market relevance:
      Hurricane → insurance (PGR, ALL, TRV), utilities (PCG, EIX), REITs
      Drought    → agriculture (WEAT, CORN, DBA, ADM, BG)
      Flood      → infrastructure, flood insurance (NFIP exposure)
    """

    @classmethod
    def fetch(cls, severity: str = "Extreme,Severe") -> Dict:
        data = _get(
            url=CONFIG.NOAA_ALERTS,
            cache_key=f"noaa_{severity.replace(',','_')}",
            ttl=900,
            params={"severity": severity, "status": "actual"},
            headers={"Accept": "application/geo+json"},
            source="noaa",
        )

        alerts_raw = []
        for feat in (data or {}).get("features") or []:
            p = feat.get("properties") or {}
            alerts_raw.append({
                "event":       p.get("event") or "",
                "severity":    p.get("severity") or "",
                "urgency":     p.get("urgency") or "",
                "headline":    (p.get("headline") or "")[:180],
                "area":        (p.get("areaDesc") or "")[:120],
                "effective":   p.get("effective") or "",
                "expires":     p.get("expires") or "",
            })

        hurricanes = [a for a in alerts_raw if "Hurricane" in a["event"]]
        floods     = [a for a in alerts_raw if "Flood" in a["event"]]
        droughts   = [a for a in alerts_raw if "Drought" in a["event"]]
        tornadoes  = [a for a in alerts_raw if "Tornado" in a["event"]]

        impact_signals = []
        if hurricanes:
            impact_signals.append({
                "type": "Hurricane", "count": len(hurricanes),
                "instruments": "PGR, ALL, TRV, CB — catastrophe insurance exposure",
            })
        if floods:
            impact_signals.append({
                "type": "Flood", "count": len(floods),
                "instruments": "ADM, BG, WEAT, CORN — agriculture; infrastructure",
            })

        return {
            "source":        "NOAA/NWS Official Weather Alerts",
            "total_alerts":  len(alerts_raw),
            "hurricanes":    len(hurricanes),
            "floods":        len(floods),
            "droughts":      len(droughts),
            "tornadoes":     len(tornadoes),
            "alerts":        alerts_raw[:20],
            "impact_signals":impact_signals,
            "timestamp":     datetime.now(timezone.utc).isoformat(),
        }


class NASAFIRMSMonitor:
    """
    NASA FIRMS — VIIRS/MODIS Satellite Thermal Anomaly Detection.
    Global coverage at 375m resolution, ~3-hour latency.
    Free API key at firms.modaps.eosdis.nasa.gov/api/

    Thermal hotspot intelligence:
      Conflict zones:   military vehicle concentration, destroyed equipment
      Industrial fires: supply chain disruption signal
      Wildfires:        insurance/utilities/agriculture impact
      Crop burns:       harvest-season commodity signal
    """

    @classmethod
    def get_status(cls) -> Dict:
        return {
            "source":           "NASA FIRMS (VIIRS SNPP + MODIS Terra/Aqua)",
            "resolution":       "375 metres (VIIRS) / 500 metres (MODIS)",
            "update_frequency": "Every 3 hours",
            "global_coverage":  True,
            "api_registration": "firms.modaps.eosdis.nasa.gov/api/ [free]",
            "intelligence_value": {
                "conflict_zones":   "Industrial-scale fires → military operations",
                "wildfires_west_us":"PCG, EIX, SRE + PGR, ALL, TRV",
                "industrial_fires": "Sector-specific supply chain disruption",
                "crop_burns":       "DBA, WEAT, CORN, SOYB — agriculture signal",
            },
        }


# =============================================================================
# §9  RISK SYNTHESIS ENGINE
# =============================================================================

# Sector → representative tradable instruments
SECTOR_INSTRUMENTS: Dict[str, str] = {
    "defense":     "ITA, XAR, LMT, RTX, NOC, GD, BA, KTOS",
    "energy":      "XLE, USO, BNO, XOM, CVX, VLO, MPC",
    "gold":        "GLD, IAU, GDX, GDXJ — safe-haven / crisis hedge",
    "bonds":       "TLT, IEF, SHY — flight-to-safety",
    "currency":    "UUP (USD), FXY (JPY), FXF (CHF) — safe havens",
    "em":          "EEM, FXI, VWO, EWZ — risk-off impact",
    "agri":        "DBA, WEAT, CORN, SOYB — commodity supply shock",
    "tech":        "XLK, SOXX, SMH — supply chain / sanctions",
    "insurance":   "PGR, ALL, TRV, CB, MKL — CAT risk",
    "shipping":    "BDRY, ZIM, SBLK, MATX, GOGL — trade flows",
    "pharma":      "XPH, IBB, XBI — pandemic / biodefense",
    "uranium":     "URA, CCJ, SRUUF — nuclear escalation signal",
    "utilities":   "XLU, PCG, EIX, SRE — climate / disaster exposure",
    "volatility":  "VXX, UVXY, VIXM — tail risk hedge",
}


class RiskSynthesisEngine:
    """
    Composite Geopolitical Risk Score (0–100).

    Methodology:
      Component 1 — Military Posture    (weight 35%)
        Aircraft count, HVI aircraft, naval assets, carrier deployments
      Component 2 — Conflict Intensity  (weight 35%)
        ACLED events/fatalities, critical sub-events (air strikes, missiles)
      Component 3 — News Sentiment      (weight 15%)
        Inverted sentiment score from Event Registry / GDELT
      Component 4 — Natural Disaster    (weight 15%)
        USGS magnitude, NOAA alert severity

    Output:
      Composite score → risk level (LOW / MODERATE / ELEVATED / HIGH / CRITICAL)
      Top-5 impacted sectors with representative instruments
      Prioritized trading signals (HIGH / MODERATE priority)
      Data confidence metric (sources active / total)
      Alert queue from all active intelligence layers
    """

    @classmethod
    def compute(cls, aircraft: Dict, naval: Dict, conflicts: Dict,
                news: Dict, seismic: Dict, weather: Dict) -> Dict:

        components: Dict[str, float] = {}

        # ── Component 1: Military Posture ─────────────────────────────────
        mil_count  = aircraft.get("military_count") or 0
        hvi_count  = aircraft.get("hvi_count")       or 0
        nav_count  = naval.get("naval_count")         or 0
        carriers   = len(naval.get("carriers") or [])

        c1 = min(100,
            mil_count  * 0.20 +
            hvi_count  * 4.00 +   # HVI aircraft = high-signal events
            nav_count  * 0.40 +
            carriers   * 12.0 +   # Carrier deployment = major signal
            len(aircraft.get("alerts") or []) * 3.0 +
            len(naval.get("alerts")   or []) * 3.0
        )
        components["military_posture"] = round(c1, 1)

        # ── Component 2: Conflict Intensity ───────────────────────────────
        intensity    = (conflicts.get("intensity_score") or
                        (conflicts.get("conflict_intensity") or {}).get("score") or 0)
        critical_n   = len(conflicts.get("critical_events") or [])
        c2 = min(100, float(intensity) + critical_n * 2.5)
        components["conflict_intensity"] = round(c2, 1)

        # ── Component 3: News Sentiment ───────────────────────────────────
        avg_sent = float(news.get("avg_sentiment") or 0)
        # Normalize: very negative sentiment → higher risk score
        # Event Registry: [-1, 1]; GDELT: [-10, 10] approx
        if abs(avg_sent) > 2:   # GDELT scale
            c3 = min(100, max(0, (-avg_sent / 10 + 0.5) * 80))
        else:                   # Event Registry scale
            c3 = min(100, max(0, (-avg_sent + 0.5) * 60))
        components["news_sentiment"] = round(c3, 1)

        # ── Component 4: Natural Disaster ─────────────────────────────────
        max_mag = float(seismic.get("max_magnitude") or 0)
        eq_cnt  = int(seismic.get("count") or 0)
        w_alerts= int(weather.get("total_alerts") or 0)
        c4 = min(100, max(0, (max_mag - 4.0)) * 14 + eq_cnt * 1.5 + w_alerts * 2.5)
        components["natural_disaster"] = round(c4, 1)

        # ── Weighted composite ─────────────────────────────────────────────
        weights = {
            "military_posture":   0.35,
            "conflict_intensity": 0.35,
            "news_sentiment":     0.15,
            "natural_disaster":   0.15,
        }
        composite = sum(components[k] * weights[k] for k in weights)
        components["composite"] = round(composite, 1)

        level = ("CRITICAL" if composite > 75 else
                 "HIGH"     if composite > 55 else
                 "ELEVATED" if composite > 35 else
                 "MODERATE" if composite > 15 else "LOW")

        # ── Sector impact vectors ──────────────────────────────────────────
        sector_scores: Dict[str, float] = defaultdict(float)
        for e in (conflicts.get("recent_events") or []):
            for s in (e.get("sectors") or []):
                sector_scores[s] += e.get("severity") or 1
        for art in (news.get("articles") or []):
            for topic in (art.get("topics") or []):
                for s in GEO_TOPICS.get(topic, {}).get("sectors", []):
                    sector_scores[s] += GEO_TOPICS[topic].get("weight", 1)
        for a in (seismic.get("affected_sectors") or []):
            sector_scores[a] += 10

        top_sectors = sorted(sector_scores.items(), key=lambda x: -x[1])[:7]

        # ── Trading signals ────────────────────────────────────────────────
        signals: List[Dict] = []

        if composite > 50:
            signals.append({
                "priority":    "HIGH",
                "action":      "INCREASE HEDGES",
                "instruments": "GLD, TLT, VXX, UVXY",
                "rationale":   (f"Composite risk {composite:.0f}/100 — "
                                "safe-haven flows historically follow this threshold"),
            })

        if c1 > 40:
            signals.append({
                "priority":    "HIGH",
                "action":      "OVERWEIGHT DEFENSE",
                "instruments": SECTOR_INSTRUMENTS["defense"],
                "rationale":   (f"Military posture score {c1:.0f} — "
                                "elevated military activity → defense sector tailwind"),
            })

        if carriers > 0:
            signals.append({
                "priority":    "CRITICAL",
                "action":      "MONITOR — CARRIER GROUP DEPLOYED",
                "instruments": SECTOR_INSTRUMENTS["defense"] + "; " + SECTOR_INSTRUMENTS["energy"],
                "rationale":   (f"{carriers} carrier strike group(s) at sea — "
                                "historically precedes or accompanies major operations"),
            })

        if c2 > 50:
            signals.append({
                "priority":    "HIGH",
                "action":      "BUY VOLATILITY",
                "instruments": SECTOR_INSTRUMENTS["volatility"],
                "rationale":   (f"Conflict intensity {c2:.0f}/100 — "
                                "VIX historically spikes 20–40% on major escalations"),
            })

        if "energy" in dict(top_sectors):
            signals.append({
                "priority":    "MODERATE",
                "action":      "MONITOR CRUDE",
                "instruments": SECTOR_INSTRUMENTS["energy"],
                "rationale":   "Energy sector elevated in impact — monitor oil supply disruption",
            })

        if c4 > 30:
            signals.append({
                "priority":    "MODERATE",
                "action":      "UNDERWEIGHT INSURANCE",
                "instruments": SECTOR_INSTRUMENTS["insurance"],
                "rationale":   "Natural disaster cluster — elevated CAT loss exposure",
            })

        # ── Aggregate alerts ───────────────────────────────────────────────
        all_alerts: List[str] = []
        all_alerts.extend(aircraft.get("alerts") or [])
        all_alerts.extend(naval.get("alerts")   or [])
        all_alerts.extend(seismic.get("alerts") or [])
        for e in (conflicts.get("critical_events") or []):
            if e.get("intel_flag"):
                all_alerts.append(
                    f"{e['intel_flag']} | {e.get('country','')} — {e.get('notes','')[:80]}")

        # ── Data confidence ────────────────────────────────────────────────
        sources_live = sum([
            1 if (aircraft.get("military_count") or 0) > 0 else 0,
            1 if (naval.get("naval_count") or 0)    > 0 else 0,
            1 if (conflicts.get("total_events") or 0) > 0 else 0,
            1,  # USGS always available
            1,  # NOAA always available
        ])
        confidence_pct = round(sources_live / 5 * 100, 0)
        confidence_lvl = "HIGH" if confidence_pct >= 80 else "MODERATE" if confidence_pct >= 60 else "LOW"

        return {
            "composite_risk":    round(composite, 1),
            "risk_level":        level,
            "components":        components,
            "top_sectors":       top_sectors,
            "sector_instruments":{s: SECTOR_INSTRUMENTS.get(s, "") for s, _ in top_sectors},
            "trading_signals":   signals,
            "alerts":            all_alerts[:20],
            "data_confidence":   {
                "score":          confidence_pct,
                "level":          confidence_lvl,
                "sources_active": sources_live,
                "sources_total":  5,
            },
            "updated":           datetime.now(timezone.utc).isoformat(),
        }


# =============================================================================
# §10  THEATER INTELLIGENCE
# =============================================================================

THEATERS = {
    "taiwan":       {
        "label":   "Taiwan Strait",
        "lat":     [21.0, 27.0], "lon": [118.0, 123.0],
        "note":    "Cross-strait military activity — semiconductor supply chain (SOXX, SMH)",
    },
    "ukraine":      {
        "label":   "Ukraine / Eastern Europe",
        "lat":     [44.0, 52.0], "lon": [22.0, 40.0],
        "note":    "Active conflict — energy (XLE), agriculture (WEAT, CORN), defense (ITA)",
    },
    "persian_gulf": {
        "label":   "Persian Gulf / Strait of Hormuz",
        "lat":     [22.0, 30.0], "lon": [48.0, 58.0],
        "note":    "20% of global oil transit — energy (XLE, USO, BNO), tanker rates",
    },
    "south_china":  {
        "label":   "South China Sea",
        "lat":     [5.0,  25.0], "lon": [105.0, 125.0],
        "note":    "$3.4T annual trade — EEM, FXI, shipping (ZIM, MATX)",
    },
    "black_sea":    {
        "label":   "Black Sea",
        "lat":     [40.0, 47.0], "lon": [27.0, 41.0],
        "note":    "Ukraine grain/oil — WEAT, CORN, energy; NATO maritime flank",
    },
    "red_sea":      {
        "label":   "Red Sea / Bab-el-Mandeb",
        "lat":     [10.0, 22.0], "lon": [32.0, 45.0],
        "note":    "Houthi threat — shipping rates (BDRY), Suez access",
    },
    "baltic":       {
        "label":   "Baltic Sea",
        "lat":     [54.0, 66.0], "lon": [10.0, 30.0],
        "note":    "NATO eastern flank — energy infrastructure (pipelines), defense",
    },
    "korea":        {
        "label":   "Korean Peninsula",
        "lat":     [34.0, 42.0], "lon": [124.0, 132.0],
        "note":    "DPRK nuclear/missile tests — defense, semiconductors, KOSPI",
    },
}


# =============================================================================
# §11  MASTER ORCHESTRATOR
# =============================================================================

class GeopoliticalIntelligencePlatform:
    """
    GIG Geopolitical Intelligence Platform v2.0

    Single entry point for all intelligence layers.
    Implements singleton with 2-minute dashboard cache.

    Data refresh cadence:
      Military aircraft:     30 seconds (ADS-B Exchange live)
      Naval vessels:         2 minutes  (AIS live)
      Conflict events:       10 minutes (ACLED)
      Geopolitical news:     15 minutes (Event Registry / GDELT)
      Seismic:               30 minutes (USGS)
      Weather:               15 minutes (NOAA/NWS)
      Sanctions:             1 hour     (OpenSanctions)
    """

    _instance: Optional["GeopoliticalIntelligencePlatform"] = None
    _lock = threading.Lock()

    def __init__(self):
        self._cache:       Optional[Dict] = None
        self._cache_ts:    float = 0.0
        self._cache_ttl:   float = 120.0  # 2 minutes
        logger.info("✅ GIG GeoPolitical Intelligence Platform v2.0 — online")

    @classmethod
    def instance(cls) -> "GeopoliticalIntelligencePlatform":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def dashboard(self, force: bool = False) -> Dict:
        """
        Full intelligence dashboard — all layers aggregated.
        Cached for 2 minutes. Pass force=True to bypass cache.
        """
        now = time.monotonic()
        if not force and self._cache and now - self._cache_ts < self._cache_ttl:
            return self._cache

        logger.info("🌐 Refreshing geopolitical intelligence dashboard...")

        # ── Fetch all layers ───────────────────────────────────────────────
        aircraft_data  = MilitaryAircraftTracker.fetch_global()
        naval_data     = NavalVesselTracker.fetch_global()
        conflict_data  = ACLEDConflictTracker.fetch(days=7)
        news_data      = EventRegistryIntelligence.fetch(hours=24)
        sanctions_data = OpenSanctionsTracker.recent_designations(days=7)
        seismic_data   = USGSSeismicMonitor.fetch(min_mag=4.5, days=7)
        weather_data   = NOAAWeatherMonitor.fetch()
        firms_data     = NASAFIRMSMonitor.get_status()

        # ── Risk synthesis ─────────────────────────────────────────────────
        risk = RiskSynthesisEngine.compute(
            aircraft_data, naval_data, conflict_data,
            news_data, seismic_data, weather_data,
        )

        # ── Theater snapshots ──────────────────────────────────────────────
        theater_ops = {}
        for tid, tcfg in list(THEATERS.items())[:4]:  # 4 most active theaters
            theater_ops[tid] = MilitaryAircraftTracker.fetch_region(
                tcfg["lat"][0], tcfg["lat"][1],
                tcfg["lon"][0], tcfg["lon"][1],
                tcfg["label"],
            )

        dash = {
            "meta": {
                "platform":  "GIG Geopolitical Intelligence v2.0",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "api_config": {
                    "adsbx_aircraft": "active" if (CONFIG.RAPIDAPI_KEY or CONFIG.ADSBX_API_KEY) else "degraded [set RAPIDAPI_KEY]",
                    "marinetraffic":  "active" if CONFIG.MARINETRAFFIC_KEY else "degraded [set MARINETRAFFIC_KEY]",
                    "aishub":         "active" if CONFIG.AISHUB_USERNAME else "not configured",
                    "acled":          "active" if CONFIG.ACLED_KEY else "fallback-GDELT [set ACLED_KEY]",
                    "event_registry": "active" if CONFIG.EVENT_REGISTRY_KEY else "fallback-GDELT [set EVENT_REGISTRY_KEY]",
                    "opensanctions":  "active [no key required]",
                    "usgs":           "active [no key required]",
                    "noaa":           "active [no key required]",
                },
            },

            # Layer 1: Kinetic
            "military_aircraft": aircraft_data,
            "naval_vessels":     naval_data,
            "theater_ops":       theater_ops,
            "chokepoints":       NavalVesselTracker.get_chokepoint_status(),

            # Layer 2: Conflict & Intelligence
            "conflict_events":    conflict_data,
            "geopolitical_news":  news_data,
            "sanctions":          sanctions_data,

            # Layer 3: Macro
            "seismic":            seismic_data,
            "weather":            weather_data,
            "satellite_firms":    firms_data,

            # Layer 4: Synthesis
            "risk":               risk,
        }

        self._cache    = dash
        self._cache_ts = now
        logger.info(
            f"✅ Dashboard refreshed — "
            f"Risk: {risk['risk_level']} ({risk['composite_risk']}/100) | "
            f"Aircraft: {aircraft_data.get('military_count', 0)} mil | "
            f"Naval: {naval_data.get('naval_count', 0)} | "
            f"Conflicts: {conflict_data.get('total_events', 0)}"
        )
        return dash

    def theater_assessment(self, theater_id: str) -> Dict:
        """
        Deep-dive assessment for a named military theater of operation.
        Combines aircraft, naval contacts, and conflict events for the region.
        """
        tcfg = THEATERS.get(theater_id.lower())
        if not tcfg:
            return {
                "error": f"Unknown theater '{theater_id}'",
                "available": list(THEATERS.keys()),
            }

        lat_min, lat_max = tcfg["lat"]
        lon_min, lon_max = tcfg["lon"]
        label = tcfg["label"]

        aircraft = MilitaryAircraftTracker.fetch_region(
            lat_min, lat_max, lon_min, lon_max, label)

        # Pull regional conflict events from ACLED
        all_conflicts = ACLEDConflictTracker.fetch(days=3)
        regional = [
            e for e in (all_conflicts.get("recent_events") or [])
            if (e.get("latitude") is not None and e.get("longitude") is not None
                and lat_min <= float(e["latitude"])  <= lat_max
                and lon_min <= float(e["longitude"]) <= lon_max)
        ]

        # Assessment text
        mil_c = aircraft.get("military_count") or 0
        hvi_c = aircraft.get("hvi_count") or 0
        n_ev  = len(regional)
        if mil_c > 50 or hvi_c > 5 or n_ev > 20:
            assessment = "⚠️  HIGH ACTIVITY — Significant military/conflict activity detected"
        elif mil_c > 20 or n_ev > 5:
            assessment = "🟡 ELEVATED — Above-normal military presence or conflict events"
        elif mil_c > 5 or n_ev > 0:
            assessment = "🟢 MONITORING — Some activity, within normal parameters"
        else:
            assessment = "⬛ QUIET — No significant activity detected"

        return {
            "theater":          label,
            "note":             tcfg["note"],
            "assessment":       assessment,
            "aircraft":         aircraft,
            "conflict_events":  regional[:20],
            "conflict_count":   n_ev,
            "timestamp":        datetime.now(timezone.utc).isoformat(),
        }


# =============================================================================
# §12  PUBLIC API
# =============================================================================

def get_platform() -> GeopoliticalIntelligencePlatform:
    return GeopoliticalIntelligencePlatform.instance()

def get_dashboard(force: bool = False) -> Dict:
    """Full geopolitical intelligence dashboard."""
    return get_platform().dashboard(force_refresh=force)

def get_risk_score() -> Dict:
    """Composite risk score only (fast — uses dashboard cache)."""
    return get_dashboard().get("risk", {})

def get_military_aircraft() -> Dict:
    """Live global military aircraft via ADS-B Exchange."""
    return MilitaryAircraftTracker.fetch_global()

def get_naval_vessels() -> Dict:
    """Live naval vessels via MarineTraffic / AISHub."""
    return NavalVesselTracker.fetch_global()

def get_conflict_events(days: int = 7) -> Dict:
    """Armed conflict events via ACLED."""
    return ACLEDConflictTracker.fetch(days=days)

def get_theater(theater_id: str) -> Dict:
    """Theater-level assessment (taiwan/ukraine/persian_gulf/etc.)."""
    return get_platform().theater_assessment(theater_id)

def search_sanctions(query: str) -> Dict:
    """Search OpenSanctions database by entity name."""
    return OpenSanctionsTracker.search(query)

def get_seismic(min_mag: float = 4.5) -> Dict:
    """Recent significant earthquakes from USGS."""
    return USGSSeismicMonitor.fetch(min_mag=min_mag)

def get_weather_alerts() -> Dict:
    """Active extreme weather alerts from NOAA/NWS."""
    return NOAAWeatherMonitor.fetch()


# =============================================================================
# §13  HELPERS
# =============================================================================

def _safe_float(v: Any) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


# =============================================================================
# §14  CLI ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  [%(levelname)-8s]  %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )

    def _bar(score: float, width: int = 30) -> str:
        filled = int(score / 100 * width)
        return "█" * filled + "░" * (width - filled)

    def _divider(title: str = "", width: int = 72) -> str:
        if title:
            pad = (width - len(title) - 2) // 2
            return "═" * pad + f" {title} " + "═" * pad
        return "═" * width

    print()
    print(_divider())
    print("  GIG — GEOPOLITICAL INTELLIGENCE PLATFORM v2.0")
    print(_divider())

    args = sys.argv[1:]
    cmd  = (args[0].lower() if args else "dashboard")

    if cmd == "aircraft":
        data = get_military_aircraft()
        print(f"\n✈  MILITARY AIRCRAFT TRACKING  [{data.get('source','')}]")
        print(f"   Total tracked:   {data.get('total_tracked', 0):,}")
        print(f"   Military:        {data.get('military_count', 0)}")
        print(f"   HVI (high-value):{data.get('hvi_count', 0)}")
        print(f"   Emergency:       {data.get('emergency_count', 0)}")
        if data.get("by_nation"):
            print(f"\n   By Nation:")
            for n, c in sorted(data["by_nation"].items(), key=lambda x: -x[1])[:6]:
                print(f"     {n:<25s} {c}")
        if data.get("type_breakdown"):
            print(f"\n   Aircraft Types:")
            for t, c in sorted(data["type_breakdown"].items(), key=lambda x: -x[1])[:8]:
                print(f"     {t:<35s} {c}")
        print(f"\n   Alerts ({len(data.get('alerts', []))}):")
        for a in data.get("alerts", []):
            print(f"     {a}")

    elif cmd == "naval":
        data = get_naval_vessels()
        print(f"\n🚢 NAVAL VESSEL TRACKING  [{data.get('source','')}]")
        print(f"   Total tracked: {data.get('total_tracked', 0):,}")
        print(f"   Naval assets:  {data.get('naval_count', 0)}")
        print(f"   HVI vessels:   {data.get('hvi_count', 0)}")
        if data.get("by_navy"):
            print(f"\n   By Navy:")
            for n, c in sorted(data["by_navy"].items(), key=lambda x: -x[1])[:8]:
                print(f"     {n:<35s} {c}")
        if data.get("by_class"):
            print(f"\n   By Hull Class:")
            for cls_name, c in sorted(data["by_class"].items(), key=lambda x: -x[1]):
                print(f"     {cls_name:<30s} {c}")
        print(f"\n   Alerts ({len(data.get('alerts', []))}):")
        for a in data.get("alerts", []):
            print(f"     {a}")

    elif cmd == "theater":
        tid = args[1] if len(args) > 1 else "taiwan"
        data = get_theater(tid)
        if "error" in data:
            print(f"\n  Error: {data['error']}")
            print(f"  Available theaters: {', '.join(data['available'])}")
        else:
            print(f"\n🎯 THEATER: {data.get('theater')}")
            print(f"   Note:       {data.get('note')}")
            print(f"   Assessment: {data.get('assessment')}")
            ac = data.get("aircraft", {})
            print(f"   Aircraft:   {ac.get('military_count', 0)} military, "
                  f"{ac.get('hvi_count', 0)} HVI")
            print(f"   Conflicts:  {data.get('conflict_count', 0)} events (72-hour window)")
            if data.get("conflict_events"):
                print(f"\n   Recent events:")
                for e in data["conflict_events"][:5]:
                    flag = e.get("intel_flag") or "  "
                    print(f"     {flag[:2]} {e.get('event_type',''):30s} {e.get('country',''):15s} "
                          f"fatalities={e.get('fatalities',0)}")

    elif cmd == "sanctions":
        query = " ".join(args[1:]) if len(args) > 1 else "Gazprom"
        data = search_sanctions(query)
        print(f"\n🚫 SANCTIONS SEARCH: '{query}'  (total hits: {data.get('total', 0)})")
        for r in data.get("results", [])[:5]:
            print(f"\n   [{r.get('schema','')}] {r.get('name','')}  (score: {r.get('score',0):.0f})")
            print(f"     Lists:    {', '.join(r.get('datasets',[])[:4])}")
            print(f"     Topics:   {', '.join(r.get('topics',[]))}")

    elif cmd == "conflicts":
        days = int(args[1]) if len(args) > 1 else 7
        data = get_conflict_events(days=days)
        print(f"\n⚔  ACLED CONFLICT FEED  [{data.get('source','')}]")
        print(f"   Period:       {days} days")
        print(f"   Total events: {data.get('total_events', 0):,}")
        print(f"   Fatalities:   {data.get('total_fatalities', 0):,}")
        intensity = data.get("intensity_score", data.get("conflict_intensity", {}).get("score", 0))
        print(f"   Intensity:    {intensity}/100 — {data.get('intensity_level','')}")
        if data.get("hotspots"):
            print(f"\n   Hotspots (by risk score):")
            for country, stats in data["hotspots"][:8]:
                print(f"     {country:<25s} score={stats['risk_score']:5.1f}  "
                      f"events={stats['events']:4d}  fatalities={stats['fatalities']:4d}")
        if data.get("critical_events"):
            print(f"\n   Critical events:")
            for e in data["critical_events"][:5]:
                print(f"     {e.get('intel_flag','')[:2]} {e.get('country',''):15s} "
                      f"{e.get('sub_type',''):35s} "
                      f"fat={e.get('fatalities',0)}")

    else:  # Full dashboard
        dash = get_dashboard()
        risk = dash["risk"]

        print(f"\n  Timestamp: {dash['meta']['timestamp']}")
        print()
        print(_divider("COMPOSITE RISK SCORE"))
        print(f"\n  {risk['risk_level']:8s}  {risk['composite_risk']:5.1f}/100")
        print(f"  {_bar(risk['composite_risk'])}")
        print()
        print("  Components:")
        for k, v in risk.get("components", {}).items():
            if k != "composite":
                print(f"    {k:<25s} {v:5.1f}/100  {_bar(v, 20)}")
        print()

        conf = risk.get("data_confidence", {})
        print(f"  Data confidence: {conf.get('level','')} "
              f"({conf.get('sources_active',0)}/{conf.get('sources_total',0)} sources live)")

        print()
        print(_divider("ACTIVE ALERTS"))
        alerts = risk.get("alerts", [])
        if alerts:
            for a in alerts[:8]:
                print(f"  {a}")
        else:
            print("  No active alerts.")

        print()
        print(_divider("TRADING SIGNALS"))
        for sig in risk.get("trading_signals", [])[:4]:
            print(f"\n  [{sig['priority']:8s}] {sig['action']}")
            print(f"    Instruments: {sig['instruments']}")
            print(f"    Rationale:   {sig['rationale']}")

        print()
        print(_divider("LAYER SUMMARY"))
        ac  = dash["military_aircraft"]
        nav = dash["naval_vessels"]
        cf  = dash["conflict_events"]
        nw  = dash["geopolitical_news"]
        sq  = dash["seismic"]
        wt  = dash["weather"]

        print(f"  ✈  Military Aircraft:  {ac.get('military_count',0):>5}  "
              f"tracked  |  HVI: {ac.get('hvi_count',0)}")
        print(f"  🚢 Naval Vessels:      {nav.get('naval_count',0):>5}  tracked")
        print(f"  ⚔  Conflict Events:    {cf.get('total_events',0):>5}  "
              f"(7-day)  |  fatalities: {cf.get('total_fatalities',0):,}")
        print(f"  📰 News Articles:      {nw.get('total_articles',0):>5}  "
              f"(24h)    |  tone: {nw.get('market_tone','N/A')}")
        print(f"  🌍 Earthquakes:        {sq.get('count',0):>5}  (7-day)  "
              f"|  max: M{sq.get('max_magnitude',0)}")
        print(f"  🌪  Weather Alerts:     {wt.get('total_alerts',0):>5}  active")

        print()
        print(_divider("API STATUS"))
        for src, status in dash["meta"]["api_config"].items():
            icon = "✅" if status.startswith("active") else "⚠️ "
            print(f"  {icon} {src:<20s} {status}")

        print()
        print(_divider())
        print("  Available commands:")
        print("    python geopolitical_engine.py aircraft")
        print("    python geopolitical_engine.py naval")
        print("    python geopolitical_engine.py theater taiwan")
        print("    python geopolitical_engine.py theater persian_gulf")
        print("    python geopolitical_engine.py theater ukraine")
        print("    python geopolitical_engine.py conflicts [days]")
        print("    python geopolitical_engine.py sanctions [entity name]")
        print(_divider())
        print()