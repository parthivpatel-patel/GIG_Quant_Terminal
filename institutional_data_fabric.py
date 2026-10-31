"""
institutional_data_fabric.py
=============================================================================
Institutional multi-provider data fabric for terminal-grade orchestration.
This module is provider-agnostic and can be used by UI, research, and
execution layers to inspect feed readiness, health, and data snapshots.
=============================================================================
"""

from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional

import requests


@dataclass(frozen=True)
class ProviderSpec:
    provider_id: str
    category: str
    transport: str
    asset_classes: List[str]
    data_domains: List[str]
    sla_target_ms: int
    auth_env_var: Optional[str] = None


class InstitutionalDataFabric:
    """Registry + health manager + lightweight snapshot aggregator."""

    def __init__(self) -> None:
        self._session = requests.Session()
        self._health_cache: Dict[str, Dict] = {}
        self._health_ttl_s = 30
        self._provider_specs: List[ProviderSpec] = [
            ProviderSpec(
                provider_id="polygon",
                category="aggregator",
                transport="rest+websocket",
                asset_classes=["equities", "options", "fx", "crypto"],
                data_domains=["ticks", "bars", "trades", "nbbo"],
                sla_target_ms=80,
                auth_env_var="POLYGON_API_KEY",
            ),
            ProviderSpec(
                provider_id="alpaca",
                category="broker+marketdata",
                transport="rest+websocket",
                asset_classes=["equities", "crypto", "options"],
                data_domains=["quotes", "bars", "orders", "positions"],
                sla_target_ms=120,
                auth_env_var="ALPACA_API_KEY",
            ),
            ProviderSpec(
                provider_id="fred",
                category="macro",
                transport="rest",
                asset_classes=["macro"],
                data_domains=["rates", "credit", "economic_series"],
                sla_target_ms=1500,
            ),
            ProviderSpec(
                provider_id="newsapi",
                category="news",
                transport="rest",
                asset_classes=["cross-asset"],
                data_domains=["headlines", "entities", "sentiment_inputs"],
                sla_target_ms=1200,
                auth_env_var="NEWS_API_KEY",
            ),
            ProviderSpec(
                provider_id="eia",
                category="satellite_proxy+energy",
                transport="rest",
                asset_classes=["commodities"],
                data_domains=["inventory", "sensor_proxies", "energy_macro"],
                sla_target_ms=2000,
                auth_env_var="EIA_API_KEY",
            ),
            ProviderSpec(
                provider_id="sec_edgar",
                category="filings",
                transport="rest",
                asset_classes=["equities", "credit"],
                data_domains=["filings", "insider", "events"],
                sla_target_ms=2200,
            ),
            ProviderSpec(
                provider_id="sentinel_hub",
                category="satellite",
                transport="rest",
                asset_classes=["alternative"],
                data_domains=["imagery", "geospatial_activity"],
                sla_target_ms=8000,
                auth_env_var="SENTINELHUB_API_KEY",
            ),
            ProviderSpec(
                provider_id="marine_ais",
                category="alternative",
                transport="rest",
                asset_classes=["shipping", "energy"],
                data_domains=["vessel_flow", "port_congestion"],
                sla_target_ms=5000,
                auth_env_var="MARINE_AIS_API_KEY",
            ),
        ]

    def _key_state(self, env_name: Optional[str]) -> Dict:
        if not env_name:
            return {"required": False, "configured": True}
        value = os.getenv(env_name, "")
        return {"required": True, "configured": bool(value), "env_var": env_name}

    def provider_manifest(self) -> Dict:
        return {
            "providers": [
                {
                    **asdict(spec),
                    "auth": self._key_state(spec.auth_env_var),
                }
                for spec in self._provider_specs
            ],
            "provider_count": len(self._provider_specs),
            "generated_at": time.time(),
        }

    def _probe(self, provider_id: str, timeout_s: float = 2.5) -> Dict:
        started = time.perf_counter()
        ok = False
        code = None
        error = None

        # Lightweight probes; no paid endpoints.
        urls = {
            "polygon": "https://api.polygon.io/v1/marketstatus/now",
            "alpaca": "https://paper-api.alpaca.markets/v2/clock",
            "fred": "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DGS10",
            "newsapi": "https://newsapi.org/v2/top-headlines?category=business&pageSize=1",
            "eia": "https://api.eia.gov/v2/petroleum/stoc/wstk/data/?length=1&api_key=DEMO_KEY",
            "sec_edgar": "https://www.sec.gov/edgar/searchedgar/companysearch.html",
            "sentinel_hub": "https://services.sentinel-hub.com/",
            "marine_ais": "https://www.marinetraffic.com/",
        }

        url = urls.get(provider_id)
        if not url:
            return {"provider_id": provider_id, "status": "unknown"}

        try:
            resp = self._session.get(url, timeout=timeout_s, headers={"User-Agent": "QuantTerminal/1.0"})
            code = int(resp.status_code)
            ok = code < 500
        except Exception as exc:  # noqa: BLE001
            error = str(exc)

        latency_ms = (time.perf_counter() - started) * 1000.0
        return {
            "provider_id": provider_id,
            "status": "ok" if ok else "degraded",
            "http_code": code,
            "latency_ms": round(latency_ms, 2),
            "error": error,
            "checked_at": time.time(),
        }

    def health_dashboard(self, force_refresh: bool = False) -> Dict:
        now = time.time()
        if not force_refresh and self._health_cache:
            if now - self._health_cache.get("_ts", 0) <= self._health_ttl_s:
                return self._health_cache.get("value", {})

        provider_ids = [spec.provider_id for spec in self._provider_specs]
        rows: List[Dict] = []
        with ThreadPoolExecutor(max_workers=min(8, len(provider_ids))) as pool:
            futures = [pool.submit(self._probe, pid) for pid in provider_ids]
            for f in as_completed(futures):
                rows.append(f.result())

        rows.sort(key=lambda x: x.get("provider_id", ""))
        ok_count = sum(1 for r in rows if r.get("status") == "ok")
        p95 = 0.0
        latencies = [r.get("latency_ms", 0.0) for r in rows if isinstance(r.get("latency_ms"), (int, float))]
        if latencies:
            latencies_sorted = sorted(latencies)
            idx = int(max(0, min(len(latencies_sorted) - 1, round(0.95 * (len(latencies_sorted) - 1)))))
            p95 = float(latencies_sorted[idx])

        dashboard = {
            "providers": rows,
            "healthy_ratio": round(ok_count / max(len(rows), 1), 3),
            "p95_latency_ms": round(p95, 2),
            "checked_at": now,
        }
        self._health_cache = {"_ts": now, "value": dashboard}
        return dashboard


_fabric: Optional[InstitutionalDataFabric] = None


def get_data_fabric() -> InstitutionalDataFabric:
    global _fabric
    if _fabric is None:
        _fabric = InstitutionalDataFabric()
    return _fabric

