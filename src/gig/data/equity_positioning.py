"""Pinned official equity-index futures positions, not observed stock trades.

The source reports aggregate S&P 500 E-mini futures contracts. No index price,
constituent history, cash-equity corporate action or historical publication clock
is supplied by this dataset. Acquisition is explicit; normal reads are offline.
"""
from __future__ import annotations

import hashlib
import json
import urllib.parse
import urllib.request
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

from gig.data.cftc_positions import (
    INFO_URL,
    MAX_BYTES,
    METADATA_URL,
    POLICY_URL,
    SP500_EMINI,
    _CFTCRedirect,
    normalize_rows,
)
from gig.timestamps import aware_timestamp

DATASET_ID = "cftc-tff-sp500-emini-2023"
QUERY = {"$where": "cftc_contract_market_code = '13874A' AND report_date_as_yyyy_mm_dd between '2023-01-01T00:00:00' and '2023-12-31T23:59:59'",
         "$order": "report_date_as_yyyy_mm_dd ASC", "$limit": "100"}
SOURCE_URL = "https://publicreporting.cftc.gov/resource/gpe5-46if.json?" + urllib.parse.urlencode(QUERY)
FIXTURE = Path(__file__).parents[1] / "research_org/fixtures/observed_cftc_sp500_2023.json"
PROVENANCE = FIXTURE.with_name("observed_cftc_sp500_2023.metadata.json")
RAW_SHA256 = "cdd469e84c79d21759cf858a246500123ffb90f82b1491480fe74f698d003eaf"
PROVENANCE_SHA256 = "9989daa91dd77ea4441f8120d42093ea6e560f4caf55a1793292cad5f332845c"


def _complete(raw: bytes) -> list[dict[str, Any]]:
    if len(raw) > MAX_BYTES:
        raise ValueError("equity positioning source exceeds bound")
    rows = normalize_rows(json.loads(raw), identity=SP500_EMINI)
    if len(rows) != 52 or rows[0]["report_date"] != "2023-01-03" or rows[-1]["report_date"] != "2023-12-26":
        raise ValueError("complete pinned 2023 equity positioning sample required")
    return rows


def load_equity_positions() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    raw, provenance = FIXTURE.read_bytes(), PROVENANCE.read_bytes()
    if (len(provenance) > 20000 or hashlib.sha256(raw).hexdigest() != RAW_SHA256
            or hashlib.sha256(provenance).hexdigest() != PROVENANCE_SHA256):
        raise ValueError("equity positioning evidence hash mismatch")
    meta = json.loads(provenance)
    if (meta["dataset_id"] != DATASET_ID or meta["source_url"] != SOURCE_URL
            or meta["license_url"] != POLICY_URL or meta["metadata_url"] != METADATA_URL
            or meta["information_url"] != INFO_URL or meta["contract_market_code"] != SP500_EMINI.code
            or meta["source_authenticity"] != "OFFICIAL_HTTPS_FETCH" or meta["raw_sha256"] != RAW_SHA256
            or meta["point_in_time_backtest_eligible"] is not False or meta["equity_cash_quotes_available"] is not False):
        raise ValueError("equity positioning provenance mismatch")
    if aware_timestamp(meta["acquired_at"]).date() <= date(2023, 12, 31):
        raise ValueError("explicit later acquisition clock required")
    return meta, _complete(raw)


def dataset_readiness() -> dict[str, Any]:
    try:
        meta, rows = load_equity_positions()
    except (OSError, ValueError, KeyError, TypeError):
        return {"state": "UNAVAILABLE", "dataset_id": DATASET_ID,
                "reason": "Pinned official equity positioning evidence missing or rejected; no replacement observations."}
    return {"state": "AVAILABLE", "dataset_id": DATASET_ID, "observations": len(rows),
            "sha256": RAW_SHA256, "acquired_at": meta["acquired_at"], "publication_time": "UNKNOWN",
            "point_in_time_backtest_eligible": False, "equity_cash_quotes_available": False}


def fetch_equity_positions_2023() -> tuple[bytes, dict[str, Any]]:
    """Explicit fixed-host public acquisition; never invoked by UI or worker."""
    request = urllib.request.Request(SOURCE_URL, headers={"User-Agent": "GIG-Quant-Terminal/0.1 public-research",
                                                         "Accept": "application/json"})
    try:
        with urllib.request.build_opener(_CFTCRedirect()).open(request, timeout=30) as response:
            if response.status != 200 or response.geturl() != SOURCE_URL:
                raise ValueError("equity positioning acquisition unavailable")
            raw = response.read(MAX_BYTES + 1)
            _complete(raw)
            return raw, {"status": 200, "url": response.geturl(), "acquired_at": datetime.now(UTC).isoformat(),
                         "raw_sha256": hashlib.sha256(raw).hexdigest(), "admitted_to_pinned_store": False}
    except Exception:
        raise ValueError("equity positioning acquisition unavailable") from None
