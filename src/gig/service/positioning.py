"""Read-only observed positioning snapshot; no order or provider connection."""
from __future__ import annotations

from typing import Any

from gig.data.cftc_positions import (
    CATEGORIES,
    CONTRACT_CODE,
    MARKET,
    UNITS,
    analyze,
    canonical_date,
    load_positions,
)
from gig.service.contracts import Contract, State


def positioning_contract(report_through: str | None = None) -> Contract:
    if report_through is not None:
        canonical_date(report_through)
    try:
        source, observations = load_positions()
    except (OSError, ValueError, KeyError, TypeError):
        return Contract.admit(State.UNAVAILABLE, None, source="CFTC", classification="UNAVAILABLE", scope="market-flow-positioning",
                              evidence_state="DATASET_UNAVAILABLE_OR_REJECTED", readiness="RESEARCH_ONLY")
    rows = analyze(observations, report_through)
    if not rows:
        return Contract.admit(State.UNAVAILABLE, None, source="CFTC", classification="UNAVAILABLE", scope="market-flow-positioning",
                              evidence_state="NO_REPORTS_IN_SELECTED_OBSERVATION_WINDOW", readiness="RESEARCH_ONLY")
    data: dict[str, Any] = {
        "title": "CFTC weekly Euro FX positioning", "state": "AVAILABLE", "classification": "HISTORICAL",
        "source": source, "instrument": {"contract_market_code": CONTRACT_CODE, "market_name": MARKET, "contract_units": UNITS},
        "freshness": {"state": "STALE", "latest_report_date": rows[-1]["report_date"],
                      "scope": "Fixed 2023 historical sample; not current position coverage", "observation_precision": "DATE_ONLY"},
        "rows": rows, "categories": [item[0] for item in CATEGORIES],
        "summary": {"observations": len(rows), "first_report_date": rows[0]["report_date"], "last_report_date": rows[-1]["report_date"]},
        "report_through": report_through, "historical_knowledge_eligible": False,
        "publication_time": "UNKNOWN", "revision_knowledge_time": source["acquired_at"],
        "orders_submitted": 0, "execution_enabled": False,
        "notes": ["CFTC published aggregate futures positions; not individual trades, options prints, dealer gamma or dark-pool flow.",
                  "Changes in net contracts are descriptive position changes, not inferred fund flows or buy/sell transactions.",
                  "Percentiles compare strictly earlier report observations; at least eight are required. They are not approved signals.",
                  "Report dates are date-only observation labels. Historical publication/revision vintages are unavailable; no point-in-time trading eligibility.",
                  "The envelope is an available historical research snapshot. Nested freshness is STALE for current-market use; no live feed is claimed."],
    }
    return Contract.admit(State.AVAILABLE, data, source="CFTC official TFF report", classification="HISTORICAL",
                          as_of=source["acquired_at"], scope="market-flow-positioning",
                          source_identity=source["raw_sha256"], instrument_identity="CFTC:099741:TFF:FUTURES_ONLY",
                          evidence_state="OFFICIAL_HISTORICAL_AGGREGATE_POSITIONS", readiness="RESEARCH_ONLY")
