"""Available historical equity-index futures aggregate positions, never trades."""
from __future__ import annotations

from typing import Any

from gig.data.cftc_positions import CATEGORIES, SP500_EMINI, analyze, canonical_date
from gig.data.equity_positioning import load_equity_positions
from gig.service.contracts import Contract, State


def equity_positioning_contract(report_through: str | None = None) -> Contract:
    if report_through is not None:
        canonical_date(report_through)
    try:
        source, observations = load_equity_positions()
    except (OSError, ValueError, KeyError, TypeError):
        return Contract.admit(State.UNAVAILABLE, None, source="CFTC", classification="UNAVAILABLE",
                              scope="market-flow-equity-positioning", evidence_state="DATASET_UNAVAILABLE_OR_REJECTED",
                              readiness="RESEARCH_ONLY")
    rows = analyze(observations, report_through)
    if not rows:
        return Contract.admit(State.UNAVAILABLE, None, source="CFTC", classification="UNAVAILABLE",
                              scope="market-flow-equity-positioning", evidence_state="NO_REPORTS_IN_SELECTED_OBSERVATION_WINDOW",
                              readiness="RESEARCH_ONLY")
    data: dict[str, Any] = {
        "title": "CFTC weekly E-mini S&P 500 positioning", "state": "AVAILABLE", "classification": "HISTORICAL",
        "source": source, "instrument": {"contract_market_code": SP500_EMINI.code, "market_name": SP500_EMINI.market,
                                          "contract_units": SP500_EMINI.units, "asset_class": "EQUITY_INDEX_FUTURES"},
        "freshness": {"state": "STALE", "latest_report_date": rows[-1]["report_date"],
                      "scope": "Fixed 2023 historical sample; not current position coverage", "observation_precision": "DATE_ONLY"},
        "rows": rows, "categories": [item[0] for item in CATEGORIES],
        "summary": {"observations": len(rows), "first_report_date": rows[0]["report_date"], "last_report_date": rows[-1]["report_date"]},
        "report_through": report_through, "historical_knowledge_eligible": False,
        "publication_time": "UNKNOWN", "revision_knowledge_time": source["acquired_at"],
        "cash_equity_observations": "UNAVAILABLE", "corporate_actions": "NOT_APPLICABLE_TO_AGGREGATE_POSITION_COUNTS",
        "equity_total_return_research_eligible": False, "orders_submitted": 0, "execution_enabled": False,
        "notes": ["Official aggregate equity-index futures positions; not cash-equity prices, options prints, executed trades, fund flows or dark-pool transactions.",
                  "Contract units are source labels. No index-level price or notional exposure is inferred from position counts.",
                  "Net-contract changes describe positions. Percentiles use strictly earlier report observations, with at least eight required; neither is an approved signal.",
                  "Historical publication/revision vintages, constituent identity and cash-equity corporate actions are unavailable; no point-in-time trading eligibility.",
                  "Available historical snapshot; nested freshness is STALE for current-market use. The terminal never fetches a provider or submits an order."],
    }
    return Contract.admit(State.AVAILABLE, data, source="CFTC official TFF equity-index futures report",
                          classification="HISTORICAL", as_of=source["acquired_at"], scope="market-flow-equity-positioning",
                          source_identity=source["raw_sha256"], instrument_identity="CFTC:13874A:TFF:FUTURES_ONLY",
                          evidence_state="OFFICIAL_HISTORICAL_AGGREGATE_POSITIONS", readiness="RESEARCH_ONLY")
