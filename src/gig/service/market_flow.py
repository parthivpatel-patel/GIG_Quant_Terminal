"""Market-flow access truth and bounded authored options research.

This workspace has no observed options/order-flow feed. It reuses the existing
European options kernel on explicit analyst assumptions; its alerts are local
model threshold crossings, not unusual-market-activity detections.
"""

from __future__ import annotations

import hashlib
import math
import re
from typing import Any

from gig.options import OptionPosition, portfolio_exposure, scenario_grid
from gig.service.contracts import Contract, State
from gig.service.serialization import dumps, loads
from gig.timestamps import aware_timestamp

VERSION = "market-flow-authored-risk-v1"
IDENTITY = re.compile(r"^[A-Z0-9][A-Z0-9._:-]{0,63}$")
REQUEST_KEYS = {"source", "classification", "valuation_at", "underlying_id", "currency", "spot",
                "rate", "dividend_yield", "contracts", "spot_shocks", "limits"}
CONTRACT_KEYS = {"contract_id", "strike", "right", "expiry", "quantity", "multiplier", "volatility"}
LIMIT_KEYS = {"max_abs_delta", "max_abs_vega", "max_scenario_loss"}


def _number(value: Any, low: float, high: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("explicit bounded numeric assumptions required")
    try:
        result = float(value)
    except OverflowError:
        raise ValueError("numeric assumption outside bounds") from None
    if not math.isfinite(result) or not low <= result <= high:
        raise ValueError("numeric assumption outside bounds")
    return result


def _identity(value: Any) -> str:
    if not isinstance(value, str) or IDENTITY.fullmatch(value) is None:
        raise ValueError("bounded explicit instrument identity required")
    return value


def _fields(value: Any, keys: set[str]) -> None:
    if not isinstance(value, dict) or set(value) != keys:
        raise ValueError("explicit supported fields required")


def capability_register() -> list[dict[str, Any]]:
    """Detached access evidence, not inferred licensing or provider connectivity."""
    rows: list[dict[str, Any]] = [
        {"id": "modeled_options_risk", "name": "Authored options scenarios and threshold alerts",
         "state": "AVAILABLE", "classification": "MODEL_ESTIMATE", "provider": "Local GIG European BSM kernel",
         "official_url": None, "cost": "$0 local CPU", "access_terms": "Repository MIT code; caller-authored assumptions only",
         "delay": "NOT_APPLICABLE", "historical_coverage": "NONE", "redistribution": "No vendor data included",
         "reason": "Explicit hypothetical inputs. No observed trades, holdings, open interest or trading approval."},
        {"id": "observed_options_flow", "name": "Unusual activity, options prints, sweeps and open interest",
         "state": "UNAVAILABLE", "classification": "UNAVAILABLE", "provider": "NONE_ADMITTED",
         "official_url": None, "cost": "UNKNOWN", "access_terms": "No approved licensed zero-cost source admitted",
         "delay": "UNKNOWN", "historical_coverage": "UNKNOWN", "redistribution": "UNKNOWN",
         "reason": "No OPRA or other options-print entitlement. Gamma exposure from dealer positioning is UNKNOWN."},
        {"id": "finra_off_exchange", "name": "FINRA aggregate off-exchange research",
         "state": "UNAVAILABLE", "classification": "UNAVAILABLE", "provider": "FINRA",
         "official_url": "https://otctransparency.finra.org/", "cost": "UNKNOWN_USE_TERMS",
         "access_terms": "Copyrighted; acceptance of FINRA terms required; no agreement granted here",
         "delay": "UNKNOWN", "historical_coverage": "NOT_ADMITTED", "redistribution": "NOT_AUTHORIZED",
         "reason": "BLOCKED_TERMS. Public website access does not authorize data reuse. No file downloaded."},
        {"id": "sec_insiders", "name": "SEC Forms 3, 4 and 5 insider disclosures",
         "state": "UNAVAILABLE", "classification": "UNAVAILABLE", "provider": "U.S. SEC",
         "official_url": "https://www.sec.gov/data-research/sec-markets-data/insider-transactions-data-sets",
         "cost": "$0 published source; access blocked", "access_terms": "Source-specific conditions; genuine contact required for EDGAR requests",
         "delay": "FILING_NOT_TRANSACTION_TIME", "historical_coverage": "Catalog lists quarters; none admitted",
         "redistribution": "NOT_VERIFIED", "reason": "BLOCKED_ACCESS. Prior official archive requests returned 403; no bypass attempted."},
        {"id": "congress_disclosures", "name": "Congressional transaction disclosures",
         "state": "UNKNOWN", "classification": "UNKNOWN", "provider": "NOT_VERIFIED",
         "official_url": None, "cost": "UNKNOWN", "access_terms": "Not verified",
         "delay": "UNKNOWN", "historical_coverage": "UNKNOWN", "redistribution": "UNKNOWN",
         "reason": "No source has been admitted. Disclosure dates must never be confused with trade dates."},
    ]
    return rows


def _admit(payload: dict[str, Any]) -> tuple[dict[str, Any], list[OptionPosition]]:
    _fields(payload, REQUEST_KEYS)
    if (payload["source"] != "USER_AUTHORED_ASSUMPTIONS" or payload["classification"] != "SYNTHETIC"
            or payload["currency"] != "USD"):
        raise ValueError("only explicit authored USD assumptions supported")
    document = loads(dumps(payload))
    underlying = _identity(document["underlying_id"])
    valued = aware_timestamp(document["valuation_at"])
    document["valuation_at"] = valued.isoformat()
    document["spot"] = _number(document["spot"], 0.000001, 1e7)
    for key in ("rate", "dividend_yield"):
        document[key] = _number(document[key], -1, 1)
    contracts = document["contracts"]
    if not isinstance(contracts, list) or not 1 <= len(contracts) <= 32:
        raise ValueError("one to 32 authored contracts required")
    positions, identities, terms, expiries = [], set(), set(), set()
    for item in contracts:
        _fields(item, CONTRACT_KEYS)
        identity = _identity(item["contract_id"])
        expiry = aware_timestamp(item["expiry"])
        duration = (expiry.value - valued.value) / (365 * 86400 * 1_000_000_000)
        if not 1e-8 <= duration <= 30:
            raise ValueError("expiry must follow valuation within 30 ACT/365F years")
        item["expiry"] = expiry.isoformat()
        expiries.add(item["expiry"])
        item["strike"] = _number(item["strike"], 0.000001, 1e7)
        if item["right"] not in ("CALL", "PUT"):
            raise ValueError("explicit CALL or PUT required")
        for key, bound in (("quantity", 10000), ("multiplier", 10000)):
            value = item[key]
            if isinstance(value, bool) or not isinstance(value, int) or abs(value) > bound:
                raise ValueError("bounded exact integer quantities and multipliers required")
            if (key == "quantity" and value == 0) or (key == "multiplier" and value <= 0):
                raise ValueError("nonzero quantities and positive multipliers required")
        item["volatility"] = _number(item["volatility"], 0.000001, 5)
        economic = (item["expiry"], item["strike"], item["right"])
        if identity in identities or economic in terms:
            raise ValueError("duplicate contract identity or economic terms")
        identities.add(identity)
        terms.add(economic)
        positions.append(OptionPosition(identity, underlying, item["strike"], duration,
            item["volatility"], item["quantity"], item["multiplier"], item["right"] == "CALL",
            "USD", document["source"]))
    if len(expiries) != 1:
        raise ValueError("common expiry required for the bounded payoff scanner")
    shocks = document["spot_shocks"]
    if not isinstance(shocks, list) or not 2 <= len(shocks) <= 41:
        raise ValueError("two to 41 explicit spot shocks required")
    document["spot_shocks"] = [_number(value, -0.95, 3) for value in shocks]
    if len(set(document["spot_shocks"])) != len(shocks):
        raise ValueError("duplicate spot shocks")
    document["spot_shocks"].sort()
    _fields(document["limits"], LIMIT_KEYS)
    for key in LIMIT_KEYS:
        document["limits"][key] = _number(document["limits"][key], 0, 1e15)
    document["contracts"].sort(key=lambda item: item["contract_id"])
    positions.sort(key=lambda position: position.instrument)
    return document, positions


def market_flow_contract(payload: dict[str, Any] | None = None) -> Contract:
    """Registry on GET; reproducible authored scenario report on explicit POST."""
    data: dict[str, Any] = {
        "title": "Market Flow Intelligence", "version": VERSION, "capabilities": capability_register(),
        "analysis": None, "observed_flow_state": "UNAVAILABLE", "orders_submitted": 0,
        "execution_enabled": False, "funded_accounting_claim": False,
        "notes": ["No options prints, dark-pool transactions, real-time sweeps or insider filings are connected.",
                  "Model alerts are scenario threshold crossings, not evidence of institutional activity."],
    }
    if payload is not None:
        document, positions = _admit(payload)
        exposure = portfolio_exposure(positions, spot=document["spot"], rate=document["rate"], div=document["dividend_yield"])
        scenarios = scenario_grid(positions, spot=document["spot"], spot_shocks=document["spot_shocks"],
            volatility_shocks=[0.0], rate=document["rate"], div=document["dividend_yield"])
        if exposure["state"] != "AVAILABLE" or scenarios["state"] != "AVAILABLE":
            raise ValueError("scenario exposures unavailable")
        price = float(exposure["price"])
        payoff = []
        for shock in document["spot_shocks"]:
            spot = document["spot"] * (1 + shock)
            value = math.fsum(max(0.0, spot-p.strike if p.call else p.strike-spot)
                              * p.quantity * p.multiplier for p in positions)
            payoff.append({"spot": spot, "terminal_payoff": value, "undiscounted_premium_difference": value-price})
        checks = {
            "max_abs_delta": abs(float(exposure["delta"])),
            "max_abs_vega": abs(float(exposure["vega"])),
            "max_scenario_loss": max(0.0, -min(row["model_pnl"] for row in scenarios["scenarios"])),
        }
        alerts = [{"id": key, "state": "BREACH" if checks[key] > document["limits"][key] else "WITHIN_RESEARCH_LIMIT",
                   "modeled_value": checks[key], "threshold": document["limits"][key],
                   "classification": "MODEL_ESTIMATE"} for key in sorted(LIMIT_KEYS)]
        data["analysis"] = {
            "inputs": document, "inputs_sha256": hashlib.sha256(dumps(document).encode()).hexdigest(),
            "classification": "MODEL_ESTIMATE", "input_classification": "SYNTHETIC", "exposure": exposure,
            "scenarios": scenarios, "payoff": payoff, "alerts": alerts,
            "observed_gamma_exposure": None, "dealer_positioning": "UNKNOWN", "open_interest": None,
            "assumptions": ["European cash-settled hypothetical contracts; common expiry; ACT/365F.",
                "Constant authored volatility and continuous rate/dividend yield; no observed quotes or implied-volatility inversion.",
                "Expiry premium difference ignores premium financing, fees and execution costs; it is not a return or funded ledger.",
                "Scenario losses cover only the supplied grid. Within a research limit is not pre-trade approval."],
        }
    return Contract.admit(State.AVAILABLE, data, source="gig.local-market-flow-research",
        classification="MODEL_ESTIMATE" if payload is not None else "SNAPSHOT", as_of=None,
        scope="market-flow", evidence_state="AUTHORED_MODEL_ONLY" if payload is not None else "CAPABILITY_REGISTER",
        model_identity="gig.options.EuropeanBSM", model_version=VERSION, readiness="RESEARCH_ONLY")
