"""One explicit research cycle. Nothing runs unless something calls this."""

from __future__ import annotations

from typing import Any

from gig.research_org.experiment import HYPOTHESIS_ID, run_seeded
from gig.research_org.ledger import ResearchLedger
from gig.research_org.roles import MAY, MUST_NOT, REGISTERED_NOT_RUN, ROLES
from gig.service.contracts import Contract, State

SCHEDULER = "EXPLICIT_ONLY"


def _public(ledger: ResearchLedger, *, last: dict[str, Any] | None = None, deduplicated: bool = False) -> dict[str, Any]:
    stored = ledger.status()
    return {
        "scheduler": SCHEDULER,
        "roles": [dict(role) for role in ROLES],
        "may": list(MAY),
        "must_not": list(MUST_NOT),
        "catalog": [
            *REGISTERED_NOT_RUN,
            {"id": HYPOTHESIS_ID, "name": "Lagged-sign momentum", "module": "gig.research_org.experiment",
             "status": "EXECUTED_ON_AUTHORED_SYNTHETIC" if stored["run_count"] else "APPROVED_NOT_YET_RUN",
             "note": "The only hypothesis this cycle executes. Synthetic. Never a market result."},
        ],
        "run_count": stored["run_count"],
        "runs": stored["runs"],
        "audit": stored["audit"],
        "last": last,
        "deduplicated": deduplicated,
        "orders_submitted": 0,
        "background_thread": False,
    }


def status(ledger: ResearchLedger) -> dict[str, Any]:
    return _public(ledger)


def run_cycle(ledger: ResearchLedger) -> dict[str, Any]:
    """Run the approved hypothesis, or return the stored run if it already exists."""
    digest, report = run_seeded()
    existing = ledger.find(digest)
    if existing is not None:
        ledger.note("orchestrator", "deduplicate", digest)
        return _public(ledger, last=existing["report"], deduplicated=True)
    ledger.append(digest, report, actor="quant+skeptic")
    return _public(ledger, last=report, deduplicated=False)


def status_contract(ledger: ResearchLedger) -> Contract:
    data = status(ledger)
    return Contract.admit(State.AVAILABLE, data, source="research-organization", classification="SYNTHETIC",
                          as_of=None, scope="research-org", evidence_state="RESEARCH_OUTPUT",
                          readiness="RESEARCH_ONLY")


def cycle_contract(ledger: ResearchLedger) -> Contract:
    data = run_cycle(ledger)
    return Contract.admit(State.AVAILABLE, data, source="research-organization", classification="SYNTHETIC",
                          as_of=None, scope="research-org", evidence_state="RESEARCH_OUTPUT",
                          readiness="RESEARCH_ONLY")
