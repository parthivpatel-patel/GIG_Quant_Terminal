"""One explicit research cycle. Nothing runs unless something calls this."""

from __future__ import annotations

from typing import Any

from gig.research_org.experiment import HYPOTHESIS_ID, run_seeded
from gig.research_org.futures_experiment import HYPOTHESIS_ID as FUTURES_ID
from gig.research_org.ledger import ResearchLedger
from gig.research_org.roles import MAY, MUST_NOT, REGISTERED_NOT_RUN, ROLES
from gig.service.contracts import Contract, State

SCHEDULER = "BACKLOG_TICK"


def _catalog(executed: set[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in REGISTERED_NOT_RUN:
        row = dict(item)
        if item["id"] == "futures_tsmom" and FUTURES_ID in executed:
            row["status"] = "EXECUTED_ON_AUTHORED_SYNTHETIC"
            row["note"] = (
                "Ran on authored-futures-excess-v1. Synthetic. The finding holds the metrics. "
                "They are not a market result."
            )
        rows.append(row)
    rows.append({
        "id": HYPOTHESIS_ID, "name": "Lagged-sign momentum", "module": "gig.research_org.experiment",
        "status": "EXECUTED_ON_AUTHORED_SYNTHETIC" if HYPOTHESIS_ID in executed else "APPROVED_NOT_YET_RUN",
        "note": "Authored synthetic series. A positive verdict is not available.",
    })
    return rows


def _public(ledger: ResearchLedger, *, last: dict[str, Any] | None = None, deduplicated: bool = False) -> dict[str, Any]:
    from gig.research_org.backlog import RESUME_COMMAND, queue_view
    stored = ledger.status()
    executed = {row["hypothesis_id"] for row in stored["runs"]}
    return {
        "scheduler": SCHEDULER,
        "resume_command": RESUME_COMMAND,
        "roles": [dict(role) for role in ROLES],
        "may": list(MAY),
        "must_not": list(MUST_NOT),
        "catalog": _catalog(executed),
        "backlog": queue_view(ledger),
        "objectives": stored.get("objectives", []),
        "run_count": stored["run_count"],
        "runs": stored["runs"],
        "audit": stored["audit"],
        "last": last if last is not None else (stored["runs"][0]["report"] if stored["runs"] else None),
        "deduplicated": deduplicated,
        "orders_submitted": 0,
        "background_thread": False,
        "continuous_process": False,
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


def tick_contract(ledger: ResearchLedger) -> Contract:
    from gig.research_org.scheduler import tick
    return Contract.admit(State.AVAILABLE, tick(ledger), source="research-organization", classification="SYNTHETIC",
                          as_of=None, scope="research-org", evidence_state="RESEARCH_OUTPUT",
                          readiness="RESEARCH_ONLY")


def objective_contract(ledger: ResearchLedger, text: str) -> Contract:
    from gig.research_org.scheduler import submit_objective
    return Contract.admit(State.AVAILABLE, submit_objective(ledger, text), source="research-organization",
                          classification="SYNTHETIC", as_of=None, scope="research-org",
                          evidence_state="RESEARCH_OUTPUT", readiness="RESEARCH_ONLY")
