"""One explicit research cycle. Nothing runs unless something calls this."""

from __future__ import annotations

from typing import Any

from gig.research_org.experiment import HYPOTHESIS_ID, run_seeded
from gig.research_org.futures_experiment import HYPOTHESIS_ID as FUTURES_ID
from gig.research_org.ledger import ResearchLedger
from gig.research_org.roles import MAY, MUST_NOT, REGISTERED_NOT_RUN, ROLES
from gig.research_org.treasury_experiment import HYPOTHESIS_ID as TREASURY_ID
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
    rows.append({
        "id": TREASURY_ID, "name": "Treasury 10-year lagged sign, 2023",
        "module": "gig.research_org.treasury_experiment",
        "status": "EXECUTED_ON_OFFICIAL_YIELDS" if TREASURY_ID in executed else "APPROVED_NOT_YET_RUN",
        "note": (
            "Official daily par yields, CC0, calendar year 2023. Not a traded-futures result. "
            "The scheduler queues it once after the two synthetic jobs."
        ),
    })
    return rows


def _scheduler_public(ledger: ResearchLedger) -> dict[str, Any]:
    state = ledger.scheduler_state()
    return {
        "desired": state["desired"],
        "cadence_seconds": state["cadence_seconds"],
        "candidates": state["candidates"],
        "reason": state["reason"],
        "lease_held": ledger.lease_blocks(None),
    }


def _public(ledger: ResearchLedger, *, last: dict[str, Any] | None = None, deduplicated: bool = False) -> dict[str, Any]:
    from gig.research_org.backlog import RESUME_COMMAND, queue_view
    from gig.research_org.director import status as director_status
    stored = ledger.status()
    executed = {row["hypothesis_id"] for row in stored["runs"]}
    return {
        "scheduler": SCHEDULER,
        "scheduler_state": _scheduler_public(ledger),
        "resume_command": RESUME_COMMAND,
        "roles": [dict(role) for role in ROLES],
        "may": list(MAY),
        "must_not": list(MUST_NOT),
        "catalog": _catalog(executed),
        "backlog": queue_view(ledger),
        "objectives": stored.get("objectives", []),
        "director": director_status(ledger),
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


def _envelope_classification(data: dict[str, Any]) -> str:
    last = data.get("last") or {}
    kind = last.get("classification")
    if kind in {"HISTORICAL", "SYNTHETIC", "MODEL_ESTIMATE", "DELAYED", "SNAPSHOT"}:
        return str(kind)
    return "SYNTHETIC"


def status_contract(ledger: ResearchLedger, scheduler: dict[str, Any] | None = None) -> Contract:
    data = status(ledger)
    if scheduler is not None:
        data["scheduler_state"] = {
            "desired": scheduler["desired"],
            "cadence_seconds": scheduler["cadence_seconds"],
            "candidates": scheduler["candidates"],
            "reason": scheduler["reason"],
            "lease_held": bool(scheduler.get("lease_held")),
            "alive": bool(scheduler.get("alive")),
            "rss_kb": scheduler.get("rss_kb"),
            "cpu_seconds": scheduler.get("cpu_seconds"),
            "db_bytes": scheduler.get("db_bytes"),
        }
        data["background_thread"] = bool(scheduler.get("alive"))
    data["continuous_process"] = False
    return Contract.admit(State.AVAILABLE, data, source="research-organization", classification=_envelope_classification(data),
                          as_of=None, scope="research-org", evidence_state="RESEARCH_OUTPUT",
                          readiness="RESEARCH_ONLY")


def cycle_contract(ledger: ResearchLedger) -> Contract:
    data = run_cycle(ledger)
    return Contract.admit(State.AVAILABLE, data, source="research-organization", classification="SYNTHETIC",
                          as_of=None, scope="research-org", evidence_state="RESEARCH_OUTPUT",
                          readiness="RESEARCH_ONLY")


def tick_contract(ledger: ResearchLedger) -> Contract:
    from gig.research_org.scheduler import tick
    data = tick(ledger)
    return Contract.admit(State.AVAILABLE, data, source="research-organization",
                          classification=_envelope_classification(data),
                          as_of=None, scope="research-org", evidence_state="RESEARCH_OUTPUT",
                          readiness="RESEARCH_ONLY")


def objective_contract(ledger: ResearchLedger, text: str) -> Contract:
    from gig.research_org.scheduler import submit_objective
    return Contract.admit(State.AVAILABLE, submit_objective(ledger, text), source="research-organization",
                          classification="SYNTHETIC", as_of=None, scope="research-org",
                          evidence_state="RESEARCH_OUTPUT", readiness="RESEARCH_ONLY")
