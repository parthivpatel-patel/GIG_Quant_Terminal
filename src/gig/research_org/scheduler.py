"""One approved job per call.

The background worker lives in ``service.py``. A manual tick still runs a
single job. A live lease makes a second caller wait instead of double-running.
"""

from __future__ import annotations

from typing import Any

from gig.research_org.backlog import APPROVED, CANDIDATE, SEED
from gig.research_org.cycle import status
from gig.research_org.experiment import run_seeded
from gig.research_org.futures_experiment import run_futures
from gig.research_org.ledger import ResearchLedger
from gig.research_org.treasury_experiment import run_treasury

RUNNERS = {"lagged_sign": run_seeded, "futures_tsmom": run_futures, "treasury_yield": run_treasury}
RUNNER_BY_JOB = {job["job_id"]: job["runner"] for job in (*SEED, CANDIDATE)}
MAX_ATTEMPTS = 3


def propose_candidate(ledger: ResearchLedger) -> str:
    """Queue the one approved historical dataset, once. Never invent a second study."""
    if int(ledger.scheduler_state()["candidates"]) >= 1:
        return "already"
    pending = [row for row in ledger.jobs() if row["job_id"] == CANDIDATE["job_id"]]
    if pending or ledger.find_hypothesis(CANDIDATE["hypothesis_id"]):
        ledger.mark_candidate()
        return "none"
    ledger.ensure_job(CANDIDATE)
    ledger.mark_candidate()
    return "added"


def tick(ledger: ResearchLedger, *, lease_owner: str | None = None) -> dict[str, Any]:
    if ledger.lease_blocks(lease_owner):
        payload = status(ledger)
        payload["tick"] = {"action": "busy", "job_id": None, "deduplicated": False}
        payload["deduplicated"] = False
        return payload
    claimed = ledger.recover_and_claim(SEED)
    if claimed is None:
        payload = status(ledger)
        payload["tick"] = {"action": "idle", "job_id": None, "deduplicated": False}
        payload["deduplicated"] = False
        return payload
    runner_name = claimed.get("runner") or RUNNER_BY_JOB.get(str(claimed["job_id"]))
    runner = RUNNERS.get(str(runner_name))
    if runner is None:
        ledger.complete_job(str(claimed["job_id"]), "FAILED", "unknown job")
        payload = status(ledger)
        payload["tick"] = {"action": "failed", "job_id": claimed["job_id"], "deduplicated": False}
        return payload
    try:
        digest, report = runner()
    except Exception:
        attempts = int(claimed["attempts"])
        state = "FAILED" if attempts >= MAX_ATTEMPTS else "QUEUED"
        ledger.complete_job(str(claimed["job_id"]), state, "experiment failed")
        payload = status(ledger)
        payload["tick"] = {"action": "failed", "job_id": claimed["job_id"], "deduplicated": False,
                           "error": "experiment failed"}
        payload["deduplicated"] = False
        return payload
    existing = ledger.find(digest)
    if existing is None:
        ledger.append(digest, report, actor=str(runner_name))
        last = report
        deduplicated = False
    else:
        ledger.note("orchestrator", "deduplicate", digest)
        last = existing["report"]
        deduplicated = True
    ledger.complete_job(str(claimed["job_id"]), "DONE", None)
    payload = status(ledger)
    payload["last"] = last
    payload["deduplicated"] = deduplicated
    payload["tick"] = {"action": "ran", "job_id": claimed["job_id"], "deduplicated": deduplicated}
    return payload


def cancel_job(ledger: ResearchLedger, job_id: str) -> bool:
    known = next((job for job in APPROVED if job["job_id"] == job_id), None)
    if known is None:
        return False
    ledger.ensure_job(known)
    return ledger.cancel(job_id)


def submit_objective(ledger: ResearchLedger, text: str) -> dict[str, Any]:
    """Store an objective. Do not invent an experiment for data the store does not have."""
    cleaned = " ".join(text.split())
    if not cleaned or len(cleaned) > 500:
        raise ValueError("objective length")
    lowered = cleaned.lower()
    blocked = ("options flow", "options-flow", "dark pool", "dark-pool", "unusual options", "sweep")
    if any(term in lowered for term in blocked):
        disposition = "BLOCKED_NO_DATASET"
        detail = "No licensed options-flow or dark-pool observations are admitted. No experiment was created."
    elif any(term in lowered for term in ("futures", "time-series momentum", "tsmom")):
        disposition = "ALREADY_ON_BACKLOG"
        detail = "Futures time-series momentum is already an approved job. No second experiment was created."
    else:
        disposition = "RECORDED_NOT_EXPANDED"
        detail = "Stored. Only the approved backlog can run, and this objective did not match it."
    ledger.record_objective(cleaned, disposition, detail)
    payload = status(ledger)
    payload["objective"] = {"text": cleaned, "disposition": disposition, "detail": detail}
    payload["deduplicated"] = False
    return payload
