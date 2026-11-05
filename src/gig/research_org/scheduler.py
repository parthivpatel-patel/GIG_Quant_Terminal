"""Drain the approved backlog one job at a time, then stop.

Nothing in here starts a thread. A crash leaves a job ``RUNNING``; the next
tick returns it to the queue and runs it again. A finished configuration is
not recomputed.
"""

from __future__ import annotations

from typing import Any

from gig.research_org.backlog import APPROVED
from gig.research_org.cycle import status
from gig.research_org.experiment import run_seeded
from gig.research_org.futures_experiment import run_futures
from gig.research_org.ledger import ResearchLedger

RUNNERS = {"lagged_sign": run_seeded, "futures_tsmom": run_futures}
MAX_ATTEMPTS = 3


def tick(ledger: ResearchLedger) -> dict[str, Any]:
    claimed = ledger.recover_and_claim(APPROVED)
    if claimed is None:
        payload = status(ledger)
        payload["tick"] = {"action": "idle", "job_id": None, "deduplicated": False}
        payload["deduplicated"] = False
        return payload
    runner = RUNNERS.get(str(claimed["runner"]))
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
        ledger.append(digest, report, actor=str(claimed["runner"]))
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
