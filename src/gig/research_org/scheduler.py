"""One approved job per call.

The background worker lives in ``service.py``. A manual tick still runs a
single job. A live lease makes a second caller wait instead of double-running.
"""

from __future__ import annotations

from typing import Any

from gig.research_org.backlog import APPROVED, CANDIDATE, SEED
from gig.research_org.cycle import status
from gig.research_org.director import FOLLOWUP_JOB, propose
from gig.research_org.experiment import run_seeded
from gig.research_org.futures_experiment import run_futures
from gig.research_org.ledger import ResearchLedger
from gig.research_org.treasury_experiment import run_treasury
from gig.research_org.treasury_followup import run_followup

RUNNERS = {"lagged_sign": run_seeded, "futures_tsmom": run_futures, "treasury_yield": run_treasury,
           "treasury_reversal": run_followup}
RUNNER_BY_JOB = {job["job_id"]: job["runner"] for job in (*SEED, CANDIDATE, FOLLOWUP_JOB)}
MAX_ATTEMPTS = 3


def propose_candidate(ledger: ResearchLedger) -> str:
    """Queue the one approved historical dataset, once. Never invent a second study."""
    if int(ledger.scheduler_state()["candidates"]) >= 1:
        return propose(ledger)
    pending = [row for row in ledger.jobs() if row["job_id"] == CANDIDATE["job_id"]]
    if pending or ledger.find_hypothesis(CANDIDATE["hypothesis_id"]):
        ledger.mark_candidate()
        return "none"
    ledger.ensure_job(CANDIDATE)
    ledger.mark_candidate()
    return "added"


def tick(ledger: ResearchLedger, *, lease_owner: str | None = None,
         lease_seconds: float = 30.0) -> dict[str, Any]:
    """Execute one registered job under a renewable lease and fenced commit."""
    import threading
    import uuid

    from gig.research_org.ledger import LeaseLost

    owner = lease_owner or uuid.uuid4().hex
    acquired = (ledger.acquire_lease(owner, lease_seconds) if lease_owner is None
                else ledger.renew_lease(owner, lease_seconds))
    if not acquired:
        return _tick_busy(ledger)
    done = threading.Event()
    heartbeat = threading.Thread(target=_renew_tick_lease,
                                 args=(ledger, owner, lease_seconds, done), daemon=True,
                                 name="gig-research-lease")
    heartbeat.start()
    try:
        return _tick_under_lease(ledger, owner)
    except LeaseLost:
        return _tick_busy(ledger)
    finally:
        done.set()
        heartbeat.join()
        ledger.release_lease(owner)


def _renew_tick_lease(ledger: ResearchLedger, owner: str, seconds: float, done: Any) -> None:
    while not done.wait(min(1.0, max(0.005, seconds / 3))):
        try:
            if not ledger.renew_lease(owner, seconds):
                return
        except Exception:
            # Failure to renew is not permission to revive or commit a lease.
            return


def _tick_busy(ledger: ResearchLedger) -> dict[str, Any]:
    payload = status(ledger)
    payload["tick"] = {"action": "busy", "job_id": None, "deduplicated": False}
    payload["deduplicated"] = False
    return payload


def _tick_under_lease(ledger: ResearchLedger, owner: str) -> dict[str, Any]:
    claimed = ledger.recover_and_claim(SEED, lease_owner=owner)
    if claimed is None:
        payload = status(ledger)
        payload["tick"] = {"action": "idle", "job_id": None, "deduplicated": False}
        payload["deduplicated"] = False
        return payload
    job_id, attempt = str(claimed["job_id"]), int(claimed["attempts"])
    runner_name = claimed.get("runner") or RUNNER_BY_JOB.get(job_id)
    runner = RUNNERS.get(str(runner_name))
    if runner is None:
        ledger.complete_job(job_id, "FAILED", "unknown job", lease_owner=owner, attempt=attempt)
        payload = status(ledger)
        payload["tick"] = {"action": "failed", "job_id": job_id, "deduplicated": False}
        return payload
    try:
        digest, report = runner()
    except Exception:
        state = "FAILED" if attempt >= MAX_ATTEMPTS else "QUEUED"
        ledger.complete_job(job_id, state, "experiment failed", lease_owner=owner, attempt=attempt)
        payload = status(ledger)
        payload["tick"] = {"action": "failed", "job_id": job_id, "deduplicated": False,
                           "error": "experiment failed"}
        payload["deduplicated"] = False
        return payload
    last, deduplicated = ledger.finish_claim(job_id, digest, report, actor=str(runner_name),
                                            lease_owner=owner, attempt=attempt)
    payload = status(ledger)
    payload["last"] = last
    payload["deduplicated"] = deduplicated
    payload["tick"] = {"action": "ran", "job_id": job_id, "deduplicated": deduplicated}
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
