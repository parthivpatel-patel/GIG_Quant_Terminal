"""Approved jobs. The list is code, not a chat session."""

from __future__ import annotations

from typing import Any

from gig.research_org.experiment import HYPOTHESIS_ID as LAGGED_ID
from gig.research_org.futures_experiment import HYPOTHESIS_ID as FUTURES_ID
from gig.research_org.ledger import ResearchLedger
from gig.research_org.treasury_experiment import HYPOTHESIS_ID as TREASURY_ID

SEED: tuple[dict[str, Any], ...] = (
    {"seq": 0, "job_id": "job-lagged-sign-v1", "hypothesis_id": LAGGED_ID, "runner": "lagged_sign"},
    {"seq": 1, "job_id": "job-futures-tsmom-v1", "hypothesis_id": FUTURES_ID, "runner": "futures_tsmom"},
)
CANDIDATE: dict[str, Any] = {
    "seq": 2, "job_id": "job-treasury-10y-2023", "hypothesis_id": TREASURY_ID, "runner": "treasury_yield",
}
APPROVED: tuple[dict[str, Any], ...] = (*SEED, CANDIDATE)
SCHEDULER = "BACKLOG_TICK"
RESUME_COMMAND = (
    "python -m gig.research_org --worker --cadence-seconds 5 "
    "--state-dir <GIG_DATA_DIR>/terminal/research-org"
)


def queue_view(ledger: ResearchLedger) -> list[dict[str, Any]]:
    stored = {row["job_id"]: row for row in ledger.jobs()}
    view = []
    for job in SEED:
        row = stored.get(job["job_id"])
        view.append({
            "job_id": job["job_id"],
            "hypothesis_id": job["hypothesis_id"],
            "status": row["status"] if row else "QUEUED",
            "attempts": int(row["attempts"]) if row else 0,
            "error": row["error"] if row else None,
        })
    for extra in stored.values():
        if extra["job_id"] in {job["job_id"] for job in SEED}:
            continue
        view.append({
            "job_id": extra["job_id"],
            "hypothesis_id": extra["hypothesis_id"],
            "status": extra["status"],
            "attempts": int(extra["attempts"]),
            "error": extra["error"],
        })
    return view
