"""Bounded, deterministic research planning over explicitly approved templates.

No generated code, LLM service, credentials, network acquisition or trading. An
approval permits one registered follow-up, not an unlimited search. Reading the
Director does not start compute or mutate the ledger.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
from contextlib import closing
from datetime import UTC, datetime
from typing import Any

from gig.research_org.ledger import ResearchLedger
from gig.research_org.roles import ROLES
from gig.research_org.treasury_experiment import HYPOTHESIS_ID as PARENT

OBJECTIVE_ID = "treasury-reversal-robustness-v1"
FOLLOWUP_ID = "treasury-10y-reversal-robustness-2023-v1"
FOLLOWUP_JOB = {
    "seq": 3, "job_id": "job-treasury-reversal-robustness-v1",
    "hypothesis_id": FOLLOWUP_ID, "runner": "treasury_reversal",
}
QUESTION = "Challenge the negative Treasury persistence finding with purged reversal robustness."
DATASET_ID = "us-treasury-daily-par-yield-2023"


def approve(ledger: ResearchLedger, objective_id: str, *, revoked: bool = False) -> dict[str, Any]:
    if objective_id != OBJECTIVE_ID:
        raise ValueError("unregistered research objective")
    now = datetime.now(UTC).isoformat()
    with ledger._lock, closing(ledger._connect()) as db, db:
        db.execute("BEGIN IMMEDIATE")
        db.execute(
            "INSERT INTO director_objectives VALUES (?, ?, ?) ON CONFLICT(objective_id) "
            "DO UPDATE SET state=excluded.state, updated_at=excluded.updated_at",
            (objective_id, "REVOKED" if revoked else "APPROVED", now))
        if revoked:
            db.execute("UPDATE jobs SET status='CANCELLED',updated_at=? WHERE job_id=? AND status='QUEUED'",
                       (now, FOLLOWUP_JOB["job_id"]))
        db.execute("INSERT INTO audit(at,actor,action,detail) VALUES (?,?,?,?)",
                   (now, "director", "revoke" if revoked else "approve", objective_id))
    return status(ledger)


def _rows(ledger: ResearchLedger) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    db = ledger._read()
    if db is None:
        return [], [], []
    try:
        try:
            objectives = [dict(row) for row in db.execute("SELECT * FROM director_objectives")]
            plans = [dict(row) for row in db.execute("SELECT * FROM director_investigations ORDER BY priority DESC, hypothesis_id")]
        except sqlite3.OperationalError:
            objectives, plans = [], []
        reports = [json.loads(row[0]) for row in db.execute("SELECT payload FROM runs ORDER BY finished_at, run_id")]
        return objectives, plans, reports
    finally:
        db.close()


def propose(ledger: ResearchLedger) -> str:
    """Review prior evidence, prioritize a registered challenge, and checkpoint once."""
    objectives, _, reports = _rows(ledger)
    if not any(row["objective_id"] == OBJECTIVE_ID and row["state"] == "APPROVED" for row in objectives):
        return "not-approved"
    parent = next((r for r in reports if r["hypothesis_id"] == PARENT), None)
    if parent is None:
        return "awaiting-parent"
    if any(r["hypothesis_id"] == FOLLOWUP_ID for r in reports):
        return "exhausted"
    # A negative or inconclusive directional result warrants falsification, not
    # a claim that the opposite strategy is profitable.
    if parent.get("verdict") not in {"INCONCLUSIVE", "NEGATIVE"}:
        return "unsupported-parent"
    now = datetime.now(UTC).isoformat()
    with ledger._lock, closing(ledger._connect()) as db, db:
        db.execute("BEGIN IMMEDIATE")
        approval = db.execute("SELECT state FROM director_objectives WHERE objective_id=?", (OBJECTIVE_ID,)).fetchone()
        if approval is None or approval[0] != "APPROVED":
            return "not-approved"
        cursor = db.execute(
            "INSERT OR IGNORE INTO director_investigations VALUES (?,?,?,?,?,?)",
            (FOLLOWUP_ID, OBJECTIVE_ID, PARENT, 100,
             "Negative persistence evidence: challenge with purged reversal, training-only baseline and independent skepticism; reused-data exploratory study.", now))
        if not cursor.rowcount:
            return "already-planned"
        db.execute("INSERT OR IGNORE INTO jobs VALUES (?,?, 'QUEUED',0,NULL,?,?)",
                   (FOLLOWUP_JOB["job_id"], FOLLOWUP_ID, FOLLOWUP_JOB["seq"], now))
        db.execute("INSERT INTO audit(at,actor,action,detail) VALUES (?,?,?,?)",
                   (now, "director", "automatic-followup", FOLLOWUP_ID))
    return "added"


def status(ledger: ResearchLedger) -> dict[str, Any]:
    objectives, plans, reports = _rows(ledger)
    jobs = {row["hypothesis_id"]: row for row in ledger.jobs()}
    report_by_id = {row["hypothesis_id"]: row for row in reports}
    approval = next((row["state"] for row in objectives if row["objective_id"] == OBJECTIVE_ID), "NOT_APPROVED")
    complete = FOLLOWUP_ID in report_by_id
    objective_state = "COMPLETED" if complete else approval
    role_outputs = report_by_id.get(FOLLOWUP_ID, {}).get("role_outputs", {})
    roles = []
    for role in ROLES:
        rid = str(role["id"])
        output = role_outputs.get(rid) if isinstance(role_outputs, dict) else None
        detail = output if isinstance(output, str) else "No completed task for this approved investigation."
        if isinstance(output, dict):
            detail = output.get("output", detail)
            if isinstance(detail, dict):
                detail = f"Independent reference checks: {detail.get('verdict', 'UNKNOWN')}. {len(detail.get('checks', {}))} checks."
        state = "COMPLETED" if output else "NOT_RUN"
        if rid in {"microstructure", "options"}:
            state = "BLOCKED_NO_DATASET"
            detail = "No approved observed prints/quotes/options dataset; no experiment or liquidity claim."
        if rid == "orchestrator":
            state = "COMPLETED" if plans else "READY"
            detail = "Deduplicates the approved hypothesis, prioritizes a falsification follow-up, and pauses when its catalog is exhausted."
        roles.append({"id": rid, "name": role["name"], "status": state, "detail": detail})
    investigations = []
    for plan in plans:
        report = report_by_id.get(plan["hypothesis_id"])
        encoded = json.dumps(report, sort_keys=True, separators=(",", ":"), allow_nan=False) if report else None
        investigations.append({**plan, "status": jobs.get(plan["hypothesis_id"], {}).get("status", "UNKNOWN"),
                               "report_hash": hashlib.sha256(encoded.encode()).hexdigest() if encoded else None})
    return {
        "policy": "BOUNDED_APPROVED_TEMPLATES", "version": 1,
        "objectives": [{"objective_id": OBJECTIVE_ID, "question": QUESTION, "status": objective_state,
                        "dataset_id": DATASET_ID,
                        "reason": "One registered follow-up; server must be running. Approval does not start compute."}],
        "investigations": investigations,
        "memory": [{"hypothesis_id": r["hypothesis_id"], "verdict": r["verdict"],
                    "classification": r.get("classification", "UNKNOWN"), "economic_claim": r.get("economic_claim", "UNKNOWN")}
                   for r in reports],
        "roles": roles, "max_followups_per_objective": 1, "max_attempts_per_job": 3,
        "server_required": True, "code_generation": False, "network_acquisition": False,
        "brokerage_execution": "DISABLED", "models_promoted": 0,
    }
