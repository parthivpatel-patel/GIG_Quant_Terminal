"""Durable approval and bounded, evidence-aware planning over finite templates.

No generated code, credentials, network acquisition or trading. One approval
permits at most one investigation at the registered dataset/method version.
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
from gig.research_org.template_catalog import (
    TEMPLATES,
    ResearchTemplate,
    dataset_readiness,
    find_template,
    selection_identity,
)

# Public compatibility identifiers for the first approved Director objective.
OBJECTIVE_ID = TEMPLATES[0].objective_id
FOLLOWUP_ID = TEMPLATES[0].hypothesis_id
FOLLOWUP_JOB = TEMPLATES[0].job()
QUESTION = TEMPLATES[0].question
DATASET_ID = TEMPLATES[0].dataset_id
PARENT = TEMPLATES[0].parent


def approve(ledger: ResearchLedger, objective_id: str, *, revoked: bool = False) -> dict[str, Any]:
    template = find_template(objective_id)
    now = datetime.now(UTC).isoformat()
    with ledger._lock, closing(ledger._connect()) as db, db:
        db.execute("BEGIN IMMEDIATE")
        db.execute(
            "INSERT INTO director_objectives VALUES (?, ?, ?) ON CONFLICT(objective_id) "
            "DO UPDATE SET state=excluded.state, updated_at=excluded.updated_at",
            (objective_id, "REVOKED" if revoked else "APPROVED", now))
        if revoked:
            db.execute("UPDATE jobs SET status='CANCELLED',updated_at=? WHERE job_id=? AND status='QUEUED'",
                       (now, template.job_id))
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


def _dependency(template: ResearchTemplate, reports: list[dict[str, Any]]) -> tuple[str, str]:
    if template.parent is None:
        return "READY", "Admitted dataset and code-registered exploratory method; no parent result required."
    parent = next((r for r in reports if r["hypothesis_id"] == template.parent), None)
    if parent is None:
        return "WAITING_PARENT", "Waiting for the registered parent investigation."
    if parent.get("verdict") not in {"INCONCLUSIVE", "NEGATIVE"}:
        return "BLOCKED_PARENT", "The registered negative/inconclusive parent condition is not met."
    hits, pairs = parent.get("hits"), parent.get("pairs")
    if (isinstance(hits, bool) or isinstance(pairs, bool) or not isinstance(hits, int)
            or not isinstance(pairs, int) or not 0 <= hits < pairs / 2
            or parent.get("classification") != "HISTORICAL"):
        return "BLOCKED_PARENT", "The parent does not contain the registered negative directional evidence."
    return "READY", "Negative persistence evidence warrants a purged reversal challenge, not a profitability claim."


def _plan_detail(raw: str) -> dict[str, Any]:
    try:
        decoded = json.loads(raw)
        if isinstance(decoded, dict) and isinstance(decoded.get("reason"), str):
            return decoded
    except (TypeError, ValueError):
        pass
    return {"reason": raw, "selection_identity": None, "dataset_sha256": None}


def propose(ledger: ResearchLedger) -> str:
    """Checkpoint all eligible approved investigations, in declared priority order.

    Priorities are registered before results are observed. Rejections, exhausted
    attempts, cancellation and data revision never silently generate another run.
    """
    objectives, _, reports = _rows(ledger)
    approved = {r["objective_id"] for r in objectives if r["state"] == "APPROVED"}
    if not approved:
        return "not-approved"
    added, outcomes = 0, []
    for template in sorted(TEMPLATES, key=lambda row: (-row.priority, row.objective_id)):
        if template.objective_id not in approved:
            continue
        if any(r["hypothesis_id"] == template.hypothesis_id for r in reports):
            outcomes.append("exhausted")
            continue
        dependency, reason = _dependency(template, reports)
        if dependency != "READY":
            outcomes.append("awaiting-parent" if dependency == "WAITING_PARENT" else "unsupported-parent")
            continue
        dataset = dataset_readiness(template.dataset_id)
        identity = selection_identity(template, dataset)
        if identity is None:
            outcomes.append("blocked-dataset")
            continue
        planned = {"reason": reason, "selection_identity": identity,
                   "dataset_sha256": dataset["sha256"], "method": template.method,
                   "method_version": template.method_version}
        now = datetime.now(UTC).isoformat()
        with ledger._lock, closing(ledger._connect()) as db, db:
            db.execute("BEGIN IMMEDIATE")
            approval = db.execute("SELECT state FROM director_objectives WHERE objective_id=?", (template.objective_id,)).fetchone()
            if approval is None or approval[0] != "APPROVED":
                outcomes.append("not-approved")
                continue
            if db.execute("SELECT 1 FROM runs WHERE hypothesis_id=?", (template.hypothesis_id,)).fetchone():
                outcomes.append("exhausted")
                continue
            cursor = db.execute(
                "INSERT OR IGNORE INTO director_investigations VALUES (?,?,?,?,?,?)",
                (template.hypothesis_id, template.objective_id, template.parent or "dataset:" + template.dataset_id,
                 template.priority, json.dumps(planned, sort_keys=True, allow_nan=False), now))
            if not cursor.rowcount:
                outcomes.append("already-planned")
                continue
            db.execute("INSERT OR IGNORE INTO jobs VALUES (?,?, 'QUEUED',0,NULL,?,?)",
                       (template.job_id, template.hypothesis_id, template.seq, now))
            db.execute("INSERT INTO audit(at,actor,action,detail) VALUES (?,?,?,?)",
                       (now, "director", "automatic-followup" if template.parent else "approved-investigation",
                        template.hypothesis_id))
            added += 1
    if added:
        return "added"
    for outcome in ("already-planned", "awaiting-parent", "unsupported-parent", "blocked-dataset", "exhausted"):
        if outcome in outcomes:
            return outcome
    return "not-approved"


def _report_hash(report: dict[str, Any]) -> str:
    encoded = json.dumps(report, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode()).hexdigest()


def admit_claim(ledger: ResearchLedger, job_id: str) -> bool:
    """Recheck local admission at execution, without changing the lease protocol."""
    template = next((row for row in TEMPLATES if row.job_id == job_id), None)
    if template is None:
        return True
    _, plans, _ = _rows(ledger)
    plan = next((row for row in plans if row["hypothesis_id"] == template.hypothesis_id), None)
    if plan is None:
        return False
    dataset = dataset_readiness(template.dataset_id)
    current = selection_identity(template, dataset)
    stored = _plan_detail(plan["reason"]).get("selection_identity")
    # The old Treasury-only plan schema did not store an identity. Its runner
    # still enforces the original pinned source hash; retain restart compatibility.
    return current is not None and (stored is None or stored == current)


def admit_result(ledger: ResearchLedger, job_id: str, report: dict[str, Any]) -> bool:
    """A completed run must still match the approved source and safe output scope."""
    template = next((row for row in TEMPLATES if row.job_id == job_id), None)
    if template is None:
        return True
    if not admit_claim(ledger, job_id):
        return False
    dataset = dataset_readiness(template.dataset_id)
    claimed_dataset = report.get("dataset")
    return (isinstance(claimed_dataset, dict)
            and claimed_dataset.get("id") == template.dataset_id
            and claimed_dataset.get("sha256") == dataset.get("sha256")
            and report.get("hypothesis_id") == template.hypothesis_id
            and report.get("economic_claim") == "NONE"
            and report.get("orders_submitted") == 0 and report.get("models_promoted") == 0)


def _roles(reports: list[dict[str, Any]], planned: bool) -> list[dict[str, Any]]:
    roles = []
    for role in ROLES:
        rid = str(role["id"])
        completed = []
        for report in reports:
            outputs = report.get("role_outputs", {})
            output = outputs.get(rid) if isinstance(outputs, dict) else None
            if not output:
                continue
            detail = output.get("output") if isinstance(output, dict) else output
            if isinstance(detail, dict):
                detail = f"Independent reference checks: {detail.get('verdict', 'UNKNOWN')}. {len(detail.get('checks', {}))} checks."
            completed.append({"hypothesis_id": report["hypothesis_id"], "detail": str(detail),
                              "state": output.get("state", "COMPLETED") if isinstance(output, dict) else "COMPLETED"})
        state = "COMPLETED" if completed else "NOT_RUN"
        detail = completed[-1]["detail"] if completed else "No completed task for an approved investigation."
        if not completed and rid in {"microstructure", "options"}:
            state = "BLOCKED_NO_DATASET"
            detail = "No approved observed prints/quotes/options dataset; no experiment or liquidity claim."
        if rid == "orchestrator":
            state = "COMPLETED" if planned else "READY"
            detail = "Ranks registered approved investigations by declared priority, verifies dataset admission and dependencies, deduplicates durable identities, and pauses on exhausted work."
        roles.append({"id": rid, "name": role["name"], "status": state, "detail": detail,
                      "workflow_count": len(completed), "workflows": completed})
    return roles


def status(ledger: ResearchLedger) -> dict[str, Any]:
    objectives, plans, reports = _rows(ledger)
    approvals = {row["objective_id"]: row["state"] for row in objectives}
    jobs = {row["hypothesis_id"]: row for row in ledger.jobs()}
    report_by_id = {row["hypothesis_id"]: row for row in reports}
    plans_by_id = {row["hypothesis_id"]: row for row in plans}
    catalog = []
    datasets: dict[str, dict[str, Any]] = {}
    for template in TEMPLATES:
        dataset = datasets.setdefault(template.dataset_id, dataset_readiness(template.dataset_id))
        approval = approvals.get(template.objective_id, "NOT_APPROVED")
        complete = template.hypothesis_id in report_by_id
        dependency, reason = _dependency(template, reports)
        identity = selection_identity(template, dataset)
        plan = plans_by_id.get(template.hypothesis_id)
        if dataset["state"] != "AVAILABLE":
            dependency, reason = "BLOCKED_DATASET", dataset["reason"]
        elif plan is not None:
            recorded_identity = _plan_detail(plan["reason"]).get("selection_identity")
            if recorded_identity is not None and recorded_identity != identity:
                dependency, reason = "BLOCKED_DATASET_REVISION", "Dataset/method identity differs from the durable plan; a newly reviewed version is required."
        catalog.append({
            "objective_id": template.objective_id, "hypothesis_id": template.hypothesis_id,
            "question": template.question, "status": "COMPLETED" if complete else approval,
            "approval_state": approval, "dataset_id": template.dataset_id, "dataset": dataset,
            "dataset_state": dataset["state"], "method": template.method,
            "method_version": template.method_version, "priority": template.priority,
            "parent_hypothesis_id": template.parent, "eligibility": dependency,
            "selection_identity": identity, "job_status": jobs.get(template.hypothesis_id, {}).get("status"),
            "reason": reason + " One registered investigation per objective; approval does not start compute.",
        })
    investigations = []
    for plan in plans:
        report = report_by_id.get(plan["hypothesis_id"])
        investigations.append({**plan, **_plan_detail(plan["reason"]),
                               "status": jobs.get(plan["hypothesis_id"], {}).get("status", "UNKNOWN"),
                               "report_hash": _report_hash(report) if report else None})
    return {
        "policy": "BOUNDED_APPROVED_TEMPLATES", "version": 2,
        "objectives": catalog, "datasets": list(datasets.values()),
        "investigations": investigations,
        "memory": [{"hypothesis_id": r["hypothesis_id"], "verdict": r["verdict"],
                    "classification": r.get("classification", "UNKNOWN"),
                    "economic_claim": r.get("economic_claim", "UNKNOWN"),
                    "dataset": r.get("dataset"), "report_hash": _report_hash(r)} for r in reports],
        "roles": _roles(reports, bool(plans)),
        "selection_policy": "Fixed preregistered priority; explicit approval, valid dataset and dependency gates; one persisted plan per hypothesis version.",
        "template_count": len(TEMPLATES), "max_followups_per_objective": 1,
        "max_investigations_per_objective": 1, "max_attempts_per_job": 3,
        "server_required": True, "code_generation": False, "network_acquisition": False,
        "brokerage_execution": "DISABLED", "models_promoted": 0,
    }
