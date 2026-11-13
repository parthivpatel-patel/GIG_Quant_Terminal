"""Finite multi-dataset planning: approval, admission, identity and recovery."""
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from dataclasses import replace
from unittest.mock import patch

import pytest

from gig.research_org import director, scheduler
from gig.research_org.ledger import ResearchLedger
from gig.research_org.service import ResearchWorker
from gig.research_org.template_catalog import TEMPLATES, dataset_readiness, selection_identity

ECB, BOC = TEMPLATES[1:]


def _ready(dataset_id):
    return {"dataset_id": dataset_id, "state": "AVAILABLE", "reason": "Authored readiness fixture; no provider observation claim.",
            "sha256": ("a" if dataset_id == ECB.dataset_id else "b") * 64,
            "provider": "TEST_FIXTURE", "publication_time": "UNKNOWN", "revision_knowledge_time": "UNKNOWN"}


def _runner(template, calls):
    def run():
        calls.append(template.hypothesis_id)
        return "test-" + template.hypothesis_id, {
            "hypothesis_id": template.hypothesis_id, "verdict": "INCONCLUSIVE", "classification": "SYNTHETIC",
            "economic_claim": "NONE", "orders_submitted": 0, "models_promoted": 0,
            "dataset": {"id": template.dataset_id, "sha256": _ready(template.dataset_id)["sha256"]},
            "role_outputs": {"quant": {"state": "COMPLETED", "output": "Authored test workflow"},
                             "skeptic": {"state": "COMPLETED", "output": {"verdict": "INCONCLUSIVE", "checks": {"fixture": True}}}},
        }
    return run


def _finish_seeds(ledger):
    worker = ResearchWorker(ledger)
    worker.start(.05)
    worker.wait(20)
    assert ledger.count() == 3


def test_catalog_distinguishes_three_templates_and_datasets_without_mutation(tmp_path):
    ledger = ResearchLedger(tmp_path / "org")
    data = director.status(ledger)
    assert data["version"] == 3 and data["template_count"] == 5
    assert len({r["dataset_id"] for r in data["objectives"]}) == 4
    assert len({r["method"] for r in data["objectives"]}) == 5
    assert data["objectives"][0]["objective_id"] == director.OBJECTIVE_ID
    assert all(r["status"] == "NOT_APPROVED" for r in data["objectives"])
    assert not ledger.path.exists()
    assert data["network_acquisition"] is False and data["models_promoted"] == 0


def test_each_registered_objective_requires_its_own_durable_approval(tmp_path):
    ledger = ResearchLedger(tmp_path)
    with patch.object(director, "dataset_readiness", side_effect=_ready):
        director.approve(ledger, ECB.objective_id)
        data = director.status(ResearchLedger(tmp_path))
    assert [r["status"] for r in data["objectives"]] == ["NOT_APPROVED", "APPROVED", "NOT_APPROVED", "NOT_APPROVED", "NOT_APPROVED"]
    assert ledger.count() == 0 and ledger.scheduler_state()["desired"] == "STOPPED"
    with pytest.raises(ValueError):
        director.approve(ledger, "arbitrary-unbounded-hypothesis")


def test_fixed_priority_and_concurrent_planning_deduplicate_both_templates(tmp_path):
    ledger = ResearchLedger(tmp_path)
    with patch.object(director, "dataset_readiness", side_effect=_ready):
        for template in (BOC, ECB):
            director.approve(ledger, template.objective_id)
        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(lambda _: director.propose(ResearchLedger(tmp_path)), range(8)))
        data = director.status(ledger)
    assert [row["hypothesis_id"] for row in data["investigations"]] == [ECB.hypothesis_id, BOC.hypothesis_id]
    assert [row["priority"] for row in data["investigations"]] == [90, 80]
    assert len(ledger.jobs()) == 2 and all(row["attempts"] == 0 for row in ledger.jobs())
    assert len({row["selection_identity"] for row in data["investigations"]}) == 2
    for row in data["investigations"]:
        assert row["parent_hypothesis_id"].startswith("dataset:")
        assert len(row["dataset_sha256"]) == 64
        assert not row["reason"].startswith("{")


@pytest.mark.parametrize("state", ["UNAVAILABLE", "ERROR"])
def test_invalid_dataset_does_not_create_a_plan_or_consume_attempts(tmp_path, state):
    ledger = ResearchLedger(tmp_path)
    denied = {"dataset_id": ECB.dataset_id, "state": state, "reason": "Fixture admission denied", "sha256": None}
    with patch.object(director, "dataset_readiness", return_value=denied):
        director.approve(ledger, ECB.objective_id)
        assert director.propose(ledger) == "blocked-dataset"
        data = director.status(ledger)
    objective = data["objectives"][1]
    assert objective["eligibility"] == "BLOCKED_DATASET"
    assert objective["status"] == "APPROVED" and objective["dataset_state"] == state
    assert not ledger.jobs() and not data["investigations"]


def test_revocation_cancels_only_the_selected_queued_template(tmp_path):
    ledger = ResearchLedger(tmp_path)
    with patch.object(director, "dataset_readiness", side_effect=_ready):
        for template in (ECB, BOC):
            director.approve(ledger, template.objective_id)
        director.propose(ledger)
        director.approve(ledger, ECB.objective_id, revoked=True)
        director.approve(ledger, ECB.objective_id)
        assert director.propose(ledger) == "already-planned"
    jobs = {r["hypothesis_id"]: r for r in ledger.jobs()}
    assert jobs[ECB.hypothesis_id]["status"] == "CANCELLED"
    assert jobs[BOC.hypothesis_id]["status"] == "QUEUED"


def test_source_or_method_revision_requires_new_review_not_duplicate_plan(tmp_path):
    ledger = ResearchLedger(tmp_path)
    with patch.object(director, "dataset_readiness", side_effect=_ready):
        director.approve(ledger, ECB.objective_id)
        director.propose(ledger)
        original = director.status(ledger)["investigations"][0]["selection_identity"]
    changed = _ready(ECB.dataset_id) | {"sha256": "c" * 64}
    with patch.object(director, "dataset_readiness", return_value=changed):
        data = director.status(ledger)
        assert data["objectives"][1]["eligibility"] == "BLOCKED_DATASET_REVISION"
        assert director.propose(ledger) == "already-planned"
        assert not director.admit_claim(ledger, ECB.job_id)
    assert len(ledger.jobs()) == 1
    assert selection_identity(replace(ECB, method_version=2), _ready(ECB.dataset_id)) != original


def test_changed_data_between_plan_and_execute_is_rejected_before_runner(tmp_path):
    ledger = ResearchLedger(tmp_path)
    _finish_seeds(ledger)
    calls = []
    with patch.object(director, "dataset_readiness", side_effect=_ready):
        director.approve(ledger, ECB.objective_id)
        director.propose(ledger)
    changed = _ready(ECB.dataset_id) | {"sha256": "c" * 64}
    with patch.object(director, "dataset_readiness", return_value=changed), patch.dict(
        scheduler.RUNNERS, {"ecb_risk": _runner(ECB, calls)}
    ):
        result = scheduler.tick(ledger)
    assert result["tick"]["action"] == "failed"
    assert not calls and ledger.count() == 3
    assert ledger.jobs()[-1]["status"] == "FAILED" and ledger.jobs()[-1]["attempts"] == 1


def test_two_approved_investigations_preserve_worker_budget_and_resume(tmp_path):
    ledger = ResearchLedger(tmp_path)
    _finish_seeds(ledger)
    calls = []
    with patch.object(director, "dataset_readiness", side_effect=_ready), patch.dict(
        scheduler.RUNNERS, {"ecb_risk": _runner(ECB, calls), "boc_volatility": _runner(BOC, calls)}
    ):
        for template in (ECB, BOC):
            director.approve(ledger, template.objective_id)
        first = ResearchWorker(ledger, max_jobs=1)
        first.start(.05)
        first.wait(20)
        assert ledger.count() == 4
        assert first.snapshot()["reason"] == "process job budget"
        second = ResearchWorker(ResearchLedger(tmp_path))
        second.start(.05)
        second.wait(20)
        data = director.status(ResearchLedger(tmp_path))
        assert ledger.count() == 5
        assert second.snapshot()["reason"] == "approved backlog is empty"
        before = json.dumps(data, sort_keys=True)
        again = ResearchWorker(ResearchLedger(tmp_path))
        again.start(.05)
        again.wait(20)
        assert json.dumps(director.status(ResearchLedger(tmp_path)), sort_keys=True) == before
    assert calls == [ECB.hypothesis_id, BOC.hypothesis_id]
    assert [r["status"] for r in data["objectives"]] == ["NOT_APPROVED", "COMPLETED", "COMPLETED", "NOT_APPROVED", "NOT_APPROVED"]
    quant = next(r for r in data["roles"] if r["id"] == "quant")
    assert quant["workflow_count"] == 2
    assert len(data["memory"]) == 5
    assert next(r for r in ledger.status()["runs"] if r["report"].get("hits") == 105)["report"]["pairs"] == 228


def test_unregistered_dataset_readiness_is_unavailable():
    assert dataset_readiness("not-registered")["state"] == "UNAVAILABLE"


def test_parent_verdict_label_cannot_substitute_for_negative_evidence(tmp_path):
    from gig.research_org.treasury_experiment import run_treasury
    ledger = ResearchLedger(tmp_path)
    digest, report = run_treasury()
    report["hits"] = 200
    ledger.append(digest, report, "tampered-fixture")
    director.approve(ledger, director.OBJECTIVE_ID)
    assert director.propose(ledger) == "unsupported-parent"
    assert director.status(ledger)["objectives"][0]["eligibility"] == "BLOCKED_PARENT"
    assert not ledger.jobs()


def test_result_provenance_mismatch_is_not_persisted_or_retried(tmp_path):
    ledger = ResearchLedger(tmp_path)
    _finish_seeds(ledger)
    calls = []
    base = _runner(ECB, calls)
    def tampered():
        digest, report = base()
        report["dataset"]["sha256"] = "0" * 64
        return digest, report
    with patch.object(director, "dataset_readiness", side_effect=_ready), patch.dict(
        scheduler.RUNNERS, {"ecb_risk": tampered}
    ):
        director.approve(ledger, ECB.objective_id)
        director.propose(ledger)
        result = scheduler.tick(ledger)
    assert calls == [ECB.hypothesis_id]
    assert result["tick"]["action"] == "failed" and ledger.count() == 3
    assert ledger.jobs()[-1]["status"] == "FAILED" and ledger.jobs()[-1]["attempts"] == 1


def test_legacy_plaintext_treasury_plan_remains_compatible(tmp_path):
    ledger = ResearchLedger(tmp_path)
    _finish_seeds(ledger)
    director.approve(ledger, director.OBJECTIVE_ID)
    assert director.propose(ledger) == "added"
    with closing(ledger._connect()) as db, db:
        db.execute("UPDATE director_investigations SET reason='Original plaintext plan' WHERE hypothesis_id=?", (director.FOLLOWUP_ID,))
    assert director.admit_claim(ledger, director.FOLLOWUP_JOB["job_id"])
    result = scheduler.tick(ledger)
    assert result["tick"]["action"] == "ran" and ledger.count() == 4
