"""Durable approval, automatic falsification, deduplication and HTTP boundaries."""
from __future__ import annotations

import asyncio
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import httpx
import pytest

from gig.research_org.director import (
    FOLLOWUP_ID,
    OBJECTIVE_ID,
    approve,
    propose,
    status,
)
from gig.research_org.ledger import ResearchLedger
from gig.research_org.service import ResearchWorker
from gig.research_org.treasury_experiment import run_treasury


def test_read_has_no_side_effects(tmp_path):
    ledger = ResearchLedger(tmp_path / "org")
    data = status(ledger)
    assert data["objectives"][0]["status"] == "NOT_APPROVED"
    assert not ledger.path.exists()
    with pytest.raises(ValueError):
        approve(ledger, "arbitrary-code")
    assert not ledger.path.exists()


def test_parent_required_and_plan_deduplicated_across_ledger_instances(tmp_path):
    ledger = ResearchLedger(tmp_path)
    approve(ledger, OBJECTIVE_ID)
    assert propose(ledger) == "awaiting-parent"
    digest, report = run_treasury()
    ledger.append(digest, report, "fixture")
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda _: propose(ResearchLedger(tmp_path)), range(8)))
    assert results.count("added") == 1
    assert len(status(ledger)["investigations"]) == 1
    assert len(ledger.jobs()) == 1
    assert ledger.jobs()[0]["attempts"] == 0


def test_automatic_followup_restarts_and_retains_negative_parent(tmp_path):
    ledger = ResearchLedger(tmp_path)
    approve(ledger, OBJECTIVE_ID)
    worker = ResearchWorker(ledger, max_jobs=3)
    worker.start(.05)
    worker.wait(30)
    assert ledger.count() == 3
    # Restart under the same durable objective; no second approval needed.
    recovered = ResearchWorker(ResearchLedger(tmp_path), max_jobs=2)
    recovered.start(.05)
    recovered.wait(30)
    data = status(ResearchLedger(tmp_path))
    assert ledger.count() == 4
    assert data["objectives"][0]["status"] == "COMPLETED"
    assert data["investigations"][0]["status"] == "DONE"
    assert len(data["investigations"][0]["report_hash"]) == 64
    assert any(r["hypothesis_id"] == FOLLOWUP_ID for r in data["memory"])
    parent = next(r["report"] for r in ledger.status()["runs"] if r["report"].get("hits") == 105)
    assert parent["pairs"] == 228 and parent["economic_claim"] == "NONE"
    before = json.dumps(data, sort_keys=True)
    again = ResearchWorker(ResearchLedger(tmp_path))
    again.start(.05)
    again.wait(30)
    assert ledger.count() == 4
    assert json.dumps(status(ResearchLedger(tmp_path)), sort_keys=True) == before
    assert again.snapshot()["reason"] == "approved backlog is empty"


def test_revoked_approval_does_not_generate_or_retry_work(tmp_path):
    ledger = ResearchLedger(tmp_path)
    approve(ledger, OBJECTIVE_ID)
    digest, report = run_treasury()
    ledger.append(digest, report, "fixture")
    assert propose(ledger) == "added"
    approve(ledger, OBJECTIVE_ID, revoked=True)
    assert ledger.jobs()[0]["status"] == "CANCELLED"
    assert propose(ledger) == "not-approved"
    approve(ledger, OBJECTIVE_ID)
    assert propose(ledger) == "already-planned"
    assert ledger.jobs()[0]["status"] == "CANCELLED"


def test_http_approval_is_separate_from_compute_and_survives_restart():
    from gig.service.app import create_app

    async def check(root):
        app = create_app(state_dir=root)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://test") as client:
            body = {"action": "approve", "objective_id": OBJECTIVE_ID}
            assert (await client.post('/api/v1/research-org/director', json=body,
                                      headers={"origin": "https://evil.invalid"})).status_code == 403
            assert (await client.post('/api/v1/research-org/director', json={**body, "extra": 1})).status_code == 422
            assert (await client.post('/api/v1/research-org/director', json=body)).status_code == 200
            data = (await client.get('/api/v1/research-org/status')).json()["data"]
            assert data["director"]["objectives"][0]["status"] == "APPROVED"
            assert data["run_count"] == 0 and not data["scheduler_state"]["alive"]
        restarted = create_app(state_dir=root)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(restarted), base_url="http://test") as client:
            data = (await client.get('/api/v1/research-org/director')).json()["data"]
            assert data["objectives"][0]["status"] == "APPROVED"
    with TemporaryDirectory() as directory:
        asyncio.run(check(Path(directory)))


def test_approval_atomic_rollback_on_failure(tmp_path):
    ledger = ResearchLedger(tmp_path)
    # A failed write cannot leave an approval or orphaned plan behind.
    with patch.object(ledger, '_connect', side_effect=OSError('fixture failure')), pytest.raises(OSError):
        approve(ledger, OBJECTIVE_ID)
    assert status(ledger)["objectives"][0]["status"] == "NOT_APPROVED"
    assert not ledger.path.exists()
