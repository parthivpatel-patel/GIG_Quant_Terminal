"""Adversarial archive diagnostics and durable approval/memory selection."""
from __future__ import annotations

import copy
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from unittest.mock import patch

import pytest

from gig.research_org import director
from gig.research_org.equity_experiments import (
    DATASET,
    HYPOTHESES,
    challenge,
    dataset_readiness,
    evaluate,
    load_equity,
    run_equity_range,
    run_equity_volume,
)
from gig.research_org.ledger import ResearchLedger
from gig.research_org.service import ResearchWorker
from gig.research_org.template_catalog import (
    REGISTRY,
    build_registry,
    investigations,
    selection_identity,
)

EQUITY = tuple(r for r in investigations() if r.dataset_id == DATASET)


def test_immutable_registry_rejects_duplicate_methods_and_execution_identities():
    rows = investigations()
    assert len(rows) == 5 and len(REGISTRY) == 5
    with pytest.raises(TypeError):
        REGISTRY['arbitrary'] = rows[0]
    with pytest.raises(ValueError, match='duplicate'):
        build_registry(rows + (rows[0],))
    duplicate = replace(rows[0], objective_id='new', hypothesis_id='new', job_id='new', seq=100)
    with pytest.raises(ValueError, match='duplicate'):
        build_registry(rows + (duplicate,))
    with pytest.raises(ValueError, match='bounded'):
        build_registry((replace(rows[0], estimated_cost_units=100),))


def test_readiness_preserves_exact_source_scope_and_quarantine():
    source, document = load_equity()
    ready = dataset_readiness()
    assert ready['observation_count'] == 125
    assert ready['state'] == 'AVAILABLE' and ready['corporate_actions_state'] == 'UNAVAILABLE'
    assert not ready['point_in_time_backtest_eligible'] and not ready['funded_allocation_eligible']
    assert ready['publication_time'] == ready['revision_knowledge_time'] == 'UNKNOWN'
    assert ready['acquired_at'] == source['source_acquired_at']
    assert all(not set(row) & set(document['quarantined_future_columns']) for row in document['rows'])
    assert len({selection_identity(t, ready) for t in EQUITY}) == 2


@pytest.mark.parametrize('runner', [run_equity_range, run_equity_volume])
def test_saved_result_replay_negative_findings_and_separate_numerical_review(runner):
    config, first = runner()
    assert (config, first) == runner()
    assert first['verdict'] == 'INCONCLUSIVE' and first['economic_claim'] == 'NONE'
    assert first['metrics']['model_mean_score'] > first['metrics']['baseline_mean_score']
    assert all(first['skeptic']['checks'].values())
    assert first['metrics']['validation_weeks'] == 15
    assert first['metrics']['panel_observations'] == 75
    assert len(first['metrics']['bootstrap_interval_95']) == 2
    assert first['multiple_testing']['family_size'] == 2
    assert first['orders_submitted'] == first['models_promoted'] == 0
    assert first['allocation_eligibility'] == 'UNAVAILABLE'
    assert first['total_return'] is None


@pytest.mark.parametrize('kind', ['range', 'volume'])
def test_future_targets_cannot_change_earlier_features_predictions_or_baselines(kind):
    _, document = load_equity()
    original = evaluate(document, kind)
    changed = copy.deepcopy(document)
    for row in changed['rows']:
        if row['week_ending_date'] >= '2011-05-27':
            row['high'] = str(float(row['high']) * 3)
            row['volume'] *= 7
    altered = evaluate(changed, kind)
    before = [r for r in original['prediction_rows'] if r['observation_date'] < '2011-05-27']
    after = [r for r in altered['prediction_rows'] if r['observation_date'] < '2011-05-27']
    assert before == after
    assert len(before) > 20


@pytest.mark.parametrize('kind', ['range', 'volume'])
def test_skeptic_rejects_tampered_forecast_score_identity_or_summary(kind):
    _, document = load_equity()
    metrics = evaluate(document, kind)
    for change in ('prediction', 'baseline_score', 'feature_date', 'aggregate', 'missing'):
        tampered = copy.deepcopy(metrics)
        if change == 'aggregate':
            tampered['model_mean_score'] = -1
        elif change == 'missing':
            tampered['prediction_rows'].pop()
        elif change == 'feature_date':
            tampered['prediction_rows'][0][change] = '2030-01-01'
        else:
            tampered['prediction_rows'][0][change] += 1
        assert challenge(document, tampered, kind)['verdict'].startswith('REJECTED')


def test_missing_security_and_invalid_numeric_values_rejected_without_imputation():
    _, document = load_equity()
    for field, value, kind in [('high', 'NaN', 'range'), ('volume', 0, 'volume'),
                               ('volume', True, 'volume'), ('volume', 1.5, 'volume')]:
        changed = copy.deepcopy(document)
        changed['rows'][0][field] = value
        with pytest.raises(ValueError):
            evaluate(changed, kind)
    changed = copy.deepcopy(document)
    changed['rows'].pop(0)
    with pytest.raises(ValueError, match='weeks|unaligned'):
        evaluate(changed, 'range')


def test_source_tampering_fails_closed_and_sanitizes_errors(tmp_path):
    from gig.research_org import equity_experiments
    bad = tmp_path / 'secret-provider-text'
    bad.write_text('{"csv_text":"sensitive raw text"}')
    with patch.object(equity_experiments, 'FIXTURE', bad):
        ready = dataset_readiness()
        assert ready['state'] == 'ERROR' and ready['sha256'] is None
        assert 'sensitive' not in json.dumps(ready) and str(bad) not in json.dumps(ready)
        with pytest.raises(ValueError):
            run_equity_range()


def test_memory_ranking_diversifies_without_favoring_positive_results(tmp_path):
    ledger = ResearchLedger(tmp_path)
    range_template, volume_template = EQUITY
    before = director.selection(volume_template, [])
    digest, report = run_equity_range()
    ledger.append(digest, report, actor='test')
    reports = [r['report'] for r in ledger.status()['runs']]
    after = director.selection(volume_template, reports)
    assert before['selection_score'] - after['selection_score'] == 30
    assert after['memory_evidence'][0]['hypothesis_id'] == range_template.hypothesis_id
    assert after['memory_evidence'][0]['verdict'] == 'INCONCLUSIVE'
    changed = copy.deepcopy(reports)
    changed[0]['verdict'] = 'POSITIVE_UNVERIFIED'
    assert director.selection(volume_template, changed)['selection_score'] == after['selection_score']
    assert director.status(ledger)['objectives'][-1]['selection_score'] == after['selection_score']


def test_concurrent_planning_explicit_approval_and_durable_ranking(tmp_path):
    ledger = ResearchLedger(tmp_path)
    assert director.propose(ledger) == 'not-approved'
    director.approve(ledger, EQUITY[1].objective_id)
    assert director.propose(ledger) == 'added'
    assert len(ledger.jobs()) == 1
    planned = director.status(ledger)['investigations'][0]
    assert planned['selection_components']['unanswered_dataset'] == 20
    assert len(planned['memory_evidence']) == 0
    director.approve(ledger, EQUITY[0].objective_id)
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda _: director.propose(ResearchLedger(tmp_path)), range(8)))
    assert len(ledger.jobs()) == 2 and all(r['attempts'] == 0 for r in ledger.jobs())
    assert ledger.jobs()[0]['hypothesis_id'] == EQUITY[0].hypothesis_id
    assert ledger.count() == 0 and ledger.scheduler_state()['desired'] == 'STOPPED'


def test_new_approved_studies_resume_after_budget_and_restart_without_duplicates(tmp_path):
    ledger = ResearchLedger(tmp_path)
    for t in EQUITY:
        director.approve(ledger, t.objective_id)
    first = ResearchWorker(ledger, max_jobs=4)
    first.start(.05)
    first.wait(20)
    assert ledger.count() == 4 and first.snapshot()['reason'] == 'process job budget'
    second = ResearchWorker(ResearchLedger(tmp_path))
    second.start(.05)
    second.wait(20)
    assert second.ledger.count() == 5
    reports = {r['hypothesis_id']: r['report'] for r in second.ledger.status()['runs']}
    assert set(HYPOTHESES.values()) <= reports.keys()
    assert reports['treasury-10y-lagged-sign-2023']['hits'] == 105
    for t in EQUITY:
        assert director.admit_result(second.ledger, t.job_id, reports[t.hypothesis_id])
    assert director.propose(second.ledger) == 'exhausted'
    before = second.ledger.count()
    third = ResearchWorker(ResearchLedger(tmp_path))
    third.start(.05)
    third.wait(20)
    assert third.ledger.count() == before


def test_planning_transaction_rolls_back_plan_when_job_admission_fails(tmp_path):
    import sqlite3
    from contextlib import closing
    ledger = ResearchLedger(tmp_path)
    director.approve(ledger, EQUITY[0].objective_id)
    with closing(ledger._connect()) as db, db:
        db.execute("CREATE TRIGGER reject_job BEFORE INSERT ON jobs BEGIN SELECT RAISE(ABORT, 'injected'); END")
    with pytest.raises(sqlite3.IntegrityError, match='injected'):
        director.propose(ledger)
    assert not ledger.jobs() and not director.status(ledger)['investigations']
    with closing(ledger._connect()) as db, db:
        db.execute('DROP TRIGGER reject_job')
    assert director.propose(ledger) == 'added'
    assert len(ledger.jobs()) == len(director.status(ledger)['investigations']) == 1


def test_new_plans_cannot_use_unfingerprinted_legacy_treasury_exception(tmp_path):
    from contextlib import closing
    ledger = ResearchLedger(tmp_path)
    director.approve(ledger, EQUITY[0].objective_id)
    director.propose(ledger)
    assert director.admit_claim(ledger, EQUITY[0].job_id)
    with closing(ledger._connect()) as db, db:
        db.execute('UPDATE director_investigations SET reason=?', ('legacy reason without identity',))
    assert not director.admit_claim(ledger, EQUITY[0].job_id)



def test_research_catalog_points_to_each_actual_registered_method_module():
    from gig.research_org.cycle import _catalog
    rows = {r["id"]: r for r in _catalog(set())}
    for template in investigations():
        assert rows[template.hypothesis_id]["module"] == template.module
    for template in EQUITY:
        assert rows[template.hypothesis_id]["module"] == "gig.research_org.equity_experiments"
