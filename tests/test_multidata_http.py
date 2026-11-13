"""Integrated approved research and observed positioning without provider calls."""
from __future__ import annotations

import asyncio
from unittest.mock import patch

import httpx

from gig.service.app import create_app

OBJECTIVES = ('treasury-reversal-robustness-v1', 'ecb-fx-correlation-stability-v1',
              'boc-usdcad-variance-calibration-v1')


def test_registered_datasets_and_positioning_are_offline_read_only(tmp_path):
    async def run():
        app = create_app(state_dir=tmp_path / 'state')
        with patch('urllib.request.OpenerDirector.open', side_effect=AssertionError('unexpected network')):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url='http://test') as client:
                director = (await client.get('/api/v1/research-org/director')).json()
                assert director['state'] == 'AVAILABLE'
                objectives = director['data']['objectives']
                assert [row['objective_id'] for row in objectives[:3]] == list(OBJECTIVES)
                assert len(objectives) == 5
                assert len({row['dataset_id'] for row in objectives}) == 4
                assert all(row['dataset_state'] == 'AVAILABLE' for row in objectives)
                assert director['data']['memory'] == []
                positions = (await client.get('/api/v1/market-flow/positioning')).json()
                assert positions['state'] == 'AVAILABLE'
                data = positions['data']
                assert data['summary']['observations'] == 52
                assert data['freshness']['state'] == 'STALE'
                assert data['publication_time'] == 'UNKNOWN'
                assert not data['historical_knowledge_eligible']
                assert data['orders_submitted'] == 0 and not data['execution_enabled']
                register = (await client.get('/api/v1/market-flow')).json()['data']
                assert register['observed_flow_state'] == 'UNAVAILABLE'
                capabilities = {r['id']: r for r in register['capabilities']}
                assert capabilities['cftc_weekly_positioning']['state'] == 'AVAILABLE'
                assert capabilities['observed_options_flow']['state'] == 'UNAVAILABLE'
        assert not (tmp_path / 'state').exists()
    asyncio.run(run())


def test_approved_distinct_research_resumes_after_budget_and_restart(tmp_path):
    async def poll(client, count):
        for _ in range(400):
            data = (await client.get('/api/v1/research-org/status')).json()['data']
            if data['run_count'] == count and data['scheduler_state']['desired'] == 'PAUSED':
                return data
            await asyncio.sleep(.025)
        raise AssertionError(f'approved research did not reach {count} durable findings: {data}')

    async def run():
        root = tmp_path / 'state'
        app = create_app(state_dir=root)
        async with app.router.lifespan_context(app), httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url='http://test') as client:
            for objective in OBJECTIVES:
                response = await client.post('/api/v1/research-org/director', json={'action': 'approve', 'objective_id': objective})
                assert response.status_code == 200
            assert (await client.get('/api/v1/research-org/status')).json()['data']['run_count'] == 0
            await client.post('/api/v1/research-org/scheduler', json={'action': 'start', 'cadence_seconds': .05})
            first = await poll(client, 4)
            assert first['scheduler_state']['reason'] == 'process job budget'
        restarted = create_app(state_dir=root)
        async with restarted.router.lifespan_context(restarted), httpx.AsyncClient(transport=httpx.ASGITransport(restarted), base_url='http://test') as client:
            held = (await client.get('/api/v1/research-org/status')).json()['data']
            assert held['run_count'] == 4 and held['scheduler_state']['desired'] == 'PAUSED'
            await client.post('/api/v1/research-org/scheduler', json={'action': 'resume'})
            completed = await poll(client, 6)
            assert completed['scheduler_state']['reason'] == 'approved backlog is empty'
            assert all(r['status'] == 'COMPLETED' for r in completed['director']['objectives'] if r['objective_id'] in OBJECTIVES)
            assert all(r['status'] == 'NOT_APPROVED' for r in completed['director']['objectives'] if r['objective_id'] not in OBJECTIVES)
            reports = {r['hypothesis_id']: r['report'] for r in completed['runs']}
            parent = reports['treasury-10y-lagged-sign-2023']
            followup = reports['treasury-10y-reversal-robustness-2023-v1']
            assert (parent['hits'], parent['pairs']) == (105, 228)
            assert (followup['hits'], followup['pairs']) == (61, 115)
            assert completed['orders_submitted'] == 0
            for report in reports.values():
                assert report['economic_claim'] == 'NONE'
            before = completed['run_count']
            await client.post('/api/v1/research-org/scheduler', json={'action': 'start', 'cadence_seconds': .05})
            again = await poll(client, before)
            assert again['run_count'] == 6
            assert len(again['director']['investigations']) == 3
    asyncio.run(run())


def test_positioning_cutoff_is_observation_date_not_invented_knowledge(tmp_path):
    async def run():
        app = create_app(state_dir=tmp_path / 'state')
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url='http://test') as client:
            path = '/api/v1/market-flow/positioning'
            for value in ('2023-02-30', '2023-1-01', 'NaN'):
                response = await client.get(path, params={'report_through': value})
                assert response.status_code == 422
                assert response.json()['data'] is None
            empty = (await client.get(path, params={'report_through': '2022-12-31'})).json()
            assert empty['state'] == 'UNAVAILABLE' and empty['data'] is None
            response = await client.get(path, params={'report_through': '2023-06-30'})
            data = response.json()['data']
            assert 20 < data['summary']['observations'] < 52
            assert all(r['report_date'] <= '2023-06-30' for r in data['rows'])
            assert data['publication_time'] == 'UNKNOWN'
            assert data['historical_knowledge_eligible'] is False
            replay = await client.get(path, params={'report_through': '2023-06-30'})
            assert replay.json() == response.json()
        assert not (tmp_path / 'state').exists()
    asyncio.run(run())
