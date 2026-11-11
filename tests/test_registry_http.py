"""Offline HTTP scope, typed serialization and durable new investigations."""
from __future__ import annotations

import asyncio
from unittest.mock import patch

import httpx

from gig.service.app import create_app


async def _poll(client, count):
    for _ in range(500):
        data = (await client.get('/api/v1/research-org/status')).json()['data']
        if data['run_count'] == count and data['scheduler_state']['desired'] == 'PAUSED':
            return data
        await asyncio.sleep(.025)
    raise AssertionError('bounded research did not finish')


def test_equity_positioning_http_uses_saved_source_and_precise_identity(tmp_path):
    async def run():
        app = create_app(state_dir=tmp_path / 'state')
        with patch('urllib.request.OpenerDirector.open', side_effect=AssertionError('unexpected network')):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url='http://test') as client:
                response = await client.get('/api/v1/market-flow/equity-positioning')
                assert response.status_code == 200
                document = response.json()
                assert document['state'] == 'AVAILABLE'
                data = document['data']
                assert data['instrument']['contract_market_code'] == '13874A'
                assert data['summary']['observations'] == 52
                assert data['freshness']['state'] == 'STALE'
                assert data['cash_equity_observations'] == 'UNAVAILABLE'
                assert not data['historical_knowledge_eligible'] and not data['execution_enabled']
                assert (await client.get('/api/v1/market-flow/equity-positioning?report_through=2023-01-03')).json()['data']['summary']['observations'] == 1
                assert (await client.get('/api/v1/market-flow/equity-positioning?report_through=2022-01-01')).json()['state'] == 'UNAVAILABLE'
                for bad in ('2023-01-03T00:00:00', '2023-02-30', 'NaN', '2023-1-3'):
                    assert (await client.get('/api/v1/market-flow/equity-positioning', params={'report_through': bad})).status_code == 422
                assert (await client.get('/api/v1/market-flow/positioning')).json()['data']['instrument']['contract_market_code'] == '099741'
                capabilities = {r['id']: r for r in (await client.get('/api/v1/market-flow')).json()['data']['capabilities']}
                assert capabilities['cftc_equity_positioning']['state'] == 'AVAILABLE'
                assert capabilities['observed_options_flow']['state'] == 'UNAVAILABLE'
        assert not (tmp_path / 'state').exists()
    asyncio.run(run())


def test_new_equity_research_http_approval_restart_and_saved_replay(tmp_path):
    async def run():
        root = tmp_path / 'state'
        app = create_app(state_dir=root)
        async with app.router.lifespan_context(app), httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url='http://test') as client:
            before = (await client.get('/api/v1/research-org/director')).json()['data']
            assert before['version'] == 3 and before['template_count'] == 5
            for obj in before['objectives'][-2:]:
                result = await client.post('/api/v1/research-org/director', json={'action': 'approve', 'objective_id': obj['objective_id']})
                assert result.status_code == 200
            assert (await client.get('/api/v1/research-org/status')).json()['data']['run_count'] == 0
            assert (await client.post('/api/v1/research-org/director', json={'action': 'approve', 'objective_id': 'arbitrary-code'})).status_code == 422
            await client.post('/api/v1/research-org/scheduler', json={'action': 'start', 'cadence_seconds': .05})
            await _poll(client, 4)
        restarted = create_app(state_dir=root)
        async with restarted.router.lifespan_context(restarted), httpx.AsyncClient(transport=httpx.ASGITransport(restarted), base_url='http://test') as client:
            assert (await client.get('/api/v1/research-org/status')).json()['data']['run_count'] == 4
            await client.post('/api/v1/research-org/scheduler', json={'action': 'resume'})
            result = await _poll(client, 5)
            reports = {r['hypothesis_id']: r['report'] for r in result['runs']}
            for hypothesis, report in reports.items():
                if hypothesis.startswith('uci-equity'):
                    assert report['verdict'] == 'INCONCLUSIVE'
                    assert report['metrics']['model_mean_score'] > report['metrics']['baseline_mean_score']
                    assert report['allocation_eligibility'] == 'UNAVAILABLE'
                    assert report['orders_submitted'] == 0
            assert result['director']['objectives'][-1]['memory_evidence']
            hashes = {k: r['result_hash'] for k, r in reports.items() if k.startswith('uci-equity')}
        again = create_app(state_dir=root)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(again), base_url='http://test') as client:
            saved = (await client.get('/api/v1/research-org/status')).json()['data']
            assert {r['hypothesis_id']: r['report']['result_hash'] for r in saved['runs'] if r['hypothesis_id'].startswith('uci-equity')} == hashes
    asyncio.run(run())
