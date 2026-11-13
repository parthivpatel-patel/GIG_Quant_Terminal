"""Observed equity-index futures positioning, provenance and rejection boundaries."""
from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import FrozenInstanceError, replace
from unittest.mock import patch

import pytest

from gig.data import cftc_positions as cftc
from gig.data import equity_positioning as equity
from gig.service.contracts import EnvelopeSchema
from gig.service.equity_positioning import equity_positioning_contract


def raw_rows():
    return json.loads(equity.FIXTURE.read_bytes())


def test_official_distinct_equity_identity_and_exact_conservation():
    source, rows = equity.load_equity_positions()
    assert source['source_authenticity'] == 'OFFICIAL_HTTPS_FETCH'
    assert hashlib.sha256(equity.FIXTURE.read_bytes()).hexdigest() == equity.RAW_SHA256
    assert hashlib.sha256(equity.PROVENANCE.read_bytes()).hexdigest() == equity.PROVENANCE_SHA256
    assert len(rows) == 52
    assert [rows[0]['report_date'], rows[-1]['report_date']] == ['2023-01-03', '2023-12-26']
    assert [rows[0]['open_interest'], rows[-1]['open_interest']] == [2010996, 2155313]
    assert [rows[0]['categories'][1]['net'], rows[-1]['categories'][1]['net']] == [436461, 878373]
    assert rows[-1]['categories'][2]['net'] == -354995
    for original, row in zip(raw_rows(), rows, strict=True):
        # Independent reference goes directly to original government integer strings.
        longs = sum(int(original[field]) for _, field, _, _ in cftc.CATEGORIES)
        shorts = sum(int(original[field]) for _, _, field, _ in cftc.CATEGORIES)
        spreads = sum(int(original[field]) for _, _, _, field in cftc.CATEGORIES if field)
        assert longs + spreads == shorts + spreads == int(original['open_interest_all'])
        assert sum(cat['net'] for cat in row['categories']) == 0
        assert row['report_id'].endswith('13874AF')
    assert equity.dataset_readiness()['equity_cash_quotes_available'] is False
    # The old default remains the exact Euro FX parser and cannot admit equities.
    with pytest.raises(ValueError):
        cftc.normalize_rows(raw_rows())
    _, euro = cftc.load_positions()
    assert euro[-1]['open_interest'] == 705627


def test_identity_is_frozen_and_unregistered_variants_are_rejected():
    with pytest.raises(FrozenInstanceError):
        cftc.SP500_EMINI.code = '099741'
    with pytest.raises(ValueError, match='unregistered'):
        cftc.normalize_rows(raw_rows(), identity=replace(cftc.SP500_EMINI, units='USD 1'))


@pytest.mark.parametrize('field,value', [
    ('cftc_contract_market_code', '13874U'), ('market_and_exchange_names', 'S&P 500 Consolidated - CHICAGO MERCANTILE EXCHANGE'),
    ('contract_units', '($5 X S&P 500 INDEX)'), ('cftc_commodity_code', '099 '), ('cftc_market_code', 'CBT'),
    ('futonly_or_combined', 'Combined'), ('id', '23010313874UF'), ('yyyy_report_week_ww', '2023 Report Week 02'),
    ('report_date_as_yyyy_mm_dd', '2023-01-03T12:00:00.000'), ('report_date_as_yyyy_mm_dd', '2024-01-02T00:00:00.000'),
    ('open_interest_all', True), ('open_interest_all', 2010996), ('open_interest_all', 'NaN'),
    ('open_interest_all', '2.010996e6'), ('open_interest_all', '-1'), ('open_interest_all', '02010996'),
    ('open_interest_all', '9999999999999'), ('tot_rept_positions_long_all', '1'),
    ('dealer_positions_long_all', '99999999999'), ('asset_mgr_positions_long', None),
])
def test_adversarial_source_fields_never_become_valid_positions(field, value):
    rows = raw_rows()
    rows[0][field] = value
    with pytest.raises(ValueError):
        cftc.normalize_rows(rows, identity=cftc.SP500_EMINI)


def test_duplicate_missing_observations_and_future_revisions():
    original = raw_rows()
    with pytest.raises(ValueError):
        cftc.normalize_rows(original + [original[0]], identity=cftc.SP500_EMINI)
    missing = copy.deepcopy(original)
    del missing[0]['dealer_positions_long_all']
    with pytest.raises(ValueError):
        cftc.normalize_rows(missing, identity=cftc.SP500_EMINI)
    changed = copy.deepcopy(original)
    for row in changed[30:]:
        row['asset_mgr_positions_long'], row['lev_money_positions_long'] = row['lev_money_positions_long'], row['asset_mgr_positions_long']
    before = cftc.analyze(cftc.normalize_rows(original, identity=cftc.SP500_EMINI))
    after = cftc.analyze(cftc.normalize_rows(changed, identity=cftc.SP500_EMINI))
    assert before[:30] == after[:30]
    assert before[30:] != after[30:]
    assert cftc.analyze(cftc.normalize_rows(changed, identity=cftc.SP500_EMINI), before[20]['report_date']) == before[:21]
    with pytest.raises(ValueError):
        equity._complete(json.dumps(original[:-1]).encode())


def test_all_prior_only_metrics_independent_reference_and_missingness():
    _, rows = equity.load_equity_positions()
    calculated = cftc.analyze(rows)
    for index, report in enumerate(calculated):
        for n, category in enumerate(report['categories']):
            history = [row['categories'][n]['net'] for row in rows[:index]]
            assert category['net_share_of_open_interest'] == category['net'] / report['open_interest']
            if index >= 8:
                assert category['prior_only_percentile'] == sum(1 if v < category['net'] else .5 if v == category['net'] else 0 for v in history) / index
            else:
                assert category['prior_only_percentile'] is None
            assert category['prior_observations'] == index
    assert all(c['weekly_net_change'] is None for c in cftc.analyze([rows[0], rows[2]])[-1]['categories'])
    assert rows[-1]['categories'][-1]['spread'] is None
    zero = raw_rows()[0]
    for key in ('open_interest_all', 'tot_rept_positions_long_all', 'tot_rept_positions_short'):
        zero[key] = '0'
    for _, long, short, spread in cftc.CATEGORIES:
        for key in (long, short, spread):
            if key:
                zero[key] = '0'
    normalized = cftc.normalize_rows([zero], identity=cftc.SP500_EMINI)
    assert all(c['net_share_of_open_interest'] is None for c in cftc.analyze(normalized)[0]['categories'])


def test_scope_clocks_permissions_and_detached_replay():
    first = equity_positioning_contract().to_dict()
    assert first == equity_positioning_contract().to_dict()
    EnvelopeSchema.model_validate(first)
    assert first['state'] == 'AVAILABLE'
    assert first['provenance']['classification'] == 'HISTORICAL'
    assert first['provenance']['instrument_identity'] == 'CFTC:13874A:TFF:FUTURES_ONLY'
    data = first['data']
    assert data['instrument']['asset_class'] == 'EQUITY_INDEX_FUTURES'
    assert data['freshness']['state'] == 'STALE'
    assert data['publication_time'] == 'UNKNOWN'
    assert data['revision_knowledge_time'].startswith('2026-11-11')
    assert all(row['observation_time'] is None and row['publication_time'] == 'UNKNOWN' for row in data['rows'])
    assert data['historical_knowledge_eligible'] is False
    assert data['equity_total_return_research_eligible'] is False
    assert data['cash_equity_observations'] == 'UNAVAILABLE'
    assert data['execution_enabled'] is False and data['orders_submitted'] == 0
    assert data['source']['cost'] == 'ZERO_NO_CREDENTIALS'
    assert 'public domain' in data['source']['license_evidence']['quote']
    assert 'without using a token' in data['source']['api_evidence']['quote']
    first['data']['rows'].clear()
    assert len(equity_positioning_contract().to_dict()['data']['rows']) == 52
    cutoff = equity_positioning_contract('2023-03-07').to_dict()['data']
    assert cutoff['summary']['last_report_date'] == '2023-03-07'
    assert cutoff['historical_knowledge_eligible'] is False
    assert equity_positioning_contract('2022-12-31').to_dict()['state'] == 'UNAVAILABLE'


@pytest.mark.parametrize('bad', ['2023-1-3', '2023-02-30', '2023-01-03T00:00:00Z', True, 20230103, 'NaN'])
def test_date_only_cutoff_rejects_invalid_inputs(bad):
    with pytest.raises(ValueError):
        equity_positioning_contract(bad)


def test_tampered_source_and_provenance_fail_closed():
    with patch.object(equity, 'FIXTURE') as source:
        source.read_bytes.return_value = b'[]'
        assert equity.dataset_readiness()['state'] == 'UNAVAILABLE'
        assert equity_positioning_contract().to_dict()['data'] is None
    meta = json.loads(equity.PROVENANCE.read_text())
    with patch.object(equity, 'PROVENANCE') as source:
        meta['publication_time'] = '2011-01-01T00:00:00Z'
        source.read_bytes.return_value = json.dumps(meta).encode()
        assert equity_positioning_contract().to_dict()['state'] == 'UNAVAILABLE'
    with patch.object(equity, 'FIXTURE') as source:
        source.read_bytes.side_effect = FileNotFoundError
        assert equity_positioning_contract().to_dict()['state'] == 'UNAVAILABLE'


def test_explicit_acquisition_is_bounded_and_does_not_admit_changed_vintage():
    raw = equity.FIXTURE.read_bytes()

    class Response:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def read(self, limit):
            assert limit == cftc.MAX_BYTES + 1
            return raw

        def geturl(self):
            return equity.SOURCE_URL

    with patch('urllib.request.build_opener') as opener:
        opener.return_value.open.return_value = Response()
        acquired, evidence = equity.fetch_equity_positions_2023()
    assert acquired == raw and evidence['admitted_to_pinned_store'] is False
    assert evidence['raw_sha256'] == equity.RAW_SHA256
    with patch('urllib.request.build_opener', side_effect=RuntimeError('private provider body')), pytest.raises(
        ValueError, match='^equity positioning acquisition unavailable$'
    ):
        equity.fetch_equity_positions_2023()
    for changed in (b'[]', b'x' * (cftc.MAX_BYTES + 1)):
        raw = changed
        with patch('urllib.request.build_opener') as opener, pytest.raises(ValueError, match='^equity positioning acquisition unavailable$'):
            opener.return_value.open.return_value = Response()
            equity.fetch_equity_positions_2023()
