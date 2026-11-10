"""Official CFTC aggregate-positioning admission and temporal reference checks."""
from __future__ import annotations

import copy
import hashlib
import json
from unittest.mock import patch

import pytest

from gig.data import cftc_positions as cftc
from gig.service.contracts import EnvelopeSchema
from gig.service.positioning import positioning_contract


def raw_rows():
    return json.loads(cftc.FIXTURE.read_bytes())


def test_actual_official_reports_and_independent_conservation_reference():
    source, rows = cftc.load_positions()
    assert source['source_authenticity'] == 'OFFICIAL_HTTPS_FETCH'
    assert source['raw_sha256'] == hashlib.sha256(cftc.FIXTURE.read_bytes()).hexdigest()
    assert len(rows) == 52
    assert rows[0]['report_date'] == '2023-01-03'
    assert rows[-1]['report_date'] == '2023-12-26'
    assert rows[0]['open_interest'] == 712030
    assert rows[-1]['open_interest'] == 705627
    assert rows[0]['categories'][1]['net'] == 349543
    assert rows[-1]['categories'][1]['net'] == 365700
    assert rows[-1]['categories'][2]['net'] == -4233
    for report in rows:
        cats = report['categories']
        # Independent raw integer identity: both sides balance to the same OI.
        assert sum(c['long'] for c in cats) + sum(c['spread'] for c in cats[:-1]) == report['open_interest']
        assert sum(c['short'] for c in cats) + sum(c['spread'] for c in cats[:-1]) == report['open_interest']
        assert sum(c['net'] for c in cats) == 0


def test_derived_metrics_use_prior_reports_and_exact_contract_counts():
    _, rows = cftc.load_positions()
    calculated = cftc.analyze(rows)
    for index, row in enumerate(calculated):
        for category_index, cat in enumerate(row['categories']):
            history = [r['categories'][category_index]['net'] for r in rows[:index]]
            assert cat['prior_observations'] == index
            assert cat['net_share_of_open_interest'] == cat['net'] / row['open_interest']
            assert cat['weekly_net_change'] == (cat['net'] - history[-1] if history else None)
            if index < 8:
                assert cat['prior_only_percentile'] is None
            else:
                expected = sum(1 if prior < cat['net'] else .5 if prior == cat['net'] else 0 for prior in history) / len(history)
                assert cat['prior_only_percentile'] == expected
    assert calculated[-1]['categories'][1]['weekly_net_change'] == 2382
    assert calculated[-1]['categories'][2]['weekly_net_change'] == 3232


def test_future_revision_cannot_change_earlier_metrics_and_cutoff_is_observation_only():
    rows = raw_rows()
    changed = copy.deepcopy(rows)
    # Swap category longs on later reports, preserving every OI identity.
    for report in changed[30:]:
        report['asset_mgr_positions_long'], report['lev_money_positions_long'] = report['lev_money_positions_long'], report['asset_mgr_positions_long']
    original = cftc.analyze(cftc.normalize_rows(rows))
    revised = cftc.analyze(cftc.normalize_rows(changed))
    assert original[:30] == revised[:30]
    assert original[30:] != revised[30:]
    cutoff = original[20]['report_date']
    assert cftc.analyze(cftc.normalize_rows(changed), cutoff) == original[:21]
    payload = positioning_contract(cutoff).to_dict()
    assert payload['data']['historical_knowledge_eligible'] is False
    assert payload['data']['summary']['last_report_date'] == cutoff
    assert positioning_contract('2022-12-31').to_dict()['state'] == 'UNAVAILABLE'


@pytest.mark.parametrize('field,value', [
    ('cftc_contract_market_code', '99741'), ('cftc_market_code', 1), ('cftc_commodity_code', None),
    ('contract_units', 'USD 1'), ('market_and_exchange_names', 'OTHER'), ('futonly_or_combined', 'Combined'),
    ('id', 'bad'), ('yyyy_report_week_ww', '2023 Report Week 02'),
    ('report_date_as_yyyy_mm_dd', '2023-02-30T00:00:00.000'),
    ('report_date_as_yyyy_mm_dd', '2023-01-03T12:00:00.000'),
    ('open_interest_all', True), ('open_interest_all', 712030), ('open_interest_all', '7.12030e5'),
    ('open_interest_all', '-1'), ('open_interest_all', 'NaN'), ('open_interest_all', 'Infinity'),
    ('open_interest_all', '9999999999999'), ('dealer_positions_long_all', '99999999999'),
    ('dealer_positions_short_all', '0001'), ('tot_rept_positions_short', '1'),
])
def test_identity_precision_and_conservation_reject_invalid_records(field, value):
    rows = raw_rows()
    rows[0][field] = value
    with pytest.raises(ValueError):
        cftc.normalize_rows(rows)


def test_duplicate_missing_record_and_unordered_analysis_rejected():
    rows = raw_rows()
    with pytest.raises(ValueError):
        cftc.normalize_rows(rows + [rows[0]])
    del rows[0]['asset_mgr_positions_short']
    with pytest.raises(ValueError):
        cftc.normalize_rows(rows)
    _, normalized = cftc.load_positions()
    with pytest.raises(ValueError):
        cftc.analyze(list(reversed(normalized)))


def test_zero_open_interest_and_missing_weeks_remain_unknown():
    report = raw_rows()[0]
    for key in ['open_interest_all', 'tot_rept_positions_long_all', 'tot_rept_positions_short']:
        report[key] = '0'
    for _, long, short, spread in cftc.CATEGORIES:
        for key in (long, short, spread):
            if key:
                report[key] = '0'
    rows = cftc.normalize_rows([report])
    assert all(c['net_share_of_open_interest'] is None for c in cftc.analyze(rows)[0]['categories'])
    _, rows = cftc.load_positions()
    gapped = cftc.analyze([rows[0], rows[2]])
    assert all(c['weekly_net_change'] is None for c in gapped[1]['categories'])
    assert rows[0]['categories'][-1]['spread'] is None


@pytest.mark.parametrize('bad', ['2023-1-3', '2023-02-30', '2023-01-03T00:00:00Z', True, 20230103, 'NaN'])
def test_report_cutoff_is_canonical_date_not_a_fabricated_knowledge_clock(bad):
    with pytest.raises(ValueError):
        positioning_contract(bad)


def test_hash_corruption_and_missing_fixture_are_honestly_unavailable():
    with patch.object(cftc, 'FIXTURE') as file:
        file.read_bytes.return_value = b'[]'
        assert cftc.dataset_readiness()['state'] == 'UNAVAILABLE'
        assert positioning_contract().to_dict()['state'] == 'UNAVAILABLE'
    with patch.object(cftc, 'FIXTURE') as file:
        file.read_bytes.side_effect = FileNotFoundError
        assert positioning_contract().to_dict()['data'] is None


def test_replay_and_detached_contract_sanitization():
    first = positioning_contract().to_dict()
    assert first == positioning_contract().to_dict()
    EnvelopeSchema.model_validate(first)
    assert first['state'] == 'AVAILABLE'
    assert first['provenance']['classification'] == 'HISTORICAL'
    data = first['data']
    assert data['freshness']['state'] == 'STALE'
    assert data['publication_time'] == 'UNKNOWN'
    assert data['orders_submitted'] == 0 and data['execution_enabled'] is False
    assert all(row['observation_time'] is None for row in data['rows'])
    assert 'public domain' in data['source']['license_evidence']['quote']
    assert 'without using a token' in data['source']['api_evidence']['quote']
    first['data']['rows'].clear()
    assert len(positioning_contract().to_dict()['data']['rows']) == 52


def test_explicit_acquisition_errors_do_not_expose_provider_exception():
    with patch('urllib.request.build_opener', side_effect=RuntimeError('private provider response')), pytest.raises(ValueError, match='^CFTC acquisition unavailable$'):
        cftc.fetch_positions_2023()


def test_redirect_cannot_leave_approved_host():
    redirect = cftc._CFTCRedirect()
    for url in ('https://evil.invalid/a', 'http://publicreporting.cftc.gov/a',
                'https://publicreporting.cftc.gov:9999/a', 'https://user@publicreporting.cftc.gov/a'):
        with pytest.raises(ValueError):
            redirect.redirect_request(None, None, 302, 'moved', {}, url)
