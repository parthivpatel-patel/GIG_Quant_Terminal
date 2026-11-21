"""Real local Chromium: pinned annotation profiles and unverified symbol mentions."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from browser_stage_a_regressions import isolated_server
from playwright.sync_api import sync_playwright


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence-dir', type=Path, required=True)
    parser.add_argument('--axe', type=Path)
    parser.add_argument('--executable')
    args = parser.parse_args()
    args.evidence_dir.mkdir(parents=True, exist_ok=True)
    checks, errors, external, posts, accessibility = [], [], [], [], []

    def check(name, passed):
        checks.append({'name': name, 'passed': bool(passed)})

    with isolated_server(args.evidence_dir) as url, sync_playwright() as pw:
        options = {'headless': True, 'args': ['--no-sandbox']}
        if args.executable:
            options['executable_path'] = args.executable
        browser = pw.chromium.launch(**options)
        version = browser.version
        context = browser.new_context(viewport={'width': 1600, 'height': 1100}, reduced_motion='reduce')

        def restrict(route):
            if route.request.url.startswith(url+'/'):
                route.continue_()
            else:
                external.append(route.request.url)
                route.abort()

        context.route('**/*', restrict)
        page = context.new_page()
        page.on('pageerror', lambda error: errors.append(str(error)))
        page.on('request', lambda request: posts.append(request.url) if request.method == 'POST' else None)
        page.goto(url+'/#research', wait_until='networkidle')
        page.wait_for_function("!document.getElementById('refresh').disabled")

        def action(name, payload=None):
            page.locator('#analytics-kind').select_option('workspace-financial-'+name)
            if payload is not None:
                page.locator('#analytics-input').fill(json.dumps(payload))
            with page.expect_response(lambda r: r.url.endswith('/api/v1/workspace/financial-intelligence')) as response:
                page.locator('#run-analysis').click()
            page.wait_for_function("!document.getElementById('run-analysis').disabled")
            return response.value.status, response.value.json()

        before = len(posts)
        page.locator('#analytics-kind').select_option('workspace-financial-entity-protocol')
        page.wait_for_timeout(50)
        check('protocol selection cannot admit text automatically', len(posts) == before)
        status, admitted = action('entity-protocol')
        check('only explicit authored SYNTHETIC text is admitted', status == 200 and admitted['data']['classification'] == 'SYNTHETIC')
        status, current = action('entities')
        check('explicit v2 profile executes', status == 200 and current['data']['nlp_profile'] == 'v2')
        data = current['data']
        qualified = [e for e in data['financial_nlp']['entity_mentions'] if e.get('mention_kind') == 'EXCHANGE_QUALIFIED_SYMBOL_SOURCE_MENTION']
        check('qualified share class and ticker remain whole mentions', {e['mention'] for e in qualified} == {'NYSE:BRK.B', 'NASDAQ:AAPL'})
        check('no invented security or issuer authority', all(e[k] == 'UNKNOWN' for e in qualified for k in ['instrument_identity_state', 'issuer_linking_state', 'exchange_membership_state', 'existence_state']))
        check('unsupported suffix is preserved as a whole rejection', any(u['provenance']['surface_text'] == 'NYSE:BRK.BOGUS' for u in data['financial_nlp']['unsupported_mentions']))
        check('each source symbol has exact byte citation', all(e['provenance']['source_quote'] == e['mention'] and e['provenance']['source_end'] > e['provenance']['source_start'] for e in qualified))
        check('synthetic protocol never becomes a market observation', data['classification'] == 'SYNTHETIC' and data['trading_approved'] is False)
        check('profile and full source symbol are readable', 'v2' in page.locator('#analysis-summary').inner_text() and 'NYSE:BRK.B' in page.locator('#analysis-summary').inner_text())
        check('unsupported mention is readable', 'UNSUPPORTED_QUALIFIED_SYMBOL_SYNTAX' in page.locator('#analysis-summary').inner_text())
        snapshot = data['snapshot_id']
        status, replayed = action('replay')
        check('v2 saved snapshot replays exactly', status == 200 and replayed == current)
        request = {'operation': 'answer', 'query': 'entity syntax protocol', 'as_of': data['as_of'], 'limit': 3, 'extract_facts': True}
        status, legacy = action('facts', request)
        check('default v1 stays byte-profile compatible', status == 200 and 'nlp_profile' not in legacy['data'] and legacy['data']['financial_nlp']['implementation'].endswith('.v1'))
        status, legacy_replay = action('replay')
        check('v1 snapshot remains independent of v2', status == 200 and legacy_replay == legacy)
        status, invalid = action('replay', {'operation': 'replay', 'snapshot_id': legacy['data']['snapshot_id'], 'nlp_profile': 'v2'})
        check('replay cannot override historical annotation profile', status == 422 and invalid['error']['code'] == 'INVALID_REQUEST')
        status, invalid = action('entities', {**request, 'nlp_profile': 'v999'})
        check('unregistered profile rejects safely', status == 422 and invalid['state'] == 'ERROR')
        status, invalid = action('entities', {**request, 'nlp_profile': 'v2', 'extract_facts': False})
        check('v2 requires explicit annotation request', status == 422 and invalid['state'] == 'ERROR')
        page.reload(wait_until='networkidle')
        page.wait_for_function("!document.getElementById('refresh').disabled")
        status, reopened = action('replay', {'operation': 'replay', 'snapshot_id': snapshot})
        check('pinned v2 survives browser restart', status == 200 and reopened == current)
        for theme in ('dark', 'light'):
            page.evaluate('theme=>document.documentElement.dataset.theme=theme', arg=theme)
            if args.axe:
                manifest = json.loads((args.axe.parent/'manifest.json').read_text())
                assert hashlib.sha256(args.axe.read_bytes()).hexdigest() == manifest['script_sha256']
                page.add_script_tag(path=str(args.axe))
                result = page.evaluate("async()=>await axe.run(document,{runOnly:{type:'tag',values:['wcag2a','wcag2aa','wcag21a','wcag21aa']}})")
                accessibility.append({'theme': theme, 'version': result['testEngine']['version'], 'violations': result['violations'], 'incomplete': result['incomplete']})
                check(theme+' automated accessibility', not result['violations'])
            page.locator('#workbench').screenshot(path=str(args.evidence_dir/('financial-profiles-'+theme+'.png')))
        for width in (320, 375, 768, 1440):
            page.set_viewport_size({'width': width, 'height': 900})
            check(str(width)+'px no overflow', page.evaluate('document.documentElement.scrollWidth<=innerWidth'))
        check('all sixteen terminal sections retained', page.locator('#section-nav a').count() == 16)
        check('no execution endpoint is submitted', all('/financial-intelligence' in p for p in posts))
        check('zero JavaScript errors', not errors)
        check('zero external browser requests', not external)
        browser.close()
    report = {'checks': checks, 'passed': sum(c['passed'] for c in checks), 'failed': sum(not c['passed'] for c in checks), 'chromium_version': version, 'accessibility': accessibility, 'javascript_errors': errors, 'external_requests': external, 'classification': 'AUTHORED_SYNTHETIC_PROTOCOL_NOT_MARKET_DATA', 'formal_accessibility_certified': False}
    (args.evidence_dir/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))
    return int(report['failed'] != 0)


if __name__ == '__main__':
    raise SystemExit(main())
