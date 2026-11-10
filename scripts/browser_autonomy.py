"""Actual Chromium workflows: Director approval, automatic follow-up and model-only flow."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from browser_stage_a_regressions import isolated_server
from playwright.sync_api import sync_playwright


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence-dir', required=True, type=Path)
    parser.add_argument('--axe', type=Path)
    args = parser.parse_args()
    out = args.evidence_dir
    out.mkdir(parents=True, exist_ok=True)
    checks, errors, blocked = [], [], []

    def check(name, passed, detail=None):
        checks.append({'name': name, 'passed': bool(passed), 'detail': detail})

    with isolated_server(out) as url, sync_playwright() as pw:
        options = {'headless': True, 'args': ['--no-sandbox']}
        if os.environ.get('GIG_CHROMIUM_EXECUTABLE'):
            options['executable_path'] = os.environ['GIG_CHROMIUM_EXECUTABLE']
        browser = pw.chromium.launch(**options)
        version = browser.version
        context = browser.new_context(viewport={'width': 1600, 'height': 1100}, reduced_motion='reduce')
        def restrict(route):
            if route.request.url.startswith(url + '/'):
                route.continue_()
            else:
                blocked.append(route.request.url)
                route.abort()
        context.route('**/*', restrict)
        page = context.new_page()
        page.on('pageerror', lambda error: errors.append(str(error)))
        page.goto(url + '/#observatory', wait_until='networkidle')
        page.wait_for_function('window.GIGObservatory.state()?.director')
        check('all sixteen sections retained', page.locator('#section-nav a').count() == 16)
        check('approval does not start compute', page.evaluate('GIGObservatory.state().run_count === 0 && !GIGObservatory.state().scheduler_state.alive'))
        page.locator('#director-approve').click()
        page.wait_for_function("GIGObservatory.state().director.objectives[0].status === 'APPROVED'")
        check('durable bounded objective approved', page.evaluate('GIGObservatory.state().run_count === 0'))
        page.locator('#scheduler-cadence').fill('0.05')
        page.locator('#scheduler-start').click()
        page.wait_for_function("GIGObservatory.state().run_count === 4 && GIGObservatory.state().scheduler_state.desired === 'PAUSED'", timeout=30000)
        data = page.evaluate('GIGObservatory.state()')
        check('one start automatically executes parent and follow-up', data['run_count'] == 4)
        check('director completed exactly one investigation', len(data['director']['investigations']) == 1 and data['director']['investigations'][0]['status'] == 'DONE')
        report = next(run['report'] for run in data['runs'] if run['hypothesis_id'] == 'treasury-10y-reversal-robustness-2023-v1')
        check('actual inconclusive observed result displayed', report['hits'] == 61 and report['pairs'] == 115 and report['verdict'] == 'INCONCLUSIVE')
        check('no trading or promotion', data['orders_submitted'] == 0 and report['models_promoted'] == 0)
        page.locator('#observatory-filter').fill('reversal')
        check('saved findings filter is functional', page.locator('#observatory-history button').count() == 1)
        page.locator('#observatory-history button').click()
        check('six analytical timeline stages', page.locator('#observatory-timeline button').count() == 6)
        page.locator('#observatory-timeline button').nth(3).click()
        check('skeptic timeline is interactive', 'skeptic' in page.locator('#observatory-trace-detail').inner_text().lower())
        page.locator('#observatory-filter').fill('')
        page.evaluate('document.activeElement.blur(); window.scrollTo(0, 0)')
        page.screenshot(path=str(out / 'research-operations-desktop.png'), full_page=True)
        page.reload(wait_until='networkidle')
        page.wait_for_function('GIGObservatory.state()?.run_count === 4')
        replay = page.evaluate('GIGObservatory.state()')
        replay_report = next(run['report'] for run in replay['runs'] if run['hypothesis_id'] == report['hypothesis_id'])
        check('saved result reload preserves exact report', replay_report == report)
        page.set_viewport_size({'width': 390, 'height': 844})
        page.evaluate('document.activeElement.blur(); window.scrollTo(0, 0)')
        page.wait_for_timeout(200)
        check('mobile has no horizontal page overflow', page.evaluate('document.documentElement.scrollWidth <= innerWidth + 1'))
        page.screenshot(path=str(out / 'research-operations-mobile.png'), full_page=True)
        page.set_viewport_size({'width': 1600, 'height': 1100})
        page.locator('#section-nav a[href="#market-flow"]').click()
        page.wait_for_function('window.GIGMarketFlow.state()')
        page.wait_for_timeout(200)
        check('Market Flow explicitly discloses missing observations', 'UNAVAILABLE' in page.locator('#market-flow-panel').inner_text())
        page.locator('#flow-example').click()
        check('loading authored assumptions does not calculate results', not page.evaluate('GIGMarketFlow.state().analysis'))
        page.locator('#flow-evaluate').click()
        page.wait_for_function('GIGMarketFlow.state().analysis')
        check('explicit local model renders scenario curve and table', page.locator('.flow-chart').count() == 1 and page.locator('.flow-table tbody tr').count() == 7)
        check('model alert is distinguished from observed activity', 'BREACH' in page.locator('#flow-output').inner_text() and 'not realized P&L' in page.locator('#flow-output').inner_text())
        page.evaluate('document.activeElement.blur(); window.scrollTo(0, 0)')
        page.screenshot(path=str(out / 'market-flow-desktop.png'), full_page=True)
        if args.axe:
            page.add_script_tag(path=str(args.axe))
            for theme in ('dark', 'light'):
                if page.locator('html').get_attribute('data-theme') != theme:
                    page.locator('#theme-toggle').click()
                for section in ('observatory', 'market-flow'):
                    page.locator(f'#section-nav a[href="#{section}"]').click()
                    page.wait_for_timeout(250)
                    result = page.evaluate("async () => await axe.run(document, {runOnly: {type:'tag', values:['wcag2a','wcag2aa','wcag21aa']}})")
                    check(f'{section} {theme} automated axe accessibility', not result['violations'], [{'id': row['id'], 'impact': row['impact'], 'nodes': len(row['nodes'])} for row in result['violations']])
                    page.evaluate('document.activeElement.blur(); window.scrollTo(0, 0)')
                    page.screenshot(path=str(out / f'{section}-{theme}.png'), full_page=True)
        check('no JavaScript errors', not errors, errors)
        check('no external browser requests', not blocked, blocked)
        browser.close()
    result = {'browser': 'Chromium', 'version': version,
              'passed': sum(c['passed'] for c in checks), 'failed': sum(not c['passed'] for c in checks),
              'checks': checks, 'limits': 'Headless Chromium only; automated axe is not complete accessibility certification.'}
    (out / 'report.json').write_text(json.dumps(result, indent=2) + '\n')
    (out / 'observed-followup.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps(result, indent=2))
    return int(result['failed'] != 0)


if __name__ == '__main__':
    raise SystemExit(main())
