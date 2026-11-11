"""Actual Chromium acceptance for approved equity investigations and shared context.

Only the isolated local service is reachable by the browser. Source acquisitions
and brokerage connectivity are not invoked. Optional axe uses a supplied pinned
local asset. Screenshots show the application, not illustrative mockups.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from browser_stage_a_regressions import isolated_server
from playwright.sync_api import sync_playwright

EQUITY = ("uci-equity-range-persistence-v1", "uci-equity-volume-persistence-v1")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-dir", type=Path, required=True)
    parser.add_argument("--axe", type=Path)
    parser.add_argument("--executable", help="Optional explicit installed Chromium executable")
    args = parser.parse_args()
    args.evidence_dir.mkdir(parents=True, exist_ok=True)
    checks, errors, blocked, screenshots = [], [], [], []

    def check(name, passed, detail=None):
        checks.append({"name": name, "passed": bool(passed), "detail": detail})

    with isolated_server(args.evidence_dir) as url, sync_playwright() as pw:
        options = {"headless": True, "args": ["--no-sandbox"]}
        if args.executable:
            options["executable_path"] = args.executable
        browser = pw.chromium.launch(**options)
        version = browser.version
        context = browser.new_context(viewport={"width": 1600, "height": 1100}, reduced_motion="reduce")

        def restrict(route):
            if route.request.url.startswith(url + "/"):
                route.continue_()
            else:
                blocked.append(route.request.url)
                route.abort()

        context.route("**/*", restrict)
        page = context.new_page()
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.goto(url + "/#command-center", wait_until="networkidle")
        page.wait_for_function("document.querySelectorAll('.research-agenda-choice').length === 5")
        check("five approved investigation choices render in the Command Center", page.locator(".research-agenda-choice").count() == 5)
        check("all sixteen existing navigation sections remain", page.locator("#section-nav a").count() == 16)
        page.locator(f'[data-objective-id="{EQUITY[0]}"]').focus()
        page.locator(f'[data-objective-id="{EQUITY[0]}"]').press("Enter")
        check("investigation context selects the equity range study", page.evaluate("GIGResearchDesk.state().objective") == EQUITY[0])
        check("source uncertainty remains explicit", "UNKNOWN" in page.locator(".research-focus-evidence").inner_text())
        check("selection preview conveys approval boundary", "approval required" in page.locator("#research-agenda").inner_text().lower())
        handle = page.locator("#analysis-split")
        before_width = int(handle.get_attribute("aria-valuenow"))
        handle.focus()
        handle.press("ArrowLeft")
        check("Command Center splitter reports its actual keyboard-adjusted width", int(handle.get_attribute("aria-valuenow")) == min(520, before_width + 24))
        page.evaluate("document.activeElement.blur(); scrollTo(0,0)")
        page.screenshot(path=str(args.evidence_dir / "command-center-workspace.png"))
        page.locator("#research-dossier").screenshot(path=str(args.evidence_dir / "command-center.png"))
        screenshots.extend(["command-center-workspace.png", "command-center.png"])
        page.get_by_role("button", name="01 Director & Laboratory").click()
        page.wait_for_function("GIGObservatory.state()?.director?.template_count === 5")
        check("Research Observatory receives the same investigation context", page.locator("#director-objective-select").input_value() == EQUITY[0])
        check("focusing an investigation does not approve or compute", page.evaluate("GIGObservatory.state().run_count === 0 && GIGObservatory.state().director.objectives.every(r=>r.status === 'NOT_APPROVED')"))
        for objective in EQUITY:
            page.locator("#director-objective-select").select_option(objective)
            page.locator("#director-approve").click()
            page.wait_for_function("(id) => GIGObservatory.state().director.objectives.find(r=>r.objective_id === id)?.status === 'APPROVED'", arg=objective)
        check("separate equity approvals do not start compute", page.evaluate("GIGObservatory.state().run_count === 0"))
        page.locator("#scheduler-cadence").fill("0.05")
        page.locator("#scheduler-start").click()
        page.wait_for_function("GIGObservatory.state().run_count === 4 && GIGObservatory.state().scheduler_state.desired === 'PAUSED'", timeout=45000)
        check("four-job process budget remains enforced", page.evaluate("GIGObservatory.state().scheduler_state.reason === 'process job budget'"))
        page.locator("#scheduler-resume").click()
        page.wait_for_function("GIGObservatory.state().run_count === 5 && GIGObservatory.state().scheduler_state.desired === 'PAUSED'", timeout=45000)
        data = page.evaluate("GIGObservatory.state()")
        reports = [run["report"] for run in data["runs"] if isinstance(run["report"].get("dataset"), dict) and run["report"]["dataset"].get("id") == "uci-dow-jones-weekly-2011"]
        check("two approved equity studies run automatically and persist", len(reports) == 2)
        check("observed equity findings retain inconclusive results and no economic claims", len(reports) == 2 and all(r["verdict"] == "INCONCLUSIVE" and r["economic_claim"] == "NONE" and r["orders_submitted"] == 0 for r in reports))
        for report in reports:
            check("negative model comparison retained: " + report["hypothesis_id"], report["metrics"]["model_mean_score"] > report["metrics"]["baseline_mean_score"])
        page.locator("#observatory-filter").fill("uci-equity-weekly-range")
        page.locator("#observatory-history button").click()
        page.locator("#observatory-compare-selected").click()
        check("Strategy Laboratory exposes actual scoring and uncertainty", "Training-only baseline" in page.locator("#observatory-finding").inner_text())
        page.locator(".observatory-laboratory").screenshot(path=str(args.evidence_dir / "strategy-laboratory.png"))
        screenshots.append("strategy-laboratory.png")
        before = [(run["run_id"], run["report"]["hypothesis_id"]) for run in data["runs"]]
        page.reload(wait_until="networkidle")
        page.wait_for_function("GIGObservatory.state()?.run_count === 5")
        after = [(run["run_id"], run["report"]["hypothesis_id"]) for run in page.evaluate("GIGObservatory.state().runs")]
        check("saved findings survive reload without duplicated runs", before == after)
        check("selected research context survives reload", page.evaluate("GIGResearchDesk.state().objective") == EQUITY[1])
        page.locator('#section-nav a[href="#market-flow"]').click()
        page.wait_for_selector("#flow-positioning-cursor")
        page.locator("#flow-positioning-market").select_option("equity-index")
        page.wait_for_function("GIGMarketFlow.state().positioningMarket === 'equity-index' && GIGMarketFlow.state().positioningReports === 52")
        check("official equity-index futures positioning is distinct from currency futures", "13874A" in page.locator(".flow-positioning-identity").inner_text() and "S&P 500" in page.locator(".flow-positioning-identity").inner_text())
        check("positioning retains historical stale disclosure", "STALE" in page.locator("#flow-positioning-panel").inner_text())
        check("equity positioning chart draws actual report series", page.locator("#flow-positioning-chart path").count() == 2)
        page.locator("#flow-positioning-cursor").focus()
        page.locator("#flow-positioning-cursor").press("Home")
        check("keyboard report cursor selects actual first date", page.locator("#flow-positioning-date").get_attribute("data-report-date") == "2023-01-03")
        page.locator("#flow-positioning-cursor").press("End")
        page.locator("#flow-positioning-panel").screenshot(path=str(args.evidence_dir / "equity-positioning.png"))
        screenshots.append("equity-positioning.png")
        page.locator("#flow-example").click()
        page.locator("#flow-evaluate").click()
        page.wait_for_function("GIGMarketFlow.state().analysis")
        check("authored options remain separated from observed equity positioning", page.locator(".flow-chart").count() == 1 and page.locator("#flow-positioning-chart").count() == 1)
        page.route("**/api/v1/market-flow/equity-positioning", lambda route: route.fulfill(status=503, content_type="application/json", body='{"state":"UNAVAILABLE","data":null}'))
        page.locator("#flow-positioning-refresh").click()
        page.wait_for_function("document.getElementById('flow-positioning-status').textContent.startsWith('UNAVAILABLE')")
        check("unavailable positioning clears previous observed chart", page.locator("#flow-positioning-chart").count() == 0)
        page.unroute("**/api/v1/market-flow/equity-positioning")
        page.locator("#flow-positioning-refresh").click()
        page.wait_for_function("GIGMarketFlow.state().positioningReports === 52")
        check("positioning recovers by reloading verified saved evidence", page.locator("#flow-positioning-chart path").count() == 2)
        page.locator('#section-nav a[href="#charts"]').click()
        page.locator("#data-mode").select_option("synthetic")
        page.wait_for_function("document.getElementById('chart-context-classification').textContent.includes('SYNTHETIC')")
        check("shared research context never reclassifies chart observations", "SYNTHETIC" in page.locator("#chart-context-classification").inner_text() and "not a synchronized instrument" in page.locator(".research-context-limit").inner_text())
        page.locator("#charts-panel").screenshot(path=str(args.evidence_dir / "market-charts.png"))
        screenshots.append("market-charts.png")
        if args.axe:
            page.add_script_tag(path=str(args.axe))
            for theme in ("dark", "light"):
                if page.locator("html").get_attribute("data-theme") != theme:
                    page.locator("#theme-toggle").click()
                for name in ("command-center", "observatory", "charts", "market-flow"):
                    page.locator(f'#section-nav a[href="#{name}"]').click()
                    page.wait_for_timeout(200)
                    result = page.evaluate("async()=>await axe.run(document,{runOnly:{type:'tag',values:['wcag2a','wcag2aa','wcag21aa']}})")
                    check(f"axe WCAG2A/AA automated scan: {theme}/{name}", not result["violations"], [{"id": item["id"], "impact": item["impact"], "nodes": [node["target"] for node in item["nodes"]]} for item in result["violations"]])
        if page.locator("html").get_attribute("data-theme") != "dark":
            page.locator("#theme-toggle").click()
        for width in (390, 768):
            page.set_viewport_size({"width": width, "height": 844})
            for name in ("command-center", "observatory", "charts", "market-flow"):
                page.evaluate("(section)=>location.hash=section", name)
                page.wait_for_timeout(150)
                check(f"no page overflow at {width}px/{name}", page.evaluate("document.documentElement.scrollWidth <= innerWidth + 1"))
            if width == 390:
                page.evaluate("location.hash='command-center'")
                page.locator("#research-dossier").scroll_into_view_if_needed()
                page.screenshot(path=str(args.evidence_dir / "research-mobile.png"))
                screenshots.append("research-mobile.png")
        page.set_viewport_size({"width": 1600, "height": 1100})
        page.evaluate("location.hash='command-center'")
        page.wait_for_timeout(150)
        page.route("**/api/v1/research-org/status", lambda route: route.fulfill(status=503, content_type="application/json", body='{"state":"UNAVAILABLE","data":null}'))
        page.evaluate("GIGResearchDesk.reload()")
        page.wait_for_function("document.getElementById('research-dossier-status')?.textContent.startsWith('UNAVAILABLE')")
        check("unavailable research memory clears apparent selection success", page.locator(".research-agenda-choice").count() == 0)
        page.unroute("**/api/v1/research-org/status")
        page.evaluate("GIGResearchDesk.reload()")
        page.wait_for_selector(".research-agenda-choice")
        check("research memory recovers without automatic approval", page.locator(".research-agenda-choice").count() == 5)
        actual = page.request.get(url + "/api/v1/research-org/status").json()
        hostile = "<img src=x onerror=window.GIGResearchInjection=true>"
        actual["data"]["director"]["objectives"][0]["question"] = hostile
        page.route("**/api/v1/research-org/status", lambda route: route.fulfill(status=200, content_type="application/json", body=json.dumps(actual)))
        page.evaluate("GIGResearchDesk.reload()")
        page.wait_for_function("document.querySelector('.research-agenda-list').textContent.includes('<img src=x')")
        check("authored hostile response text is rendered without HTML execution", page.locator("#research-dossier img").count() == 0 and page.evaluate("typeof window.GIGResearchInjection === 'undefined'"))
        page.unroute("**/api/v1/research-org/status")
        check("reduced motion preference remains active", page.evaluate("matchMedia('(prefers-reduced-motion: reduce)').matches"))
        check("no uncaught browser JavaScript errors", not errors, errors)
        check("no external browser requests", not blocked, blocked)
        browser.close()
    report = {"browser": {"name": "Chromium", "version": version, "executable": args.executable or "Playwright-managed", "physical_arm64": False}, "checks": checks, "passed": sum(row["passed"] for row in checks), "failed": sum(not row["passed"] for row in checks), "errors": errors, "blocked_requests": blocked, "screenshots": screenshots, "limitations": ["Automated Chromium and axe checks do not establish full accessibility conformance, Firefox/Safari compatibility or physical ARM64 acceptance.", "Browser replay verifies persisted service results; worker process restart and lease safety require backend recovery tests."]}
    (args.evidence_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"passed": report["passed"], "failed": report["failed"], "browser": version, "report": str(args.evidence_dir / "report.json")}))
    return 1 if report["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
