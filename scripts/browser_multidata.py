"""Chromium acceptance for distinct approved research and observed positioning."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from browser_stage_a_regressions import isolated_server
from playwright.sync_api import sync_playwright

OBJECTIVES = (
    "treasury-reversal-robustness-v1",
    "ecb-fx-correlation-stability-v1",
    "boc-usdcad-variance-calibration-v1",
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-dir", type=Path, required=True)
    parser.add_argument("--axe", type=Path, required=True)
    args = parser.parse_args()
    out = args.evidence_dir
    out.mkdir(parents=True, exist_ok=True)
    checks, errors, blocked = [], [], []

    def check(name, passed, detail=None):
        checks.append({"name": name, "passed": bool(passed), "detail": detail})

    with isolated_server(out) as url, sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True, args=["--no-sandbox"])
        version = browser.version
        context = browser.new_context(
            viewport={"width": 1600, "height": 1100}, reduced_motion="reduce"
        )

        def restrict(route):
            if route.request.url.startswith(url + "/"):
                route.continue_()
            else:
                blocked.append(route.request.url)
                route.abort()

        context.route("**/*", restrict)
        page = context.new_page()
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.goto(url + "/#observatory", wait_until="networkidle")
        page.wait_for_function("GIGObservatory.state()?.director?.objectives.length === 3")
        check(
            "three distinct dataset objectives are visible",
            page.locator("#director-objective-select option").count() == 3,
        )
        check(
            "no research runs before authorization",
            page.evaluate("GIGObservatory.state().run_count === 0"),
        )
        for objective in OBJECTIVES:
            page.locator("#director-objective-select").select_option(objective)
            page.locator("#director-approve").click()
            page.wait_for_function(
                '(id) => GIGObservatory.state().director.objectives.find(r => r.objective_id === id).status === "APPROVED"',
                arg=objective,
            )
        check(
            "independent approval does not start compute",
            page.evaluate("GIGObservatory.state().run_count === 0"),
        )
        page.locator("#scheduler-cadence").fill("0.05")
        page.locator("#scheduler-start").click()
        page.wait_for_function(
            'GIGObservatory.state().run_count === 4 && GIGObservatory.state().scheduler_state.desired === "PAUSED"',
            timeout=45000,
        )
        check(
            "four-job resource budget is retained",
            page.evaluate('GIGObservatory.state().scheduler_state.reason === "process job budget"'),
        )
        page.locator("#scheduler-resume").click()
        page.wait_for_function(
            'GIGObservatory.state().run_count === 6 && GIGObservatory.state().scheduler_state.desired === "PAUSED"',
            timeout=45000,
        )
        data = page.evaluate("GIGObservatory.state()")
        check("six studies complete without per-job submissions", data["run_count"] == 6)
        check(
            "all approved objectives complete and deduplicate",
            len(data["director"]["investigations"]) == 3
            and all(r["status"] == "COMPLETED" for r in data["director"]["objectives"]),
        )
        reports = {r["hypothesis_id"]: r["report"] for r in data["runs"]}
        check(
            "original Treasury negative finding retained",
            reports["treasury-10y-lagged-sign-2023"]["hits"] == 105,
        )
        check(
            "new sources remain historical without economic claims",
            all(
                reports[h]["classification"] == "HISTORICAL"
                and reports[h]["economic_claim"] == "NONE"
                for h in (
                    "ecb-fx-correlation-stability-2023-v1",
                    "boc-usdcad-variance-calibration-2023-v1",
                )
            ),
        )
        for query in ("ecb-fx", "boc-usdcad"):
            page.locator("#observatory-filter").fill(query)
            page.locator("#observatory-history button").click()
            page.locator("#observatory-compare-selected").click()
        check(
            "saved experiment comparison retains method-specific scores",
            "QLIKE" in page.locator("#observatory-comparison").inner_text()
            and "covariance" in page.locator("#observatory-comparison").inner_text().lower(),
        )
        page.locator("#observatory-filter").fill("")
        splitter = page.locator("#observatory-rail-splitter")
        before = splitter.get_attribute("aria-valuenow")
        splitter.focus()
        splitter.press("ArrowRight")
        check(
            "Laboratory rail resizes with keyboard",
            splitter.get_attribute("aria-valuenow") != before,
        )
        page.evaluate("document.activeElement.blur(); scrollTo(0,0)")
        page.screenshot(path=str(out / "research-laboratory-desktop.png"), full_page=True)
        page.reload(wait_until="networkidle")
        page.wait_for_function("GIGObservatory.state()?.run_count === 6")
        check(
            "six findings survive page reload",
            page.evaluate("GIGObservatory.state().run_count === 6"),
        )
        page.locator('#section-nav a[href="#market-flow"]').click()
        page.wait_for_selector("#flow-positioning-cursor")
        check(
            "52 observed weekly reports are available",
            page.locator("#flow-positioning-cursor").get_attribute("max") == "51",
        )
        check(
            "latest official report identity is visible",
            page.locator("#flow-positioning-date").get_attribute("data-report-date")
            == "2023-12-26",
        )
        check(
            "historical positioning carries stale disclosure",
            "STALE" in page.locator("#flow-positioning-panel").inner_text(),
        )
        cursor = page.locator("#flow-positioning-cursor")
        cursor.focus()
        cursor.press("Home")
        cursor.press("ArrowRight")
        check(
            "weekly inspection uses real report dates",
            page.locator("#flow-positioning-date").get_attribute("data-report-date")
            == "2023-01-10",
        )
        cursor.press("End")
        check(
            "observed aggregate positioning chart renders",
            page.locator("#flow-positioning-chart path").count() == 2,
        )
        page.evaluate("document.activeElement.blur(); scrollTo(0,0)")
        page.screenshot(path=str(out / "observed-positioning-desktop.png"), full_page=True)
        page.locator("#flow-example").click()
        page.locator("#flow-evaluate").click()
        page.wait_for_function("GIGMarketFlow.state().analysis")
        check(
            "authored model remains separate from observed positioning",
            page.locator(".flow-chart").count() == 1
            and page.locator("#flow-positioning-chart").count() == 1,
        )
        page.locator('#section-nav a[href="#charts"]').click()
        page.locator("#data-mode").select_option("synthetic")
        page.wait_for_function(
            'GIGCharts.state()?.data?.bars?.length > 0 || document.getElementById("chart-context-classification").textContent.includes("SYNTHETIC")'
        )
        page.locator("#chart-evidence-toggle").click()
        check(
            "charts retain explicit synthetic provenance",
            "SYNTHETIC" in page.locator("#chart-evidence").inner_text(),
        )
        page.locator("#chart-focus-toggle").focus()
        page.keyboard.press("Alt+w")
        check(
            "chart focus mode keyboard shortcut works",
            page.locator("#chart-focus-toggle").get_attribute("aria-pressed") == "true",
        )
        page.evaluate("document.activeElement.blur(); scrollTo(0,0)")
        page.screenshot(path=str(out / "market-charts-desktop.png"), full_page=True)
        page.add_script_tag(path=str(args.axe))
        for theme in ("dark", "light"):
            if page.locator("html").get_attribute("data-theme") != theme:
                page.locator("#theme-toggle").click()
            for section in ("observatory", "market-flow", "charts"):
                page.locator(f'#section-nav a[href="#{section}"]').click()
                page.wait_for_timeout(200)
                result = page.evaluate(
                    "async () => await axe.run(document, {runOnly:{type:'tag', values:['wcag2a','wcag2aa','wcag21aa']}})"
                )
                check(
                    f"{section} {theme} axe rules",
                    not result["violations"],
                    [
                        {"id": v["id"], "impact": v["impact"], "nodes": len(v["nodes"])}
                        for v in result["violations"]
                    ],
                )
        page.set_viewport_size({"width": 390, "height": 844})
        for section in ("observatory", "market-flow", "charts"):
            page.locator(f'#section-nav a[href="#{section}"]').click()
            page.wait_for_timeout(200)
            check(
                f"{section} mobile no page overflow",
                page.evaluate("document.documentElement.scrollWidth <= innerWidth + 1"),
            )
            page.evaluate("document.activeElement.blur(); scrollTo(0,0)")
            page.screenshot(path=str(out / f"{section}-mobile.png"), full_page=True)
        check("no JavaScript errors", not errors, errors)
        check("no external browser requests", not blocked, blocked)
        browser.close()
    result = {
        "browser": "Chromium",
        "version": version,
        "passed": sum(c["passed"] for c in checks),
        "failed": sum(not c["passed"] for c in checks),
        "checks": checks,
        "limits": "Automated Chromium only; axe is not manual accessibility certification.",
    }
    (out / "report.json").write_text(json.dumps(result, indent=2) + "\n")
    (out / "research-evidence.json").write_text(
        json.dumps(reports, indent=2, allow_nan=False) + "\n"
    )
    print(json.dumps(result, indent=2))
    return int(result["failed"] != 0)


if __name__ == "__main__":
    raise SystemExit(main())
