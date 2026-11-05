"""Chromium check for the Research Observatory.

Starts an isolated local service. The browser may reach only that origin.
One click runs the lagged-sign job. The next runs futures time-series
momentum. A third finds an empty queue. An options-flow objective is stored
as blocked and does not create a run.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from browser_stage_a_regressions import isolated_server
from playwright.sync_api import sync_playwright

OBJECTIVE = (
    "Investigate whether options-flow anomalies predict subsequent equity "
    "volatility after realistic costs."
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-dir", type=Path, required=True)
    args = parser.parse_args()
    out = args.evidence_dir
    shots = out / "screenshots"
    shots.mkdir(parents=True, exist_ok=True)
    checks, errors, blocked = [], [], []

    def check(name, passed, detail=None):
        checks.append({"name": name, "passed": bool(passed), "detail": detail})

    with isolated_server(out) as url, sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True, args=["--no-sandbox"])
        context = browser.new_context(viewport={"width": 1440, "height": 1100})

        def route(r):
            if r.request.url.startswith(url + "/"):
                r.continue_()
            else:
                blocked.append(r.request.url)
                r.abort()

        context.route("**/*", route)
        page = context.new_page()
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.goto(url + "/#observatory", wait_until="networkidle")
        page.wait_for_function("document.getElementById('observatory-roles').children.length === 9")
        page.wait_for_function("document.getElementById('observatory-backlog').children.length === 2")
        strip = page.evaluate("[...document.querySelectorAll('#truth-strip > *')].map(e => e.textContent.trim())")
        check("observatory replaces the truth strip", "research-organization" in " ".join(strip) and "LIVE" not in strip[0], strip)
        check("nine roles are listed", page.locator("#observatory-roles li").count() == 9)
        check("registered strategies are not marked executed", "REGISTERED_NOT_EXECUTED" in page.locator("#observatory-catalog").inner_text())
        page.locator("#observatory-objective-text").fill(OBJECTIVE)
        page.locator("#observatory-objective button").click()
        page.wait_for_function("document.getElementById('observatory-objective-result').textContent.includes('BLOCKED_NO_DATASET')")
        check("options-flow objective is blocked", page.evaluate("window.GIGObservatory.state().run_count === 0"))
        page.locator("#observatory-run").click()
        page.wait_for_function("window.GIGObservatory.state() && window.GIGObservatory.state().run_count === 1")
        check("first job records an inconclusive finding", "INCONCLUSIVE" in page.locator("#observatory-finding").inner_text())
        page.locator("#observatory-run").click()
        page.wait_for_function("window.GIGObservatory.state() && window.GIGObservatory.state().run_count === 2")
        history = page.locator("#observatory-history").inner_text()
        finding = page.locator("#observatory-finding").inner_text()
        check("second job records the futures experiment", "futures-tsmom-authored-excess-v1" in history and "INCONCLUSIVE" in finding, finding[:280])
        check("futures equity curve is drawn", page.locator(".observatory-spark").count() == 1)
        check("futures strategy is marked executed and the others are not",
              "EXECUTED_ON_AUTHORED_SYNTHETIC" in page.locator("#observatory-catalog").inner_text()
              and page.locator("#observatory-catalog").inner_text().count("REGISTERED_NOT_EXECUTED") == 2)
        page.reload(wait_until="networkidle")
        page.wait_for_function("window.GIGObservatory.state() && window.GIGObservatory.state().run_count === 2")
        check("reload keeps both cycles", page.locator("#observatory-history p").count() == 2)
        page.locator("#observatory-run").click()
        page.wait_for_function("document.getElementById('observatory-status').textContent.includes('No queued job')")
        check("a further tick finds an empty queue", "No queued job remains." in page.locator("#observatory-status").inner_text())
        page.screenshot(path=str(shots / "01-two-research-cycles.png"), full_page=True)
        check("no JavaScript errors", not errors, errors)
        check("no external requests", not blocked, blocked)
        browser.close()
    report = {"passed": sum(c["passed"] for c in checks), "failed": sum(not c["passed"] for c in checks), "checks": checks}
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"{report['passed']} passed, {report['failed']} failed")
    for row in checks:
        if not row["passed"]:
            print("FAIL", row["name"], row["detail"])
    return int(report["failed"] != 0)


if __name__ == "__main__":
    raise SystemExit(main())
