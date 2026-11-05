"""Chromium check for the Research Observatory.

Starts an isolated local service. The browser may reach only that origin.
The cycle button runs the one approved synthetic experiment and a second
press returns the stored record.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from browser_stage_a_regressions import isolated_server
from playwright.sync_api import sync_playwright


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
            context = browser.new_context(viewport={"width": 1440, "height": 1000})

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
            strip = page.evaluate("[...document.querySelectorAll('#truth-strip > *')].map(e => e.textContent.trim())")
            check("observatory replaces the truth strip", "research-organization" in " ".join(strip) and "LIVE" not in strip[0], strip)
            check("nine roles are listed", page.locator("#observatory-roles li").count() == 9)
            check("registered strategies are not marked executed", "REGISTERED_NOT_EXECUTED" in page.locator("#observatory-catalog").inner_text())
            page.locator("#observatory-run").click()
            page.wait_for_function("document.getElementById('observatory-finding').textContent.includes('INCONCLUSIVE')")
            finding = page.locator("#observatory-finding").inner_text()
            check("the cycle records an inconclusive synthetic finding", "INCONCLUSIVE" in finding and "Orders submitted" in finding and "0" in finding, finding[:240])
            page.locator("#observatory-run").click()
            page.wait_for_function("document.getElementById('observatory-status').textContent.includes('Already recorded')")
            check("a second cycle returns the stored record", "Deduplicated" in page.locator("#observatory-finding").inner_text())
            page.screenshot(path=str(shots / "01-research-observatory.png"))
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
