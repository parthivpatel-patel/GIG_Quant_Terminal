"""Chromium captures of the Command Center, Market Charts, and Observatory desk.

The shared data-environment controls stay hidden on the Observatory. The
Command Center splitter changes the context width. No provider host is contacted.
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
    checks: list[dict] = []

    def check(name: str, passed: bool, detail=None) -> None:
        checks.append({"name": name, "passed": bool(passed), "detail": detail})

    with isolated_server(out) as url, sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True, args=["--no-sandbox"])

        def open_page(width: int, height: int):
            context = browser.new_context(viewport={"width": width, "height": height})
            context.route("**/*", lambda route: route.continue_() if route.request.url.startswith(url + "/") else route.abort())
            page = context.new_page()
            return context, page

        context, page = open_page(1440, 1100)
        page.goto(url + "/#command-center", wait_until="networkidle")
        page.wait_for_function("document.getElementById('content').dataset.section === 'command-center'")
        before = page.evaluate("getComputedStyle(document.getElementById('analysis-grid')).gridTemplateColumns")
        page.locator("#analysis-split").focus()
        page.keyboard.press("ArrowLeft")
        page.keyboard.press("ArrowLeft")
        after = page.evaluate("getComputedStyle(document.getElementById('analysis-grid')).gridTemplateColumns")
        check("splitter widens the context column", before != after, {"before": before, "after": after})
        page.screenshot(path=str(shots / "01-command-center-desktop.png"), full_page=True)
        context.close()

        context, page = open_page(390, 844)
        page.goto(url + "/#command-center", wait_until="networkidle")
        page.wait_for_function("document.getElementById('content').dataset.section === 'command-center'")
        overflow = page.evaluate("document.documentElement.scrollWidth <= document.documentElement.clientWidth")
        check("command center has no horizontal overflow at 390px", overflow)
        page.screenshot(path=str(shots / "02-command-center-mobile.png"), full_page=True)
        context.close()

        context, page = open_page(1440, 1100)
        page.goto(url + "/#charts", wait_until="networkidle")
        page.wait_for_function("document.getElementById('content').dataset.workspace === 'charts'")
        page.locator("#data-mode").select_option("synthetic")
        page.wait_for_function("document.getElementById('charts-status').textContent.includes('SYN001') || document.getElementById('state-badge').textContent === 'AVAILABLE'")
        page.wait_for_timeout(500)
        page.screenshot(path=str(shots / "03-market-charts-desktop.png"), full_page=True)
        context.close()

        context, page = open_page(390, 844)
        page.goto(url + "/#charts", wait_until="networkidle")
        page.locator("#data-mode").select_option("synthetic")
        page.wait_for_timeout(500)
        overflow = page.evaluate("document.documentElement.scrollWidth <= document.documentElement.clientWidth")
        check("charts have no horizontal overflow at 390px", overflow)
        page.screenshot(path=str(shots / "04-market-charts-mobile.png"), full_page=True)
        context.close()

        context, page = open_page(1440, 1100)
        page.goto(url + "/#observatory", wait_until="networkidle")
        page.wait_for_function("document.getElementById('observatory-roles').children.length === 9")
        hidden = page.evaluate("getComputedStyle(document.querySelector('.view-controls')).display === 'none'")
        check("observatory hides the shared data controls", hidden)
        page.screenshot(path=str(shots / "05-observatory-desktop.png"), full_page=True)
        context.close()

        context, page = open_page(390, 844)
        page.goto(url + "/#observatory", wait_until="networkidle")
        page.wait_for_function("document.getElementById('scheduler-start')")
        overflow = page.evaluate("document.documentElement.scrollWidth <= document.documentElement.clientWidth")
        check("observatory has no horizontal overflow at 390px", overflow)
        page.screenshot(path=str(shots / "06-observatory-mobile.png"), full_page=True)
        context.close()
        browser.close()

    log = out / "stage-a-server.txt"
    if log.exists():
        log.unlink()
    report = {"passed": sum(c["passed"] for c in checks), "failed": sum(not c["passed"] for c in checks), "checks": checks}
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"{report['passed']} passed, {report['failed']} failed")
    for row in checks:
        if not row["passed"]:
            print("FAIL", row["name"], row["detail"])
    return int(report["failed"] != 0)


if __name__ == "__main__":
    raise SystemExit(main())
