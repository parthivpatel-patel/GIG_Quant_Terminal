"""Actual local Chromium saved-dataset GP workflow; no providers or orders."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from browser_stage_a_regressions import isolated_server
from playwright.sync_api import sync_playwright


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-dir", type=Path, required=True)
    parser.add_argument("--axe", type=Path)
    parser.add_argument("--executable")
    args = parser.parse_args()
    args.evidence_dir.mkdir(parents=True, exist_ok=True)
    checks, errors, external = [], [], []

    def check(name, value):
        checks.append({"name": name, "passed": bool(value)})

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
                external.append(route.request.url)
                route.abort()

        context.route("**/*", restrict)
        page = context.new_page()
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.goto(url + "/#research", wait_until="networkidle")
        page.wait_for_function("!document.getElementById('refresh').disabled")
        responses = {}
        for action, endpoint in [("pipeline-bars", "bars"), ("pipeline-universe", "universe"),
                                 ("research-dataset", "research-dataset"), ("research-gpr", "research-model"),
                                 ("research-predict", "research-predict")]:
            page.locator("#analytics-kind").select_option("workspace-" + action)
            if action == "research-gpr":
                check("explicit fixed-kernel posterior caveat", "not calibrated confidence" in page.locator("#workbench-scope").inner_text())
            with page.expect_response(lambda response, endpoint=endpoint: response.url.endswith("/api/v1/workspace/" + endpoint)) as exchange:
                page.locator("#run-analysis").click()
            page.wait_for_function("!document.getElementById('run-analysis').disabled")
            result = exchange.value.json()
            responses[action] = result
            check(action + " admitted explicit browser action", exchange.value.status == 200 and result["state"] == "AVAILABLE")
            if result["state"] != "AVAILABLE":
                raise AssertionError(action + " failed: " + json.dumps(result))
        check("saved inference does not retrain", not responses["research-predict"]["data"]["metrics"]["training_performed"])
        run_id = responses["research-gpr"]["data"]["run_id"]
        check("run identity is shown in traceability strip", run_id in page.locator("#workflow-identities").inner_text())
        page.locator("#analytics-kind").select_option("workspace-research-allocate")
        with page.expect_response(lambda response: response.url.endswith("/api/v1/workspace/research-allocate")) as exchange:
            page.locator("#run-analysis").click()
        page.wait_for_function("!document.getElementById('run-analysis').disabled")
        check("unapproved GP allocation remains unavailable", exchange.value.json()["state"] == "UNAVAILABLE" and exchange.value.json()["data"] is None)
        page.reload(wait_until="networkidle")
        page.wait_for_function("!document.getElementById('refresh').disabled")
        page.locator("#analytics-kind").select_option("workspace-research-predict")
        # Workbench references are intentionally tab-memory only. An analyst
        # restores saved identities explicitly after reload; do not invent an ID.
        page.locator("#analytics-input").fill(json.dumps({"run_id": run_id, "dataset_id": responses["research-dataset"]["data"]["dataset_id"]}))
        with page.expect_response(lambda response: response.url.endswith("/api/v1/workspace/research-predict")) as exchange:
            page.locator("#run-analysis").click()
        page.wait_for_function("!document.getElementById('run-analysis').disabled")
        check("explicit saved identities replay after page reload", exchange.value.json() == responses["research-predict"])
        # Return to the saved GP model example for a readable genuine capture.
        page.locator("#analytics-kind").select_option("workspace-research-gpr")
        for theme in ("dark", "light"):
            page.evaluate("theme=>document.documentElement.dataset.theme=theme", arg=theme)
            if args.axe:
                manifest = json.loads((args.axe.parent / "manifest.json").read_text())
                assert hashlib.sha256(args.axe.read_bytes()).hexdigest() == manifest["script_sha256"]
                page.add_script_tag(path=str(args.axe))
                result = page.evaluate("async()=>await axe.run(document,{runOnly:{type:'tag',values:['wcag2a','wcag2aa','wcag21a','wcag21aa']}})")
                check(theme + " automated accessibility", not result["violations"])
            page.screenshot(path=str(args.evidence_dir / ("gaussian-process-" + theme + ".png")), full_page=True)
        for width in (320, 375, 768, 1440):
            page.set_viewport_size({"width": width, "height": 900})
            check(str(width) + "px no document overflow", page.evaluate("document.documentElement.scrollWidth<=innerWidth"))
        check("all sixteen existing sections retained", page.locator("#section-nav a").count() == 16)
        check("no JavaScript exceptions", not errors)
        check("no external browser requests", not external)
        browser.close()
    report = {"checks": checks, "passed": sum(c["passed"] for c in checks), "failed": sum(not c["passed"] for c in checks),
              "chromium_version": version, "javascript_errors": errors, "external_requests": external,
              "classification": "AUTHORED_SYNTHETIC_RESEARCH_ONLY", "orders_submitted": 0,
              "formal_accessibility_certified": False}
    (args.evidence_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return int(report["failed"] != 0)


if __name__ == "__main__":
    raise SystemExit(main())
