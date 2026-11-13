"""Genuine local Chromium workflow for both variants of bounded family09."""
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

        def action(name, endpoint, payload=None):
            page.locator("#analytics-kind").select_option("workspace-"+name)
            if payload is not None:
                page.locator("#analytics-input").fill(json.dumps(payload))
            with page.expect_response(lambda r: r.url.endswith("/api/v1/workspace/"+endpoint)) as response:
                page.locator("#run-analysis").click()
            page.wait_for_function("!document.getElementById('run-analysis').disabled")
            return response.value.status, response.value.json()

        for name, endpoint in (("pipeline-bars", "bars"), ("pipeline-universe", "universe"), ("research-dataset", "research-dataset")):
            status, result = action(name, endpoint)
            check(name+" is admitted only on explicit action", status == 200 and result["state"] == "AVAILABLE")
        dataset = result["data"]["dataset_id"]
        stamp = result["provenance"]["as_of"]
        for variant in ("rf", "xt"):
            status, result = action("research-"+variant, "research-model", {"dataset_id": dataset, "model": "random_forest" if variant == "rf" else "extra_trees", "min_train": 20, "test_size": 10, "embargo": 1, "seed": 17})
            check(variant+" actual CPU training is available", status == 200 and result["state"] == "AVAILABLE")
            if result["state"] != "AVAILABLE":
                raise AssertionError(json.dumps(result))
            check(variant+" never receives allocation approval", result["data"]["usable_for_portfolio"] is False)
            check(variant+" uncertainty limits remain visible", "not calibrated confidence" in page.locator("#workbench-scope").inner_text())
            run_id = result["data"]["run_id"]
            status, prediction = action("research-predict", "research-predict")
            check(variant+" saved inference does not retrain", status == 200 and prediction["state"] == "AVAILABLE" and not prediction["data"]["metrics"]["training_performed"])
            status, rejected = action("research-allocate", "research-allocate")
            check(variant+" allocator explicitly rejects research-only output", status == 200 and rejected["state"] == "UNAVAILABLE" and rejected["data"] is None)
            page.reload(wait_until="networkidle")
            page.wait_for_function("!document.getElementById('refresh').disabled")
            status, replay = action("research-predict", "research-predict", {"dataset_id": dataset, "run_id": run_id})
            check(variant+" exact saved inference survives reload", status == 200 and replay == prediction)
        page.locator("#replay-asof").fill(stamp)
        page.locator("#replay-asof").press("Tab")
        page.wait_for_function("!document.getElementById('refresh').disabled")
        page.locator("#analysis-engine").select_option("persisted")
        page.wait_for_function("!document.getElementById('refresh').disabled")
        page.locator('#section-nav a[href="#models"]').click()
        page.wait_for_function("document.getElementById('state-badge').textContent==='AVAILABLE' && !document.getElementById('refresh').disabled")
        check("Models labels derived estimates and synthetic inputs", "MODEL_ESTIMATE" in page.locator("#source-label").inner_text() and "SYNTHETIC INPUTS" in page.locator("#truth-notice").inner_text())
        for theme in ("dark", "light"):
            page.evaluate("theme=>document.documentElement.dataset.theme=theme", arg=theme)
            if args.axe:
                manifest = json.loads((args.axe.parent/"manifest.json").read_text())
                assert hashlib.sha256(args.axe.read_bytes()).hexdigest() == manifest["script_sha256"]
                page.add_script_tag(path=str(args.axe))
                result = page.evaluate("async()=>await axe.run(document,{runOnly:{type:'tag',values:['wcag2a','wcag2aa','wcag21a','wcag21aa']}})")
                check(theme+" automated accessibility", not result["violations"])
            page.screenshot(path=str(args.evidence_dir/("forest-models-"+theme+".png")), full_page=True)
        for width in (320, 375, 768, 1440):
            page.set_viewport_size({"width": width, "height": 900})
            check(str(width)+"px no document overflow", page.evaluate("document.documentElement.scrollWidth<=innerWidth"))
        check("sixteen existing navigation sections retained", page.locator("#section-nav a").count() == 16)
        check("zero JavaScript errors", not errors)
        check("zero external browser requests", not external)
        browser.close()
    report = {"checks": checks, "passed": sum(c["passed"] for c in checks), "failed": sum(not c["passed"] for c in checks),
              "chromium_version": version, "javascript_errors": errors, "external_requests": external,
              "classification": "AUTHORED_SYNTHETIC_RESEARCH_ONLY", "orders_submitted": 0, "formal_accessibility_certified": False}
    (args.evidence_dir/"report.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report, indent=2))
    return int(report["failed"] != 0)


if __name__ == "__main__":
    raise SystemExit(main())
