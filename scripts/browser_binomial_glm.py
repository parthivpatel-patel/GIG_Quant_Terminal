"""Actual local Chromium saved-dataset Binomial GLM workflow; no providers or orders."""
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
    checks, errors, external, responsive, axe_reports = [], [], [], [], []

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
                                 ("research-dataset", "research-dataset"), ("research-glm", "research-model"),
                                 ("research-predict", "research-predict")]:
            page.locator("#analytics-kind").select_option("workspace-" + action)
            if action == "research-glm":
                check("explicit Binomial GLM uncertainty caveat", "uncalibrated directional probabilities" in page.locator("#workbench-scope").inner_text())
            with page.expect_response(lambda response, endpoint=endpoint: response.url.endswith("/api/v1/workspace/" + endpoint)) as exchange:
                page.locator("#run-analysis").click()
            page.wait_for_function("!document.getElementById('run-analysis').disabled")
            result = exchange.value.json()
            responses[action] = result
            check(action + " admitted explicit browser action", exchange.value.status == 200 and result["state"] == "AVAILABLE")
            if result["state"] != "AVAILABLE":
                raise AssertionError(action + " failed: " + json.dumps(result))
        check("saved inference does not retrain", not responses["research-predict"]["data"]["metrics"]["training_performed"])
        check("actual Binomial GLM artifact keeps allocation disabled", responses["research-glm"]["data"]["model"] == "binomial_glm" and responses["research-glm"]["data"]["usable_for_portfolio"] is False)
        check("Binomial GLM parameter uncertainty remains unavailable", all(row["uncertainty"]["value"] is None and row["uncertainty"]["calibrated"] is False for row in responses["research-glm"]["data"]["rows"]))
        check("probability model rejects return-MSE interpretation", responses["research-glm"]["data"]["metrics"]["mse"] is None and responses["research-glm"]["data"]["metrics"]["brier_score"] >= 0)
        check("directional output kind is explicit", responses["research-glm"]["data"]["prediction_kind"] == "FORWARD_DIRECTION_PROBABILITY")
        run_id = responses["research-glm"]["data"]["run_id"]
        check("run identity is shown in traceability strip", run_id in page.locator("#workflow-identities").inner_text())
        page.locator("#analytics-kind").select_option("workspace-research-allocate")
        with page.expect_response(lambda response: response.url.endswith("/api/v1/workspace/research-allocate")) as exchange:
            page.locator("#run-analysis").click()
        page.wait_for_function("!document.getElementById('run-analysis').disabled")
        check("unapproved Binomial GLM allocation remains unavailable", exchange.value.json()["state"] == "UNAVAILABLE" and exchange.value.json()["data"] is None)
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
        # Inspect actual saved models under the declared historical cutoff.
        page.locator("#replay-asof").fill(responses["research-dataset"]["provenance"]["as_of"])
        page.locator("#replay-asof").press("Tab")
        page.wait_for_function("!document.getElementById('refresh').disabled")
        page.locator("#analysis-engine").select_option("persisted")
        page.wait_for_function("!document.getElementById('refresh').disabled")
        with page.expect_response(lambda r: "/api/v1/terminal/models?" in r.url):
            page.locator('#section-nav a[href="#models"]').click()
        page.wait_for_function("document.getElementById('content').dataset.section==='models' && !document.getElementById('refresh').disabled && document.getElementById('state-badge').textContent === 'AVAILABLE'")
        check("Models renders saved Binomial GLM with source classification", "MODEL_ESTIMATE" in page.locator("#source-label").inner_text() and "SYNTHETIC INPUTS" in page.locator("#truth-notice").inner_text())
        page.locator("#analytics-kind").select_option("workspace-research-glm")
        page.locator("#analytics-input").fill(json.dumps({"dataset_id": responses["research-dataset"]["data"]["dataset_id"], "model": "binomial_glm", "min_train": 20, "test_size": 10, "embargo": 1, "seed": 17}, indent=2))
        for theme in ("dark", "light"):
            page.evaluate("theme=>document.documentElement.dataset.theme=theme", arg=theme)
            if args.axe:
                manifest = json.loads((args.axe.parent / "manifest.json").read_text())
                assert hashlib.sha256(args.axe.read_bytes()).hexdigest() == manifest["script_sha256"]
                page.add_script_tag(path=str(args.axe))
                result = page.evaluate("async()=>await axe.run(document,{runOnly:{type:'tag',values:['wcag2a','wcag2aa','wcag21a','wcag21aa']}})")
                axe_reports.append({"theme": theme, "violations": result["violations"], "incomplete": result["incomplete"]})
                check(theme + " automated accessibility", not result["violations"])
            page.screenshot(path=str(args.evidence_dir / ("binomial-" + theme + ".png")), full_page=True)
        for width in (320, 375, 768, 1440):
            page.set_viewport_size({"width": width, "height": 900})
            layout = page.evaluate("""() => ({width:innerWidth,document_width:document.documentElement.scrollWidth,
              overflowing_elements:[...document.querySelectorAll('main *')].filter(e=>{
                const r=e.getBoundingClientRect();return !e.closest('.table-scroll,pre')&&r.width>0&&r.left>=0&&r.right>innerWidth+1;
              }).slice(0,16).map(e=>({tag:e.tagName,id:e.id,class_name:e.className,
                text:e.textContent.slice(0,90),right:e.getBoundingClientRect().right}))})""")
            responsive.append(layout)
            check(str(width) + "px no document overflow", layout["document_width"] <= layout["width"])
            if layout["document_width"] > layout["width"]:
                page.screenshot(path=str(args.evidence_dir / ("overflow-"+str(width)+".png")), full_page=True)
        check("all sixteen existing sections retained", page.locator("#section-nav a").count() == 16)
        check("no JavaScript exceptions", not errors)
        check("no external browser requests", not external)
        browser.close()
    report = {"checks": checks, "passed": sum(c["passed"] for c in checks), "failed": sum(not c["passed"] for c in checks),
              "chromium_version": version, "javascript_errors": errors, "external_requests": external,
              "responsive": responsive, "axe": axe_reports,
              "classification": "AUTHORED_SYNTHETIC_RESEARCH_ONLY", "orders_submitted": 0,
              "formal_accessibility_certified": False}
    (args.evidence_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return int(report["failed"] != 0)


if __name__ == "__main__":
    raise SystemExit(main())
