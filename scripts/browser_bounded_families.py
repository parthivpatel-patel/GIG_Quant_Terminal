"""Actual local Chromium saved-dataset workflow for every registered bounded family; no providers or orders."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from browser_stage_a_regressions import isolated_server
from playwright.sync_api import sync_playwright

from gig.ml.bounded import FAMILIES

# The authored browser fixture has signed features and fewer than 100 rows per
# instrument, so these two must refuse visibly instead of fitting.
REFUSED = {"nmf", "evt_pot"}


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

        def run(kind, endpoint):
            page.locator("#analytics-kind").select_option(kind)
            scope = page.locator("#workbench-scope").inner_text()
            with page.expect_response(lambda response: response.url.endswith("/api/v1/workspace/" + endpoint)) as exchange:
                page.locator("#run-analysis").click()
            page.wait_for_function("!document.getElementById('run-analysis').disabled")
            return exchange.value.json(), scope

        dataset = None
        for action, endpoint in [("pipeline-bars", "bars"), ("pipeline-universe", "universe"), ("research-dataset", "research-dataset")]:
            dataset, _ = run("workspace-" + action, endpoint)
            check(action + " admitted explicit browser action", dataset["state"] == "AVAILABLE")
        offered = page.locator("#analytics-kind option[value^='workspace-research-family-']").count()
        check("every registered family is offered once", offered == len(FAMILIES))
        # A family whose optional library is absent must refuse visibly, never substitute.
        refused = REFUSED | {key for key, family in FAMILIES.items() if not family.available()}
        fitted = 0
        for key, family in FAMILIES.items():
            result, scope = run("workspace-research-family-" + key, "research-model")
            check(key + " scope states the research-only boundary", scope == family.scope)
            if key in refused:
                check(key + " refuses unsupported data or a missing library without an artifact",
                      result["state"] != "AVAILABLE" and result.get("data") is None)
                continue
            data = result.get("data") or {}
            ok = result["state"] == "AVAILABLE" and data.get("model") == key
            check(key + " actual fit is saved and stays research-only", ok and data["usable_for_portfolio"] is False
                  and data["prediction_kind"] == family.kind and isinstance(data["metrics"]["family"], dict)
                  and all(r["uncertainty"]["calibrated"] is False and not r["usable_for_portfolio"] for r in data["rows"]))
            if not ok:
                continue
            fitted += 1
            replay, _ = run("workspace-research-predict", "research-predict")
            check(key + " saved replay performs no training", replay["state"] == "AVAILABLE"
                  and replay["data"]["metrics"]["training_performed"] is False and replay["data"]["usable_for_portfolio"] is False)
            if key in {"huber", "arima", "gmm"}:
                rejected, _ = run("workspace-research-allocate", "research-allocate")
                check(key + " allocation remains unavailable", rejected["state"] == "UNAVAILABLE" and rejected["data"] is None)
        check("all supported families fitted in the browser", fitted == len(FAMILIES)-len(refused) and fitted >= 16)
        page.locator("#analytics-kind").select_option("workspace-research-family-quantile")
        for theme in ("dark", "light"):
            page.evaluate("theme=>document.documentElement.dataset.theme=theme", arg=theme)
            if args.axe:
                manifest = json.loads((args.axe.parent / "manifest.json").read_text())
                assert hashlib.sha256(args.axe.read_bytes()).hexdigest() == manifest["script_sha256"]
                page.add_script_tag(path=str(args.axe))
                result = page.evaluate("async()=>await axe.run(document,{runOnly:{type:'tag',values:['wcag2a','wcag2aa','wcag21a','wcag21aa']}})")
                check(theme + " automated accessibility", not result["violations"])
            page.screenshot(path=str(args.evidence_dir / ("bounded-" + theme + ".png")), full_page=True)
        for width in (320, 375, 768, 1440):
            page.set_viewport_size({"width": width, "height": 900})
            check(str(width) + "px no document overflow", page.evaluate("document.documentElement.scrollWidth<=innerWidth"))
        check("all sixteen existing sections retained", page.locator("#section-nav a").count() == 16)
        check("no JavaScript exceptions", not errors)
        check("no external browser requests", not external)
        browser.close()
    report = {"checks": checks, "passed": sum(c["passed"] for c in checks), "failed": sum(not c["passed"] for c in checks),
              "families": len(FAMILIES), "families_fitted": fitted, "families_refused": sorted(refused), "chromium_version": version, "javascript_errors": errors,
              "external_requests": external, "classification": "AUTHORED_SYNTHETIC_RESEARCH_ONLY", "orders_submitted": 0,
              "formal_accessibility_certified": False}
    (args.evidence_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return int(report["failed"] != 0)


if __name__ == "__main__":
    raise SystemExit(main())
