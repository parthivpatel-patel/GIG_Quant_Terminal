"""Actual local Chromium flow: every registry calculator runs its example and shows a readable summary."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from browser_stage_a_regressions import isolated_server
from playwright.sync_api import sync_playwright

from gig.service.quant_analytics import ANALYSES


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-dir", type=Path, required=True)
    parser.add_argument("--axe", type=Path)
    parser.add_argument("--executable")
    args = parser.parse_args()
    args.evidence_dir.mkdir(parents=True, exist_ok=True)
    checks, errors, external, posts = [], [], [], []

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
                if route.request.method == "POST":
                    posts.append(route.request.url)
                route.continue_()
            else:
                external.append(route.request.url)
                route.abort()

        context.route("**/*", restrict)
        page = context.new_page()
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.goto(url + "/#research", wait_until="networkidle")
        page.wait_for_function("!document.getElementById('refresh').disabled")

        def run(kind, change=None):
            page.locator("#analytics-kind").select_option(kind)
            payload = json.loads(page.locator("#analytics-input").input_value())
            if change:
                payload.update(change)
                page.locator("#analytics-input").fill(json.dumps(payload))
            with page.expect_response(lambda response: "/api/v1/analytics/" in response.url or response.url.endswith("/api/v1/workspace/calculator-journal")) as exchange:
                page.locator("#run-analysis").click()
            page.wait_for_function("!document.getElementById('run-analysis').disabled")
            summary = page.locator("#analysis-summary")
            return exchange.value, exchange.value.json(), summary.inner_text() if summary.is_visible() else ""

        offered = page.locator("#analytics-kind option").evaluate_all("options=>options.map(option=>option.value)")
        check("every registered calculator is offered", all(key in offered for key in ANALYSES))
        count = len(posts)
        page.locator("#analytics-kind").select_option("hierarchical-risk-parity")
        check("selection sends no request", len(posts) == count)
        for key, analysis in ANALYSES.items():
            response, body, summary = run(key)
            data = body.get("data") or {}
            check(key + " computes its example as a research-only model estimate",
                  response.status == 200 and response.url.endswith("/api/v1/analytics/" + key) and body["state"] == "AVAILABLE"
                  and body["provenance"]["classification"] == "MODEL_ESTIMATE" and body["provenance"]["readiness"] == "RESEARCH_ONLY")
            check(key + " shows its headline and that nothing was saved",
                  all(label in summary for label, _ in analysis.headline) and "Nothing was saved or sent" in summary
                  and len(data.get("headline", [])) == len(analysis.headline))
        response, body, summary = run("minimum-cvar", {"minimum_return": 5})
        check("an unreachable return floor is refused with no summary", response.status == 422 and body["state"] == "ERROR" and not summary)
        response, body, summary = run("american-option", {"broker": "live"})
        check("an unknown field is refused", response.status == 422 and body["state"] == "ERROR" and not summary)
        response, body, summary = run("american-option")
        check("American put shows a positive early-exercise premium beside the European value",
              body["data"]["early_exercise_premium"] > 0 and "Early-exercise premium" in summary)
        check("only stateless analytics endpoints were called", all("/api/v1/analytics/" in post for post in posts))
        response, saved, summary = run("workspace-calculator-save")
        record = (saved.get("data") or {}).get("record_id")
        check("a calculation is saved only by the explicit journal action", response.status == 200 and saved["state"] == "AVAILABLE"
              and isinstance(record, str) and len(record) == 64 and saved["data"]["operation"] == "calculator"
              and "Saved calculation" in summary and "Nothing was saved" not in summary)
        page.locator("#analytics-kind").select_option("workspace-calculator-replay")
        handed = json.loads(page.locator("#analytics-input").input_value())
        check("the saved record is handed to replay", handed.get("record_id") == record)
        response, replayed, summary = run("workspace-calculator-replay")
        check("the saved calculation replays from its pinned inputs", response.status == 200
              and replayed["data"]["replay_state"] == "REPRODUCED_FROM_PINNED_INPUTS" and replayed["data"]["record_id"] == record
              and "REPRODUCED_FROM_PINNED_INPUTS" in summary)
        for theme in ("dark", "light"):
            page.evaluate("theme=>document.documentElement.dataset.theme=theme", arg=theme)
            if args.axe:
                manifest = json.loads((args.axe.parent / "manifest.json").read_text())
                assert hashlib.sha256(args.axe.read_bytes()).hexdigest() == manifest["script_sha256"]
                page.add_script_tag(path=str(args.axe))
                result = page.evaluate("async()=>await axe.run(document,{runOnly:{type:'tag',values:['wcag2a','wcag2aa','wcag21a','wcag21aa']}})")
                check(theme + " automated accessibility", not result["violations"])
            page.locator("#workbench").screenshot(path=str(args.evidence_dir / ("quant-analytics-" + theme + ".png")))
        for width in (320, 375, 768, 1440):
            page.set_viewport_size({"width": width, "height": 900})
            check(str(width) + "px no document overflow", page.evaluate("document.documentElement.scrollWidth<=innerWidth"))
        check("all sixteen existing sections retained", page.locator("#section-nav a").count() == 16)
        check("no JavaScript exceptions", not errors)
        check("no external browser requests", not external)
        browser.close()
    report = {"checks": checks, "passed": sum(c["passed"] for c in checks), "failed": sum(not c["passed"] for c in checks),
              "chromium_version": version, "javascript_errors": errors, "external_requests": external,
              "classification": "CALLER_SUPPLIED_EXAMPLES_RESEARCH_ONLY", "orders_submitted": 0,
              "formal_accessibility_certified": False}
    (args.evidence_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return int(report["failed"] != 0)


if __name__ == "__main__":
    raise SystemExit(main())
