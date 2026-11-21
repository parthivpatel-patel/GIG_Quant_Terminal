"""Actual Chromium byte-provenance audit, with inert authored source text."""
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
    checks, errors, external, other_posts, axe = [], [], [], [], []
    def check(name, value):
        checks.append({"name": name, "passed": bool(value)})
    with isolated_server(args.evidence_dir) as url, sync_playwright() as pw:
        options = {"headless": True, "args": ["--no-sandbox"]}
        if args.executable:
            options["executable_path"] = args.executable
        browser = pw.chromium.launch(**options)
        version = browser.version
        context = browser.new_context(viewport={"width": 1600, "height": 1100}, reduced_motion="reduce")
        def route(request):
            if not request.request.url.startswith(url+"/"):
                external.append(request.request.url)
                request.abort()
            else:
                if request.request.method == "POST" and not request.request.url.endswith("/api/v1/workspace/financial-intelligence"):
                    other_posts.append(request.request.url)
                request.continue_()
        context.route("**/*", route)
        page = context.new_page()
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.goto(url+"/#research", wait_until="networkidle")
        page.wait_for_function("!document.getElementById('refresh').disabled")
        posts = []
        page.on("request", lambda r: posts.append(r.url) if r.method == "POST" else None)
        def action(kind, payload):
            page.locator("#analytics-kind").select_option("workspace-"+kind)
            page.locator("#analytics-input").fill(json.dumps(payload, ensure_ascii=False))
            with page.expect_response(lambda r: r.url.endswith("/api/v1/workspace/financial-intelligence")) as exchange:
                page.locator("#run-analysis").click()
            page.wait_for_function("!document.getElementById('run-analysis').disabled")
            return exchange.value.status, exchange.value.json()
        text = "Café revenue USD 7 million. NaN. <script>window.citationInjected=1</script>"
        admitted = {"operation": "admit-authored", "source_id": "citation-browser-protocol",
                    "title": "GIG-authored synthetic citation protocol", "license": "GIG-AUTHORED", "text": text}
        status, receipt = action("financial-entity-protocol", admitted)
        check("explicit synthetic source admission", status == 200 and receipt["data"]["classification"] == "SYNTHETIC")
        status, saved = action("financial-hybrid", {"operation": "answer", "query": "revenue", "as_of": receipt["provenance"]["as_of"],
            "retrieval_mode": "hybrid", "latent_dimensions": 2, "extract_facts": True, "nlp_profile": "v2"})
        check("saved hybrid v2 query available", status == 200 and saved["data"]["nlp_profile"] == "v2")
        payload = {"operation": "verify-citation", "snapshot_id": saved["data"]["snapshot_id"], "passage_id": saved["data"]["rows"][0]["passage_id"]}
        count = len(posts)
        page.locator("#analytics-kind").select_option("workspace-financial-verify")
        check("selection sends no request", len(posts) == count)
        check("query and passage identity handoff", json.loads(page.locator("#analytics-input").input_value()) == payload)
        check("explicit read-only button", page.locator("#run-analysis").inner_text() == "Verify saved citation")
        status, verified = action("financial-verify", payload)
        proof = verified["data"]
        check("immutable saved query and source verified", status == 200 and proof["saved_query_replay_verified"] and proof["document_integrity_verified"])
        check("actual authored byte hash", proof["quoted_bytes_sha256"] == hashlib.sha256(text.encode()).hexdigest())
        check("authored source bytes match exact text", proof["direct_text_byte_match"] and proof["source_excerpt"] == text)
        check("truth identity freshness and signature remain limited", proof["content_truth_state"] == "UNKNOWN" and proof["issuer_identity_state"] == "UNKNOWN" and proof["freshness_state"] == "NOT_ASSESSED" and proof["signature_state"] == "UNSIGNED_LOCAL_HASH_CHECK")
        check("no new state certification or trading", not proof["new_state_written"] and not proof["certified"] and not proof["trading_approved"])
        check("readable quote and truth limitations", text in page.locator("#analysis-summary").inner_text() and "UNKNOWN / UNKNOWN" in page.locator("#analysis-summary").inner_text())
        check("source script is inert", page.evaluate("window.citationInjected===undefined"))
        status, rejected = action("financial-verify", {**payload, "certified": True})
        check("caller cannot assert certification", status == 422 and rejected["state"] == "ERROR")
        status, _ = action("financial-entity-protocol", {**admitted, "text": "Revenue USD 999 million. Later authored revision."})
        check("later revision explicitly admitted", status == 200)
        status, old = action("financial-verify", payload)
        check("later correction cannot rewrite old proof", status == 200 and old == verified)
        page.reload(wait_until="networkidle")
        page.wait_for_function("!document.getElementById('refresh').disabled")
        status, again = action("financial-verify", payload)
        check("explicit identities survive page reload", status == 200 and again == verified)
        for theme in ("dark", "light"):
            page.evaluate("theme=>document.documentElement.dataset.theme=theme", arg=theme)
            if args.axe:
                manifest = json.loads((args.axe.parent/"manifest.json").read_text())
                assert hashlib.sha256(args.axe.read_bytes()).hexdigest() == manifest["script_sha256"]
                page.add_script_tag(path=str(args.axe))
                report = page.evaluate("async()=>await axe.run(document,{runOnly:{type:'tag',values:['wcag2a','wcag2aa','wcag21a','wcag21aa']}})")
                axe.append({"theme": theme, "violations": report["violations"], "incomplete": report["incomplete"]})
                check(theme+" automated accessibility", not report["violations"])
            page.locator("#workbench").screenshot(path=str(args.evidence_dir/("citation-"+theme+".png")))
        for width in (320, 375, 768, 1440):
            page.set_viewport_size({"width": width, "height": 900})
            check(str(width)+"px no overflow", page.evaluate("document.documentElement.scrollWidth<=innerWidth"))
        check("all sixteen sections", page.locator("#section-nav a").count() == 16)
        check("no JavaScript errors", not errors)
        check("no external requests", not external)
        check("no nonfinancial POST or orders", not other_posts)
        browser.close()
    report = {"checks": checks, "passed": sum(c["passed"] for c in checks), "failed": sum(not c["passed"] for c in checks),
              "chromium_version": version, "axe": axe, "javascript_errors": errors,
              "external_requests": external, "other_posts": other_posts,
              "classification": "AUTHORED_SYNTHETIC_BYTE_PROVENANCE_ONLY", "formal_accessibility_certified": False}
    (args.evidence_dir/"report.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps({k: v for k, v in report.items() if k != "axe"}, indent=2))
    return int(report["failed"] != 0)


if __name__ == "__main__":
    raise SystemExit(main())
