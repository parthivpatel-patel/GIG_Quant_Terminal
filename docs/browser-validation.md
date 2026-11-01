# Browser validation

The recovered terminal was exercised in **real headless Chromium 151.0.7922.34** through Playwright on 2026-11-01. The final integrated run passed **87 checks, zero failures**, with zero browser JavaScript exceptions. The separate Node DOM harness passed **40 smoke checks**; those checks are not browser rendering evidence.

## Reproducing the browser run

Install the project's service, data, research and CPU sequence dependencies, plus Playwright and its Chromium browser, in a permitted local environment. No brokerage credentials are needed. From the repository root:

```bash
python scripts/browser_terminal_check.py --evidence-dir /tmp/gig-browser-evidence
```

Optional automated accessibility scans accept `--axe /path/to/axe.min.js`. The adjacent `manifest.json` must contain the script's `script_sha256`; the harness verifies it before execution. The verified run used axe-core 4.10.3 (MPL-2.0), downloaded from the official npm package with verified npm SHA512 integrity. Its script SHA256 was `880970c081707360e64f34cea25ff91892f5bc95675b0776925b9709dd8a68bb`. Browser binaries and third-party scripts are not embedded in the source repository.

The harness launches an isolated temporary-state HTTP service bound to `127.0.0.1`, removes known provider credential variables from its child environment, disables brokerage connectivity, and aborts browser requests outside that loopback origin. It stops the service and removes its temporary state afterward. It does not submit broker orders, acquire provider data, deploy publicly, or distribute model weights.

## Verified scope

- All twelve sections render explicit local and synthetic truth states and source disclosures.
- Actual browser POST requests admit authored bars and news, persist research, and run offline simulation. Persisted section views are queried with an explicit replay cutoff.
- Additional controls execute offline IEX protocol ingestion, point-in-time universe persistence and selection, real CPU BiLSTM training, saved-model inference using the returned identity, and preparation of a durable offline execution intent. Preparing an intent does not dispatch it.
- An unconfigured FinBERT path produces HTTP 503, `ERROR`, `DEPENDENCY_UNAVAILABLE`, and no model output. Actual FinBERT inference with licensed, verified assets is covered separately by backend tests. The UI identifies the selected ProsusAI weights as CC-BY-NC-SA-4.0 noncommercial research only.
- Local evidence selectors route sequence, sentiment, provider and execution requests to their backend contracts. Synthetic mode disables those selectors. A point-in-time universe identity is explicit and length bounded.
- Persisted SYNTHETIC Treasury fixtures render as a rotating yield surface with a 2D projection, exact quote table, same-date spreads and explicit units. The public-acquisition control requires submission and never fetches on selection; actual public Treasury acquisition is verified separately in backend evidence.
- Authored factor-space and options-volatility charts render; keyboard rotation changes actual canvas pixels, and a 2D alternative is available. Synthetic chart evidence remains labeled synthetic.
- Dark and light themes pass the harness's computed visible-text contrast checks and the pinned axe WCAG A/AA automated rules on the tested pages.
- 320, 375, 768, 1024 and 1920-pixel viewports have no horizontal document overflow; all workbench controls are geometrically inside the viewport. Focused skip navigation remains visible and preserves the selected section. Loading disables refresh; intercepted request failures clear stale successful metrics and show a safe error.

Reproducible UI defects repaired during this run included skip-link hash navigation resetting the selected section, workbench controls clipped by narrow grid tracks, insufficient contrast for unknown states and disabled export controls, and a misleading unimplemented shortcut glyph. Decorative separators are hidden from assistive technology.

## Evidence and limits

The validation command writes `report.json`, `server.txt`, desktop screenshots for both themes and `mobile-320.png` to the requested evidence directory. This session's final evidence is `/workspace/work/evidence/browser-curve-final/`; earlier failing reports remain preserved in sibling directories. Evidence should be included in the independently verified recovery archive by the checkpoint owner.

This is automated Chromium evidence and limited screenshot inspection, not formal WCAG certification, screen-reader testing, cross-browser certification, ARM64 browser execution, provider connectivity validation, economic validation or production readiness. Axe incomplete findings remain in the report for manual review. T4 remains `T4_NOT_CERTIFIED`.
