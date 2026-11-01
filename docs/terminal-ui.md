# Local terminal interface

The buildless terminal consists of `src/gig/service/static/index.html`, `styles.css`, and `app.js`. It needs no JavaScript package install, external fonts, CDN assets, or separate frontend build. The Python service serves the assets locally.

Navigation exposes Command Center, Markets, Signals, Portfolio, Risk, Models, NLP / News, Statistical Arbitrage, Options, Execution, Research, and Operations. Each requests `/api/v1/terminal/{section}?mode=local|synthetic`. Local queries also send an explicit maximum-data-age policy (one hour, one day, or one week; the visible default is one day), plus an optional timezone-aware point-in-time cutoff. Default mode is `local`; the user must explicitly select synthetic research data. A navigation change retains the selected mode, freshness policy, and replay cutoff. Cutoff text is URL-encoded verbatim; it never passes through browser `Date` conversion that could truncate nanoseconds. The controls explain that an empty cutoff means current time and that the freshness policy applies to local data. Unsupported local capabilities display the backend's unavailable state; the interface never creates fallback prices or brokerage performance.

## Contract and evidence handling

Caller-supplied workspace provenance is explicitly marked as vendor-unverified. A source name, classification, or content hash does not establish that a provider supplied the content. The terminal admits version `1` envelopes containing an explicit `AVAILABLE`, `UNAVAILABLE`, `UNKNOWN`, `ERROR`, or `STALE` state, provenance, data, and public errors. Available/stale responses require a data object. Unknown/unavailable/error responses require null data; errors require a code and message. Stale snapshots require a source timestamp and display an explicit freshness warning. A valid typed HTTP error retains its public backend message. Network failures and invalid envelope shapes display a generic local error and clear previous data.

Null and missing metric values display `Unknown`; numeric zero remains `0`, and signed zero displays `−0`. Tagged `{$integer:"..."}` and `{$decimal:"..."}` scalar values display their exact original strings. Table sorting compares their signs, decimal orders, and significands without converting them to binary64 or expanding exponent-sized buffers. Mixed ordinary numeric and tagged values sort numerically; textual values retain textual sorting. Unknown values remain last in either sort direction. Nanosecond timestamp text is displayed verbatim.

Source and article content enters the DOM through text nodes. No `innerHTML`, dynamic script evaluation, or raw HTML provider rendering is used. Analytics inputs are parsed only for local JSON syntax/object checking; the exact textarea source is sent to the backend. Duplicate keys, whitespace, original research text, and identity-bearing integer lexemes therefore reach the server unchanged for strict admission. The UI does not imply that locally parseable JSON is a valid backend request.

CSV export contains the currently filtered/sorted observations. Fields and headers are quoted, string formula prefixes are escaped, and exact scalar tags are exported as apostrophe-prefixed text to reduce spreadsheet identity rounding. CSV import behavior depends on the spreadsheet application; exported tagged scalars intentionally carry the text marker.

## Local workspace actions

The workbench is available from every terminal section. Its four explicit workspace actions post raw JSON to `/api/v1/workspace/bars`, `/news`, `/research`, and `/simulate`. They admit caller-supplied observations or financial text, save purged research experiments, or run the strictly offline simulator. These operations persist evidence and research state on the local service; the interface states this before submission. They cannot place brokerage orders. Ordinary model calculations continue to use `/api/v1/analytics/{kind}`.

All bundled examples are authored synthetic fixtures with September 2026 timestamps and fictional SYN001/SYN002 instruments. Loading an example only fills the editor. It does not send a request, admit data, switch the data environment, or place an order. To inspect stored examples with a current freshness policy, select an explicit replay cutoff corresponding to their dated observations. The examples do not create a default demo inside local mode.

Workspace requests use `Content-Type: application/json`. Original textarea bytes are transmitted unchanged, preserving duplicate-key evidence, original source whitespace, and integer lexemes for server admission. A valid `ERROR`, `UNKNOWN`, or stale result remains that state in the output; a rejected write does not display a successful-save message or trigger a success refresh. An available write refreshes the selected section only when the user already selected local mode.

## Analytical controls and accessibility

The self-contained canvas renderer supports factor scatter plots and sampled volatility/risk surfaces. Axes, units, tooltips, and source evidence come from the backend. Surface lines connect existing samples sharing an axis value; the renderer creates no interpolated observations. Only finite points are plotted, with a 1,600-point drawing cap. Table rendering is capped at 500 visible rows, while export uses all filtered rows.

Controls include drag rotation, wheel zoom, arrow-key rotation, plus/minus zoom, reset, and a 2D projection. The interface provides dark and light themes, keyboard focus styling, explicit navigation names, a skip link, a keyboard-scrollable table, semantic headings, and status/error announcements. The numerical table remains the textual alternative to charts. These controls are implemented; a formal accessibility-conformance claim has not been established.

## Verification

Run from the repository root:

```sh
node --check src/gig/service/static/app.js
node --check scripts/check_terminal.mjs
node scripts/check_terminal.mjs
```

The retained Node VM harness performs 27 smoke checks, including all twelve paths in both modes, truth states, nanosecond text, signed zero, precise tags, large integer and mixed decimal sorting, source-text rendering, filter/export behavior, chart/2D/keyboard controls, exact analytics request source, explicit freshness and replay precision, all four workspace write paths without automatic admission, rejected-write truth handling, caller provenance warnings, and invalid-contract/network failures. It also checks HTML IDs, label targets, static DOM references, and absence of external shell assets.

The harness uses a DOM/canvas stub and synthetic responses. It does not launch a browser, evaluate layout, verify contrast, establish screen-reader behavior, test browser compatibility, validate screenshots, or test live HTTP integration. Those remain separate checks. No Site registration, hosting, or deployment was performed.
