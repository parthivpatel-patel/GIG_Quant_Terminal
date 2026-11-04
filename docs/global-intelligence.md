# Global Intelligence workspace

The 13th terminal section. It shows natural hazards, cited geopolitical and
delayed public defense records, and approximate maritime reference geometry, and
links them to financial research themes. It is a research tool for
financial-risk awareness, not a live, operational or tactical feed.

Code: `src/gig/intel/` (sources, normalization, store, fetch, defense policy,
research), `src/gig/service/intel_workspace.py`, routes in
`src/gig/service/app.py`, client `src/gig/service/static/intel.js`.
Tests: `tests/test_global_intelligence.py`, `scripts/browser_global_3d.py`.

## Sources and permissions

| Source | Layer | Status | Terms / reason |
| --- | --- | --- | --- |
| USGS Earthquake Hazards GeoJSON (M2.5+, 7 days) | hazards | INTEGRATED | U.S. Government public domain; fixed HTTPS URL |
| NASA EONET v3 open events (30 days) | hazards | INTEGRATED | NASA data without copyright restriction; upstream sources keep their own terms |
| GDACS event list | hazards | CANDIDATE | Endpoint reachable 2026-11-04; reuse terms not yet reviewed, so not ingested |
| NASA FIRMS | hazards | BLOCKED | Needs a free MAP_KEY credential; none configured. Thermal detections are not evidence of military activity |
| Global AIS | maritime | BLOCKED | Free streams require keys and restrict redistribution; no vessel positions shown or simulated |
| OpenSky Network | aviation | BLOCKED | Connection timed out from this environment; non-commercial terms; no aircraft positions shown or simulated |
| Copernicus Sentinel / USGS Landsat | imagery | BLOCKED | Registered accounts required; the terminal loads no external tiles; imagery is periodic, never live video |
| GIG reference geometry | reference | REFERENCE | Approximate chokepoint and port locations; no traffic volumes asserted |
| Analyst-admitted reports | geopolitical / defense | INTEGRATED | Each record cites its publisher; reuse is governed by that publisher |
| Basemap | — | BUNDLED | Natural Earth 1:110m land and boundaries, public domain, rounded to 0.1° (`static/world-110m.json`) |

Every registry entry is returned by `GET /api/v1/intel/overview` and rendered in
the "Source availability" panel, so unavailable layers state their blocker in
the UI instead of being hidden.

## Data semantics

Labels used: DELAYED, HISTORICAL, CACHED, DEMONSTRATION, UNAVAILABLE. LIVE is
never emitted: no integrated source has verified real-time semantics.

- A provider is DELAYED after a successful acquisition within its staleness
  window, CACHED after a failure or once stale, UNAVAILABLE before any success.
- A replay cutoff more than one hour in the past makes every provider HISTORICAL
  (or UNAVAILABLE if nothing had been acquired by then). Replay returns only
  revisions acquired at or before the cutoff.
- Authored demonstration events and admitted SYNTHETIC records are labeled
  DEMONSTRATION; they are drawn with dashed markers.

Each event carries provider, dataset, license, observation time, publication
time (or "not supplied"), acquisition time, reporting delay, geographic
uncertainty with its basis, verification state, revision count and hash, and
citations.

## Provider requests

Reads never contact a provider. `POST /api/v1/intel/refresh` is the only path:
it accepts only a registry source, uses that source's fixed HTTPS URL, allows
redirects only to the same host, bounds the response size, enforces the
source's minimum refresh interval and exponential failure backoff, and appends
immutable revisions (a changed provider record becomes a new revision; an
unchanged one is not duplicated). Deferred and failed attempts return a status
record; previously acquired events remain available as CACHED.

## Geopolitical and defense research

Categories ARMY, AIR FORCE, NAVY and GEOPOLITICAL accept announcements, notices
and historical aggregates only. Admission (`gig.intel.defense.admit_record`)
enforces:

- at least 24 hours between publication and admission;
- coordinates coarsened to a 1° grid (about 111 km);
- at least one HTTPS citation; CONFIRMED_OFFICIAL requires an official or
  intergovernmental publisher, MULTI_SOURCE_REPORTED at least two publishers;
- a closed field set: unit identifiers, callsigns, hull or tail numbers,
  headings, speeds and tracks are rejected;
- interpretation fixed to NOT_EVIDENCE_OF_IMPENDING_CONFLICT.

Verification states distinguish confirmed official, multi-source reported,
unverified, historical observation and analytical scenario. The workspace shows
weekly counts by branch and region; it provides no positions, tracks,
forecasts, targeting or interception support.

## Financial research links

- Theme links (`research_links`) map an event category and proximity to a
  chokepoint (500 km) to themes such as crude oil, natural gas/LNG, tanker and
  container shipping, aviation, defense, insurance, regional FX and equities.
  They are labeled RESEARCH_HYPOTHESIS_NOT_CAUSAL.
- The Geopolitical Market Risk Monitor counts linked events per theme, split
  into confirmed and unverified; counts are attention indicators, not forecasts.
- Event studies (mean-adjusted or market model) estimate strictly before the
  event window and are labeled DESCRIPTIVE_SINGLE_EVENT_NOT_CAUSAL.
- Theme exposure accepts long-only holdings with caller-supplied tags; scenario
  shocks apply the most adverse shock once per holding and are labeled
  ANALYTICAL_SCENARIO.

## World Monitor and WorldMirror evaluation

Both repositories were reviewed through GitHub metadata on 2026-11-04 for
design ideas. No code, data or assets were copied from either.

| Project | License | Language | Activity | Disposition |
| --- | --- | --- | --- | --- |
| koala73/worldmonitor | AGPL-3.0 | TypeScript | ~90k stars, pushed 2026-11-04, 344 open issues | Not reused |
| houalexdev/WorldMirror | MIT | Python | 2 stars, last push 2026-03-02 | Not reused |

World Monitor is AGPL-3.0. GIG declares MIT in `pyproject.toml`. Incorporating
AGPL code would make the combined served application AGPL, including the
network-use source obligation, which conflicts with the declared license. Its
stack (TypeScript, external map tiles and many live feeds) also does not fit the
terminal's no-external-asset and lazy-loading constraints. Only general,
non-copyrightable ideas are shared: a layered map, per-source health and a
chronological feed.

WorldMirror is MIT and therefore license-compatible, but it is small and
inactive; GIG's own adapters and provenance model already cover what it would
provide, so nothing was taken.

## Limitations

- Coverage is limited to the integrated sources and admitted reports; absence
  of an event is not evidence of absence.
- Event locations are provider-reported points or polygon centroids; USGS
  summary feeds omit horizontal error, so uncertainty is shown as unknown.
- Theme links and risk-monitor counts are not validated predictors of prices.
