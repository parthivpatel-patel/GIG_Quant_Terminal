# Next development phase

Follows PR #3 (Sprint 9 reconstruction, deep-black redesign, six 3D modules,
Global Intelligence). PR #3 stays the review checkpoint; this phase lands on
follow-up branches based on it.

Boundaries carried unchanged: strict $0 additional spending, no brokerage
orders (live or paper), no fabricated data, no reuse of third-party code without
a compatible license, no real-time tracking of military units, aircraft or
vessels, no targeting or interception support. `fully_funded_certified=false`,
`T4_NOT_CERTIFIED` and Historical §139 UNSUPPLIED remain.

## Admission gate for any new source

A source moves from CANDIDATE to INTEGRATED only when all of these are recorded
in `gig.intel.sources` (or the market-data provenance registry) and tested:

1. Reuse terms read and quoted with their URL and review date; redistribution
   and caching permitted for this use.
2. Fixed HTTPS endpoint reachable from the CI and agent environments without a
   paid plan; any free key comes from a secret, never the repository.
3. Provider rate limit and refresh interval encoded; backoff tested.
4. Normalizer tested against a recorded real response, with malformed,
   non-finite and redirect cases failing closed.
5. Time semantics stated (DELAYED, HISTORICAL or aggregated period); LIVE only
   with verified real-time semantics.

Listed sources below are candidates. None is asserted usable until it passes
the gate.

## Workstreams

| Area | Candidate free sources (terms to verify) | Deliverable | Main risk |
| --- | --- | --- | --- |
| Permitted real market data | ECB reference rates (integrated); U.S. Treasury Fiscal Data yield curves; SEC EDGAR filings (declared User-Agent, fair-access rate); FRED (free key) | Observed daily rates and filings admitted through `data/provenance.py`; vol-surface and correlation builders run on admitted series where coverage exists | No free licensed equity or option quote feed; most vendor sites forbid redistribution |
| Maritime | NOAA/BOEM Marine Cadastre historical AIS (U.S. waters, archived); IMF PortWatch aggregates | Historical, aggregated port-call and chokepoint-transit counts by day; no vessel positions | Files are large; aggregate terms need review; global live AIS stays BLOCKED |
| Civil aviation | BTS on-time and traffic statistics; OpenSky historical (non-commercial, unreachable here) | Monthly and daily aggregated traffic by airport and region | OpenSky network access and terms; no per-aircraft display |
| Satellite imagery | NASA GIBS WMTS tiles; Sentinel-2 open-data STAC catalogue (not requester-pays) | Optional, user-triggered periodic imagery layer with acquisition date; off by default | External tiles conflict with the current no-external-asset rule, so they need an explicit opt-in; requester-pays buckets excluded |
| Environmental | NWS alerts API (public domain, User-Agent); GDACS after terms review; FIRMS with a free MAP_KEY secret | Additional DELAYED hazard layers through the same adapter contract | FIRMS key must be supplied by the user as a secret |
| Historical defense intelligence | UCDP datasets (CC BY 4.0); SIPRI military expenditure (citation terms) | Annual and event-level historical aggregates by country and region, admitted through `intel/defense.py` | Must stay aggregated and delayed; ACLED excluded unless its terms allow this use |
| Geopolitical market risk | GDELT event aggregates (citation required) | Daily theme-attention counts joined to the risk monitor, labelled not causal | Volume and noise; must not be presented as forecasts |
| Portfolio exposure and event studies | Funded paper portfolio and admitted observed series already in the repository | Event studies run on admitted ECB/UCI series and funded holdings; theme exposure read from funded positions instead of caller tags where metadata exists | Short observed history limits statistical power |
| Cross-browser and accessibility | Playwright WebKit and Firefox; axe-core 4.10.3 (pinned) | Same acceptance suites on WebKit and Firefox in CI; axe on all thirteen sections; manual keyboard and screen-reader pass recorded | Firefox failed to start in this VM (uid_map EROFS); may need CI-only runs |
| Performance | Local measurement only | Budgets for first paint, section switch, 3D frame time and `/api/v1/intel/overview` latency, recorded in CI; lazy section loading kept | Headless timing noise; budgets must be relative |

## Order of work

1. Cross-browser and axe coverage for all sections, since every later feature
   depends on that acceptance harness.
2. Market data (Treasury, EDGAR) and event-study integration with funded
   holdings, because they extend existing provenance code.
3. Environmental and historical defense aggregates (NWS, UCDP), then maritime
   and aviation aggregates.
4. Opt-in imagery layer last, because it changes the external-asset policy.
5. Performance budgets in parallel once the suites are stable.

## Done for this phase on `cursor/gig-next-phase-26c4`

- Intel acquisition hardening: redirects off the registry host refused before
  following, strict JSON, payloads that are malformed or contain non-finite
  values fail closed with backoff, USGS `provider_updated_at`, conservative
  event-study day 0, and partitioned risk-monitor counts.
- 3D builders take explicit inputs, with input-sensitivity and no-look-ahead
  tests and a browser check that rendering follows the payload.
- 3D label fitting, halos, clamping and collision avoidance.
- axe-core WCAG A/AA scans of Options and Global Intelligence in both themes,
  pinned and SHA-256 verified in CI.
