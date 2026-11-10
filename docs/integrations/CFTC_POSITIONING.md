# CFTC historical positioning: supported observed vertical

The admitted source is the CFTC Public Reporting Environment's **Traders in
Financial Futures — Futures Only** report, dataset `gpe5-46if`. A bounded,
credential-free HTTPS query retrieved 52 official Euro FX reports for 2023,
contract market code `099741`, CME, units `(CONTRACTS OF EUR 125,000)`.
The source sample runs from 2023-01-03 through 2023-12-26. It is not current
position coverage and is visibly classified **HISTORICAL / STALE**.

This is an actual observed aggregate-positioning dataset. It is not options
prints, sweeps, individual institutional transactions, fund flows, dealer gamma
exposure, or off-exchange trading. The CFTC's “Dealer” classification names a
reporter category; it does not establish an options dealer's gamma inventory.

## Verified access and reuse scope

The official [CFTC Web Policy](https://www.cftc.gov/WebPolicy/index.htm) returned
HTTP 200 with the application's descriptive public-research User-Agent. Its
Copyright section states:

> Government information at the CFTC website is in the public domain. Public
> domain information may be freely distributed and copied, but it is requested
> that in any subsequent use the CFTC be given appropriate acknowledgement.

The policy separately warns about privately contributed/licensed resources.
This integration admits only the CFTC government aggregate report. It does not
vendor CFTC seals, photographs or unrelated third-party assets. Attribution is
preserved; GIG's descriptive modifications do not imply CFTC endorsement.

The official [COT report page](https://www.cftc.gov/MarketReports/CommitmentsofTraders/index.htm)
returned HTTP 200 and explicitly describes the Public Reporting Environment API
and downloadable report formats. Its API FAQ says:

> Currently, we are not providing tokens to use the Public Reporting APIs.
> People who are using the API are generally using it successfully without
> using a token. As long as you are not overusing the API, you should be able
> to use the API without a token.

No token, identity, account, paid entitlement, brokerage connection or agreement
was introduced. Dataset metadata's license field was null; permission is based
on the explicit official government-information policy and the identified CFTC
report, not inferred from website accessibility. An uncustomized Python
User-Agent received 403 on the policy pages; the ordinary application
User-Agent received 200. No identity impersonation or access bypass was used.

Pinned metadata records acquisition time, exact query URL, SHA-256 of source
bytes, permission/API quotations and hashes of the retrieved official pages.
The raw JSON is preserved without reformatting in
`research_org/fixtures/observed_cftc_2023.json`. Provenance is adjacent in
`observed_cftc_2023.metadata.json`. Dataset metadata and raw source were fetched
successfully; the first selected-field query returned 400 because the provider's
short-total field is `tot_rept_positions_short`. The final bounded query used
the complete official records; no missing field was fabricated.

## Measurement and temporal contracts

Five category rows retain exact integer long/short counts and reported spread
positions where supplied. Nonreportable spread is **unknown**, not zero.
For every report, reportable plus nonreportable positions balance to open
interest on both sides; category net positions sum to zero. Invalid contract
identity, noninteger/negative/oversized counts, duplicates and conservation
failures are rejected.

Derived values are descriptive:

- Net contracts = long minus short.
- Net share of open interest = net / open interest; zero open interest yields
  an unknown ratio.
- Weekly net change compares adjacent official report weeks. Actual source
  week 27 is dated July 3, 2023, so a hard seven-calendar-day rule would wrongly
  discard that holiday-shifted report. Gaps in reporting weeks yield unknown
  weekly changes.
- Prior-only percentile uses midranks against strictly earlier report
  observations, at least eight required. The scale is 0–1. It is not a trading
  signal, predictive result or independent historical knowledge reconstruction.

The source encodes date labels as `T00:00:00.000`. That is not retained as an
asserted observation or publication timestamp. Report dates remain dates;
observation/publication clocks remain UNKNOWN. The acquisition timestamp is the
known revision clock for this downloaded snapshot. Earlier historical
publication/revision vintages are not supplied, and point-in-time trading
eligibility is false. `report_through=YYYY-MM-DD` selects a retrospective
observation-date sample only.

`positioning_contract(report_through=None)` returns an available **historical
research snapshot**, with nested `freshness.state=STALE` for current-market use.
The envelope's `as_of` is explicitly the recorded acquisition timestamp, not a
manufactured midnight market timestamp. Missing/corrupt source returns an
UNAVAILABLE contract; no synthetic observations replace it. Reading status or
opening the workspace makes no provider request. The fixed-host bounded fetch
function runs only when explicitly invoked by development/ingestion code.

## Reproduction and verification

```python
from gig.service.positioning import positioning_contract
snapshot = positioning_contract().to_dict()
```

The pinned source's first open interest is 712,030 contracts; last is 705,627.
Asset-manager net positions are 349,543 first and 365,700 last. Last leveraged
money net is −4,233. Last asset-manager weekly change is +2,382; leveraged-money
change is +3,232. These are observed report aggregates and derived differences,
not inferred actual buys, sales or capital transfers.

Tests cover independent integer conservation/reference metrics, source hashes,
identity and precision admission, holiday/report-week boundaries, future
perturbations, observation cutoff honesty, zero/unknown ratios, corrupt source,
restart/replay, defensive contract reads and sanitized acquisition failures.
