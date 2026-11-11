# Official equity-index futures positioning

The terminal adds a fixed 2023 **E-mini S&P 500 futures aggregate-positioning**
sample. This is an observed equity-market derivatives research vertical. It does
not provide cash-equity prices, stock-return histories, constituent universes,
executed trades, options flow, dark-pool activity or dollar fund flows.

## Verified source and permission

The source is the CFTC's official [Traders in Financial Futures — Futures Only
public API](https://publicreporting.cftc.gov/resource/gpe5-46if.json), dataset
`gpe5-46if`. The exact filtered, ordered request is recorded in
`src/gig/research_org/fixtures/observed_cftc_sp500_2023.metadata.json`.
An actual HTTPS 200 response was acquired on **2026-11-11T13:06:18.210685+00:00**.
It contains 52 reports, January 3–December 26, 2023, 162,849 original bytes.

Original response SHA-256:
`cdd469e84c79d21759cf858a246500123ffb90f82b1491480fe74f698d003eaf`.
The loader independently pins both raw-response and metadata hashes in code,
then validates exact contract identities, counts and open-interest conservation.
It rejects altered evidence without substituting another dataset.

The [CFTC Web Policy](https://www.cftc.gov/WebPolicy/index.htm) was fetched
successfully on 2026-11-11 and explicitly states:

> Government information at the CFTC website is in the public domain. Public
> domain information may be freely distributed and copied, but it is requested
> that in any subsequent use the CFTC be given appropriate acknowledgement.

The grant applies here to CFTC government aggregate reports. It does not cover
third-party photos, logos or licensed contributions. Attribution accompanies the
derived view. The official [COT information
page](https://www.cftc.gov/MarketReports/CommitmentsofTraders/index.htm) was also
fetched successfully and confirms public API access without tokens for
nonexcessive use. URL, response status, acquisition clock, page SHA-256 and exact
relevant quotation are preserved in the metadata. No credentials, paid service,
exchange feed or redistribution of vendor trading records is involved.

## Identity and accounting of counts

- CFTC contract-market code: **13874A**.
- Exact name: **E-MINI S&P 500 - CHICAGO MERCANTILE EXCHANGE**.
- Source units: **($50 X S&P 500 INDEX)**.
- Market code: **CME**; commodity code: **138**; report type: **FutOnly**.
- Report IDs encode date, contract and futures-only suffix. Consolidated S&P,
  micro E-mini, options-combined reports and other identities are rejected.
- Dealers, asset managers, leveraged money, other reportable and nonreportable
  categories are preserved. Spread counts are not added to net positions.
  Nonreportable spread is unavailable, not imputed as an observed zero.
- Both long and short sides, including reportable spreads, independently
  reconcile to reported open interest. Integer counts are parsed exactly.

On the first report, open interest was 2,010,996 contracts and asset-manager net
positions were +436,461. On the final report, open interest was 2,155,313 and
asset-manager net was +878,373; leveraged-money net was −354,995. These are
**descriptive counts**, not independent predictive evidence or inferred flows.

## Time, research and corporate-action boundaries

Report dates are source observation dates; the API's midnight encoding is not an
assertion of observation or publication time. Historical publication and
revision vintages are **UNKNOWN**. Only the later acquisition clock is verified.
The optional `report_through` filter selects an observation-date sample and does
not reconstruct what a trader knew on that date.

Reported net changes and prior-only percentiles use earlier observation rows;
at least eight prior rows are required for a percentile. Missing reporting
weeks yield unavailable weekly changes. Neither quantity is an approved signal.
Earlier calculations remain unchanged when later observation rows are perturbed,
which is tested, but this does not establish historical knowledge availability.

Cash-equity corporate actions, underlying constituent changes and total-return
adjustments are absent. They are not needed to add reported contract counts, but
that fact does not authorize a cash-equity return backtest, portfolio exposure
estimate or dollar-notional conversion. No index-level prices are supplied.
Historical knowledge and equity-total-return eligibility remain false.

## Integration and recovery

`equity_positioning_contract(report_through=None)` returns the same envelope and
chart/table structure as the existing Euro FX positioning contract. The fixed
historical snapshot is AVAILABLE/HISTORICAL, while current-market freshness is
STALE. The new scope is `market-flow-equity-positioning`; instrument identity is
`CFTC:13874A:TFF:FUTURES_ONLY`. Normal reads, browser use and worker operations are
entirely offline. Explicit bounded fixed-host reacquisition does not automatically
admit a changed revision to the pinned store. This prevents silent replacement.

The old Euro FX normalizer and source remain the default and are regression
tested. A frozen, code-reviewed identity parameter allows only the two supported
contracts. Tests cover direct raw-count references, exact identities,
conservation, corruption of data or metadata, numeric attacks, missingness,
future perturbation, independent percentile references, deterministic detached
replay and sanitized acquisition failure.
