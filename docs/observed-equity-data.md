# Permitted weekly historical equity archive

The fixed provider `UCI_DOW_JONES_WEEKLY_2011` acquires the official
[UCI Dow Jones Index archive](https://archive.ics.uci.edu/dataset/312/dow+jones+index),
DOI [10.24432/C5788V](https://doi.org/10.24432/C5788V). The official dataset page
explicitly identifies CC BY 4.0. GIG retains attribution, license URL, raw/data
SHA-256 hashes, actual acquisition timestamps and the original weekly dates.
The adapter verifies that license declaration on each actual acquisition.

Attribution: Michael Brown, *Dow Jones Index* [Dataset], UCI Machine Learning
Repository, 2013. Brown, M. S., Pelosi, M. and Dirska, H. (2013), *Dynamic-radius
Species-conserving Genetic Algorithm for the Financial Forecasting of Dow Jones
Index Stocks*, LNAI 7988, 27–41. Source data are available under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). GIG selects and parses
weekly observations and quarantines future-label/dividend columns; these
modifications are GIG's, not UCI or exchange outputs.

Actual official HTTPS acquisition returned HTTP 200, parsed and persisted the
selected source, then reproduced it after workspace restart. The original
archive has 750 rows, 30 source stock labels and 25 weekly dates from January 7
through June 24, 2011. The verified AA/IBM/KO selection has 75 rows. Original ZIP
SHA-256: `08ad8e30915f071b46f6f5bef2b1ccdfe70aacfa31aee116117048ad2adec0bf`.
The permitted raw ZIP, acquisition metadata and normalized snapshot are preserved
under `docs/evidence/sprint8/equity/` with the attribution above.

```json
{"operation":"fetch","provider":"UCI_DOW_JONES_WEEKLY_2011","symbols":["AA","IBM","KO"],"start":"2011-01-07","end":"2011-06-24"}
```

`POST /api/v1/workspace/equity-reference` accepts this fixed-source request.
`admit` additionally accepts CSV text and an acquisition timestamp but is always
SYNTHETIC; callers cannot choose classification, URL, credentials or filesystem
paths. `replay` accepts the returned `snapshot_id`. Markets, Research and
Operations expose saved evidence through engine `observed-equity`. Original
source/model workspaces remain separate. Raw source and canonical snapshot hashes,
index timestamps and immutable bytes are verified on replay. Future acquisition
cannot appear at an earlier terminal cutoff; missing data does not create state.

This is repository-distributed, source-reported weekly aggregate equity data,
not direct exchange verification or present-day quotes. Source labels use
`DJIA_MEMBER_LABEL:<label>` and do not establish permanent security identities.
The original dollar sign is preserved as a source unit; currency is UNKNOWN.
Weekly source dates and actual volume remain observed archive values. Intraday
times, historical publication/revision clocks, corporate actions, adjustment
basis and authoritative historical universe membership remain UNKNOWN/UNAVAILABLE.
No timestamp, daily bar, missing volume or absent security is invented.

The archive includes `next_weeks_open`, `next_weeks_close`,
`percent_change_next_weeks_price`, `days_to_next_dividend` and
`percent_return_next_dividend`. These future fields are preserved only in raw
source evidence and never emitted as normalized features. Tests adversarially
change those fields and verify that normalized observations remain identical.

The dataset is admitted for bounded historical archive inspection. It is
ineligible for point-in-time backtests, funded allocation and approved trading;
it does not establish equity alpha or bias-free model selection. Permanent
identity, corporate-action and historical membership evidence must be supplied
before broader use. The existing funded account rejects this reference namespace.

Other investigated sources were not admitted: yfinance documentation limits Yahoo
data to personal use, and the investigated historical Nasdaq WIKI URLs returned
404. Neither an installed client library nor an HTTP response alone proves a
license or usable coverage. No Alpaca credential or brokerage connection was used.
