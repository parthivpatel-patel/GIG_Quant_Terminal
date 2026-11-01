# Durable local terminal workspace

`gig.service.workspace.TerminalWorkspace(Path(...))` integrates admitted local
observations, news, reproducible research, and strictly offline execution with
the versioned terminal contracts. The constructor's directory is selected by
the application, not by an HTTP payload. No method imports a broker client,
reads brokerage credentials, downloads a model, or contacts a provider.

All contracts carry `CALLER_SUPPLIED_NOT_VENDOR_VERIFIED` evidence and
`RESEARCH_ONLY` readiness. A supplied `HISTORICAL` or `REAL_TIME` classification
remains the caller's assertion; a hash does not authenticate a provider. The
workspace never inserts demo data into local mode.

## Write interfaces

The HTTP owner can bind these methods to the corresponding workspace POST
routes. Unknown request fields, including filenames and paths, are rejected.

| Method | Required payload | Optional fields |
| --- | --- | --- |
| `ingest_bars` | `rows`, `source`, `classification`, `corporate_action_policy`, `currency` | None |
| `ingest_news` | `articles`, `as_of` | `aliases` mapping alias text to instrument identity |
| `run_research` | `rows`, `source`, `classification` | `model`, `min_train`, `test_size`, `embargo`, `ridge_alpha`, `seed`, `as_of` |
| `simulate` | `scores`, `prices`, `sectors`, `run_id`, `asof`, `now`, `source_id`, `classification` | `nav`, `n_long`, `n_short`, `currency` |

Bar rows contain exactly `instrument_id`, `timestamp`, `available_at`, `open`,
`high`, `low`, `close`, and integer `volume`. An admission contains 1–2,000 bars
and at most 32 instrument identities. Currency is an explicit three-letter
ASCII code; adjustment is `RAW`, `SPLIT_ADJUSTED`, or `TOTAL_RETURN_ADJUSTED`.
The existing revision-aware store preserves source, classification, adjustment,
currency, observation timestamp, and revision knowledge time independently.
Volumes beyond binary64 precision remain exact integer strings in responses.

News articles contain the existing evidence fields: `source`, `article_id`,
`url`, `license`, `title`, `body`, `published_at`, `available_at`, and optional
`symbols`. Each request admits 1–100 articles. Text is preserved verbatim;
the local analysis explicitly identifies lexical sentiment and dictionary entity
linking. This interface does not claim FinBERT inference. Alias mappings are
preserved in an immutable manifest and survive restart. Duplicate/revision
handling comes from `NewsStore`; original attribution remains visible.

Research rows contain exactly `timestamp`, `features` (a mapping of stable
feature names to finite numbers), `target`, `label_end`, and `available_at`.
Require 20–1,000 strictly chronological observations, 1–16 features, and 1–20
complete walk-forward folds. Defaults are ridge, 10 training observations,
5 test observations, one-observation embargo, ridge alpha one, and seed 42.
All feature availability must precede its observation; labels must end at or
after the observation and be knowable by the result cutoff. An omitted cutoff
means the latest supplied label-end time, recorded as a caller assumption.
The supplied data are not automatically linked to a market-data vendor.

The existing purged workflow trains transforms on training samples only and
stores immutable model states and a complete reproducible artifact. This layer
preserves a source receipt and a manifest linking source, dataset, configuration,
run, and model identities. The terminal rechecks the actual artifact/model file
hashes when reading stored results. LightGBM is an explicit model selection;
missing optional dependencies never cause a silent ridge substitution.

Simulation maps identities to scores, reference prices, and explicit sectors.
It accepts 2–32 instruments, `SYNTHETIC` or `HISTORICAL` classification only,
positive finite prices within the OMS exact-decimal precision, NAV between
1,000 and 1e12 caller currency units, and positive side counts up to 16.
The default NAV is one million caller units. An omitted currency stays null;
it never becomes USD merely because prices are present.

Simulation timestamps require explicit timezones and at most microsecond
precision. A nanosecond timestamp with a nonzero submicrosecond remainder is
rejected rather than truncated before entering the OMS. Inputs go through the
existing signal-to-portfolio, pre-trade risk, exact-decimal OMS, simulated fills,
positions, and reconciliation flow. Fills are explicitly `SIMULATED` and broker
connectivity remains false. A completed request replays without new orders;
changed intent under a run ID fails. The first immutable report is preserved
when a later response reports that the request was replayed. Interrupted runs
require reconciliation and are not resubmitted.

## Local section reads

Call `section(name, as_of, max_age_seconds)`. Supported names are
`command-center`, `markets`, `signals`, `portfolio`, `risk`, `stat-arb`, `news`,
`research`, and `execution`. Unsupported sections return `None` for the parent
application's existing fallback. Missing supported data returns `UNAVAILABLE`;
reading a missing workspace creates no directory or database.

The cutoff must be timezone aware; nanoseconds are retained. Freshness must be
an explicit finite duration from zero to 366 days. Passing `None` returns
`UNKNOWN` with null data. Stale observations retain their original timestamp
and return `STALE` with their research data. Execution errors stay outer
`ERROR`; a stale simulation cannot be promoted by selecting a looser UI policy.
Run selection uses the latest knowable result timestamp, breaking equal-time
ties by durable admission order.

Market-derived sections require a single source/classification/adjustment/
currency combination and at most 32 identities. Conflicting combinations
produce `UNKNOWN` rather than mixing prices. Analytics require a complete
aligned price panel; missing observations are never forward-filled. Analytics
also reject snapshots exceeding 32,000 visible rows. Markets show each
instrument's actual latest admitted bar and its source knowledge evidence.

Signals use 20-observation arithmetic momentum, negative 5-observation return,
20-observation sample return volatility, and cross-sectional momentum rank.
The three factors form an instrument-level analytical plot with explicit units.
They are descriptive research scores, not approved trading signals.

The statistical-arbitrage section uses the first two sorted instrument
identities for a documented descriptive example. OLS uses a training prefix;
held-out spreads use the fixed fitted coefficients. AR(1) diagnostics do not
claim cointegration, significance, or economic validation. Half-life is shown
in observations; no unsupported calendar cadence is inferred.

Portfolio and risk require a successful stored offline rehearsal whose source,
classification, currency, and instrument identities agree with admitted bars.
They also require consistent valuation timestamps and an OMS ledger that still
matches the manifest with no working/interrupted execution. Unknown currency
or mismatched identity produces `UNKNOWN`. NAV is a simulated mark from caller
NAV and supplied fill-price assumptions; it is never brokerage performance.
Risk uses actual aligned historical arithmetic returns at constant current
simulated weights and an annualization factor of one. Frequency and calendar
assumptions are not invented.

## State and evidence files

The directory contains `bars.sqlite`, `news.sqlite`, `execution.sqlite`,
`workspace.sqlite`, immutable content-addressed `receipts/` and `manifests/`,
source `snapshots/`, and reproducible `research/` artifacts/models. These are
local state, not external backups. Canonical receipts preserve admitted value
identity and original text, not original HTTP whitespace or numeric lexemes.
Existing paths cannot redirect reads or writes through symbolic links or parent
traversal. No response exposes application filesystem paths.

The underlying SQLite stores commit their batches atomically, and immutable
files publish atomically. There is no global cross-store transaction: a crash
between storage and manifest publication may leave valid unreferenced evidence.
Replaying admission is safe; an interrupted multi-order scenario remains blocked
for reconciliation. Recovery must preserve those artifacts and ledger markers.

## Verification

```sh
PYTHONPATH=src python -m unittest discover -s tests -p test_terminal_workspace.py -v
```

Initial result: 19 tests passed, no failures or skips. Later boundary/model
review expanded the workspace suite to 25 tests, with eight additional HTTP
integration tests. Checks cover missing
read side effects, durable ingestion/replay, nanosecond point-in-time revisions,
freshness, exact large volumes, source/currency isolation, complete-panel
analytics, original news text and aliases, purged research/model hashes,
corruption rejection, simulator fills/replay/restart, unknown currency,
timestamp truncation rejection, blocked/stale truth states, and path guards.

## Response bounds and registered models

News admission performs verified read-only candidate analysis and full response
serialization before writing state, both at the requested cutoff and subsequent
revision knowledge times. Text whose expanded explanations exceed the wire
contract is rejected; original text is never truncated. Rejection preserves the
existing workspace, including when no workspace existed before the request.

Provenance lists display at most 256 snapshot hashes and include total count,
an explicit completeness flag, and a versioned aggregate SHA-256 covering every
hash. This bounds responses while preserving the identity of the full evidence
set. Statistical-arbitrage views require at most 2,000 aligned observations;
larger sources remain stored and the view reports UNKNOWN.

The Models section now reads registered states from the latest knowable research
manifest, verifies artifact/model hashes, and reports actual trained states with
trading approval false. A workspace without research models falls back to the
general capability view. Research and model envelopes carry the run identity in
their model provenance.

HTTP writes use `/api/v1/workspace/{bars,news,research,simulate}` with JSON bodies.
Cross-origin browser writes are rejected. The application selects one fixed
state directory (`create_app(state_dir=...)`); requests cannot choose paths.
Local reads accept explicit `as_of` and `max_age_seconds` query parameters. The
terminal exposes both controls and sends the original timestamp text without
browser date conversion, preserving nanoseconds for replay.
