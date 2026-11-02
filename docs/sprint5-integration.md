# Sprint 5 integrated research and offline simulation

Development continues on `development/gig-ui-redesign` from
`7e42c93820f4d10328706fe702fbf8ce75396efb`. Interrupted source was preserved and
independently downloaded from GitHub at
`cd2b34ea9871f17614b7db6eac0728f5f92fb810`: 426 file hashes verified, archive
SHA-256 `9bf6511e34d5fdd5154650013dce4c1a4bfcfe656d02c5f61f27567a960ab457`.
The quantitative branch remains at `202b1a2c4fb3fe8acc734f29cbaa09f17857fddb`.

## Connected workflows

Admitted point-in-time datasets feed saved pairs experiments through
`pairs-research` and `pairs-replay`. Training-only Engle–Granger screening,
Bonferroni correction, descriptive ADF, causal Kalman innovations, missing-data
diagnostics, explicit costs and original-decision price marks support unapproved
pair allocation proposals. See [pairs assumptions](sprint5-pairs.md).

`options-admit`, `options-research` and `options-replay` preserve historical
European contract identity, quote/underlying knowledge and acquisition times,
immutable revisions and saved valuations. BSM inversion/Greeks, admitted-point
CALL surfaces, term/skew diagnostics, PCR and documented VTS accompany explicit
hypothetical contract scenarios. CSM remains UNSUPPLIED/UNAVAILABLE. No OPRA
coverage or vendor authenticity is inferred. See [options contracts](options-pipeline.md).

Saved forecasting/ranking runs feed `research-signals` and `signals-replay`.
Prior out-of-sample label vintages alone determine IC-aware research weights;
confidence remains uncalibrated. Allocation accepts exactly one CPU run, signal
or pair identity, with original hedge ratios, turnover/cost limits and retained
asynchronous exposure constraints. Preparation, acknowledgement, explicit fills
and Decimal cash/position reconciliation remain separate offline steps.
See [signal/proposal contracts](signal-proposals.md).

Execution-linked risk admission and kill control now share one OMS transaction.
Timestamp precision is checked before mutation; the research ledger retains
nanoseconds. See [failure/recovery semantics](execution-risk-atomicity.md).

All 12 terminal sections and the existing design system remain. Three additional
evidence engines, explicit workbench actions, actual saved-result table choices,
modeled IV/hedge plots and session-only experiment comparisons expose these
workflows. Input classifications, provenance, uncertainty and missingness remain
visible. No browser action runs automatically or sends a broker order.

The next integrated batch adds `options-overlay` and `options-overlay-replay`:
a published, current reconciled simulated equity checkpoint and a saved options
experiment form a single-underlying hypothetical scenario report. Exact source
mark/clock alignment, unchanged current fills/positions, quote revisions and
explicit delta/gamma/vega/scenario USD caps are checked. Other unmodeled holdings
are rejected. The original OMS remains unchanged. Financing and outstanding-order
exposure remain UNKNOWN; combined modeled value is never labeled funded NAV.
Historical tail risk remains UNAVAILABLE rather than inferred from a scenario
grid. See [overlay contracts and limits](options-overlay.md).

## Checkpoint verification

The first integrated source checkpoint passed 40 new focused tests and the
68-test execution safety/regression selection (overlapping counts). Ruff,
Pyflakes, compilation and configured mypy on 134 source files pass. Mypy retains
the repository's import-skipping/untyped-body limitations; it is not a strict
whole-program proof. Complete coverage and installed-wheel results will be
recorded at the final validation checkpoint.

The overlay source checkpoint additionally passed 85 tests against the installed
native wheel, with all 140 packaged source/static/data files verified against
source. Wheel SHA-256 is
`de3a1cc57e395e1ae9140aa80d94eb019d0d57efa47dad720a86cd0356391ed5`.
Twenty-two focused native/operations tests pass. A 20,000-observation C++20 rolling
comparison has maximum absolute error `3.469446951953614e-18`; this x86 host run
does not establish ARM64 parity. Timing evidence was collected with concurrent
test work and is not a controlled performance comparison.

Actual Chromium 151.0.7922.34 passed 87 retained baseline, 32 persisted-flow,
24 depth/motion and 44 Sprint 5 checks. The suites overlap. No JavaScript errors
or automated axe 4.10.3 violations occurred in tested dark/light views; viewport
checks cover 320, 375, 768, 1024 and 1920 px. Node's 78 smoke checks are separate.
Expanded scrollable provenance received keyboard focus/labels; long artifacts
start collapsed. Manual accessibility, screen-reader and broader browser review
remain outstanding. Evidence is under `docs/evidence/sprint5`.

The final expanded Chromium workflow passed 60 checks including reconciled
equity-to-options overlay, deterministic replay, unchanged OMS response, explicit
unknown financing, dark/light accessibility and responsive layout. It supersedes
the 44-check Sprint 5 workflow and must not be summed with it.

## Actual scope and remaining acceptance

Numerical and workflow verification uses labeled authored fixtures. No provider,
broker, entitlement, observed liquidity or economic value is fabricated.
Options positions are researcher-declared hypothetical contracts, not OMS option
positions. Pairs backtests are fractional-weight research, not share/cash ledgers.
Regime labels are explicit trailing-volatility heuristics, not HMM output.
Restricted FinBERT weights remain outside Git/archives and noncommercial only.
Physical Ampere and full ARM64 runtime remain unverified. Sprint 4's image is
not retroactively claimed to contain this sprint. No public deployment occurred.
T4 remains `T4_NOT_CERTIFIED`; historical §139 remains `UNSUPPLIED`.
