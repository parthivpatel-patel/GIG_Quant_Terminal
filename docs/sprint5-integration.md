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

## Checkpoint verification

The first integrated source checkpoint passed 40 new focused tests and the
68-test execution safety/regression selection (overlapping counts). Ruff,
Pyflakes, compilation and configured mypy on 134 source files pass. Mypy retains
the repository's import-skipping/untyped-body limitations; it is not a strict
whole-program proof. Complete coverage and installed-wheel results will be
recorded at the final validation checkpoint.

Actual Chromium 151.0.7922.34 passed 87 retained baseline, 32 persisted-flow,
24 depth/motion and 44 Sprint 5 checks. The suites overlap. No JavaScript errors
or automated axe 4.10.3 violations occurred in tested dark/light views; viewport
checks cover 320, 375, 768, 1024 and 1920 px. Node's 78 smoke checks are separate.
Expanded scrollable provenance received keyboard focus/labels; long artifacts
start collapsed. Manual accessibility, screen-reader and broader browser review
remain outstanding. Evidence is under `docs/evidence/sprint5`.

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
