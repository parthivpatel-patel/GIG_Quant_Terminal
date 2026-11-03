# GIG engineering completion matrix

Authority: [current accelerated directive](directives/sprint7-master-20261103.md),
the supplied [Sprint 4/UI directive](directives/sprint4-ui-20261102.md), and the
scoped contracts/evidence below. This is the single maintained requirements
disposition. COMPLETE applies only to the stated supported scope; it never means
economic validation or institutional certification. Historical §139 is unsupplied.

Baseline: `b04f168ddf4f704a1e90c00f09146d024d8d831d`, 821 backend tests, 81.43%
coverage, 110 installed-wheel checks. These counts verify the baseline rather than
proving completion of new requirements. No baseline source was discarded.

| Requirement / authoritative source | Status | Implementation / verification evidence | Dependency and remaining work | Priority / external blocker |
| --- | --- | --- | --- | --- |
| Preserve branches, historical workspaces and source (§1,14) | COMPLETE | Sprint 6 publication and recovery proofs; clean starting HEAD | Verify every subsequent checkpoint; quantitative branch/main unchanged | P0; none |
| Strict $0, offline brokerage, no public deployment (§15) | COMPLETE | Offline OMS/outbox and API safety tests; no brokerage integration enabled | Maintain restriction in every batch | P0; permanent boundary |
| Immutable OHLCV, revisions, acquisition/knowledge clocks (§4) | COMPLETE | `data/validated.py`, persisted-data/adversarial tests | Source authenticity is caller asserted until provider evidence verified | P0; scoped admission |
| Permitted observed data source/permission inventory (§4) | COMPLETE | Official ECB HTTPS fetch and reuse terms; `observed-data.md` and actual six-model proof | FX reference scope verified; genuine equity coverage remains blocked | P0; licensing and source availability |
| Observed data research without invented OHLC/volume/time (§4,5) | COMPLETE | Saved actual ECB observations and six CPU model programs; source/model replay | Volume and intraday/publication clocks must remain UNKNOWN when absent | P0; historical vintage clocks may be unavailable |
| Historical point-in-time universes / no survivorship imputation (§4,5) | COMPLETE | Persisted datasets, universe store, missing held-security and whole-cross-section tests | Authoritative external historical membership coverage remains data dependent | P0; external coverage |
| Corporate-action research adjustment / late revisions (§4,5) | COMPLETE | Action store, original-decision datasets, label vintages and pair correction tests | Connect actual account lifecycle without rewriting earlier evidence | P0; scope extension below |
| LightGBM ranking / saved inference (§5,8) | COMPLETE | Real CPU library-backed models, model registry and replay tests | Execute on admitted observed reference data; no profitability assumption | P1; observed inputs |
| CPU BiLSTM training/inference (§5,8) | COMPLETE | Actual Torch CPU sequence and saved-state tests | Restore disposable runtime dependencies; observed runs | P1; temporary dependency restoration |
| GARCH MLE, HMM and causal Kalman (§5,8) | COMPLETE | Numerical state-model and persisted-model tests | Observed research integration; heuristics must remain separately labeled | P1; observed inputs |
| FinBERT inference/provenance (§8,15) | COMPLETE | Pinned real CPU research inference and license tests | Restore external noncommercial assets for tests; no commercial claim | P1; CC-BY-NC-SA-4.0 restriction |
| Purged/embargoed folds, training-only selection (§5) | COMPLETE | Saved CPU/pair folds, vintage and future-prefix adversarial tests | Verify new observed reconstruction scope without false past availability | P0; source clock evidence |
| IC adaptive scoring, uncertainty and provenance (§5,8) | COMPLETE | Prior OOS label gates, saved ensemble IDs, calibrated-confidence UNKNOWN | Observed reference scoring must not become approved trading signals | P1; economic validation distinct |
| Costs/slippage, turnover/capacity, benchmark/factors (§5) | COMPLETE | Sprint 6 economic evaluation and negative synthetic finding | Observed data evaluation; missing executed volume remains UNKNOWN | P0; actual volume coverage |
| Multiplicity, HAC/bootstrap, robustness/attribution/degradation (§5) | COMPLETE | Economic tests and declared-family reports; unsupported training degradation UNKNOWN | No family-completeness, stationarity or alpha certification | P1; research assumptions |
| Observed-market economic usefulness / alpha (§5,16) | BLOCKED | Synthetic result is negative and not economic validation | Appropriate observed data, vintage clocks and independent evaluation | P0; data + statistical evidence; no guaranteed alpha |
| Paid long equities and long European cash options (§6) | COMPLETE | Funded account, exact premiums/multipliers/FIFO/fees and conservation tests | Unsupported debt/short/physical/early exercise remain rejected | P0; bounded scope only |
| Trade/settlement-date cash and reservations (§6) | COMPLETE | Explicit dated cash and XNYS T+1, unsettled-proceeds and restart tests | Explicit settlement dates and unavailable unsettled proceeds; legacy preservation | P0; settlement-calendar/evidence contracts |
| Dividend entitlement and pay-date cash (§6) | COMPLETE | Frozen pre-ex entitlement, pay-date cash and rate-revision tests | Freeze record-date entitlement and knowledge; no arbitrary cash credits | P0; action evidence |
| Dividend receivable valuation between ex-time and payment (§6) | NOT STARTED | Explicit unmodeled accrual makes NAV/P&L/reconciliation UNKNOWN; no invented zero | Model accrual/revisions before claiming complete intermediate valuation | P0; financial scope limitation |
| Splits, position offsets and action revisions (§6) | COMPLETE | Atomic whole-lot splits, exact FIFO offsets and revision tests | Atomic compensating events; fractional entitlement policy explicit | P0; action evidence |
| Expiry / automatic supported cash exercise (§6) | COMPLETE | Expiry/fee/cash checks, duplicate settlement and replay | Physical delivery, assignment and early exercise unsupported | P0; permanent supported-scope boundary |
| Atomic funded events, duplicate/concurrent requests, restart (§6,11) | COMPLETE | Process-death tests and writer-lock receipt recheck | Extend same guarantees to lifecycle events | P0; none |
| Independent statement fixtures / discrepancy reports (§7) | COMPLETE | Hand-calculated literal statements, discrepancy/UNKNOWN and replay tests | Compare balances/orders/fills/positions/obligations/reservations; unresolved is not MATCH | P0; local authorized fixtures only |
| Actual brokerage reconciliation (§7,15) | BLOCKED | Deliberately disconnected; external reconciliation UNKNOWN | No account connection authorized; fixtures cannot establish broker truth | P0; explicit prohibition |
| Cointegration/ADF, Kalman hedge, spreads and pair replay (§8) | COMPLETE | Persisted pairs, original-mark returns and gap invalidation tests | Observed capacity and regime evidence remain scoped research | P1; observed inputs |
| Options admission, IV/Greeks, surface/skew/term/PCR/VTS (§8) | COMPLETE | Immutable European options and numerical references | No invented OPRA observations or entitlement | P1; observed option quotes absent |
| CSM (§8) | BLOCKED | Public CSM result explicitly UNAVAILABLE | Authoritative definition must be supplied | P2; unsupplied definition |
| OFI, Lee–Ready estimate, Kyle, VPIN (§8) | COMPLETE | Persisted quote/print fixtures and scalar references | True print/quote feeds for observed validation; depth is not executions | P1; permitted source evidence |
| Executed-cost / market-impact / capacity evidence (§8) | IN PROGRESS | Existing execution-cost kernels and conservative volume assumptions | Independent statement/print matching; unsupported metrics UNKNOWN | P1; matching observed prints/executions |
| Actual funded Greeks, scenarios, concentration/funding (§9) | COMPLETE | Actual positions override hypothetical counts; working-order budgets and mark guards | Extend optional sector/factor/drawdown contracts | P0; scoped model estimates |
| Funded sector/factor constraints and drawdown (§9) | COMPLETE | Delta-equivalent sector/factor budgets and immutable NAV drawdown tests | Explicit exposure/knowledge metadata and saved history; no inferred zero beta | P0; metadata availability |
| Historical VaR/ES (§9) | COMPLETE | Empirical reference on 21 comparable daily funded NAV marks; gaps unavailable | New lifecycle must preserve capital/mark comparability | P1; sufficient admissible history |
| Atomic risk policy/kill, pre-trade/post-event recovery (§9) | COMPLETE | Risk-index/kill rollback, stale post-fill and admission tests | Lifecycle/reconciliation hooks must retain guarantees | P0; none |
| All twelve sections / typography / themes / 3D / 2D (§10) | COMPLETE | Chromium 78 + retained suites; Node 106 baseline checks | Wire new verified workspaces through existing design | P1; integration work |
| New observed/lifecycle/reconciliation terminal workflows (§10) | COMPLETE | Chromium 151: 77 Sprint 7 checks; retained suites 87/24/32; Node 128 | Backend contracts then real Chromium/restart/accessibility checks | P1; backend readiness |
| Bounds, serialization, errors, secrets, corruption, recovery (§11) | COMPLETE | Existing API, persistence and recovery tests | Repeat focused new-boundary checks, not a full institutional audit | P0; none |
| CPU C++20 / wheel / installation (§11,13) | COMPLETE | x86 native parity, wheel content hashes/import tests | Reverify new release; no GPU or infrastructure expansion | P1; none |
| Fresh current container lifecycle (§11) | BLOCKED | Previous Sprint 4 image/restart evidence is scoped to that source | Fresh image only with safe resource reserve; current writable disk <1 GiB | P2; local disk capacity, no source/evidence deletion |
| Physical Ubuntu 24.04 ARM64/Ampere (§11) | BLOCKED | Prior QEMU evidence explicitly separate | Actual hardware/runtime acceptance required; no spending/provisioning | P2; hardware unavailable |
| Browser/manual accessibility / cross-browser (§13) | IN PROGRESS | Chromium/axe and selected manual inspection | Chromium/axe 4.10.3 restored; broader browser/manual coverage remains open | P1; available tools/hardware |
| Final comprehensive engineering acceptance (§13,16) | NOT STARTED | This matrix is the available requirements disposition | One final acceptance review after substantive work, not repeated audits | P0; feature completion first |
| Verified publication and independently restored archives (§14) | COMPLETE | Published Sprint 6 510-file restore, SHA-256 and modes; local complete Git bundle | Repeat for each Sprint 7 substantial checkpoint | P0; GitHub sole verified external provider |
| Historical §139 / full T4 acceptance (§3,15,16) | BLOCKED | §139 UNSUPPLIED; T4_NOT_CERTIFIED | Required authoritative acceptance contract and independent evidence | P0; unsupplied requirements |

## Critical path and forecast

As of November 3, 2026: the observed-reference, lifecycle, independent
statement, factor-risk and terminal workflows are integrated and focused tests
pass. Publish and finish broad release checks, then advance observed evaluation, operational acceptance
and one comprehensive final review. Restore disposable model/browser dependencies
without deleting source, prior archives or irreplaceable dependencies.

The directive's preferred three-to-four-week engineering target is November 24
through December 1; December 8 is the five-week planning ceiling, with Week 5
reserved for contingency. Finish feasible work earlier when evidence permits.
These are conditional targets, not guarantees of data availability, hardware,
profitable results or certification. Reforecast from actual delivered checkpoints
and the external blockers above. Feature completion, operational acceptance,
economic validation and institutional certification remain separate milestones.
