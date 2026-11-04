# GIG engineering completion matrix

Sprint 9 delivery: [sprint9-delivery.md](sprint9-delivery.md).

Authority: [Sprint 8 direct-chat directive](directives/sprint8-direct-chat-20261103.md),
[accelerated master directive](directives/sprint7-master-20261103.md),
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
| Permitted observed equity archive (§4) | COMPLETE | UCI CC BY4.0 official HTTPS ZIP, 750 weekly source rows; selected75 persisted/replayed; future labels quarantined | Jan–Jun2011 weekly source labels only; CA, permanent identity, currency and vintage clocks unknown; not PIT/funded eligible | P1; broader data evidence blocked |
| Model-appropriate observed scoring (§5,8) | COMPLETE | Saved v1/v2 dispatch, ranking NDCG/IC, variance QLIKE/NLL, HMM partial diagnostics and independent scalar references;31 focused tests | Unadjusted uncertainty, Gaussian assumptions, no economic validation; negative findings preserved | P1; limited acquired window |
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
| Saved observed-reference predictive baselines (§5) | COMPLETE | Acquired ECB model reports; paired-date baseline losses, bootstrap and immutable replay; Chromium 82 checks | v1 evidence preserved; v2 NDCG/IC and QLIKE/calibration with training-only baselines verified | P1; not economic validation |
| Observed-market economic usefulness / alpha (§5,16) | BLOCKED | Actual acquired reference return forecasts underperform baselines; rank/variance diagnostics do not establish economic value | Appropriate observed data, vintage clocks and independent evaluation | P0; data + statistical evidence; no guaranteed alpha |
| Paid long equities and long European cash options (§6) | COMPLETE | Funded account, exact premiums/multipliers/FIFO/fees and conservation tests | Unsupported debt/short/physical/early exercise remain rejected | P0; bounded scope only |
| Trade/settlement-date cash and reservations (§6) | COMPLETE | Explicit dated cash and XNYS T+1, unsettled-proceeds and restart tests | Explicit settlement dates and unavailable unsettled proceeds; legacy preservation | P0; settlement-calendar/evidence contracts |
| Dividend entitlement and pay-date cash (§6) | COMPLETE | Frozen pre-ex entitlement, pay-date cash and rate-revision tests | Freeze record-date entitlement and knowledge; no arbitrary cash credits | P0; action evidence |
| Dividend receivable valuation between ex-time and payment (§6) | COMPLETE | Exact face-value accrual, frozen entitlement, cash conversion, revisions/cancel, same-OMS rollback and literal statement tests | Declared USD cash dividends only; credit/default/tax models and external confirmation unavailable | P0; bounded simulation scope |
| Splits, position offsets and action revisions (§6) | COMPLETE | Atomic whole-lot splits, exact FIFO offsets and revision tests | Atomic compensating events; fractional entitlement policy explicit | P0; action evidence |
| Expiry / automatic supported cash exercise (§6) | COMPLETE | Expiry/fee/cash checks, duplicate settlement and replay | Physical delivery, assignment and early exercise unsupported | P0; permanent supported-scope boundary |
| Atomic funded events, duplicate/concurrent requests, restart (§6,11) | COMPLETE | Process-death tests and writer-lock receipt recheck | Extend same guarantees to lifecycle events | P0; none |
| Independent statement fixtures / discrepancy reports (§7) | COMPLETE | Hand-calculated literal statements, discrepancy/UNKNOWN and replay tests | Compare balances/orders/fills/positions/obligations/reservations; unresolved is not MATCH | P0; local authorized fixtures only |
| Actual brokerage reconciliation (§7,15) | BLOCKED | Deliberately disconnected; external reconciliation UNKNOWN | No account connection authorized; fixtures cannot establish broker truth | P0; explicit prohibition |
| Cointegration/ADF, Kalman hedge, spreads and pair replay (§8) | COMPLETE | Persisted pairs, original-mark returns and gap invalidation tests | Observed capacity and regime evidence remain scoped research | P1; observed inputs |
| Options admission, IV/Greeks, surface/skew/term/PCR/VTS (§8) | COMPLETE | Immutable European options and numerical references | No invented OPRA observations or entitlement | P1; observed option quotes absent |
| CSM (§8) | BLOCKED | Public CSM result explicitly UNAVAILABLE | Authoritative definition must be supplied | P2; unsupplied definition |
| OFI, Lee–Ready estimate, Kyle, VPIN (§8) | COMPLETE | Persisted quote/print fixtures and scalar references | True print/quote feeds for observed validation; depth is not executions | P1; permitted source evidence |
| Executed-cost / market-impact / capacity evidence (§8) | COMPLETE | Sprint 9 `execution/cost_evidence.py`: class-separated ESTIMATED/SIMULATED/OBSERVED_HISTORICAL reports, exact shortfall identity, independent print matching, measured non-causal impact, print-derived capacity; 8 tests; Chromium workbench | No observed executions/prints in repository, so observed class is fixture-exercised only; broker-reconciled class rejected | P1; observed executions/prints absent |
| Market-data lineage / point-in-time eligibility (§4) | COMPLETE | Sprint 9 `data/provenance.py`: license, permission, five clocks, revision history, CA basis, identity, quality and verified lineage hash; purpose-specific ELIGIBLE/UNKNOWN/INELIGIBLE with parent inheritance; 9 tests | Documented ECB/UCI sources are not point-in-time eligible; vintage evidence required | P0; source clock evidence |
| Free continuous integration (§13,15) | COMPLETE | Sprint 9 `.github/workflows/ci.yml`: GitHub-hosted 3.11/3.12 backend matrix, native build, lint/types/tests/coverage, Node smoke and Chromium workflow; read-only token | Whole-package mypy drift (9 pre-existing errors) keeps typing scoped to new modules | P1; no paid runners |
| Actual funded Greeks, scenarios, concentration/funding (§9) | COMPLETE | Actual positions override hypothetical counts; working-order budgets and mark guards | Extend optional sector/factor/drawdown contracts | P0; scoped model estimates |
| Funded sector/factor constraints and drawdown (§9) | COMPLETE | Delta-equivalent sector/factor budgets and immutable NAV drawdown tests | Explicit exposure/knowledge metadata and saved history; no inferred zero beta | P0; metadata availability |
| Historical VaR/ES (§9) | COMPLETE | Empirical reference on 21 comparable daily funded NAV marks; gaps unavailable | New lifecycle must preserve capital/mark comparability | P1; sufficient admissible history |
| Atomic risk policy/kill, pre-trade/post-event recovery (§9) | COMPLETE | Risk-index/kill rollback, stale post-fill and admission tests | Lifecycle/reconciliation hooks must retain guarantees | P0; none |
| All twelve sections / typography / themes / 3D / 2D (§10) | COMPLETE | Chromium 78 + retained suites; Node 106 baseline checks | Wire new verified workspaces through existing design | P1; integration work |
| New observed/lifecycle/reconciliation terminal workflows (§10) | COMPLETE | Chromium 151: 77 Sprint 7 checks; retained suites 87/24/32; Node 128 | Backend contracts then real Chromium/restart/accessibility checks | P1; backend readiness |
| Bounds, serialization, errors, secrets, corruption, recovery (§11) | COMPLETE | Existing API, persistence and recovery tests | Repeat focused new-boundary checks, not a full institutional audit | P0; none |
| CPU C++20 / wheel / installation (§11,13) | COMPLETE | x86 native parity, wheel content hashes/import tests | Reverify new release; no GPU or infrastructure expansion | P1; none |
| Fresh current container lifecycle (§11) | BLOCKED | Previous Sprint 4 image/restart evidence is scoped to that source | Read-only VFS gate: 0 available Docker-store bytes; requires at least 3 GiB container reserve / 6 GiB plus budget for build | P2; local disk capacity, no source/evidence deletion |
| Physical Ubuntu 24.04 ARM64/Ampere (§11) | BLOCKED | Prior QEMU evidence explicitly separate | Actual hardware/runtime acceptance required; no spending/provisioning | P2; hardware unavailable |
| Browser/manual accessibility / cross-browser (§13) | IN PROGRESS | Chromium 151 / axe 4.10.3; Sprint 9 Chromium 156 operations workflow 28/0; twelve-section HTTP workflows and selected manual inspection | Chromium194 checks; Firefox153 assets installed but startup timeout/uid_map EROFS/compositor errors; formal accessibility and broader browser acceptance remain open | P1; available tools/hardware |
| Final comprehensive engineering acceptance (§13,16) | NOT STARTED | This matrix is the available requirements disposition | One final acceptance review after substantive work, not repeated audits | P0; feature completion first |
| Verified publication and independently restored archives (§14) | COMPLETE | Sprint8 637-file release and643-file follow-up GitHub restores, every hash/mode; rebuilt native wheel and13 numerical checks; local Git bundle | Maintain pinned-source archives at each substantial checkpoint; GitHub is the sole verified external provider | P0; GitHub sole verified external provider |
| Historical §139 / full T4 acceptance (§3,15,16) | BLOCKED | §139 UNSUPPLIED; T4_NOT_CERTIFIED | Required authoritative acceptance contract and independent evidence | P0; unsupplied requirements |

## Critical path and forecast

As of November 3, 2026: Sprint 8 receivable accounting/reconciliation, versioned
model scoring, permitted weekly equity evidence and the Docker capacity gate are
integrated. Finish frozen-source release checks and publish the checkpoint, then
advance remaining operational acceptance, security/performance checks and one
comprehensive final review after substantive engineering is complete. Restore disposable model/browser dependencies
without deleting source, prior archives or irreplaceable dependencies.

The directive's preferred three-to-four-week engineering target is November 24
through December 1; December 8 is the five-week planning ceiling, with Week 5
reserved for contingency. Finish feasible work earlier when evidence permits.
These are conditional targets, not guarantees of data availability, hardware,
profitable results or certification. Reforecast from actual delivered checkpoints
and the external blockers above. Feature completion, operational acceptance,
economic validation and institutional certification remain separate milestones.
