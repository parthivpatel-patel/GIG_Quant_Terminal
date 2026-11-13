# GIG engineering completion matrix

## PR #13 integrated; first ML expansion batch (2026-11-13)

PR #13 is COMPLETE at merge `53405c066d5ef63573c0a570ad781d7f66cdbae2`.
All three post-merge checks passed in Actions run `43433467105`: Python 3.11,
Python 3.12 and frontend. Both backends report 1,235 passed,11 skipped,673
subtests and 82.92% combined coverage. The merged tree exactly equals the
verified PR head `25148d2b79e179a3dd265715b80e8417185edefc`.

Independent codeload restoration verified all 990 tracked file hashes/modes;
62 focused tests pass from that restored source. Archive SHA-256:
`c48cee3fc3a957cfd793ca5d34d1e620b2740bbe455fe38afa8fc31267c58516`.
Git recovery bundle verified with prerequisite base `1f9a3e1`; source is also
recoverable from GitHub. Main and protected quantitative refs remain unchanged.

Only after this gate passed, `development/gig-ml-expansion-batch1` added the
[40-family inventory](ml/EXPANSION_INVENTORY.md) and the first bounded
[Gaussian-process adapter](ml/GAUSSIAN_PROCESS.md). Current batch acceptance
results are recorded below as they complete; earlier evidence retains its exact
source scope. No production approval, observed-market alpha or certification is
implied.

### First ML batch acceptance

| Capability | Verified result and limit |
| --- | --- |
| Forty-family inventory | Seven existing families identified; Gaussian process implemented as family08;32 additions remain planned. No automatic model or objective approval. |
| Persisted Gaussian process | Actual sklearn CPU fit, fixed RBF/no optimizer,128 training rows/16 fits/16 features; JSON inference, strict numeric/state admission, purged vintage selection. Portfolio admission remains false. |
| Negative research finding | SYNTHETIC fixture only:72 OOS rows/6 states. GP MSE9.31549e-5 is worse than train-only mean baseline7.25312e-5; nominal95% coverage86.1111%, not calibrated confidence. |
| Final backend regression |1,247 passed,9 optional FinBERT skips,0 failures;83.1%combinedcoverage. Counts do not include overlapping reruns/wheel tests. |
| Installed native wheel |172 passed,1 optional standalone C-ABI skip;217 packaged source/data/UI byte comparisons; native pybind loaded independently. |
| Static and frontend |Ruff,Pyflakes,compileall,mypy22configured modules plus GP kernel;150 Node DOM checks. All original CI suites retained. |
| Actual Chromium |20 passed,0 failures;151.0.7922.173;dark/light axe checks,four responsive widths,Models rendering and explicit replay. No formal accessibility/cross-browser certification. |
| Restart and numerical evidence |Separate source/installed-wheel processes reproduce exact prediction hash without training/orders; native parity6.94e-18 on200,000samples/11repeats,x86_64 only. |
| Operational and institutional gates |Docker BLOCKED_STORAGE:0free bytes on/workspace,0Docker commands. Physical ARM64/Ampere unverified;fully_funded_certified=false;T4_NOT_CERTIFIED;§139UNSUPPLIED. |

PR #14 targets development only. Exact final-head CI and independent recovery
verification are recorded on the PR after publication. Evidence:
[ML batch1](evidence/ml-batch1/README.md). All prior observed research, screenshots,
worker lease controls and protected branches remain preserved.

## PR #13 recovered release acceptance (2026-11-13)

Verified feature checkpoint: `5b8cf871a7551d747953d726fd643fb76b3243fe` on
`development/gig-research-equity-registry`. PR #13 targets
`development/gig-ui-redesign`; main and the quantitative branch are unchanged.
The interrupted frontend correction was already published and all six PR/push
checks are green. Final evidence-only commit checks and the merge/post-merge
results are recorded on PR #13 before this release is treated as integrated.

| Requirement | Verified result and limit |
| --- | --- |
| Ubuntu/Playwright correction | Standard free Ubuntu 24.04 runners; bundled Playwright Chromium download without redundant APT/fonts. No browser, axe, responsive or security check removed. Both frontend contexts passed. |
| Full clean-runner regressions | Python 3.11 and 3.12 each: 1,235 passed,11 skipped,673 passing subtests;82.92%combinedcoverage. Skips:9FinBERT assets/runtime,2legacy ECB acquisition-store checks. Suites overlap. |
| Fresh installed-wheel acceptance |164passed,1optional standalone C-ABI skip. Native pybind imported from independent installation;217packaged source/data/UI files match source bytes. |
| Fresh browser acceptance |47actual Chromium151.0.7922.173 checks;8axe scans and8responsive checks,0failures. Original CI Chromium158.0.8127.18 completed all retained suites and47new checks. Automated accessibility only. |
| Worker and risk safety |62focused lease, stale-owner, registry, HTTP and provider checks pass. The corrected clock-driven expiry tests remain intact; runtime controls are unchanged. |
| Independent recovery |Published archive restored:978tracked file hashes and executable modes match; archive SHA256accf4789356905dc8df9f6c325bd7784a70de9f4d566a699edad6aad83b3e68f.353source/test/script/config hashes verified. Older19local changes separately preserved. |
| Research and licensing |UCI studies remain retrospective/INCONCLUSIVE and worse than training-only baselines. CFTC data are historical aggregate futures positions. FinBERT remains noncommercial research-only; no restricted weights in archives. |
| Remaining acceptance |No PIT-admitted cash-equity universe/actions/vintages or observed options-print entitlement. Docker store full; physical ARM64/Ampere, Firefox/Safari and full manual accessibility unverified. fully_funded_certified=false;T4_NOT_CERTIFIED;§139UNSUPPLIED. |

Evidence: `docs/evidence/pr13-recovery-20261113/` and
[the original delivery report](research/equity-registry-delivery.md). Counts across
local, installed-wheel, repeated CI and browser suites are not added together.
No ML expansion is started before verified PR #13 merge and post-merge CI.

## Approved equity investigations and connected workspaces (2026-11-11)

PR #12 is merged at `1f9a3e1cfa7363db242e4d3a9f3fb8802acc9a13`.
The next milestone is PR #13 on `development/gig-research-equity-registry`,
targeting `development/gig-ui-redesign`. Main and the quantitative branch remain
unchanged. This is a bounded research extension, not autonomous production or a
public application launch.

| Requirement | Disposition | Evidence and scope |
| --- | --- | --- |
| Registry and memory-informed selection | IMPLEMENTED / TESTED | Five immutable approved investigations, four datasets, transparent score components and saved report hashes; explicit per-objective approval, checkpointed queue order and source/version deduplication. Runtime code/template generation remains disabled. |
| Useful observed equity research | IMPLEMENTED / INCONCLUSIVE | Existing permitted UCI archive extended to125 observations/five labels, two distinct purged range/log-volume diagnostics, training-only median benchmarks, weekly-block uncertainty and separate Decimal Skeptic. Both models are worse than their baselines; no economic claim. |
| Official equity-market positioning | IMPLEMENTED / HISTORICAL |52 official CFTC E-mini S&P 500 weekly futures-only reports, exact contract/count conservation and source/permission evidence. Aggregate derivatives positions, not cash-equity prices, options prints or executed flows. |
| Research-to-decision traceability and UX | IMPLEMENTED / TESTED | Persistent Investigation Desk across Command Center, Laboratory/Observatory, Charts and Market Flow; ranked agenda, source limits, saved fold/uncertainty evidence, keyboard splitter/catalogue, offline risk boundary.16 sections and six 3Dmodules retained. |
| New adversarial and recovery tests | IMPLEMENTED / TESTED | Concurrent planning, transaction failure rollback, invalid numerics, source tampering, future-data perturbation, separate approvals, fenced identity and restart/deduplication tests. Existing worker safety suite retained. |
| Full backend / native / wheel | VERIFIED x86 host scope |1,235 passed,9 FinBERT-asset skips,0 failures;82.99% combinedcoverage. Final independently installed wheel147 passed/1 optional C-ABI skip, plus5 factor tests;217 packaged source/data/UI bytes match. x86C++20 pybind and standalone builds,200,000sample rolling parity. |
| Browser / static | VERIFIED automated scope |47 new Chromium checks,8 axe scans,8 responsive checks; retained suites28/37/15/35/22/30/105allpass (overlap).149 Node smoke checks. Ruff,Pyflakes,mypy 22 modules,compilation pass. Genuine screenshots retained. |
| Remaining research / acceptance | PARTIAL / BLOCKED | Registry remains finite and code-reviewed; no arbitrary hypothesis execution. Cash-equity permanent IDs, corporate actions, historical membership/publication vintages unavailable. No licensed observed options/print feed. Container capacity, physical ARM64, cross-browser and full/manual accessibility acceptance remain open. |

A subsequent CI run exposed a scheduling assumption in the existing 90ms lease
test. Its deterministic clock/heartbeat correction passes75stress cases and41focused
tests without changing runtime safety. Original full-regression artifacts are scoped
to `c5112cf`; final-head CI results are recorded in PR #13 and the delivery report
explains the test-only overlay.

The read-only catalogue now declares the actual module for each registered
method, correcting the inherited equity module label. Models,results,data,UI
and worker controls are unchanged. Last complete local regression at `7f3f382`:1,236
passed/9skipped/0failures,82.98%coverage. Current catalogue passes42focused checks
and164installed-wheel tests with1optional C-ABI skip. Full final-head CI results
are recorded in PR #13.

Final application head `e7d8939`:1,237local backend passes,9FinBERT skips,0failures;
Python3.11/3.12 CI:1,235passes/11skips each,82.92%coverage. Frontend installation
failed on Ubuntu26.04 APT DNS before browser tests. CI now pins the documented
Ubuntu24.04 runner with preinstalled Chrome runtime; all browser suites remain
required. Final check runs are recorded on PR #13.

Exact methods, source rights and negative results: [equity-registry.md](research/equity-registry.md).
Release tests, hashes and screenshots: [delivery report](research/equity-registry-delivery.md)
and `docs/evidence/registry-equity-20261111/validation.json`.
`fully_funded_certified=false`; `T4_NOT_CERTIFIED`; historical §139 UNSUPPLIED.

## Multi-dataset research and observed positioning (2026-11-10)

PR #11 was inspected with no unresolved reviews and six green CI checks, then
merged into the development branch at `a82a528f743fd61dffe095b18b386e68538f8337`.
Its 880-file source archive was independently downloaded and hash/mode verified.
The next milestone is proposed separately in PR #12 on
`development/gig-research-multidata`; no public launch is authorized.

| Requirement | Disposition | Evidence and scope |
| --- | --- | --- |
| Multiple approved templates and distinct datasets | IMPLEMENTED / TESTED | Treasury reversal, ECB FX covariance and BoC USD/CAD variance; separate approval, fixed priorities, source/method fingerprints and revision gates |
| Stronger research validation and Skeptic | IMPLEMENTED / INCONCLUSIVE | Purged chronology, covariance/QLIKE scoring, block uncertainty, local comparison correction and independent numerical implementations; no economic claim |
| Lawful zero-cost data | VERIFIED for pinned sources | ECB255 dates, BoC248 dates and CFTC52 weekly reports; original artifacts, actual acquisition clocks, official terms, attribution and integrity checks retained |
| Actual Market Flow analytics | IMPLEMENTED, bounded scope | Observed CFTC aggregate futures positions, open-interest identities, reporting-week net changes and prior-only percentiles; not trades, sweeps, cash flows or dealer positioning |
| Strategy Laboratory / Observatory / Charts / Market Flow UX | IMPLEMENTED / TESTED | Persistent resizable research rail, method-aware saved comparisons, chart source/focus controls and interactive observed positioning; all16 sections and 3D modules retained |
| Comprehensive regression | VERIFIED | 1,185 passed,9 FinBERT-asset skips,0 failures;82.83% combined statement/branch coverage (85.82% statements,74.49% branches) |
| Installed wheel / numerical kernels | VERIFIED within x86 host scope | 121 passed,1 optional standalone C-ABI skip; compiled native extension and source kernels exercised; no ARM64 claim |
| Browser and static checks | VERIFIED within automated scope | 30 new Chromium workflow checks,6 axe scans,3 mobile overflow checks;149 Node checks;Ruff,Pyflakes,mypy18 modules and compilation passed; suites overlap |
| Remaining acceptance | BLOCKED / REMAINING | Fixed historical samples, unknown historical publication/revision clocks, no current feed or options-print entitlement; Docker capacity, physical Ampere and broader/manual accessibility acceptance unresolved |

Exact source permissions and research results are in
[multidata-observed.md](research/multidata-observed.md) and
[CFTC_POSITIONING.md](integrations/CFTC_POSITIONING.md). Genuine screenshots and
validation records are under `docs/evidence/multidata-20261110/`.
`fully_funded_certified=false`; `T4_NOT_CERTIFIED`; historical §139 remains UNSUPPLIED.

## Recovered Research Director milestone (2026-11-09)

This bounded milestone follows PR #10's verified merge at `ed8f0712` and is
proposed on its own feature branch. It does not certify the full platform.

| Capability | Disposition | Evidence / remaining boundary |
| --- | --- | --- |
| Approved Director objective and automatic follow-up | IMPLEMENTED / TESTED, bounded scope | One registered Treasury falsification template; durable deduplication, approval/revocation, role outputs and memory; server must be running |
| Research lease and failure recovery | IMPLEMENTED / TESTED | Renewable leases, fenced atomic result/job commits, retry persistence, SQLite failure injection and competing-process tests |
| Additional observed research | IMPLEMENTED / INCONCLUSIVE | Purged Treasury reversal 61/115 versus baseline58/115; reused archive, unknown publication/revision clocks; no economic claim |
| Investigation timeline and Laboratory | IMPLEMENTED / TESTED | Saved findings, model-appropriate metrics and interactive evidence-to-disabled-execution timeline; existing 12 sections plus prior extensions retained |
| Market Flow vertical | PARTIAL / TESTED | Authored European option scenarios, Greeks, payoff and model alerts; observed flow unavailable, no monitored watchlists or dealer-position assumption |
| Regression / wheel / browser | VERIFIED for recorded scope | 1,110 backend passes,20optional skips,82.56%coverage;118installed-wheelpasses,1optionalC-ABI skip;22newChromiumchecks including dark/lightaxe,15legacyObservatorychecks,149Nodechecks; suites overlap |
| Research Autonomy 3.0 beyond registered template | REMAINING | Further approved dataset/template coverage and specialized investigations; no unrestricted code generation or automatic model promotion |
| Operations and certification | BLOCKED / UNVERIFIED | Docker store remains full; no container run attempted. Physical ARM64/Ampere unverified; FinBERT assets absent/research-only; fully_funded_certified=false,T4_NOT_CERTIFIED,§139UNSUPPLIED |

See [research-organization.md](research-organization.md) and the milestone evidence
under `docs/evidence/autonomy-20261110/` for exact validation scope and screenshots.

Sprint 9 delivery: [sprint9-delivery.md](sprint9-delivery.md). Redesign, 3D lab and Global Intelligence:
[global-intelligence.md](global-intelligence.md), [analytics-lab-3d.md](analytics-lab-3d.md); backend regression at
`5ffd805` and again at `47e3180`: 1584 JUnit cases, 0 failed, 11 skipped, 81.89% branch coverage. Follow-up hardening on
`cursor/gig-next-phase-26c4`: see "Data classes" below and [next-phase plan](next-phase-plan.md).

## Data classes

| Class | Where it appears | What it means |
| --- | --- | --- |
| OBSERVED (DELAYED) | USGS earthquakes, NASA EONET events | Acquired from the provider by an explicit refresh; provider feeds lag real time, so never LIVE |
| CACHED | USGS/EONET after a failed refresh or past the staleness window | Last acquired revision, shown with its acquisition time |
| HISTORICAL | Replay cutoffs more than one hour in the past; admitted HISTORICAL_OBSERVATION reports | Only revisions acquired at or before the cutoff |
| CALLER-ASSERTED | Analyst-admitted geopolitical/defense reports | Citation verified as HTTPS and policy-checked; content truth asserted by the admitting analyst |
| SYNTHETIC / DEMONSTRATION | All six 3D module inputs; authored hazard demonstration events; admitted SYNTHETIC records | Authored data, labelled in the envelope, legend and footer; dashed markers on the globe |
| UNAVAILABLE | AIS, OpenSky ADS-B, satellite imagery, FIRMS, admitted-local 3D inputs | Listed with blocker; nothing drawn or simulated |
| CANDIDATE | GDACS | Reachable; reuse terms not reviewed; not ingested |

No sanctions source is in the registry. Sanctions material can enter only as a cited admitted report.

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
| Free continuous integration (§13,15) | COMPLETE | Sprint 9 `.github/workflows/ci.yml`: GitHub-hosted 3.11/3.12 backend matrix, native build, lint/types/tests/coverage, Node smoke and Chromium workflow; read-only token; green on `47e3180` (runs 41775322422, 41775328878) after `b3c0668` (`python -m pytest`) and `47e3180` (section-ready wait); follow-up branch adds pinned SHA-256-verified axe-core | Whole-package mypy drift (9 pre-existing errors) keeps typing scoped to new modules | P1; no paid runners |
| Actual funded Greeks, scenarios, concentration/funding (§9) | COMPLETE | Actual positions override hypothetical counts; working-order budgets and mark guards | Extend optional sector/factor/drawdown contracts | P0; scoped model estimates |
| Funded sector/factor constraints and drawdown (§9) | COMPLETE | Delta-equivalent sector/factor budgets and immutable NAV drawdown tests | Explicit exposure/knowledge metadata and saved history; no inferred zero beta | P0; metadata availability |
| Historical VaR/ES (§9) | COMPLETE | Empirical reference on 21 comparable daily funded NAV marks; gaps unavailable | New lifecycle must preserve capital/mark comparability | P1; sufficient admissible history |
| Atomic risk policy/kill, pre-trade/post-event recovery (§9) | COMPLETE | Risk-index/kill rollback, stale post-fill and admission tests | Lifecycle/reconciliation hooks must retain guarantees | P0; none |
| All twelve sections / typography / themes / 3D / 2D (§10) | COMPLETE | Chromium 78 + retained suites; Node 106 baseline checks | Wire new verified workspaces through existing design | P1; integration work |
| Institutional deep-black redesign, thirteen sections (UI directive) | COMPLETE | `#050608` palette layer in `styles.css`; twelve original sections preserved plus Global Intelligence; retained Chromium suites 87/24/32/60/78/82/194/28 (axe-core) and global-3d acceptance pass; at `47e3180` the global-3d suite had no axe scan; the follow-up branch adds axe WCAG A/AA scans of Options and Global Intelligence in dark and light themes and fixes 3D label clipping/overlap found in screenshot review | Visual review on further hardware/browsers | P1; none |
| Six interactive 3D analytics modules (UI directive) | COMPLETE | `viz.py` + `viz3d.js`/`models3d.js`: vol surface, order-book landscape, risk terrain, correlation network, stat-arb landscape, model evaluation; orbit/zoom/pan/hover/select/filter, axes/legends, keyboard, pause, reduced motion, 2D table fallback, listener cleanup; stored outputs equal recomputation; `tests/test_viz_modules.py`; Chromium global-3d checks; follow-up branch: builders take explicit inputs, input-sensitivity and no-look-ahead tests, and a browser check that a changed payload changes the rendering | Inputs are labeled SYNTHETIC/DEMONSTRATION; observed option chains, depth and holdings absent | P1; observed inputs |
| Global Intelligence workspace (UI directive) | COMPLETE | `gig/intel` + `intel_workspace.py` + `intel.js`: orthographic globe and flat map from local Natural Earth geometry, layers, search, timeline playback, feed, cited details, freshness/health, saved investigations; USGS and NASA EONET adapters with revisions and backoff; event study and theme exposure; [design and source disposition](global-intelligence.md) | Maritime AIS, ADS-B, imagery and FIRMS shown UNAVAILABLE and GDACS CANDIDATE pending permitted free access; no sanctions source; follow-up branch refuses off-host redirects before following, fails closed on malformed payloads and records USGS update time separately; no World Monitor (AGPL-3.0) code reused | P1; source permissions/keys |
| Geopolitical & defense research workspace (UI directive) | COMPLETE | `intel/defense.py`: ≥24h publication delay, 1° coarsening, closed fields, HTTPS citations, exercise not treated as conflict; ARMY/AIR FORCE/NAVY/GEOPOLITICAL aggregates and market risk monitor | Admitted public reports only; no unit or platform tracking, interception or targeting support by design | P0; permanent boundary |
| New observed/lifecycle/reconciliation terminal workflows (§10) | COMPLETE | Chromium 151: 77 Sprint 7 checks; retained suites 87/24/32; Node 128 | Backend contracts then real Chromium/restart/accessibility checks | P1; backend readiness |
| Bounds, serialization, errors, secrets, corruption, recovery (§11) | COMPLETE | Existing API, persistence and recovery tests | Repeat focused new-boundary checks, not a full institutional audit | P0; none |
| CPU C++20 / wheel / installation (§11,13) | COMPLETE | x86 native parity, wheel content hashes/import tests | Reverify new release; no GPU or infrastructure expansion | P1; none |
| Fresh current container lifecycle (§11) | BLOCKED | Previous Sprint 4 image/restart evidence is scoped to that source | Read-only VFS gate: 0 available Docker-store bytes; requires at least 3 GiB container reserve / 6 GiB plus budget for build | P2; local disk capacity, no source/evidence deletion |
| Physical Ubuntu 24.04 ARM64/Ampere (§11) | BLOCKED | Prior QEMU evidence explicitly separate | Actual hardware/runtime acceptance required; no spending/provisioning | P2; hardware unavailable |
| Browser/manual accessibility / cross-browser (§13) | IN PROGRESS | Chromium 151 / axe 4.10.3; Sprint 9 Chromium 156 operations workflow 28/0; Chromium 156 thirteen-section retained suites (585/0) and global-3d acceptance 102/0 with one USGS/EONET request and 100/0 offline at `47e3180`; follow-up branch 107/0 live (USGS 339, EONET 44 acquired) and 105/0 offline including axe WCAG A/AA with 0 violations (1 incomplete item per scan, manual review) on Options and Global Intelligence in both themes; 23 screenshots in `evidence/global-3d/` inspected individually | Chromium194 checks; Firefox153 assets installed but startup timeout/uid_map EROFS/compositor errors; formal accessibility and broader browser acceptance remain open | P1; available tools/hardware |
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
