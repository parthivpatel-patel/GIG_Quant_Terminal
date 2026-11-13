# Approved equity research milestone: delivery evidence

Feature branch: `development/gig-research-equity-registry`, PR[#13](https://github.com/parthivpatel-patel/GIG_Quant_Terminal/pull/13),
base `development/gig-ui-redesign` at `1f9a3e1cfa7363db242e4d3a9f3fb8802acc9a13`.
The scientific experiment/UI implementation is commit `f749380fb8162eb8b7b2cb1d30f90ef139a12c60`;
later commits add release evidence, a deterministic lease-test correction and
explicit registered catalogue module metadata.
Worker, expiry and fencing controls and scientific results are unchanged. Validation records
hash every tracked source, test, script and CI/config input. The final publication
SHA is the PR's current head; this avoids a self-referential commit hash in a file.

## Implemented and tested

The immutable investigation registry now contains five distinct methods across
four admitted datasets. Planning uses saved research memory, dataset novelty,
registered falsification and estimated resource cost, with transparent components
and a checkpointed order. Approvals remain per investigation; selecting a question
or reading status neither approves nor starts compute. Resource caps, live leases,
fenced commits, bounded retries, pause/resume and restart behavior are retained.
The planner transaction rolls back the investigation if job insertion fails.
The unfingerprinted compatibility exception applies only to the original Treasury
objective, not new investigations.

Two new studies extend the existing licensed UCI archive. Each has125 observations
on five fixed source labels, three purged chronological folds and75 held-out panel
observations across15 weeks. Range and log-volume persistence both underperform
training-only medians. Their independent numerical Skeptic checks pass; verdicts
remain INCONCLUSIVE. Original Treasury, ECB, BoC and synthetic negative/inconclusive
findings remain unchanged. See [methods and permissions](equity-registry.md).
These retrospective diagnostics cannot establish adjusted equity returns, a
survivorship-free universe or historical trading eligibility.

The official CFTC equity-index futures slice adds 52 weekly E-mini S&P 500 reports
for 2023, with exact identity, integer conservation, pinned data/metadata, original
acquisition clocks and verified public-domain reuse. It has its own offline HTTP
contract and Market Flow selector. No dollar flow, options print, cash-equity
price, dark-pool trade or historical publication timestamp is inferred.
See [source evidence](../integrations/EQUITY_INDEX_POSITIONING.md).

The Investigation Desk connects Command Center, Market Charts, Observatory /
Strategy Laboratory, Market Flow and Research using a persisted research focus.
It explicitly retains each workspace's independent instrument and data identity.
The Laboratory exposes actual fold scores, units, uncertainty, design limits,
Skeptic findings and provenance-to-disabled-execution timeline. The original
sixteen sections and analytical3Dmodules remain. Splitter ARIA state and keyboard
access to the catalogue were repaired after actual browser checks found defects.

## Frozen runtime and regression snapshots

Evidence directory: `docs/evidence/registry-equity-20261111/`.

- Previous full backend snapshot at `c5112cf`: **1,235 passed, 9 skipped, 0 failures**, 485.66 seconds. The nine skips
  require actual pinned FinBERT assets/runtime, which are not provisioned or
  redistributed. No DuckDB, CPU sequence-model or observed-data test is blocked.
- Combined statement/branch coverage: **82.99%** (85.96% statements, 74.69% branches); exact counts are in
  `coverage-summary.json`. Suites below overlap and are not added to backend totals.
- Final installed x86 CPython 3.12 wheel: **147 passed, 1 skipped**, plus **5 factor tests**.
  The skip is the optional standalone C-ABI library not included in the wheel.
  The compiled pybind extension is imported from the independent target install.
  All 217 packaged source/data/UI files match frozen source bytes.
- C++20 pybind and standalone rolling builds pass. A 200,000-sample NumPy/standalone
  native comparison passes with 11 timing repetitions. Numeric error, host platform
  and raw timings are recorded in `native-parity.json`; timings are local CPU
  measurements, not application-level acceleration or economic evidence.
  Twelve additional installed-wheel categorical neutralization cases match NumPy
  least-squares with maximum absolute error 1.281e-9, including masking, absent
  baseline groups, large offsets and both demean settings.
- Ruff, Pyflakes, compilation and the expanded configured mypy check
  (**22 modules**) pass. Node verifies 149 DOM smoke checks; these are not browser
  rendering or accessibility certification.
- New genuine Chromium 151.0.7922.173 suite: **47 passed, 0 failed**, including8 axe-core
  scans across four views and both themes,8 responsive checks at 390/768 pixels,
  approvals, automatic bounded cycles, saved replay, source failures, hostile
  response text, keyboard controls and shared context. No uncaught JavaScript
  errors or external browser requests.
- Retained Chromium 151.0.7922.34 suites: Sprint 9 **28**, visual regressions **37**,
  Observatory **15**, Charts **35**, prior autonomy **22**, multidata **30** and
  Global Intelligence/3D **105**, all with zero failures. These suites overlap.

Actual screenshots and their SHA256records are linked from
[browser evidence](../evidence/registry-equity-20261111/browser/README.md).
The screenshots show real saved results and accurately unavailable states,
not generated illustrations. Automated axe does not establish complete WCAG
conformance; manual accessibility, Firefox/Safari and broader device acceptance
remain incomplete.

## CI lease-test correction

A Python 3.12 push-CI run at `c5112cf` failed one existing lease test while the
Python 3.11 and frontend jobs passed. The test used a 90ms lease and a fixed
sleep, assuming the host would schedule heartbeat threads before expiry. That
assumption is invalid on a loaded runner. Runtime expiry and fencing correctly
remained strict; no safety grace period or ownership bypass was introduced.

The test now advances a logical lease clock only after observing an actual
persisted heartbeat renewal. It still crosses the original expiry boundary,
checks concurrent callers and shutdown, and adds exact-expiry rejection of
renewal and stale commits. Seventy-five stress cases (25 repetitions of three
cases) and the focused 41-test suite pass. `lease-repair.json` pins the changed
test and the earlier CI failure. The final-head full rerun and its exact counts
are recorded in PR #13; original regression artifacts retain their source scope.
UI, installed-wheel and numerical evidence remain valid because their inputs did
not change. The current hash manifest includes the test correction.

## Final catalogue metadata correction and acceptance

The expanded registry initially inherited the reference-study module path for
its two equity methods in the read-only catalogue. Their actual runners and
saved reports already identified `gig.research_org.equity_experiments` correctly.
Each registered method now declares its implementation module explicitly; a
regression check verifies the new catalogue paths. No calculation, model output,
source identity, approval or worker control changed.

The last complete local regression before this metadata correction, at `7f3f382`,
is **1,236 passed,9 skipped,0 failures**, with **82.98% combined coverage**.
Its full output is preserved as `pre-catalogue-regression.txt` and XML. The
corrected catalogue passes42focused checks; its rebuilt independently installed
wheel passes **164 tests,1 optional standalone C-ABI skip**. All217packaged
source/data/UI files match current source. Ruff,Pyflakes,compilation and configured
mypy22modules pass. Complete current-head Python3.11/3.12 and frontend CI results
are recorded in PR #13; the current source manifest includes every correction.
Original earlier regression artifacts keep their original source scope.

## Final application regression and CI runner correction

The exact application head `e7d8939` passes **1,237 backend tests,9 FinBERT skips,
0 failures** locally in427.73seconds; full logs are retained in
`final-application-regression.txt` and XML. Python3.11 and3.12 CI each pass
**1,235 tests,11 skips** (the same9FinBERT plus2legacy ECB acquisition-store
checks), with82.92%combinedcoverage. Those clean-runner skips do not affect the
new packaged equity/positioning studies. Suites overlap.

Frontend CI failed before application tests while APT could not resolve
`security.ubuntu.com` for font packages. GitHub's official runner manifest confirms
the November2026 migration of `ubuntu-latest` to26.04 and that the standard free
Ubuntu24.04 image already includes Chrome/Chromium runtime libraries. The workflow
now pins24.04, downloads Playwright's Chromium, and retains every actual browser
suite. No host DNS/security control is altered and no test is skipped or converted
to a DOM-only substitute. Final-head CI results are recorded in PR #13.

## Recovery, restrictions and remaining work

The original full filesystem and its older checkout were left intact. Its 19 surviving
modified/untracked files were copied into a SHA256-verified local preservation
archive before implementation. Development uses an independent `/tmp` checkout.
No protected historical workspace, main, quantitative branch or independent C++
Microstructure Engine repository was modified.

GitHub source checkpoints preserve the baseline's 914 tracked files through normal
ancestry. The frozen implementation's independently downloaded full-source archive has
940 tracked files with every byte and executable mode verified. Its SHA256 is
`6ec8657c59a8cd64119a59f07fc5b82584816c37066ae7f0c35f4fff2b15ac32`.
The final documentation head is downloaded and verified separately during release;
the PR records its exact SHA and recovery result. A local Git recovery bundle
records its base prerequisite explicitly. A local bundle is not a separately verified external provider. GitHub is
currently the sole verified external recovery provider.

Container creation was not attempted: the existing filesystem has zero writable
capacity. Host tests do not substitute for current-source container acceptance.
Physical ARM64/Ampere remains unverified. FinBERT remains noncommercial research
only; restricted weights are excluded from Git and recovery artifacts.
`fully_funded_certified=false`; `T4_NOT_CERTIFIED`; §139 UNSUPPLIED. No brokerage
connection/order, paid service, production promotion or public deployment occurred.

Next work should broaden independently approved scientific questions when suitable
source evidence exists, establish authoritative corporate-action/security identity
and historical knowledge clocks before equity-return research, and add observed
Market Flow sources only with verified reuse permissions. The registry is a finite
reviewed allowlist, not an unrestricted self-directed scientist. Operational disk
capacity, actual ARM64, cross-browser/manual accessibility and independent economic
validation remain separate acceptance gaps.
