# Sprint 8 recovered integration and release

The interrupted work resumed from `development/gig-ui-redesign` at
`0cffd576c66f795fa807e41d8379932a4c8431ba`, rather than the previously reported
`23de36d…`. The surviving terminal-boundary fix had already been published.
All five workstreams left source, but dividend adversarial tests, browser
integration and some model/provenance cases were incomplete. These were finished
from their existing changes. No completed component was rebuilt from baseline.

Preservation commit `2eb8350…` published an internally verified 589-file source
archive before integration. The original full `/workspace` checkout and its
uncommitted release files remain preserved. Work continued in the independent
healthy `/tmp/gig-sprint7-publication` checkout. Main, the quantitative branch and
historical protected workspaces were not modified. Model weights and runtime
account state are excluded from source archives.

## Frozen integrated source

`50f7443e10352e8a935e591545e13a4b2182ce49` is the frozen source tested below.
Independent GitHub comparison matched all 660 tracked file hashes and modes.
Subsequent release commits add documentation/evidence/recovery records only.

Saved evaluation artifacts now use their research contract in the terminal
section reader. Previously, the reader treated an evaluation as a raw snapshot
and failed on classification. HTTP and Chromium tests cover Models, Research and
Signals, ordinary snapshots, earlier cutoffs, service restart and exact replay.

Dividend accounting recognizes exact declared receivables on frozen ex-date
entitlement and known record/settlement evidence. NAV includes receivables at face
value; buying power excludes them. Payment exchanges receivable for cash without
recognizing income again. Rate changes/cancellation append compensation events;
source, accrued income, paid cash and outstanding entitlement remain distinct.
Immutable old snapshots and original raw OMS fills remain unchanged. Ambiguous
late fills/splits and unsupported financing reject. Unfundable corrections retain
their admitted source as unresolved and fail closed rather than borrowing.
Independent literal statements distinguish MATCH, DISCREPANCY and UNKNOWN.

Fourteen new dividend regressions cover conservation, pre/post-payment correction,
cancellation, exact precision, concurrent duplicates, actual process death,
exception rollback, paid-index corruption, legacy migration and independent
statement comparisons. Risk integration verifies that a receivable increases NAV
without supplying cash admission and that a failed risk publication rolls back
the source and financial transition atomically. See [lifecycle](funded-lifecycle.md)
and [statement contracts](statement-reconciliation.md) for the bounded scope.
Credit/default/tax adjustments and externally confirmed settlement are not modeled.

## Research and admitted equity evidence

Versioned evaluations preserve all six original v1 identities. V2 adds
training-only ranking baselines, tie-aware NDCG/IC, GARCH QLIKE/Gaussian diagnostic
scores, HMM partial/convergence reporting and explicit missing-baseline states.
Independent scalar references and all six saved acquired ECB reports were checked
without refitting or provider access. The four original negative return-estimator
findings remain. Ranking comparison intervals include zero. The limited GARCH
window has lower sampled QLIKE than its training-only baseline; calibrated
confidence remains UNKNOWN and no economic profitability follows.
See [model evaluation](reference-model-evaluation.md).

The official UCI Dow Jones Index ZIP was fetched, parsed, validated, persisted and
replayed under its verified CC BY 4.0 conditions. It contains 750 weekly records
for 30 source labels in January–June 2011. The AA/IBM/KO selection contains 75
observations. Future-label/dividend columns remain quarantined in raw evidence.
The published ZIP and normalized snapshot were independently downloaded from
pinned GitHub source and re-parsed through the fresh installed wheel; all fields
matched. This is source-reported archive data, not direct exchange verification.
Currency, permanent IDs, historical vintages, corporate actions and authoritative
universe membership remain unknown/unavailable. No PIT/funded eligibility or
equity alpha is claimed. See [equity scope and license](observed-equity-data.md).

## Frozen-source verification

| Check | Result |
| --- | --- |
| Complete backend regression | 945 passed; 0 failed; 0 skipped |
| Package branch-aware coverage | 81.99% |
| Fresh independent installed wheel | 163 passed; 1 optional C-ABI skip |
| Wheel source/install hash comparison | 156 packaged files matched |
| Native and operations source suite | 22 passed; 0 failed/skipped |
| Actual Chromium 151.0.7922.34 | 194 passed; 0 failures/JavaScript exceptions |
| Node smoke checks | 147 passed |
| Ruff / Pyflakes / compilation | Passed |
| Configured mypy | Passed, 150 source files |

These selections overlap and must not be added together as unique coverage.
The wheel's separately built rolling C-ABI library is absent, so its optional
wheel test is skipped; that kernel's source-build parity test passed. The wheel's
pybind native extension imported and executed independently of the source tree.

C++20 x86 checks used GCC 14.2.0. On 200,000 samples over 31 repetitions, rolling
maximum absolute parity error was 6.94e-18; native median was 4.86 ms and the
NumPy/native median ratio was 15.86 on this shared host. This is a kernel measure,
not total application throughput or economic performance.

Browser checks exercise actual local HTTP workflows and restarts, accruals and
independent statements, saved ranking/variance tables and 3D/2D keyboard views,
equity fixtures, all twelve sections, five viewport widths, both themes and
reduced motion. Axe 4.10.3 reported zero violations; contrast incomplete items
are disclosed and require human interpretation. Selected desktop/mobile and
scoring screenshots were manually inspected. This is Chromium coverage, not
cross-browser or formal accessibility certification. No provider/broker fetches
occurred in browser tests. Exact results are in `docs/evidence/sprint8/`.

## Remaining critical path

The Docker store uses VFS and has zero available bytes. A read-only preflight
now accounts for image-copy allocation and filesystem-specific reserves, so a
healthy `/tmp` filesystem cannot incorrectly admit a Docker-store write. No new
container/build was attempted. Container startup/restart/health/crash acceptance
remains blocked; physical ARM64/Ampere acceptance is also unverified. Source
HTTP/process-death checks and prior QEMU evidence do not establish those results.

Next acceptance work is current-image/runtime validation on adequate storage,
physical ARM64 testing when available, broader accessibility/browser coverage,
and permitted equity identity/action/vintage evidence before PIT research use.
Continue using the [single completion matrix](completion-matrix.md); this delivery
is not whole-platform completion or an institutional certification.

The three-to-four-week preferred engineering target remains November 24–December
1, with December 8 as the five-week planning ceiling, conditional on the explicit
external blockers. FinBERT remains noncommercial research only;
`fully_funded_certified=false`, `T4_NOT_CERTIFIED`, and historical §139 UNSUPPLIED.

## Follow-on acceptance preparation

After protecting Sprint 8, the browser runner was extended with an explicit
Chromium/Firefox selector, a 30-second launch bound and task-local writable cache.
The library, frontend and backend-test source remain byte-identical to frozen
commit `50f7443…`. Ruff, Pyflakes and compilation passed for the modified runner;
the final Chromium run again passed all 194 workflows with no JavaScript errors.

Firefox 153.0 assets downloaded successfully, but actual launch timed out before
any workflow ran. The final attempt logged `Sandbox: writing /proc/self/uid_map:
EROFS` and `RenderCompositorSWGL failed mapping default framebuffer`. This is a
failed startup/blocked runtime check, not Firefox validation. No host namespace
or filesystem permissions were changed to force a pass. Cross-browser acceptance
therefore remains open; the failure evidence is preserved separately.

The 637-file Sprint 8 source/evidence archive was independently downloaded from
pinned GitHub commit `8f60faaeb1b0e37b66c83c631d47981b7245780f`, restored into a
fresh directory, verified against every SHA-256 and file mode, and compiled.
Archive SHA-256:
`e81bc3a567d02da57b51c8a1c24d650b821686f242318da2b16e4325e16cd8bf`.
That independently restored source then built and installed a fresh native wheel.
All 155 Python/static/schema package files matched the original wheel, the rebuilt
extension imported from its independent installation, and 13 native numerical
tests passed with zero skips. Compiled binary bit identity is not claimed.
These checks overlap the release tests and are not additional unique coverage.

GitHub is the sole verified external storage provider. Local temporary copies
and complete-history Git bundles do not constitute another durable provider.
Recovery records identify exact source/archive commits; later documentation and
runner commits do not rewrite earlier evidence or model findings.
