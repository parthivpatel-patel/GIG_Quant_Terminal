# Sprint 9 delivery (reconstructed)

Starting point: `development/gig-ui-redesign` at
`7f22036f07cab26d41f4519dc841cfe0b386adef` (local and remote agreed). No Sprint 9
commit, branch, worker output, evidence file or recovery record existed on any
local or remote ref, so Sprint 9 was reconstructed from the completion-matrix
rows left IN PROGRESS rather than recovered. Nothing earlier was rewritten.

Checkpoint source: `51014c2` on `cursor/sprint9-recovery-26c4`
(`c905323` reconstruction, `51014c2` CI scoping).

## Scope delivered

| Area | Implementation | Verification |
| --- | --- | --- |
| Executed-cost evidence | `gig.execution.cost_evidence.cost_evidence_report`: ESTIMATED, SIMULATED, OBSERVED_HISTORICAL classes; BROKER_RECONCILED rejected because brokerage is disabled. Exact Decimal spread/slippage/shortfall identity; market impact only as a measured post-trade mid move labelled `MEASURED_PRICE_MOVE_NOT_CAUSAL`, otherwise UNKNOWN; independent print matching MATCHED/UNMATCHED/AMBIGUOUS/SAME_SOURCE_UNVERIFIED; observed classes aggregate only when corroborated; capacity derived only from independent prints; deterministic `report_sha256` | `tests/test_sprint9_cost_evidence.py` (8) |
| Market-data provenance | `gig.data.provenance`: lineage records with license, access permission, observation/publication/acquisition/ingestion clocks, historical-availability evidence, revision history (null = unknown), corporate-action basis, security identity, quality and content hash; verified `lineage_sha256`; eligibility for archive inspection / point-in-time research / funded allocation with inherited parent failures. Documented ECB and UCI sources are never point-in-time eligible | `tests/test_sprint9_provenance.py` (9) |
| Terminal exposure | Workbench analytics kinds `cost-evidence` and `provenance` | Chromium workflow below |
| Browser / operations | `scripts/browser_sprint9.py`: isolated server, static security headers, cross-origin 403, malformed JSON 400, invalid request 422, both workbench analyses in Chromium, restart with deterministic replay, twelve sections render, no JS errors, no external requests | 28 passed, 0 failed, Chromium 156.0.8078.4 ([report](evidence/sprint9/browser/report.json)) |
| Free CI | `.github/workflows/ci.yml`: GitHub-hosted runners only, read-only permissions, Python 3.11/3.12 backend matrix (ruff, pyflakes, compileall, scoped mypy, pytest with coverage, native build) and a frontend job (Node smoke + Chromium Sprint 9 workflow + offline 3D/Global Intelligence acceptance) | Runs on push/PR; no paid runners. First green on `47e3180` (runs 41775322422, 41775328878), after `b3c0668` switched to `python -m pytest` so `scripts` imports resolve and `47e3180` made the browser check wait for the clicked section to become current |

## Regression at the checkpoint

Run from a clean worktree of `51014c2` with native extensions built (GCC 13.3):
1547 test cases reported by JUnit (951 passed, 11 skipped, 0 failed, 0 errors,
585 subtests passed), branch coverage 81.61% against the 70% gate. The JUnit file
is [evidence/sprint9/regression.xml](evidence/sprint9/regression.xml).

Skips (all environmental, none hidden): 9 FinBERT tests (pinned
noncommercial weights not provisioned) and 2 observed-scoring tests (actual ECB
acquisition evidence not in this environment). Node terminal smoke: 147 checks
passed.

## Known limitations

- mypy on the whole package reports 9 errors in files this sprint did not touch
  (`ranker.py`, `ledger.py`, `persisted_pairs.py`, `contracts.py`,
  `store/__init__.py`); the Sprint 8 mypy version was not recorded, so CI type
  checks only the new modules until that drift is resolved.
- No observed historical executions or independent prints are in the repository,
  so OBSERVED_HISTORICAL evidence is exercised on fixtures only.
- Docker lifecycle, physical ARM64 and Firefox remain blocked as recorded in
  Sprint 8; they were not re-attempted.
- `fully_funded_certified=false`, `T4_NOT_CERTIFIED`, Historical §139
  UNSUPPLIED are unchanged.
