# Sprint 7 integrated delivery

The interrupted lifecycle, statement, factor-risk, observed-reference and terminal
work was recovered without resetting source. Preservation commit
`d5a9622bccf46f48d20de9f51fa24e8028669795` preceded integration. Integrated source
`92951605090a9047e6aee5f632608ee8c5868c45` was published and independently compared
with GitHub: all 581 tracked blob identities and modes matched.

The supported new workflows are documented in [observed data](observed-data.md),
[funded lifecycle](funded-lifecycle.md), [statement comparison](statement-reconciliation.md)
and [factor risk](funded-factor-risk.md). All twelve terminal sections remain.
The [completion matrix](completion-matrix.md) records remaining scope and blockers.

Full backend regression: **880 passed, zero failed/skipped**, branch-aware coverage
**81.68%**. Configured Ruff, Pyflakes, compilation and mypy passed. The independent
native wheel installed and matched 153 packaged file hashes; installed checks:
108 passed, one skipped because the separately built rolling C-ABI library is
not in that wheel. The native extension itself imported from the installed wheel.
The source native/operations suite passed 22/22. These suites overlap and must
not be added together as unique test coverage.

The x86 C++20 rolling benchmark used 200,000 samples and 31 repetitions: maximum
absolute parity error 6.94e-18; native median 4.67 ms versus NumPy 114.28 ms in this
shared environment. This measures a kernel, not complete application throughput.

Chromium 151.0.7922.34 release workflows passed 77/77; retained baseline, depth and
legacy flows passed 87/24/32, respectively, without JavaScript exceptions. Tests
include actual local HTTP restart/replay, five viewport widths, both themes,
keyboard/3D behavior and axe 4.10.3. There were no automated axe violations;
some contrast checks remain incomplete and require human interpretation.
Node smoke checks passed 128/128. This is Chromium coverage, not cross-browser
or formal accessibility certification. Exact evidence is in `docs/evidence/sprint7/`.

Recovery archive `recovery/sprint7-integrated-source.tar.gz` contains 539 source
and evidence files from the integrated source commit, excluding nested recovery,
runtime state and model binaries. Archive SHA-256:
`cbc0f871f7fbfca27bc7450bd4f3c49f4221074bf6766fec8fc1f7d33c4216af`.
It was published, downloaded independently from pinned GitHub commit
`f6eb53f82eb560c0fbdb4f701c3fbe86f5559a58`, restored into a fresh directory,
verified file-by-file including modes, and compiled. A verified complete-history
Git bundle is local; no second independent storage provider is claimed.

Known limitations include unmodeled pre-payment dividend receivable valuation,
no genuine admitted equity/option vendor coverage, unknown historical ECB
publication clocks, one nonconverged actual HMM fold, poor actual BiLSTM error,
fixture-only external statement reconciliation and no economic validation.
FinBERT remains noncommercial research only and its restricted weights are outside
Git and recovery archives. Current fresh container build is blocked by safe disk
reserve; earlier QEMU evidence does not verify physical ARM64/Ampere acceptance.
T4 remains T4_NOT_CERTIFIED and historical §139 remains UNSUPPLIED.

The next batch evaluates saved observed-reference predictions against training-only
baselines, retaining unavailable fits and negative findings. It does not turn
date-only FX reference research into equity alpha or account P&L.
