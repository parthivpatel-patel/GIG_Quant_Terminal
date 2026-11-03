# Sprint 6 recovered integration

Work continues on `development/gig-ui-redesign` from
`63074fc750f242ce59fe352bd1bcb5b24a2cb8d2`. The interrupted implementation was
preserved at `69be154` in a verified 473-file archive before recovery edits.
Its SHA-256 is `143eba0d8389437e2768e998f152024985d28c6c8f9ba1c8f71d0012726cbb2b`.
No surviving implementation commit, publication or active worker was found.

Funded accounting, economics, microstructure and partial terminal work survived.
The advanced-risk module and Sprint 6 browser script had not survived; they were
completed after the existing files were recovered. The quantitative branch and
main remain unchanged; historical protected workspaces are untouched.

## Integrated capabilities

The isolated funded cash account reuses the existing OMS, outbox and offline
counterparty protocol. Fully paid long USD equities and long European cash-settled
options have exact multiplier-aware premiums, fees, FIFO P&L, available/reserved
cash, ownership checks, expiry obligations and explicit cash settlement. Opening
shorts, borrowing, physical delivery and unmodeled exercise styles reject. Actual
OMS fill quantities and supplemental settlement offsets remain separately visible.
No broker or counterparty-position reconciliation is fabricated, and
`fully_funded_certified` remains false. See [accounting scope](funded-accounting.md).

Economic evaluation consumes hash-verified CPU/pairs runs and their admitted data.
It retains purged folds and point-in-time universes, cost/slippage grids, turnover,
capacity assumptions, declared-family Bonferroni diagnostics, HAC/block-bootstrap
uncertainty, benchmark/factor proxies, attribution and degradation diagnostics.
Missing held securities or entire OOS cross-sections remain UNKNOWN. Insufficient
train-performance evidence does not become an invented degradation statistic.
Synthetic fixtures remain synthetic; profitable alpha and economic validation
are not certified. See [evaluation contract](economic-validation.md).

Persisted quotes and prints retain acquisition and revision knowledge clocks,
immutable source identities, causal matching/classification, best-level OFI,
Kyle regression and VPIN diagnostics. Unknown signs and missing execution costs
remain explicit. Displayed depth does not supply executed volume or fills.
See [microstructure contract](persisted-microstructure.md).

Actual funded positions and outstanding reservations connect to saved-model
Greeks, multi-underlying stress budgets, concentration, funding and optional
executed-volume policies. Policy/index/kill changes share the OMS transaction;
pre-trade admission and post-event reconciliation use existing safety controls.
Historical replay is not current-account approval. See [risk scope](funded-risk.md).

All twelve terminal sections retain their design, themes, responsive navigation,
provenance and 3D camera controls. Explicit workspace actions, saved references,
reservations, OFI/period/liquidity tables and cost/stress plots use actual results.
The new Chromium workflow exercises premium cash, duplicate fills, expiry,
research/microstructure/risk replay and an actual service restart.

## Verified release

The complete final regression passes **821 tests**, with **zero failures, errors,
skips or deselections**, at **81.43% coverage** against the configured 70% minimum.
No dependency-blocked tests remain. Ruff, Pyflakes, compilation and configured
mypy on 141 source files pass; mypy retains its import-skipping and untyped-body
limitations and is not a strict whole-program proof. The installed native wheel
passes **110 selected tests**, and all **146 packaged source/static/data files**
match the checkout. Its SHA-256 is
`50a86eb692043d7bb934472d8c9ea4d5f41bbdcb240a46ac8a92c28a504e044e`.
The 22 focused native/operations tests pass; a 20,000-observation x86 C++20 rolling
comparison matches the reference within `3.469446951953614e-18` absolute error.
Concurrent test load makes the timing evidence unsuitable for controlled
performance claims.

Actual Chromium 151.0.7922.34 passes 78 Sprint 6, 87 retained baseline,
24 depth/motion and 32 legacy persisted-flow checks, with no JavaScript errors or
tested axe 4.10.3 violations. Node passes 106 smoke checks. Counts overlap; Node
is not browser certification. Responsive checks cover 320, 375, 768, 1024 and
1920 px; manual screen-reader/cross-browser review remains outstanding.

Further adversarial verification reproduced a concurrent duplicate-intent race:
one of two simultaneous callers received a uniqueness error despite a single
safe order. Receipt and clock checks now repeat under the OMS writer lock, and
both callers receive the same immutable receipt. Force-exit, duplicate fill,
unfunded liability, price/fee, expiry, precision and risk/kill rollback tests also
pass. Intermediate failures were repaired; the final complete suite ran after
the final source change.

Comparable funded NAV history now supports empirical 99% VaR and fractional-tail
ES only with 21 complete marks on a regular daily UTC grid, fixed initial capital
and current mark clocks. Insufficient/irregular inputs stay UNAVAILABLE. The
historical source snapshot IDs are frozen and hash-verified on replay.

The recorded synthetic CPU economics example produced -4.616% compounded return
at zero costs and -6.314% at 10 bps, with adjusted positive-mean p-value 1.0 in both
cases. This verifies cost sensitivity and a negative research finding, not
observed-market alpha or profitability. Exact report identity/values are in
`docs/evidence/sprint6/sprint6-economic-fixture-result.json`.

Validated implementation milestones were published at `0edb250` and
`3599e3b7f5cb719199c42bbf8ed947c73dd53fc8`. The latter's complete 529-file GitHub
tree matches local blob hashes/modes. The interrupted archive was independently
downloaded from GitHub and all 473 member hashes checked. A complete Git bundle
is verified locally; final source/evidence receives an additional externally
verified recovery archive. GitHub is the verified external provider, not a claim
of a second independent storage service.

FinBERT remains noncommercial research
only, with restricted weights excluded from Git and recovery archives. There were
no broker/provider connections, paid services, paper/live orders or public
deployment. Physical ARM64/Ampere acceptance is unverified. T4 remains
`T4_NOT_CERTIFIED`; historical §139 remains `UNSUPPLIED`.
