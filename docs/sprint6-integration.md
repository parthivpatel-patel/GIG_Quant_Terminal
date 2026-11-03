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

## Verification checkpoint

The first integrated selection passes 48 focused tests. Ruff and configured mypy
pass. Actual Chromium 151.0.7922.34 passes 78 Sprint 6, 87 retained baseline,
24 depth/motion and 32 legacy persisted-flow checks, with no JavaScript errors or
tested axe 4.10.3 violations. Node passes 106 smoke checks. Counts overlap; Node
is not browser certification. Responsive checks cover 320, 375, 768, 1024 and
1920 px; manual screen-reader/cross-browser review remains outstanding.

Complete regression, installed wheel, numerical and final recovery evidence are
recorded at the final release checkpoint. FinBERT remains noncommercial research
only, with restricted weights excluded from Git and recovery archives. There were
no broker/provider connections, paid services, paper/live orders or public
deployment. Physical ARM64/Ampere acceptance is unverified. T4 remains
`T4_NOT_CERTIFIED`; historical §139 remains `UNSUPPLIED`.
