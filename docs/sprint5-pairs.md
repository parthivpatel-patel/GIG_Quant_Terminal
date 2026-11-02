# Persisted statistical-arbitrage research

`PairsResearchWorkspace` extends the existing admitted dataset workspace. It never
connects to a provider, account or broker. A successful calculation is research
evidence and cannot approve trading.

## Requests and saved state

Run `pairs-research` with an admitted `dataset_id`. Optional settings are
`training_window` (default 60, minimum 30), `test_size` (20), `embargo` (1,
minimum 1 source-event observation), `alpha` (0.05), `entry_z` (1.5),
`cost_bps` (10), `borrow_bps` (0), `max_gross` (1), `process_variance` (0.00001)
and `observation_variance` (0.01). Replay `pairs-replay` with only `run_id`.
The resulting immutable run, every saved model and the source dataset are hash
verified on load and after restart. Replay reads these objects without fitting.

Every fold screens all pairs among instruments present throughout its training
window. The preceding embargo is excluded from fitting. Cointegration uses the
existing library-backed Engle–Granger implementation on log prices, with a
Bonferroni correction across that fold's complete training-eligible pair family.
Missing significance dependencies, invalid tests and degenerate fits cannot
select a pair. Screening results retain rejection reasons, raw and adjusted
p-values, descriptive residual ADF and mean-reversion half-life diagnostics.
ADF after pair selection does not establish independently valid inference.
Bonferroni is not a correction across all subsequent experiments or settings.

## Time, identity and corporate actions

The source panel already applies effective and knowledge-time universe
membership. Pair inputs freeze each observation at its original decision time;
they are not rewritten to the latest research-cutoff vintage. A model's
training evidence, fit and identity depend only on preceding decision evidence.
A correction learned later may change later research information, but cannot
change an earlier decision, saved model or immutable experiment. Experiments
are reconstructed research, not claims of historically published models.

RAW inputs use the admitted raw price at each decision. They have no corporate
action adjustment or independently complete action coverage. Adjusted datasets
use an explicitly labeled causal index of their action-aware `return_1`,
anchored at 100 at the beginning of a contiguous instrument segment. Raw closes
remain raw and are never presented as adjusted levels. This index preserves
known action timing; it is not a retrospectively revised adjusted-price series.
A membership gap breaks continuity. No price or membership is imputed.
The admitted archive's event grid is not an inferred exchange-session calendar.

## Causal estimates and hypothetical economics

Selected pairs use the existing causal Kalman `[intercept, beta]` filter,
initialized from the preceding OLS training fit. The last three training
observations warm the filter; subsequent observations are processed in order.
No smoothing, future fit or test-outcome-based selection is used. Process and
observation variances are declared assumptions per processed observation, not
calibrated probabilities or wall-clock diffusion estimates. Innovation variance
and standardized innovations are retained as diagnostics.

The regime label is a trailing-return-volatility heuristic relative to training
volatility, with an explicit 0.5 gross-weight haircut in the high-volatility
regime. It is **not an HMM output**. A threshold-crossing innovation produces
opposing hedge-elasticity dollar weights, scaled to the declared gross budget
and shared across selected pairs. A negative hedge ratio may produce same-sign
legs; dollar neutrality is not guaranteed. Downstream allocation must enforce
its own concentration, sector, factor, turnover, cost and execution controls.

The backtest models fractional constant-dollar weights between consecutive
frozen decision marks. RAW returns use ratios of the actual frozen marks;
they do not substitute a later correction to a preceding mark as the entry
price. Adjusted returns use the causal action-aware index. Turnover incurs
explicit basis-point costs, short weights incur ACT/365.25 borrow costs, and the
final observation incurs liquidation turnover. Unknown held outcomes make the
aggregate return unknown. No selected experiment means no fabricated zero
return. Nonpositive modeled period equity also makes compounded return unknown.
Latest target proposals are research outputs separate from the backtest's
terminal liquidation convention.

This hypothetical path does not model cash/share lots, margin, financing on
long positions, bid/ask liquidity, slippage beyond the declared cost rate, order
fills, corporate-action cash settlement, or economic validation. The existing
portfolio ledger and offline outbox remain the authority for actual **simulated**
position/cash reconciliation and explicit order-event workflows.

## Bounds and verification

Experiments accept 2–8 instrument identities, at most 500 decision clocks and
500 pair evaluations. Oversized experiments reject explicitly before publication.
Public states retain AVAILABLE / UNAVAILABLE / UNKNOWN / STALE semantics.
Caller classification and source integrity do not establish vendor authenticity.

Focused tests cover real statsmodels significance, retained model references,
transaction/borrow costs, future-prefix causality, delayed corrections,
corporate-action indexes, missing membership, immutable replay/restart,
defensive copies, tamper rejection, freshness and invalid requests. They use
authored SYNTHETIC cointegrated fixtures and make no provider-coverage claim.
