# Scoped risk for actual offline funded-account positions

`funded-risk` evaluates the isolated `funded-account/oms.sqlite` book using the
existing OMS. Saved European option runs supply IV and model assumptions; their
hypothetical contract counts are ignored. Actual effective positions, lifecycle
settlements and all outstanding orders determine exposure. Legacy OMS positions
and the Sprint 5 hypothetical overlay remain separate.

The request requires `now`, `max_age_seconds` (0–86400), exact-decimal `marks`,
`options_run_ids`, `scenarios`, and all eight USD policy caps: `max_gross_notional`,
`max_name_notional`, `max_abs_delta_notional`, `max_abs_gamma_dollar`,
`max_abs_vega_per_point`, `max_abs_theta_per_day`, `max_scenario_loss`, and
`min_available_cash`. Each scenario has a unique name, explicit per-underlying
`spot_shocks` and `vol_shocks`, and `elapsed_days`. Missing required coordinates,
models, identities, current marks or knowledge cutoffs cannot become zero risk.
Execution clocks reject nanosecond remainders before mutation; source timestamps
and freshness comparisons retain nanoseconds.

Contract terms, multipliers, exercise style, expiry, underlying and saved model
coordinates must agree. Underlying spot/observation clocks must match the admitted
option evidence. Prices in the funded ledger may be rounded quotes; the difference
from the BSM model baseline is explicit. Greeks use the actual contract quantities:
delta notional in USD, gamma times spot squared in USD, vega per volatility
percentage point and theta per calendar day. Full BSM stress repricing uses fixed
admitted IV/rates/dividends and explicit time/spot/vol shocks. Scenarios reaching
expiry or nonpositive volatility are unsupported, not extrapolated.

Budgets use conservative per-leg absolute exposure, grouped underlying gross
concentration, and worst-case asynchronous working quantities. Working-order fees
and adverse limit-to-mark entry costs also enter the scenario bound. Candidate
orders must retain minimum available cash after their full reservation. These
assumptions are deliberately conservative and do not prove executable liquidity.
Optional `liquidity` entries require executed-print volume, observation timestamp,
source identity and `kind: EXECUTED_PRINTS`. Volume is in native instrument units;
caller authenticity is unverified. A requested `max_participation_rate` rejects
unknown/stale/insufficient evidence. Displayed depth never supplies this volume.
Without a participation policy, missing liquidity remains visibly UNKNOWN.

A completed risk run installs its policy in the funded OMS transaction domain.
The existing reserve path rechecks policy coverage, freshness, cash and worst-case
budgets before admitting an order. Policy-free accounts retain mandatory funding
controls and explicitly incomplete advanced-risk status. Changed marks, unavailable
post-fill models and actual breaches preserve the incoming simulated accounting
event and atomically admit ERROR risk evidence with the existing durable kill
switch. Pre-trade rejection creates no order. A successful report does not release
an engaged switch or approve trading.

Historical VaR/ES is admitted only from 21 complete, positive, published funded
NAV snapshots on an exact 24-hour UTC observation grid ending at the risk clock.
Capital bases must match, held-position mark clocks must equal each snapshot's
clock, and unsettled, stale or missing valuations are excluded. Irregular or
insufficient history is UNAVAILABLE. The resulting 20 simulated account returns
use empirical 99% loss VaR and fractionally weighted ES from the existing risk
kernel; they are not market-wide observations or a profitability claim. Original
snapshot IDs and NAV inputs are frozen in the report and verified on replay.

Risk files are immutable SHA-256 artifacts; their authoritative index, policy and
kill control commit in the same OMS database transaction. Saved account inputs
and model identities support deterministic historical replay. Replay does not
reapprove a changed current account. Insufficient comparable funded NAV history
leaves historical VaR/ES UNAVAILABLE; a scenario grid is not a probability sample.
No complete funding, economic, broker or institutional certification is claimed.
