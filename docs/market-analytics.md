# Event-data microstructure, options, and pair research

These local numerical modules do not contact a provider, invent market data,
grant broker authority, or establish economically validated performance. Event
identity, licensing, timestamps and point-in-time alignment remain input authority.

## Microstructure

`gig.microstructure.Quote` and `Trade` preserve integer nanosecond timestamps,
instrument, source and data-kind identities. Numeric inputs must be finite;
crossed quotes, fractional timestamps and duplicate identities are rejected. An
event identity is `(instrument, timestamp_ns, sequence)`.

- `match_quotes(trades, quotes, lag_ns=0, max_age_ns=5000000000)` performs an
  instrument-specific as-of match. It never uses a quote newer than trade time
  minus the lag. Zero lag permits an equal-timestamp quote; strictly preceding
  timestamps require a positive lag. Sequence resolves equal-timestamp quotes.
  No-prior and stale quotes are distinct diagnostics; trade order is retained.
- `lee_ready(matches)` uses the midpoint rule, then a per-instrument tick rule.
  Missing-quote fallback is labelled a tick estimate. Unresolved direction stays
  unknown, rather than becoming an observed buy or sell.
- `order_flow_imbalance(quotes)` implements top-of-book event-size OFI for a
  chronological single-instrument sequence, including quote price changes.
  It does not infer full order-book depth from bars.
- `kyle_lambda(price_changes, signed_volumes)` fits an intercept and signed-volume
  slope. Unidentifiable volume or insufficient samples returns an unavailable
  coefficient rather than zero market impact.
- `vpin(sizes, signs, bucket_volume=..., window_buckets=50)` splits trades into
  equal-volume buckets. Unknown signs and unfinished volume are reported and
  excluded. This is event-sign VPIN, not bulk-volume-classification VPIN. Inputs
  are capped at one million events and 100,000 completed buckets. Compensated
  rolling updates with periodic rebases avoid quadratic window summation.
- `spread_metrics(quote)` reports quoted spread and top-of-book imbalance; empty
  depth has unknown imbalance. `execution_costs(...)` reports signed arrival-price
  slippage and explicit fees, without inferring unobserved execution prices.

## Options

`gig.options.black_scholes(spot, strike, rate=0, div=0, vol=.2, ttm=1, call=True)`
uses the existing European BSM kernel behind strict admission. Times are years
with a caller-selected day count; volatility is annualized decimal. Vega/rho are
per percentage point and theta per calendar day. Results say `MODEL_ESTIMATE`.
Expired and zero-volatility smooth-Greek boundaries are excluded explicitly.

`implied_volatility(price, spot, strike, ...)` uses bracketed Brent inversion and
European discounted no-arbitrage bounds. Intrinsic maps to zero volatility. The
upper price boundary implies infinite volatility and is rejected. Roots above
`max_vol` are not clipped; each answer must reprice within the declared tolerance.

`diagnose_call_slice` checks price bounds, strike monotonicity and discrete
convexity. `SurfacePoint` carries log-forward-moneyness, maturity, volatility and
source. `validate_surface` checks duplicates and calendar total variance only at
exactly matching moneyness coordinates; absent comparisons stay unknown.
`interpolate_total_variance` interpolates inside the sampled convex hull and
refuses extrapolation. Source identities are retained; butterfly-arbitrage
certification is explicitly absent. Surface inputs are capped at 10,000 points.

`OptionPosition(instrument, underlying, strike, ttm, volatility, quantity, ...)`
requires contract and underlying identities. `portfolio_exposure` aggregates
signed contract quantities and multipliers for one underlying and currency.
`scenario_grid` fully reprices relative spot shocks and absolute volatility
shocks. Its result dictionary contains `scenarios` rows with `spot_shock`,
`volatility_shock`, `value` and `model_pnl`. These are model differences, never
realized brokerage P&L. The combined grid-by-position budget is 100,000 repricings,
with at most 10,000 grid cells.

VTS and CSM remain `AUTHORITY_UNSUPPLIED`; no historical definition is invented.
There is no OPRA entitlement, options-feed or observed-flow claim.

## Statistical arbitrage

`gig.research.stat_arb.fit_pair(y, x, dependent="Y", independent="X",
training_rows=None, log_prices=True)` returns immutable `PairModel` parameters
`alpha`, `beta`, `spread_mean`, `spread_std` and a training digest. They depend
only on the selected prefix; the digest includes instrument identities.
`evaluate_pair(model, y, x)` returns fixed-model spreads and z-scores; zero
training dispersion has unknown z-score. Log beta is an elasticity, not a share
hedge ratio; level beta has price-ratio units.

`mean_reversion_diagnostics(spread)` returns a dictionary of descriptive AR(1)
parameters. A continuous OU interpretation requires `0 < phi < 1`; this is not a
stationarity significance test. `cointegration_test` and `adf_test` require
optional statsmodels. Missing dependency returns `UNAVAILABLE` with no statistic
or p-value; correlation never substitutes for significance. `select_pairs`
applies Bonferroni correction across all screened pairs and never selects with
unavailable significance evidence. Post-selection economic validity is unverified.

`kalman_hedge` exposes causal filtered intercept/beta estimates and innovations,
without full-sequence smoothing. `walk_forward_pair` fits before each evaluation
fold and is capped at 1,000 folds. It computes research spreads, not P&L.

`backtest_pair` adds a bounded rehearsal: a close-t z-score target executes at
close t+1, then earns the following interval's returns. A newly computed signal
cannot receive same-bar returns. Dollar weights distinguish log elasticity from
level-price share hedges. Turnover includes drift of prior holdings and explicit
basis-point costs. The model uses constant initial-capital notional without
compounding, explicitly leaves terminal positions open and excludes borrow,
financing, market impact and capacity validation. A caller-supplied boolean
eligibility mask can block new exposure; its regime validity is not inferred.
Inputs are capped at 20,000 rows. Outputs remain unapproved research.

## Verification

The resumed batch passes 38 event/options and 14 pair-research tests with no
skips. References cover published BSM price, put-call parity, independent
finite-difference Greeks, IV repricing, OFI/VPIN hand calculations, nanosecond
matching, identity, training/test causality, next-close execution lag and
independent turnover/cost arithmetic.

```sh
PYTHONPATH=src python -m unittest discover -s tests -p test_market_analytics.py -v
PYTHONPATH=src python -m unittest discover -s tests -p test_stat_arb_research.py -v
```

No provider, broker, paid service, deployment or protected workspace was used.
