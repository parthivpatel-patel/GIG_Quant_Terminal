# Persisted economic research diagnostics

`POST /api/v1/workspace/economic-research` evaluates original saved CPU or pairs
experiments against their hash-verified admitted dataset. It does not retrain,
select the best candidate, generate market observations, admit orders, or certify
profitability. The `economic-research` terminal engine exposes completed reports
in Research, Models, Statistical Arbitrage and Signals.

The exact request requires:

```json
{
  "kind": "CPU",
  "run_ids": ["<saved 64-character run identity>"],
  "now": "2026-11-02T15:49:00Z",
  "annualization": 98280,
  "cost_bps_grid": [0, 10],
  "slippage_bps_grid": [0, 5],
  "notional": 1000,
  "participation_limit": 0.1,
  "declared_trials": 4,
  "bootstrap_repetitions": 100,
  "block_size": 3,
  "seed": 17,
  "benchmark_instrument_id": "SYN101",
  "factor_instrument_ids": {"authored_proxy": "SYN102"}
}
```

`kind` is CPU or PAIRS. One to four distinct runs must share a dataset. Each cost
axis accepts one to three distinct values in [0,1000] bps. Notional is positive,
at most 1e12; participation is in (0,1]. Annualization is an explicit researcher
assumption, not an inferred exchange calendar. The declared family must contain
at least every evaluated run/cost/slippage combination. Optional bootstrap count
defaults to 100 (100–500); block length defaults to 5 (1–64); seed defaults to 17.
Benchmark and up to four distinct admitted instrument factor proxies are optional.

## Time, membership and missingness

CPU evaluation supports one-step horizons only. Saved purged fold boundaries and
maximum training-label end precede evaluation, and each prediction must reference
its saved out-of-sample model fold. HMM probabilities and GARCH variances cannot
be treated as return forecasts. Saved pair decisions retain their original
training and embargo provenance; volatility regime labels remain the existing
descriptive heuristics, not independently estimated HMM states.

The dataset freezes observations and effective/knowledge-time membership at each
original decision. Returns use consecutive frozen decision marks or the existing
causal action-aware index; a later correction cannot silently revise the entry
price. There is no imputation of unavailable prices, absent securities or unknown
exits. Missing individual predictions and an entirely missing cross-section in a
declared CPU fold remain UNKNOWN periods. An incomplete held-security outcome
makes the full scenario's performance unknown; it is never dropped to improve
the backtest. An unavailable terminal future mark likewise remains unknown.

This protects the admitted sample from several reproducible look-ahead and
survivorship mistakes; it does not establish a complete authoritative delisting
or historical security master. The archive's event grid is not a certified
exchange-session calendar. Survivorship completeness stays UNVERIFIED.

## Economic and statistical definitions

CPU weights are current out-of-sample centered cross-sectional ranks normalized
to unit gross. Pair weights are the saved causal proposals aggregated by clock.
These are fractional, constant-notional research positions, not the funded
accounting ledger. Gross attribution is the sum of instrument weighted returns.
Net returns subtract `(cost_bps + slippage_bps) / 10000 * absolute_weight_turnover`,
including final liquidation turnover. Borrow, financing, margin, nonlinear impact,
liquidity and unreported corporate-action cash flows are outside this framework.

Capacity is a declared participation fraction of admitted bar volume times raw
price per required weight change. It is a retrospective sample diagnostic, not
executable liquidity or order admission. Zero volume is a known zero capacity;
no turnover means NOT_REQUIRED_NO_TURNOVER, rather than unknown or an invented
infinite capacity. Missing required marks/volume remain UNKNOWN. Terminal-exit
volume is explicitly a retrospective check, not knowledge available at entry.

At least 20 complete observations and two block lengths are required for the
approximate Newey–West/HAC positive-mean t diagnostic and seeded circular moving
block bootstrap mean interval. Zero sampling variance leaves significance
UNKNOWN. Stationarity, effective sample size and inference validity are not
certified. Bonferroni adjustment multiplies the diagnostic p-value by the entire
caller-declared model/cost family, including unobserved trials. Completeness of
that declared family is not independently verified.

Benchmark comparison uses only complete admitted proxy returns, with benchmark
transaction costs excluded explicitly. Factor coefficients use descriptive OOS
OLS on complete admitted price-proxy returns; a missing risk-free series is not
replaced with a claim of risk-adjusted alpha. Fold/regime summaries and early-to-
late OOS mean change are descriptive. Training-to-OOS degradation stays UNKNOWN
because a comparable economic training path was not preserved.

The report always retains `economically_validated=false`,
`production_suitable=false`, and `trading_approved=false`. A small adjusted
p-value does not establish economic usefulness. Synthetic input reports remain
explicitly labeled NOT_ESTABLISHED_SYNTHETIC_FIXTURE. Selecting a candidate after
seeing these results would require a new independent holdout.

## Persistence and tests

`economic-replay` accepts only `report_id`. Fixed-path immutable JSON, a bounded
FULL-synchronous SQLite publication index, source/model hashes, exact requests,
library versions and deterministic recomputation protect restart/replay. Up to
1,000 reports and 1,000 retained experiment periods per report are admitted.
Caller paths and URLs are unsupported. Returned objects are detached copies.
Report availability is separate from nested evaluation UNKNOWN, capacity BREACH,
or unavailable uncertainty. Terminal freshness requires an explicit policy.

Focused tests cover actual saved ridge and statsmodels-backed pairs research,
scalar HAC/cost/attribution references, bootstrap seeds, missing securities and
whole missing cross-sections, training overlap, capacity/zero volume, declared
family bounds, late corrections, stale state, tampering and deterministic replay.

