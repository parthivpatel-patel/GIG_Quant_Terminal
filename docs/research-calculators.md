# Stateless research calculators

`POST /api/v1/analytics/<kind>` computes one result from the JSON request and
returns it as a `MODEL_ESTIMATE` with readiness `RESEARCH_ONLY`. A calculator
reads no account, saved dataset or market feed, saves nothing and places no
order. Unknown fields, booleans or strings in place of numbers, nonfinite
values and oversized inputs are refused with 422.

The kinds, their workbench options and their examples come from one registry,
`gig.service.quant_analytics.ANALYSES`. `scripts/sync_bounded_families.py`
writes it into the static shell and CI fails if the two differ.

Running a calculator still saves nothing. **Save calculation** writes the kind
and the exact inputs to a local content-addressed journal.
**Replay saved calculation** runs the calculator again from that record and
rejects the record if the result differs. The plain calculation and the
saved record stay `MODEL_ESTIMATE` / `RESEARCH_ONLY`.

**Calculate from saved dataset** copies numbers from an already saved
point-in-time dataset and stores those exact numbers. One instrument's
`return_1` or `close` binds to `cornish-fisher-var`, `stationary-bootstrap`,
or `roll-spread`. An aligned `return_1` panel of 2–64 instruments binds to
`ledoit-wolf` or `minimum-cvar`. A timestamp missing every requested return
is omitted and counted. A timestamp missing only some of them is refused.
Nothing is imputed. Replay recomputes from the pinned numbers even if the
dataset file is later removed. The dataset classification is kept beside the
record and is not upgraded. **Allocate from saved covariance** runs
`hierarchical-risk-parity` or `maximum-diversification` on the covariance
inside a saved Ledoit-Wolf record and pins that matrix. The weights remain
a research estimate. `POST /api/v1/analytics/<kind>` still reads no dataset.

| Kind | Method | Independent reference in `tests/test_quant_analytics.py` |
| --- | --- | --- |
| `ledoit-wolf` | Ledoit-Wolf (2004) shrinkage to a scaled identity | scikit-learn `ledoit_wolf` |
| `hierarchical-risk-parity` | Lopez de Prado (2016): single linkage, leaf order, recursive bisection | the published quasi-diagonal and recursion, written separately |
| `minimum-cvar` | Rockafellar-Uryasev linear program, long-only, weight cap, optional return floor | two-asset grid over the definition |
| `efficient-frontier` | long-only minimum variance across attainable returns | closed-form two-fund weights; one-dimensional scan under a cap |
| `parametric-var` | normal VaR and expected shortfall with Euler components | quadrature of the quantile function; finite-difference marginal |
| `cornish-fisher-var` | fourth-order Cornish-Fisher quantile | SciPy moments and the written-out expansion |
| `historical-var-components` | scenario VaR, expected shortfall and per-asset tail shares | the tail-loss definition minimized over every sample loss |
| `monte-carlo-var` | seeded Gaussian or covariance-matched Student-t | the normal formula, within 2% at 200,000 paths |
| `var-backtest` | Kupiec coverage, Christoffersen independence, Basel-style zone | binomial log-likelihoods; the Basel 250-day table; hand-counted transitions |
| `almgren-chriss` | Almgren-Chriss (2000) discrete optimal liquidation and frontier | tridiagonal first-order conditions; random perturbations never improve the objective |
| `participation-schedule` | whole-share volume-proportional slices under a participation cap | conservation, cap and proportionality properties |
| `shortfall-simulation` | seeded implementation shortfall under the same price model | analytic mean and standard deviation |
| `american-option` | Cox-Ross-Rubinstein tree with early exercise | a plain recursion; Hull's five-step put (4.49); put-call symmetry |
| `option-greeks-advanced` | closed-form vanna, volga, charm, speed, zomma | finite differences of the existing Black-Scholes kernel |
| `svi-smile` | raw SVI least squares, butterfly function, wing slopes, implied density | recovery of generating parameters; a published arbitrage example |
| `variance-swap` | discrete log-contract replication | flat-volatility Black-Scholes strip |
| `backtest-overfitting` | combinatorially symmetric cross-validation | direct enumeration of every split |
| `track-record-length` | minimum track record length | inversion of the existing probabilistic Sharpe ratio |
| `fama-macbeth` | period regressions, Newey-West standard errors on mean slopes | statsmodels OLS and HAC |
| `trade-bars` | time, tick, volume and dollar bars from ordered prints | pandas group-by; conservation of prints |
| `bar-quality` | structural errors and statistical warnings for a bar series | one planted defect of each kind |
| `heston-option` | Heston (1993) European value, one Fourier integral | Black-Scholes limit as the variance of variance vanishes; seeded full-truncation Monte Carlo |
| `barrier-option` | Reiner-Rubinstein single barrier; Broadie-Glasserman-Kou discrete shift | Brownian-bridge Monte Carlo for every up/down, in/out, call/put branch |
| `option-strategy` | exact piecewise-linear expiry profit | hand-computed bull call spread, covered call and long call |
| `conditional-stress` | zero-mean joint-normal shock propagation | two-asset regression formula and a chi-squared tail probability |
| `liquidity-var` | equal-sale liquidation VaR plus an exogenous half spread | the closed horizon factor and the Bangia spread term |
| `cost-aware-rebalance` | mean-variance rebalance with proportional costs and a turnover cap | a one-dimensional scan; a book that is already optimal does not trade |
| `maximum-diversification` | long-only Choueifaty-Coignard ratio | inverse-volatility weights and a simplex grid under a cap |
| `tracking-portfolio` | long-only minimum tracking variance | the benchmark itself when it is feasible; an unreachable active-return floor is refused |
| `stationary-bootstrap` | Politis-Romano (1994) geometric blocks, circular wrap, percentile interval | a direct block-by-block reconstruction; unit blocks match the iid resample variance |
| `roll-spread` | Roll (1984) spread from the autocovariance of price changes | the written covariance formula; a seeded bid-ask bounce recovers its spread |
| `corwin-schultz` | Corwin-Schultz (2012) high-low spread on consecutive bars | the published two-day expression on each window |
| `risk-parity` | long-only risk budgeting | two-asset closed form in `tests/test_risk_allocation.py`; registry weights checked in `tests/test_quant_analytics.py` |
| `black-litterman` | Gaussian Black-Litterman posterior; diagonal view variance unless Omega is supplied | a view equal to the equilibrium spread leaves the prior unchanged; one-asset Gaussian update |
| `fractional-kelly` | capped `fraction * Sigma^{-1} * mu` | unconstrained two-asset weights and a binding gross cap |
| `drawdown` | compounded wealth against a peak that starts at one | a three-return path: peak, 20 percent drawdown, partial recovery |

## Limits

- Inputs are whatever the caller supplies. Covariances, expected returns,
  impact coefficients, volume forecasts and option prices are assumptions, and
  no result is evidence about a market.
- `svi-smile` is a local fit from 27 fixed starts. It reports butterfly
  arbitrage; it does not constrain the fit to be free of it, and it does not
  check calendar arbitrage across expiries.
- `american-option` supports a continuous dividend yield only.
- `variance-swap` is biased by coarse strike spacing and truncated wings; the
  share of the two edge strikes is reported.
- `monte-carlo-var` does not estimate its own sampling error.
- `var-backtest` uses asymptotic tests with low power in short samples.
- `bar-quality` is calendar-unaware and repairs nothing.
- `fama-macbeth` needs a balanced panel and applies no errors-in-variables or
  multiple-testing correction.
- `heston-option` integrates caller-supplied parameters. It does not calibrate
  them, and a failed integral is refused rather than returned.
- `barrier-option` has constant volatility, one barrier and no rebate. Discrete
  monitoring uses a barrier shift, not an exact discrete-barrier price.
- `option-strategy` values one shared expiry only. It is not a mark-to-market.
- `conditional-stress` propagates shocks linearly. It does not apply
  stress-period correlations.
- `liquidity-var` sells in equal daily pieces and adds a spread you supply. It
  does not estimate price impact.
- `cost-aware-rebalance`, `maximum-diversification` and `tracking-portfolio`
  are long-only and fully invested. A solution is returned only after its
  optimality conditions have been re-checked.
- `stationary-bootstrap` resamples the supplied series. The interval is not a
  hypothesis test and does not correct for having tried other series.
- `roll-spread` needs a negative autocovariance. A trend is reported as not
  identified rather than given an imaginary spread.
- `corwin-schultz` uses overlapping pairs. Negative windows are counted and
  also shown floored at zero.
- `risk-parity` is long-only and fully invested. A binding cap is labelled an
  approximation rather than an exact budget.
- `black-litterman` returns a posterior mean and covariance. It does not
  optimize a portfolio, and market weights must be long-only and sum to one.
- `fractional-kelly` is a small-return Gaussian approximation. Name and gross
  caps change that result, and shorting is off unless requested.
- `drawdown` compounds the supplied arithmetic returns. It is not a forecast.
