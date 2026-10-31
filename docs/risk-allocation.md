# Validated research risk and additional allocation methods

`gig.risk.analytics` accepts complete finite return panels rather than silently
dropping missing observations. Arithmetic returns use fractions, not percentage
points. Asset identities and timestamps must already be aligned upstream.

- `historical_loss_risk(returns, confidence=.99)` uses a positive-loss convention:
  `loss=-return`. Empirical VaR uses inverse CDF; Expected Shortfall integrates
  the worst `(1-confidence)*n` observations with a fractional boundary weight.
  Negative values represent a gain in that tail and are preserved. At least 20
  observations are required by default; this is a computational floor, not a
  statistical sufficiency claim. The older `gig.risk.var.historical_var_es`
  retains its documented signed-return convention for compatibility.
- `drawdown_report(returns)` compounds wealth from one, includes the initial
  wealth in the high-water mark, reports negative fractional drawdowns and
  underwater duration in observations. A -100% return permanently exhausts
  wealth. Levered portfolios that can lose more than NAV need a distinct NAV
  accounting model and are rejected here.
- `stress_positions(quantities, prices, shocks, nav=...)` fully revalues constant
  linear equity positions, including shorts. It excludes nonlinear derivatives,
  financing, transaction costs and rebalancing. Shocks have dimensions scenarios
  by instruments, and cannot reduce a stock price below zero.
- `portfolio_risk(returns, weights)` reconciles Euler variance contributions,
  reports their signed fractions (hedges can contribute negatively), historical
  loss estimates, drawdowns and annualized volatility. Annualization defaults
  to 252 and is explicit. It assumes constant periodically rebalanced weights.

`gig.portfolio.allocation` adds missing research methods without replacing the
existing mean-variance or factor-neutral construction engines. Covariances must
be finite, symmetric and strictly positive definite; any regularization must be
an explicit upstream decision.

- `risk_parity(covariance, budgets=None, max_weight=1)` solves long-only fully
  invested risk budgeting. It verifies constraints and reports the maximum
  deviation from desired risk fractions. Binding caps can make exact parity
  impossible: the result then says `CONSTRAINED_APPROXIMATION`. Solver failure
  raises; there is no unreported allocation fallback.
- `black_litterman(covariance, market_weights, views, view_returns, ...)`
  implements the Gaussian posterior using linear solves. It reports equilibrium
  prior, posterior excess returns, uncertainty in that mean, and predictive
  covariance. Supplied view covariance must be positive definite. The documented
  default is diagonal prior view variance. All returns/covariances/views must
  share a horizon; the function never mixes daily and annual units or turns a
  posterior forecast into an approved order.
- `fractional_kelly(expected_excess, covariance, fraction=.25, max_gross=1,
  max_name=.25, allow_short=False)` uses a Gaussian small-return approximation
  with explicit fraction, name and gross caps. Shorting requires a boolean
  opt-in. Capped weights are an approximation rather than a universal Kelly
  optimum; model uncertainty, loss tails and insolvency risk remain.

Numerical tests use hand-computed fractional-tail ES, independent covariance
reconciliation, long/short stress arithmetic, inverse-volatility parity under
diagonal covariance, a scalar conjugate Gaussian posterior, and explicit Kelly
formula/cap checks. These establish implementation correctness for those cases,
not trading suitability or economic validation. Run:

The resumed review reproduced and repaired three numerical admission defects:
Kelly solve overflow could be hidden by later clipping, Black-Litterman could
return nonfinite results from finite inputs, and a representable tail mean could
overflow its unnormalized sum. Regression evidence is preserved in the batch
evidence. Kelly now checks the unconstrained result before caps; Black-Litterman
checks prior, view system and posterior arithmetic; ES divides before summation.
Stress tests also reject a finite P&L whose division by tiny NAV overflows.
Allocation is capped at 512 assets/views. Risk vectors are capped at one million
observations, with at most 512 assets and four million panel/scenario elements.
The focused risk/allocation suite now has 19 passing tests and no skips.

```sh
PYTHONPATH=src python -m unittest discover -s tests -p test_risk_allocation.py -v
```
