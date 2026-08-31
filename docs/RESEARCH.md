# Equity long/short — research note

**Strategy:** dollar-neutral cross-sectional long/short  
**Universe:** synthetic 80-name panel (replace with PIT vendor data for live research)  
**Horizon:** next-session close-to-close  
**Rebalance:** every 5 sessions  
**Book:** long 20 / short 20, 5% name cap, gross 2  

## Hypothesis

Residual 12–1 momentum (Jegadeesh–Titman), 21-day reversal, and low idiosyncratic volatility (Ang et al.) retain rank-IC after sector neutralization. Combining them with lagged expanding-window IC weights should produce a positive after-cost Sharpe on a dollar-neutral book, with turnover that is not explosive.

The synthetic panel is generated as

```
r_i,t = β_i r_m,t + γ_{s(i)} r_{s,t} + λ mom_{i,t-1} + ε_i,t
```

so a passing test is evidence that the *pipeline* recovers a planted premium after neutralization — not evidence of a live edge.

## Construction

1. Tradable mask: min 252 sessions of history, price > $1.
2. Raw factors from prices through close t only.
3. Cross-sectional OLS residual vs sector dummies (drop first).
4. Spearman IC vs r_{t→t+1}; combination weights are EWM IC shifted by one day.
5. Long 15 / short 15, equal weight, gross 2, name cap 5%.
6. One-way costs: 4 bp spread + 1 bp commission + η √(Q/ADV) impact.

## What we report

- Annualized excess return, vol, Sharpe, Sortino, Calmar, max drawdown
- Annualized one-way turnover
- Per-factor mean IC, Newey–West t-stat, hit rate
- Deflated Sharpe with n_trials = number of strategies in this repo (3)

## Failure modes we actually test

| Test | File |
|---|---|
| Listing dates cannot be traded early | `tests/test_universe.py` |
| Momentum IC vs *next* bar, not same bar | `tests/test_factors.py` |
| Perfect same-bar scores do not explode Sharpe | `tests/test_backtest.py` |
| Walk-forward embargo | `tests/test_research.py` |
| Put-call parity / IV round-trip | `tests/test_assets_risk_exec.py` |

## Production gap list (honest)

- No survivorship-free CRSP universe
- No borrow fees or locate
- No open auction / close-auction execution model
- Impact is a one-parameter square-root, not a calibrated propagator
- Options VRP uses a unit long-vol PnL series, not a full surface

Those gaps are the difference between a hiring-quality research platform and a live book. The code is structured so each gap is a drop-in adapter, not a rewrite.
