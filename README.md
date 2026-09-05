# GIG Trading Algorithm

Research-to-production platform for systematic trading (**GIG_Quant_Terminal**).

The research loop is: point-in-time universe → factors → neutralization → walk-forward backtest → risk-constrained portfolio → costs → risk limits → Alpaca paper execution.

Python is the research layer. Tight loops (cross-sectional neutralization) also ship as optional **C++** (`src/cpp/speed.cpp`, module `gig._speed`). If a compiler is missing, the same math runs in Python.

Operating guide: [docs/USAGE.md](docs/USAGE.md).

## Components

| Area | Where | Why it matters |
|---|---|---|
| No lookahead | `src/gig/backtest/simulator.py` | Scores at close *t* earn the *next* session’s return |
| Point-in-time universe | `src/gig/data/universe.py` | Listing date, history, price, ADV filters |
| Factor research | `src/gig/factors/` | 12-1 momentum, short-term reversal, idiosyncratic vol |
| C++ kernels | `src/cpp/speed.cpp` | Sector neutralization in native code |
| IC / Newey–West | `src/gig/research/ic.py` | Rank IC, IR, HAC t-stats |
| Multiple testing | `src/gig/research/multiple_testing.py` | Deflated Sharpe (Bailey–López de Prado) |
| Costs | `src/gig/backtest/costs.py` | Spread + commission + square-root impact |
| Factor risk model | `src/gig/risk/factor_model.py` | `Σ = BB' + D` from principal components; ex-ante vol and its decomposition |
| Portfolio construction | `src/gig/portfolio/optimize.py` | Closed-form factor-neutral book at a volatility target |
| Risk limits | `src/gig/risk/` | Gross/net/name/sector limits, historical VaR/ES |
| Paper execution | `src/gig/execution/` | Reconcile, pre-trade gate, sliced limit orders, audit trail |
| Multi-asset | `src/gig/assets/` | Futures TSMOM, Black–Scholes + IV, VRP tilt |
| Live data + cache | `src/gig/data/providers/yahoo.py`, `store/` | Yahoo bars persisted in DuckDB |
| Walk-forward ML | `src/gig/ml/ranker.py` | GBDT predictions only on embargoed OOS folds |
| News NLP | `src/gig/nlp/` | Headline sentiment joined as-of publication date |
| Terminal | `src/gig/service/` | Read-only API + 3D factor-space UI over the same code path |

The flagship strategy is a **dollar-neutral equity long/short** (`src/gig/strategies/equity_ls.py`). Futures time-series momentum and an options variance-risk-premium tilt sit behind the same result object so the research process is identical across asset classes.

The evaluation surface is `src/gig/` and `tests/`. Root-level `*_engine.py` files are the pre-rebuild prototype.

## Install

Python 3.11+.

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux

pip install -e ".[dev]"
python -m gig doctor
python -m gig db init
python -m gig ingest
pytest
```

Live prices are the default. CI uses `--source synthetic` so tests never hit the network.

Full US tape (NYSE / Nasdaq / AMEX common stocks, not the 36-name demo):

```bash
python -m gig universe refresh   # ~5k listed commons, a few seconds
python -m gig ingest             # Yahoo bars in batches — 30–90 min first time
python -m gig backtest --source yahoo --no-ml
```

`--mega` keeps the old 36-name sleeve. The book uses names with 21-day ADV ≥ $1M (small+mid+large liquid). Micros stay in DuckDB but are not traded. This is the **current** listing tape, not CRSP: dead names are missing, so historical Sharpe on this universe is survivorship-biased.

```bash
pip install -e ".[broker,nlp,transformers]"
```

Copy `.env.example` to `.env` only if you need a broker. Keys never belong in source.

## Run the research loop

```bash
python -m gig ingest
python -m gig backtest --source yahoo
# deterministic (tests / no network):
python -m gig backtest --source synthetic
```

Vendor APIs (FRED, EDGAR, Alpaca IEX, Finnhub): [docs/APIS.md](docs/APIS.md).

yfinance is a convenience tape. Swap it for a point-in-time vendor (CRSP, Compustat, Norgate, internal) through the `DataProvider` interface. Model policy: [docs/MODELS.md](docs/MODELS.md).

## Terminal

```bash
pip install -e ".[web]"
python -m gig serve            # http://127.0.0.1:8000
```

A read-only view of the latest cross-section: every name plotted in neutralized
factor space (momentum × idiosyncratic vol × reversal, coloured by alpha score,
sized by ADV), the book it implies, risk limits against their thresholds, and
per-factor IC with Newey–West t-stats. The vertical axis switches to live percent
change so the cloud deforms as quotes arrive.

The 3D view is rendered with a projection written for this repo rather than a
WebGL dependency, so the UI has no external assets and works offline.

Constraints worth stating: the terminal exposes no order-entry route, and the tape
labels its own source — an Alpaca IEX or Finnhub *snapshot*, or the last stored
close when no vendor key is set. Neither is a consolidated real-time feed. Trading
lives in a separate command with its own confirmation, not behind a button in a
research UI.

## Portfolio construction

Counting dollars is not neutrality. An equal-weight quantile book that is exactly
dollar-neutral can still hold most of its variance in common-factor exposure,
because the alpha is correlated with the factor loadings: low idiosyncratic vol is
a low-beta bet and momentum is a style bet, so ranking on them and taking the
extremes puts on a factor position deliberately. Measured against a fitted model,
the equal-weight book ran at 94% systematic variance while calling itself
market-neutral — which is how a 40-name-per-side book reached 40% annualized vol.

`src/gig/portfolio/optimize.py` solves the constrained mean-variance problem

```
max_w  a'w − (λ/2) w'Σw     s.t.  C'w = 0,  |w_i| ≤ m,  Σ|w_i| ≤ G
```

where `C` holds cash, sector dummies, and the model's factor exposures. Because
`Σ = BB' + D` and `C` spans `B`, the solution is closed-form — a GLS residual of
alpha on the unwanted exposures, tilted by inverse specific variance — so
neutrality holds to machine precision and there is no solver to fail mid-run. `λ`
is pinned by an annualized ex-ante volatility target rather than chosen. Box
limits are imposed by alternating projection onto the box and the neutral
subspace, which converges because both sets are convex.

`--no-optimizer` runs the equal-weight rule instead; it is kept as the baseline
the constrained book has to beat. `python scripts/ab_construction.py` runs both on
one set of scores. On 300 names over 1002 sessions, with the signal held fixed:

| | Equal-weight quantiles | Risk-constrained |
|---|---|---|
| Sharpe (after cost) | 0.56 | **1.18** |
| Annualized vol | 30.8% | **9.0%** |
| Max drawdown | −41.9% | **−7.8%** |
| Systematic share of variance | — | 6e-26 |

Return fell from 17.3% to 10.6% — part of the old return *was* the factor bet.
Volatility fell by more than three times as much.

One gap is open and worth stating: the model predicted 4.2% ex-ante vol against
9.0% realized. A diagonal specific-risk matrix understates portfolio risk when
residuals stay correlated after five principal components, and that error is
largest for a diversified book. It caused no harm here because the gross cap bound
before the volatility target, so the book was sized by leverage rather than by the
understated figure — but that is luck, not design. The ex-ante number is currently
sound for comparing books and not yet for sizing to a promise. See
[docs/USAGE.md §4](docs/USAGE.md).

## Paper trading

```bash
pip install -e ".[broker]"
python -m gig trade plan          # target, trade list, risk gate. Sends nothing.
python -m gig trade run --yes     # submits to the Alpaca paper account
python -m gig trade status        # held vs intended, per name
python -m gig trade flatten --yes # kill switch
python -m gig trade history       # audit trail
```

The target book comes from the same `scores()` path the backtest uses, so every
position traces back to a factor value and a weight the research code chose. The
account is the source of truth for what is held; each run re-reads positions and
differences them, so a partial fill or a missed session self-corrects.

Between plan and order sits a gate (`src/gig/execution/pretrade.py`). It blocks
the run on a stale lake, a limit breach in the target, ex-ante vol above ceiling,
a drawdown past the halt level, or implausible turnover; it drops individual
orders for name-specific reasons such as unavailable borrow. `--force` downgrades
exactly one check, market hours. Limit breaches and the drawdown halt are not
overridable from the command line — if a limit is wrong, change it in
configuration where the change is visible.

`AlpacaBroker` refuses to construct against a non-paper URL unless
`allow_live=True` is passed in code. Orders go out as marketable limits sliced by
notional, never as market orders. Intent, gate findings, and every child order are
written to DuckDB.

## Design rules

1. **Information available at t only.** Forward returns used for IC and PnL are shifted. Combination weights are lagged one day so today’s IC cannot size today’s book. The risk model is refit on a schedule from data ending at *t*.
2. **Costs are not optional.** Every rebalance pays spread, commission, and impact.
3. **Risk is constraints, not a score.** Gross, net, name, and sector limits are checked as data. Callers halt or resize.
4. **Experiments are hashed.** Config + metrics append to `results/experiments.jsonl`.
5. **Trading is opt-in at every step.** Dry run is the default, live orders need an explicit flag, and the gate is not bypassable by flag.
6. **No secrets in git.** `.env` is gitignored.

## Repository layout

```
src/gig/             library
src/gig/portfolio/   book construction and the optimizer
src/gig/risk/        factor risk model and hard limits
src/gig/execution/   target book, reconciliation, pre-trade gate, brokers
src/gig/service/     read-only API + terminal UI
src/cpp/             optional C++ kernels
configs/             strategy YAML
docs/                usage, research note, data and model policy
scripts/             one-off comparisons (construction A/B)
tests/               pytest (synthetic data, no network)
examples/            one-command research run
.github/workflows/   CI
```

## License

MIT. Research code is not investment advice. Past backtests, especially on synthetic data, do not predict live performance.
