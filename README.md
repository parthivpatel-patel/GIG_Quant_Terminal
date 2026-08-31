# GIG Trading Algorithm

Research-to-production platform for systematic trading (**GIG_Quant_Terminal**). Built so a hiring manager at a multi-strat or quantitative hedge fund can clone the repo, run tests, and see a complete research loop: point-in-time universe → factors → neutralization → walk-forward backtest → costs → risk limits.

Python is the research layer. Tight loops (cross-sectional neutralization) also ship as optional **C++** (`src/cpp/speed.cpp`, module `gig._speed`). If a compiler is missing, the same math runs in Python.

This is not a “Renaissance clone,” not a live money printer, and not a dashboard with forty loosely named engines. It is a small, tested library that does a few things correctly.

## What a reviewer should look at

| Area | Where | Why it matters |
|---|---|---|
| No lookahead | `src/gig/backtest/simulator.py` | Scores at close *t* earn the *next* session’s return |
| Point-in-time universe | `src/gig/data/universe.py` | Listing date, history, price, ADV filters |
| Factor research | `src/gig/factors/` | 12-1 momentum, short-term reversal, idiosyncratic vol |
| C++ kernels | `src/cpp/speed.cpp` | Sector neutralization in native code |
| IC / Newey–West | `src/gig/research/ic.py` | Rank IC, IR, HAC t-stats |
| Multiple testing | `src/gig/research/multiple_testing.py` | Deflated Sharpe (Bailey–López de Prado) |
| Costs | `src/gig/backtest/costs.py` | Spread + commission + square-root impact |
| Risk | `src/gig/risk/` | Gross/net/name/sector limits, historical VaR/ES |
| Multi-asset | `src/gig/assets/` | Futures TSMOM, Black–Scholes + IV, VRP tilt |
| Live data + cache | `src/gig/data/providers/yahoo.py`, `store/` | Yahoo bars persisted in DuckDB |
| Walk-forward ML | `src/gig/ml/ranker.py` | GBDT predictions only on embargoed OOS folds |
| News NLP | `src/gig/nlp/` | Headline sentiment joined as-of publication date |

The flagship strategy is a **dollar-neutral equity long/short** (`src/gig/strategies/equity_ls.py`). Futures time-series momentum and an options variance-risk-premium tilt sit behind the same result object so the research process is identical across asset classes.

Root-level `*_engine.py` files (if still present) are the pre-rebuild prototype. They are not the evaluation surface. Interviewers should read `src/gig/` and `tests/` only.

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

Free vendor APIs (FRED, EDGAR, Alpaca IEX, Finnhub): [docs/APIS.md](docs/APIS.md).

## Design rules (non-negotiable)

1. **Information available at t only.** Forward returns used for IC and PnL are shifted. Combination weights are lagged one day so today’s IC cannot size today’s book.
2. **Costs are not optional.** Every rebalance pays spread, commission, and impact.
3. **Risk is constraints, not a score.** Gross, net, name, and sector limits are checked as data. Callers halt or resize.
4. **Experiments are hashed.** Config + metrics append to `results/experiments.jsonl`.
5. **No secrets in git.** `.env` is gitignored. The previous README in this workspace contained live API keys; rotate them.

## Repository layout

```
src/gig/          library
configs/             strategy YAML
tests/               pytest (synthetic data, no network)
examples/            one-command research run
.github/workflows/   CI
```

## What this is not

- It will not get you hired by claiming Medallion-like returns.
- LSTM-on-price or LLM buy/sell as the production signal. See `docs/MODELS.md`.
- yfinance is a convenience tape, not CRSP. The `DataProvider` interface is the swap.

A serious desk will replace the synthetic provider with a point-in-time vendor (CRSP, Compustat, Norgate, internal). The interfaces are written for that swap.

## License

MIT. Research code is not investment advice. Past backtests, especially on synthetic data, do not predict live performance.
