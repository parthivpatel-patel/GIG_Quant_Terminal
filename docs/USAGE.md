# Using GIG Trading Algorithm

Operating guide for the research loop and the terminal. Reference for the
research design is [RESEARCH.md](RESEARCH.md); vendor setup is
[APIS.md](APIS.md); model policy is [MODELS.md](MODELS.md).

---

## 1. Install

Python 3.11+.

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux

pip install -e ".[dev,web]"
```

`web` adds FastAPI and uvicorn for the terminal. The library and CLI work
without it. A C++ compiler is optional — the neutralization kernel falls back to
NumPy.

Verify the install:

```bash
python -m gig doctor
```

Each line is a capability, not a warning to ignore. `MISSING` next to a vendor
key means that feed will be skipped silently during ingest.

---

## 2. Build the data lake

Everything is a local DuckDB file. There is no server to run.

```bash
python -m gig db init            # create data/gig.duckdb and its tables
python -m gig universe refresh   # NYSE/Nasdaq/AMEX common stocks, seconds
python -m gig ingest             # Yahoo bars in batches — 30-90 min first time
python -m gig db stats           # row counts per table
```

`ingest` is incremental: it re-reads the tape, tops up bars, and pulls whichever
vendor feeds have keys (FRED macro, EDGAR 8-K item codes, Alpaca IEX snapshots,
Finnhub calendar). Re-run it daily; it does not start over.

Scope control:

| Command | Universe |
|---|---|
| `python -m gig ingest` | Whatever `configs/universe.yaml` selects (default: all US listed commons) |
| `python -m gig ingest --all` | Force the full US tape |
| `python -m gig ingest --mega` | 36-name mega-cap sleeve, fast for iteration |
| `python -m gig ingest --no-news` | Skip headline pulls |

The DuckDB file is gitignored and can reach several hundred MB. It is
reproducible from `ingest`, so it is never committed.

---

## 3. Run the research loop

```bash
python -m gig backtest --source yahoo --no-ml    # cached bars, factors only
python -m gig backtest --source yahoo            # adds the walk-forward GBDT
python -m gig backtest --source synthetic        # deterministic, no network
```

What happens, in order:

1. **Point-in-time universe.** Listing date, minimum history, price floor, and
   ADV floor are applied as of each date (`src/gig/data/universe.py`).
2. **Factors.** 12-1 momentum, 21-day reversal, 63-day idiosyncratic vol, plus
   news sentiment and 8-K item counts when those tables have rows.
3. **Neutralization.** Each factor is residualized on a constant plus sector
   dummies, per date.
4. **Optional GBDT.** Walk-forward only, embargoed; in-sample rows stay NaN.
   Skipped automatically above 800 names.
5. **Combination.** Weights are the EWM rank IC of each factor, **lagged one
   day**, clipped at zero, renormalized.
6. **Book.** A factor risk model is fit on a trailing window ending at *t*, and
   the book is solved to be neutral to cash, sectors, and the model's factors
   at a volatility target, subject to the 5% name cap and 2.0 gross cap. See
   section 4.
7. **PnL.** Scores at close *t* earn the return from *t* to *t+1*. Every
   rebalance pays spread, commission, and square-root impact.

Flags: `--seed`, `--no-ml`, `--no-news`, `--no-persist`, `--no-optimizer`.

Each run appends to `results/experiments.jsonl` with a config hash, and
overwrites `results/equity_ls_last.json`.

---

## 4. Portfolio construction and risk

### Why the weighting rule changed

The original rule was: long the top `n` names, short the bottom `n`, equal
weight, gross 2.0. That is dollar-neutral by *count*. It is not risk-neutral,
and on live data the difference was not small — a fitted factor model attributed
**94% of the book's variance to common-factor exposure**, which is why a book
holding 40 names on each side ran at 40% annualized volatility.

The mechanism is worth understanding because it is not a bug anywhere in the
code. The alpha is *correlated with the factor loadings*: low idiosyncratic vol
is a low-beta bet, momentum is a style bet. Ranking on that signal and taking
the two extremes therefore puts on a large factor position on purpose. Counting
dollars cannot detect it, because the exposure is in betas, not in notional.

### What replaced it

`src/gig/portfolio/optimize.py` solves

```
max_w  a'w − (λ/2) w'Σw     s.t.  C'w = 0,  |w_i| ≤ m,  Σ|w_i| ≤ G
```

with `Σ = BB' + D` from `src/gig/risk/factor_model.py` and `C` holding a cash
column, sector dummies, and the model's factor exposures. Because `C` spans the
columns of `B`, the constrained problem has a closed-form solution: the GLS
residual of alpha on those exposures, tilted by inverse specific variance. Two
things follow that matter in practice — neutrality is exact rather than
approximate, and there is no iterative solver that can fail to converge on a bad
cross-section halfway through a backtest.

`λ` is not tuned. Scaling the book scales its volatility proportionally, so `λ`
is pinned by the annualized ex-ante volatility target and then capped by gross.
The target is a **ceiling reached from below**: when the gross or name cap binds
first, the book runs quieter than target and reports which constraint bound.

The risk model is refit every `GIG_RISK_REFIT_EVERY` sessions from a trailing
`GIG_RISK_LOOKBACK` window ending at *t* — point-in-time like everything else.
Refitting daily is wasted work, which is why commercial risk models ship on a
monthly cycle. When the model cannot be fit (early in the sample, or too thin a
cross-section) the run falls back to the equal-weight rule for those dates
rather than reusing a stale model from a different regime.

### Comparing the two

```bash
python -m gig backtest --source yahoo --no-ml --no-optimizer   # baseline
python -m gig backtest --source yahoo --no-ml                  # constrained
python scripts/ab_construction.py --limit 300                  # both, one set of scores
```

The A/B script computes scores once and varies only the weighting rule, so the
difference in vol and systematic share is attributable to construction alone.
Reported per run:

| Field | Meaning |
|---|---|
| `ex_ante_vol` | Annualized vol the model predicts for the book it built |
| `systematic_share` | Share of predicted variance from common factors. Should be ~0 |
| `effective_names` | `1 / Σ share²`. Independent bets, not name count |
| `binding` | Which constraint set the book size: `vol_target`, `gross`, or `name_cap` |

A systematic share that is not near zero means the constraint set is not doing
its job and the number should be investigated before the Sharpe is believed.

### Measured effect

`scripts/ab_construction.py --limit 300`, 300 names, 1002 sessions, identical
scores on both sides:

| | Equal-weight quantiles | Risk-constrained |
|---|---|---|
| Sharpe (after cost) | 0.56 | **1.18** |
| Annualized return | 17.3% | 10.6% |
| Annualized vol | 30.8% | **9.0%** |
| Max drawdown | −41.9% | **−7.8%** |
| One-way turnover | 9.3× | 10.1× |
| Systematic share | — | 6e-26 |
| Effective names | 100 | 156 |
| Average gross | 1.37 | 2.00 |

Return fell, which is the expected trade: part of the old return *was* the
factor bet. Volatility fell by more than three times as much, so the risk-
adjusted number roughly doubled and the drawdown became something a real
allocator would tolerate.

### The gap that is still open

In that run the model predicted **4.2%** ex-ante vol and the book realized
**9.0%**. The risk model understates realized risk by roughly a factor of two,
and that is a real limitation, not a rounding difference.

The likely cause is the diagonal `D`. After removing five principal components
the residuals are not actually independent — industry and sub-industry effects
survive — and treating them as independent understates portfolio risk *most*
for a heavily diversified book, which at 156 effective names is exactly what
this is.

It did not do damage in this run, because the gross cap bound before the
volatility target: the book was sized by leverage, not by the underestimated
risk figure. That is luck, not design. If the target were the binding
constraint, the optimizer would lever up to a predicted 10% and realize closer
to 20%. So the ex-ante number is currently usable as a *relative* measure
(comparing books) and not yet as an *absolute* one (sizing to a promise).

Closing it means more factors, a residual-correlation adjustment, or shrinkage
toward a longer-window covariance. Until then, compare `ann_vol` against
`ex_ante_vol` on every run and treat a persistent gap as the risk model's
problem rather than the book's good fortune.

---

## 5. The terminal

```bash
python -m gig serve
# http://127.0.0.1:8000        UI
# http://127.0.0.1:8000/api/docs   OpenAPI
```

Read-only. There is no order-entry route: a research UI that can send orders is
a production system without any of the controls of one.

### Reading the 3D factor space

Each point is one name on the latest cross-section, positioned by its
**neutralized** factor values, so you are looking at what the alpha sees rather
than at price levels.

| Channel | Meaning |
|---|---|
| x | 12-1 momentum, cross-sectional z-score |
| y | idiosyncratic vol (switchable to live % change or alpha score) |
| z | 21-day reversal, cross-sectional z-score |
| colour | combined alpha score — green long side, red short side |
| size | 21-day average dollar volume |
| ring | name is actually held in the book |
| expanding pulse | that name just printed a new quote |

Drag to orbit, scroll to zoom, hover for the name's factor values and weight.
`spin` toggles auto-rotation. The sector dropdown dims everything else.

Switch the vertical axis to **live % change** to watch the cloud deform as
quotes arrive: momentum and reversal hold the horizontal plane while the tape
moves names vertically.

### Panels

- **Book** — gross, net, longs, shorts, construction label, and both legs sorted by conviction.
- **Paper account** — equity, buying power, open/closed market, and largest drift vs target (read-only).
- **Sector net** — net exposure by sector against the sector-limit budget.
- **Experiment path** — Sharpe trail from stored backtest metrics.
- **Risk limits** — gross, net, and max-name against the limits in `Settings`.
  Breaches are listed as data; nothing is auto-corrected.
- **Ex-ante risk** — predicted vol, systematic/specific split, top MCTR names.
- **Last backtest** — metrics from `results/equity_ls_last.json`.
- **Factor IC** — mean rank IC per factor with the Newey–West t-stat.
- **Research desk** — local Ollama briefing grounded in the live snapshot JSON.
  It answers questions about the book and risk; it does **not** emit BUY/SELL
  or send orders.

### Ollama research desk

```bash
# Install from https://ollama.com then:
ollama pull llama3.2
# optional overrides in .env:
# GIG_OLLAMA_HOST=http://127.0.0.1:11434
# GIG_OLLAMA_MODEL=llama3.2
```

The terminal polls `/api/ollama/status` and posts questions to `/api/ollama/brief`.
If the daemon is offline, the desk shows a hint and the rest of the UI keeps
working.

### Honesty in the status bar

The tape label states its true source. `SNAPSHOT · alpaca-iex` is a vendor
snapshot endpoint. `LAST CLOSE (DELAYED)` means no vendor key is configured and
the strip is showing the newest stored close. Neither is a consolidated
real-time feed, and the UI never claims otherwise.

The `feed` pill shows which vendor is live. The `db` pill shows bar count.

### Tuning

```bash
python -m gig serve --host 0.0.0.0 --port 9000 --limit 400
```

`--limit` is how many names (ranked by ADV) enter the cloud. 220 is smooth on a
laptop; the factor computation is cached for 90 seconds server-side.

---

## 6. Paper trading

```bash
pip install -e ".[broker]"
```

`.env` needs a paper key pair and the paper endpoint:

```
ALPACA_API_KEY=...
ALPACA_SECRET_KEY=...
ALPACA_BASE_URL=https://paper-api.alpaca.markets
```

`AlpacaBroker` refuses to construct if that URL is not a paper endpoint. Paper
and live differ by one word in a string at Alpaca, so the guard is at
construction time — the run fails before a target book is even computed rather
than after orders are already out.

### The four commands

```bash
python -m gig trade plan            # build the target, print the trade list, run the gate
python -m gig trade run --yes       # same, then submit
python -m gig trade status          # held vs intended, per name
python -m gig trade flatten --yes   # close everything
python -m gig trade history         # past runs from the audit trail
python -m gig trade loop --every 900 --yes    # rebalance on a schedule
# Outside RTH (rehearsal / weekend): add --force. DAY orders sit ACCEPTED at
# Alpaca until the next regular session and fill then.
```

`plan` sends nothing. `run` without `--yes` refuses and tells you so. Start with
`plan` every time; the output is identical to what `run` would do.

Outside the regular session, add `--force` (overrides only the market-hours
check). An empty account’s first book build is allowed to exceed the per-run
turnover cap once; after that the 35% cap binds again.

Keep the lake fresh — a bar older than five calendar days blocks the gate:

```bash
python -m gig ingest
# or, faster for the trade sleeve:
python scripts/refresh_liquid.py
```

### What one run does

1. **Target.** `build_target_book` runs the same `scores()` path as the backtest
   over the `--limit` most liquid names, fits the risk model, and solves for
   weights. It is the only place a live target is produced, so a position in the
   account always traces back to a factor value the research code computed.
2. **Reconcile.** Positions and NAV are read from the broker — the account is
   the source of truth, never a local record of intent. Target minus held gives
   the trade list, so a partial fill or a missed session self-corrects on the
   next pass instead of compounding.
3. **Gate.** Every order passes `src/gig/execution/pretrade.py` before the
   broker sees it.
4. **Execute.** Surviving orders go out as marketable limits, sliced by
   notional.
5. **Record.** Intent, gate findings, and every child order are written to
   DuckDB (`trade_runs`, `target_book`, `orders`).

### The pre-trade gate

A check either blocks the whole run or drops one order — never something in
between, because a partly-applied risk policy produces a book matching no
decision anyone made.

| Blocks the run | Trigger |
|---|---|
| `stale_data` | Last bar older than 5 calendar days. Run `gig ingest` |
| `market_closed` | Regular session closed |
| `limit_breach` | Target violates gross, net, name, or sector limits |
| `ex_ante_vol` | Target predicts more than 25% annualized vol |
| `drawdown_halt` | NAV below the high-water mark by more than `GIG_MAX_DRAWDOWN` |
| `turnover_cap` | Plan turns over more than 35% of NAV in one run |
| `nav_floor` | Account too small to build the book |
| `order_count` | More orders than the per-run cap |

| Drops one order | Trigger |
|---|---|
| `not shortable at broker` | No borrow available. Only checked for sells that would open or increase a short |

Limits are re-derived from the target inside the gate rather than read off the
optimizer's own report, because the optimizer is the thing being checked.

`--force` downgrades **only** `market_closed`, for rehearsing outside the
session. Limit breaches and the drawdown halt are deliberately not overridable
from the command line: if a limit is wrong, change it in configuration where the
change is visible and recorded.

### Order handling

Orders are marketable limits, priced through the touch by `limit_offset_bps`,
not market orders. A market order accepts whatever the book offers, and in a
thin name that can be far from the last print; the resulting slippage shows up
as unexplained drag against the backtest. A limit fills in normal conditions and
simply does not fill in abnormal ones, which is the right behaviour for a
strategy whose edge is basis points per name.

Three filters keep the trade list sane:

| Filter | Default | Why |
|---|---|---|
| No-trade band | 0.25% of NAV | Correcting drift inside tolerance costs more in spread than it saves in tracking error |
| Minimum notional | $250 | Odd lots pay a fixed cost for almost no risk reduction |
| Participation cap | 2% of ADV | The square-root impact term is only credible at low participation |

An exit always trades regardless of the band: a name the strategy has dropped is
a position with no thesis.

### What this is not

- Not a scheduled participation algo. Child orders go out back to back rather
  than worked against a volume curve. At paper size that is fine; at real size
  it is not.
- Not colocated or low-latency. Signals are daily and computed from stored
  closes.
- Not a substitute for reconciliation against a broker statement.
- Not validated for live money. The backtest that justifies it runs on the
  current listing tape, which is survivorship-biased, on a convenience price
  feed. Paper first, for long enough to compare realized vol against the
  ex-ante number the model predicted.

---

## 7. Configuration

**Risk, costs, paths** — environment variables with a `GIG_` prefix, or `.env`:

```
GIG_MAX_GROSS_LEVERAGE=2.0
GIG_MAX_NET_EXPOSURE=0.10
GIG_MAX_NAME_WEIGHT=0.05
GIG_MAX_SECTOR_NET=0.10
GIG_MAX_DRAWDOWN=0.12
GIG_SPREAD_BPS=4.0
GIG_COMMISSION_BPS=1.0
GIG_IMPACT_ETA=0.10
GIG_DB_PATH=data/gig.duckdb
```

**Construction and risk model:**

```
GIG_USE_OPTIMIZER=true      # false reverts to equal-weight quantiles
GIG_TARGET_VOL=0.10         # annualized ex-ante, a ceiling reached from below
GIG_RISK_FACTORS=5          # principal components in the risk model
GIG_RISK_LOOKBACK=252       # sessions in the covariance window
GIG_RISK_REFIT_EVERY=21     # sessions between refits
```

The gross and name caps the optimizer respects are the same settings the limits
check reads, so it cannot be configured to build a book the gate would reject.

**Vendor keys** — unprefixed names also work, so a shared `.env` is fine:

```
FRED_API_KEY=...
FINNHUB_API_KEY=...
ALPACA_API_KEY=...
ALPACA_SECRET_KEY=...
EDGAR_USER_AGENT=GIG research you@example.com
```

`.env` is gitignored. Keys never belong in source or in a README.

**Universe** — `configs/universe.yaml`. `source: us_listed` uses the full tape;
the explicit symbol map is the mega-cap sleeve.

---

## 8. Reading the output

| Metric | Read it as |
|---|---|
| `sharpe` | After-cost annualized Sharpe of the dollar-neutral book |
| `ann_vol` | Realized vol. Compare it against `ex_ante_vol`: a large gap means the risk model is misspecified, not that the book got lucky |
| `ex_ante_vol` | What the risk model predicted for the book it built |
| `systematic_share` | Predicted variance from common factors. Near zero is the design; anything else needs explaining before the Sharpe means anything |
| `max_drawdown` | Peak-to-trough on the equity curve |
| `ann_turnover` | One-way turnover per year. Above ~10x, costs and capacity dominate |
| `dsr` | Deflated Sharpe (Bailey–López de Prado) given the number of strategies tried |
| `ic_mean` | Mean daily rank IC. Single equity factors live around 0.01–0.04 |
| `ic_tstat_nw` | Newey–West t on that IC. Below ~2, treat the factor as unproven |

Sanity rules before you believe any run:

- A dollar-neutral equity book with 40% vol is a factor bet, a concentration
  problem, or bad prices. Check `systematic_share` first — that is exactly the
  diagnosis that produced section 4.
- Realized vol far from `ex_ante_vol` means the covariance estimate is stale or
  the window is the wrong length, not that the book outperformed its risk.
- `total_return` in the hundreds of percent on four years of daily data is a
  red flag, not a result.
- The tape is the **current** listing tape. Delisted names are absent, so
  historical performance on this universe is survivorship-biased upward.
- yfinance is a convenience feed. Splits, halts, and stale prints on small caps
  are not cleaned. Swap in a point-in-time vendor through `DataProvider` before
  trusting a level.
- `--source synthetic` has a **planted** premium. A good Sharpe there measures
  the pipeline, not an edge.

---

## 9. Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `terminal MISSING` in doctor | `pip install -e ".[web]"` |
| `broker MISSING` in doctor | `pip install -e ".[broker]"` |
| `Broker not configured` | No `ALPACA_API_KEY` / `ALPACA_SECRET_KEY` in `.env` |
| `not a paper endpoint` | `ALPACA_BASE_URL` points at live. Set it to `https://paper-api.alpaca.markets` |
| `trade plan` blocked on `stale_data` | Lake is behind. `python -m gig ingest` |
| `trade plan` blocked on `market_closed` | Outside the session. Add `--force` to rehearse |
| `trade plan` blocked on `limit_breach` | The target itself is out of policy. Do not force it; investigate the target |
| Every order dropped as not shortable | Paper account has shorting disabled, or the names are hard to borrow |
| `systematic_share` well above zero | The risk model could not be fit and the run fell back to equal weights. Check the lookback window has enough history |
| UI shows "No bars in DuckDB yet" | `python -m gig universe refresh && python -m gig ingest` |
| Tape says `LAST CLOSE (DELAYED)` | No Alpaca/Finnhub key in `.env` |
| `walk-forward GBDT skipped` | Over 800 names. Use `--mega`, or a smaller `--limit`, or accept factors-only |
| Empty book in the terminal | Too few names clear the ADV floor. The service drops the floor and reports `adv_floor: 0` |
| Backtest reads the wrong lake | `Settings` falls back to `data/qalpha.duckdb` when `GIG_DB_PATH` does not exist. Set the path explicitly |
| `cpp Python fallback` | No compiler. Results are identical; only speed differs |

---

## 10. Daily rhythm

```bash
python -m gig ingest                              # top up the lake
python -m gig backtest --source yahoo --no-ml     # refresh metrics
python -m gig serve                               # inspect the cross-section
python -m gig trade plan                          # see today's trade list
python -m gig trade run --yes                     # paper only
```

Read the plan before running it. The gate will stop the obvious failures, but it
cannot tell you that the target looks nothing like yesterday's for a reason
worth understanding.

Everything else — new factors, a different ranker, a real vendor — plugs into
the same interfaces.
