# Transformation plan

Audit date: 2026-10-08. This document describes the repository as it runs, not as the root-level engine filenames imply.

Readiness, stated without a production-ready claim:

| Gate | Status |
|---|---|
| Research capable | Partial. The maintained `src/gig` path can fit factors, walk-forward rankers, HMM, GARCH(1,1), and Kalman on data it actually has. Promotion to a book is not automatic. |
| Paper ready | No. `gig trade run --yes` exists and refuses a non-paper URL unless `allow_live=True` is passed in code. Alpaca connectivity was not tested in this change because no paper credentials are present. |
| Operationally ready | No. One process, one DuckDB file, no replay log, no paging rotation beyond what the host provides. |
| Live-trading ready | No. Live trading stays disabled. It needs separate credentials, an explicit code flag, and a human authorization that this repository does not grant. |

## 1. Current architecture

Two trees share the repository and do not share a runtime.

**Maintained package (`src/gig`).** Installed as `gig`. Entry point: `python -m gig`. DuckDB lake, factor long/short research, closed-form factor-neutral optimizer, historical VaR/ES, optional paper broker, FastAPI terminal on port 8000. Optional C++ extension `gig._speed` (`src/cpp/speed.cpp`) neutralizes a cross-section; Python is the fallback when the extension is missing.

**Prototype (`*_engine.py`, `live_data_server.py`).** A large Flask-style server and a set of engines that import each other. `src/gig` and `tests/` do not import them. They are not the process `gig serve` starts. They were left in place because deleting them without a dependency audit would drop code that `live_data_server.py` still calls.

```
laptop browser
    → FastAPI  src/gig/service/app.py   :8000
        → snapshot / factors / risk / optimizer   (same code as the backtest)
        → /api/models                         code registry
        → /api/models/market                  HMM + GARCH fit on stored bars
        → DuckDB bars
    → gig trade …                             separate CLI, Alpaca paper if configured

live_data_server.py :5001                   not started by gig serve
    → math_engine, execution_engine, …        not on the book path
```

## 2. Current runtime architecture

`gig serve` builds a FastAPI app, warms a snapshot cache, and runs uvicorn. Optional in-process paper trading exists on the CLI flags of `serve` and is off by default. The terminal has no order-entry route.

Research models added in this change (`src/gig/models/`) are imported by `/api/models/market` and by tests. They are not imported by `strategies/equity_ls.py`, the optimizer, or the order path.

## 3. Current data flow

```
provider (synthetic | yahoo | optional FRED, Finnhub, EDGAR, Alpaca IEX)
    → schema checks in the provider
    → DuckDB bars / universe / fundamentals
    → load_liquid_panel (most-liquid names, window ending at the last stored bar)
    → factor panel (prices through t)
    → snapshot for the UI
```

The lake does not store a separate ingestion timestamp distinct from the bar date. Adjustment status is whatever the provider wrote. There is no futures, FX, or listed-option chain in the maintained lake. Empty or short history returns `DATA UNAVAILABLE` from the new model routes. It does not synthesize a regime.

## 4. Current signal flow

```
close panel
    → 12-1 momentum, 21-day reversal, 63-day idiosyncratic vol
    → cross-sectional neutralization (C++ or Python)
    → lagged rank-IC combination          (today's IC does not weight today's book)
    → optional walk-forward GBDT scores   (OOS folds only; LightGBM if the ml extra is installed)
    → factor-neutral optimizer            (the book)
    → pre-trade gate                      (only if someone runs `gig trade`)
    → marketable limit slices            (only after `--yes`)
```

`/api/signals` maps the current book weights into a signal contract (`BUY` / `SELL` / `HOLD` / `NO_TRADE`). `confidence_kind` is `relative_score`. It is not a probability.

HMM, GARCH, Kalman, Black-Litterman, fractional Kelly, and the Almgren-Chriss schedule do not enter this chain.

## 5. Existing model inventory

See [MODEL_INVENTORY.md](MODEL_INVENTORY.md). Short version: the book is a lagged-IC factor model plus an optional walk-forward ranker and a factor-risk optimizer. Historical VaR/ES is computed and displayed. The new statistical models are research fits.

## 6. Broken modules

- `GET /api/openapi.json` returned HTTP 500 because a route annotated `Request` while FastAPI was imported inside `create_app`, so the schema could not resolve the forward reference. The route no longer takes `Request` or `Body`. The schema builds.
- CI `ruff check src tests` was already failing on the tree this branch started from (import order, a few SIM/UP rules, an unused pandas import). Those findings are pre-existing. This change does not lower the ruff configuration.
- `pytest --cov=gig` is gated at 70% in `pyproject.toml`. The last full run before this branch was 144 passed and about 64% coverage, so the coverage job fails on `main` as well. The threshold was not lowered.
- `Dockerfile` previously ran `qalpha doctor`, a command this package does not install. The image command is now `python -m gig doctor`.
- `docker-compose.yml` and `start.sh` still point at `live_data_server.py` and ports 5001/3000. They do not start `gig serve`. Treat them as prototype launchers.

## 7. Duplicated modules

| Concern | Maintained | Prototype (not on the book path) |
|---|---|---|
| HMM | `src/gig/models/hmm.py` (EM updates start, transitions, means, stds) | `math_engine.HiddenMarkovRegime` (hardcoded bear/neutral/bull means; M-step never updates the transition matrix) |
| Volatility | `src/gig/models/garch.py` MLE plus EWMA | `VolatilityAnalysis.garch_proxy_vol` with fixed α, β |
| Kalman | `src/gig/models/kalman.py` | `math_engine.KalmanFilter` |
| Kelly / Black-Litterman | `src/gig/portfolio/kelly.py`, `black_litterman.py` | `portfolio_optimizer.py` |
| Almgren-Chriss | `src/gig/execution/schedule.py` (does not submit) | `execution_engine.py`, `research_agent.py` |
| Sentiment | lexicon, optional FinBERT function | `nlp_engine.FinBERTAnalyzer` falls back to a lexicon |
| Execution | `src/gig/execution/` | `execution_engine.py` |

The maintained copies are the ones tests import. The prototype copies stay until a later pass confirms nothing operational still shells out to `live_data_server.py`.

## 8. Legacy modules

Everything matched by `*_engine.py` at the repository root, plus `live_data_server.py`, `start.sh` as a launcher for that server, and the compose file's 5001/3000 wiring. `deep_learning_engine.py` contains a BiLSTM that the maintained runtime never loads.

## 9. Missing modules

Not implemented in `src/gig`, and not fabricated in the UI:

- EGARCH, GJR-GARCH, Student-t GARCH.
- BiLSTM training or inference on the book path.
- FinBERT weights loaded by default (the function returns `None` when `transformers` is absent).
- Cointegration pair book, PCA residual stat-arb, half-life trading rules.
- Tick-native OFI, Lee-Ready, Kyle's lambda, VPIN. Bar-level proxies in the prototype are not labeled as tick analytics and are not wired here.
- Listed option chain, IV surface, gamma exposure.
- Event NLP, transcript intelligence, RAG, research grids.
- Autonomous research agent with a sandbox and a promotion queue.
- Kafka, Kubernetes, a separate GPU training service.
- WebGL regime orb, model network, IV surface, risk landscape, correlation graph, order-book landscape. The existing canvas is a projection of the factor cloud, driven by the snapshot.

## 10. Security concerns

- No broker keys are in source. `.env.example` lists optional names. This environment has none of them set.
- Live and paper credentials are not separated by a second secret store. Separation today is the paper URL check inside `AlpacaBroker`.
- The terminal API has no authentication. It is a localhost research UI. Do not expose port 8000 on a public interface without a front door.
- `POST /api/ollama/brief` forwards a question to a local Ollama process. It does not place orders. Untrusted text must not be treated as a tool command; the desk does not have a shell tool.
- Root engines that scrape or call brokers are outside this process. Leaving them on disk is not the same as running them.

## 11. Data-quality concerns

- Yahoo history is a current-listing tape. Delisted names are absent. Historical Sharpe on that universe is survivorship-biased. The UI says so on synthetic versus Yahoo research notes.
- No corporate-action adjustment ledger beyond what the provider returns.
- No point-in-time index membership.
- Fundamentals, where present, use filing dates. They are not in the flagship book.
- The HMM/GARCH diagnostic uses an equal-weight mean of up to 40 liquid stored names. That is a market proxy, not a tradable index, and the fit is in-sample.

## 12. Research-validity concerns

- One backtest does not promote a model. The registry raises if any row is `CHAMPION`.
- GARCH versus EWMA uses QLIKE on the last 20% of the same sample that was fit. The payload says this is not a purged backtest, and the promotion string is `none`.
- HMM states are unlabeled during EM. `mean_rank_*` is assigned after the fit. Bull/bear is not a fitted label.
- IC combination is lagged. That property is tested in the existing factor tests and was not changed.
- Deflated Sharpe exists (`src/gig/research/multiple_testing.py`) for the backtest report. It is not applied to the new research fits.
- The README construction table (Sharpe 1.18 on 300 names) is a prior experiment, not a result of this change, and it is not a live claim.

## 13. Risk-model concerns

- Portfolio variance is `w'Σw` with `Σ = BB' + D`. The README records that ex-ante vol understated realized vol on one 300-name run (4.2% versus 9.0%). Specific risk is diagonal. Do not size to the ex-ante number as if it were calibrated.
- VaR and expected shortfall in `src/gig/risk/var.py` are historical, not GARCH-filtered.
- `exposure_scale` is a logistic suggestion. `applied_to_orders` is false. It cannot raise a limit.

## 14. Execution concerns

- Paper orders are marketable limits sliced by notional (`src/gig/execution/algo.py`). The module docstring states this is not a participation schedule.
- `almgren_chriss_schedule` returns a trajectory and a TWAP comparison. It does not submit.
- No fill was sent to Alpaca in this change. Idempotency, restart reconciliation, and the kill switch exist in the maintained execution package and were not re-verified against a live paper account.
- Market-hours can be downgraded with `--force`. Limit breaches and the drawdown halt cannot.

## 15. UI concerns

The terminal is a dark, dense research screen: factor cloud, book, risk bars, IC, attribution, blotter, research desk. This change adds a model-council panel that lists registry rows and, when the lake has enough bars, the causal HMM filtered probabilities plus GARCH and EWMA vols.

What it is not: a multi-workspace terminal, a light theme, saved layouts, a command palette, or a set of WebGL scenes. Empty modules were not added to the navigation.

Numbers in the council come from `/api/models` and `/api/models/market`. If the fit cannot run, the panel says `DATA UNAVAILABLE`.

## 16. Performance concerns

- Neutralization is the only kernel moved to C++, and only because it is a tight cross-section loop. No new hot path was profiled in this change, so nothing else was rewritten in C++.
- `/api/models/market` refits HMM and GARCH on each request (up to 40 names, 800 days). That is acceptable for a manual refresh and too heavy to poll. The UI loads it at boot and on Refresh, not on the tape interval.
- No latency budget was measured against Alpaca. The broker round trip dominates anything this process can shave. This repository does not claim HFT.

## 17. Recommended architecture

Keep one process for the laptop and for a small always-on VM:

```
ingest (scheduled, bounded) → DuckDB
inference of approved factors → signal contract → risk gate → paper OMS
research jobs (HMM/GARCH sweeps, ranker retrains, later NLP) in a separate process with a timeout
```

Do not add Kafka or Kubernetes for a single VM. Add an append-only event log (a table, or a log file) before any second service, so a restart can replay intents. Split research onto another machine only when a profile shows the paper loop missing its schedule. GPU training stays optional and remote; CPU inference of a small GBDT is the path that fits the free VM.

## 18. Implementation priorities

1. Done in this change: honest registry, causal HMM, GARCH(1,1) versus EWMA, Kalman, Kelly, Black-Litterman, exposure scale, Almgren-Chriss planner, signal contract, terminal routes, OpenAPI fix, Docker command fix.
2. Next: point the compose file and `start.sh` at `gig serve`, or delete them once nothing calls port 5001.
3. Persist model fits (parameters, as-of, git commit) instead of refitting inside the request.
4. Out-of-sample test of HMM and GARCH against a static baseline before either may affect a weight.
5. Paper-account rehearsal with real credentials: plan, one tiny order, reconcile, flatten. Not done here.
6. Coverage and ruff so CI on `main` is green. Do not drop the 70% gate to hide the gap.
7. Only after (4) and (5): consider wiring a validated vol forecast into the risk target. Still no champion jump.

## 19. Acceptance criteria

- `model_inventory()` contains no `CHAMPION`.
- A short return sample does not produce an HMM.
- An empty lake yields `DATA UNAVAILABLE` from `/api/models/market` and `/api/signals`.
- Signal confidence is labeled `relative_score`.
- GARCH persistence is strictly below 1, and the likelihood beats a constant-variance model on a heteroskedastic sample (unit test).
- Kalman beta tracks a drifting slope on a synthetic pair (unit test).
- Almgren-Chriss trades sum to the parent quantity and are not submitted.
- Exposure scale is monotone in drawdown and is not applied to orders.
- The UI council renders registry text from the API, or `DATA UNAVAILABLE`.

## 20. Verification commands

```bash
.venv/bin/python -m gig doctor
.venv/bin/python -m pytest tests/test_research_models.py tests/test_service.py -q --tb=short
.venv/bin/python -c "from gig.service.app import create_app; create_app().openapi()"
.venv/bin/ruff check src tests
.venv/bin/python -m pytest --cov=gig --cov-report=term-missing
.venv/bin/python -m gig backtest --source synthetic
.venv/bin/python -m gig serve --host 127.0.0.1 --port 8000
```

`gig trade run` is intentionally absent. Do not run it without paper keys and a human decision to send orders.

## Files in this change

Created: `src/gig/models/` (HMM, GARCH, Kalman, diagnostics), `src/gig/governance/registry.py`, `src/gig/signals/contract.py`, `src/gig/portfolio/kelly.py`, `src/gig/portfolio/black_litterman.py`, `src/gig/risk/deleverage.py`, `src/gig/execution/schedule.py`, `tests/test_research_models.py`, this document, `docs/MODEL_INVENTORY.md`, `docs/CLOUD_DEPLOYMENT.md`.

Edited: `src/gig/service/app.py`, `src/gig/service/static/index.html`, `src/gig/service/static/app.js`, `src/gig/service/static/styles.css`, `tests/test_service.py`, `Dockerfile`, `README.md`, `docs/MODELS.md`.

Removed: none. Prototype engines were not deleted.

## Diagrams

### Data flow

```
Yahoo | synthetic | optional vendors
        → DuckDB
        → liquid panel (ends at last bar)
        → factors → lagged IC → optimizer → snapshot
        → equal-weight returns → HMM / GARCH diagnostic (display only)
```

### Model pipeline

```
registry (code, status ladder, champion forbidden)
    → research fit on stored returns
    → comparison vs EWMA
    → UI
weights ← factor model and optional ranker only
```

### Execution

```
gig trade plan → pre-trade gate → (human --yes) → Alpaca paper
schedule.almgren_chriss_schedule is not on this arrow
```

### UI

```
index.html panels
    app.js fetch /api/cloud /api/models /api/models/market …
    canvas factor cloud (projection, not WebGL)
```

### C++ boundary

`src/cpp/speed.cpp` exports neutralization. Research math in this change is Python (NumPy, SciPy L-BFGS-B). No pybind11 wrapper was added for HMM or GARCH because they are request-scoped research fits, not a profiled bottleneck.

## Backtest and risk limitations

The synthetic backtest checks the pipeline. The planted premium is not an edge. Yahoo backtests omit delistings. Risk is not calibrated to a VaR exceedance history. New models have unit tests and no strategy backtest.

## Security and operational limitations

Unauthenticated API, single DuckDB file, no tested restore drill, no Alpaca verification, prototype servers still in the tree.

## Next highest-value improvement

An out-of-sample, costed comparison of GARCH versus EWMA for position scaling, stored as an experiment row, with the scale still not applied until that row beats the baseline. That is smaller and more honest than a new visual workspace.
