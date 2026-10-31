# Model policy

The standard this stack holds itself to: **one model class, walk-forward, no
leakage, stored with a train-end date.** Breadth of model families is not a
substitute for any of those four.

## What is implemented

| Model | Role | Leakage rule |
|---|---|---|
| 12-1 momentum, ST reversal, ivol | Baseline cross-sectional factors | Prices through close t only |
| HistGradientBoosting (sklearn) | Tabular ranker on those factors | **OOS folds only**; train window scores are NaN |
| LightGBM `lambdarank` | Same walk-forward wrapper; preferred when `gig[ml]` is installed | **OOS folds only**; relevance bins within each date |
| Headline lexicon (+ optional VADER / FinBERT) | NLP factor | Headline timestamp ≤ date t |
| IC-weighted combination | Blend | Weights lagged one day |
| Ollama (local LLM) | Research-desk briefing over the live snapshot | **Never** generates BUY/SELL or orders |

Run live:

```bash
python -m gig ingest
python -m gig backtest --source yahoo
```

Tests and CI use `--source synthetic` so they never need the network.

Install the optional ranker:

```bash
pip install -e ".[ml]"          # LightGBM lambdarank (auto-selected)
# without lightgbm, walk_forward_scores falls back to sklearn HistGBDT
```

## Additions that would add depth

1. **Fundamentals (value/quality)** — SEC companyfacts book equity + shares
   (`Store.upsert_fundamentals`, `BookToPrice`). Filing date is the as-of.
   Compustat/Norgate remain the gold standard for full history.
2. **Filings NLP** — 8-K/10-K from SEC EDGAR, parsed at *file datetime*, not “the document mentions AAPL.” Same as-of join as news.
3. **CatBoost ranker** — same walk-forward wrapper once LightGBM paper drift looks clean.
4. **Earnings surprise** — SUE against a stale consensus snapshot, which must be point-in-time.
5. **Short interest / borrow** — capacity and crowding, if licensable.
6. **Options IV surface** — BSM pricing already exists; a VRP factor from a live chain is the next step.

## Excluded by policy

- LSTM/Transformer on raw OHLCV as the production signal
- An LLM that outputs BUY/SELL
- Ensembles of many neural nets with hand-set weights
- Social-media scrapers without timestamps and a bot filter
- Retraining on the full sample and reporting that Sharpe

Each is excluded for the same reason: none of them can be given an honest
train-end date and an out-of-sample score on this data.

FinBERT is available as `transformer_sentiment()` under
`pip install gig[transformers]`. It stays optional; the lexicon path is what CI
runs, because a 400MB download is not a research dependency.

## Summary of the design

A cross-sectional long/short whose features are neutralized momentum, reversal,
and idiosyncratic vol; a walk-forward ranker (LightGBM lambdarank when installed,
else sklearn HistGBDT); news treated as a dated feature joined as-of. yfinance is
a convenience tape and the `DataProvider` interface is where a point-in-time
vendor replaces it.
