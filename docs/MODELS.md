# Models that belong in this stack (and ones that do not)

Hiring managers at multi-strats do not want “we added LSTM + FinBERT + GPT.” They want **one model class, walk-forward, no leakage, stored with a train-end date**.

## What we implemented

| Model | Role | Leakage rule |
|---|---|---|
| 12-1 momentum, ST reversal, ivol | Baseline cross-sectional factors | Prices through close t only |
| HistGradientBoosting (sklearn) | Tabular ranker on those factors | **OOS folds only**; train window scores are NaN |
| Headline lexicon (+ optional VADER / FinBERT) | NLP factor | Headline timestamp ≤ date t |
| IC-weighted combination | Blend | Weights lagged one day |

Run live:

```bash
python -m gig ingest
python -m gig backtest --source yahoo
```

Tests and CI still use `--source synthetic` so they never need the network.

## What to add next if you want depth (not more logos)

These are the additions that actually show up in QR/QD interviews:

1. **Fundamentals (value/quality)** — point-in-time book equity, earnings, accruals. Needs Compustat/Norgate, not yfinance `.info`.
2. **Filings NLP** — 8-K/10-K from SEC EDGAR, parsed at *file datetime*, not “the document mentions AAPL.” Same as-of join as news.
3. **LightGBM / CatBoost ranker** (`lambdarank`) — same walk-forward wrapper, swap the estimator. Standard on equity desks.
4. **Earnings surprise** — SUE vs a stale consensus snapshot (must be PIT).
5. **Short interest / borrow** — if you can license it. Capacity and crowding, not a price LSTM.
6. **Options IV surface** — we already price BSM; a VRP factor from a live chain is the real next step.

## What not to add (it reads as retail)

- LSTM/Transformer on raw OHLCV as “the alpha”
- An LLM that outputs BUY/SELL
- Combining 12 neural nets with hand-waved weights
- Social-media scrapers without timestamps and a bot filter
- Retraining on the full sample and reporting that Sharpe

FinBERT is available as `transformer_sentiment()` if you `pip install gig[transformers]`. It is optional. The lexicon path is what CI runs, because downloading a 400MB model is not research discipline.

## How to talk about this in an interview

“I built a CS long/short. Features are neutralized momentum/reversal/ivol. I added a GBDT whose predictions are walk-forward only, embargoed, and I treat news as a dated feature joined as-of. Yahoo is a convenience tape; I would swap the provider for CRSP.”
