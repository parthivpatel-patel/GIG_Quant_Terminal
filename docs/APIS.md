# Free data APIs

Yahoo already works with no key. Add these for macro, filings, and live quotes.
None of them require a credit card for the personal/paper tiers.

## 1. FRED (macro)

1. Open https://fred.stlouisfed.org/docs/api/api_key.html  
2. Create a free account and copy the key  
3. In `.env`:

```
FRED_API_KEY=your_key_here
```

Used for VIX, the 10y–2y curve, HY spreads, unemployment, CPI.

## 2. Finnhub (quotes + earnings calendar)

1. Open https://finnhub.io/register  
2. Copy the API token  
3. In `.env`:

```
FINNHUB_API_KEY=your_token_here
```

Free personal use. Live quotes for a small list; earnings dates.
`/news` and `/calendar/earnings` return HTTP 401 if the token is truncated or the
plan does not include those routes. Yahoo already covers headlines; calendar stays 0.

## 3. Alpaca IEX (fastest free US quotes)

1. Open https://alpaca.markets and create a **paper** account  
2. In `.env`:

```
ALPACA_API_KEY=PKxxxxxxxx
ALPACA_SECRET_KEY=your_secret
ALPACA_BASE_URL=https://paper-api.alpaca.markets
ALPACA_DATA_URL=https://data.alpaca.markets
```

This is **IEX only**, not the full NYSE+Nasdaq tape. Still the best free live feed.

## 4. SEC EDGAR (8-K / 10-K)

No key. The SEC requires a User-Agent with **your email**:

```
EDGAR_USER_AGENT=GIG research your.email@gmail.com
```

Without an email, SEC returns 403.

## Apply keys

Put them in `C:\Quant Algo\.env` (copy from `.env.example` if needed).

Then:

```powershell
.\.venv\Scripts\Activate.ps1
python -m gig doctor
python -m gig db init
python -m gig ingest
python -m gig quotes
python -m gig backtest --source yahoo
```

`ingest` still works if some keys are missing — those feeds are skipped.

## What “live” means

| Source | Speed |
|---|---|
| Yahoo bars | Daily, delayed |
| Alpaca IEX / Finnhub quotes | Seconds, IEX or vendor quote |
| FRED | Daily / monthly macro |
| EDGAR | Minutes after a filing hits SEC |
