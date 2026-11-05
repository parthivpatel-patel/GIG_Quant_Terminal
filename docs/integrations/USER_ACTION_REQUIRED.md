# Actions required from you

Reviewed 2026-11-04.

**None.** The terminal, the synthetic and admitted chart paths, USGS, NASA
EONET, and the research cycle all run without a new account, key, or payment
method.

Do not create provider accounts, and do not enter a credit card, until a later
milestone names one credential here. The full inventory, including providers
that are deliberately not requested yet, is in
[API_REQUIREMENTS.md](API_REQUIREMENTS.md).

When a later milestone does need something, it will name the provider, the
official signup URL, the environment variable, and whether the integration is
blocked without it. Expected later, and not requested now:

- `EDGAR_USER_AGENT` — a contact email, before any EDGAR request. No account.
- `FRED_API_KEY` — a free FRED account, before any FRED request.
- `ALPACA_API_KEY` and `ALPACA_SECRET_KEY` — only if you later approve an IEX
  data account. Not required for charts. Not an approval to send orders.
