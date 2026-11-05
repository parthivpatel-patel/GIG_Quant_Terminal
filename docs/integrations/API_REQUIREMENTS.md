# API requirements register

Reviewed against the providers' own pages on 2026-11-04. Where a page did not
state a price, a quota, or a card requirement, this register says so. It does
not fill the gap.

Nothing in this register is a request to create an account. Outstanding
actions, if any, are only in [USER_ACTION_REQUIRED.md](USER_ACTION_REQUIRED.md).

Status values: INTEGRATED (used by the running terminal), ADAPTER (code exists,
not called by the current workspace), NOT INTEGRATED, CANDIDATE (reachable or
known, terms not accepted).

## Market data

### Alpaca Market Data

| Field | Value |
| --- | --- |
| Website | https://alpaca.markets |
| Documentation | https://docs.alpaca.markets/docs/about-market-data-api (reviewed 2026-11-04) |
| Registration | An Alpaca account. The docs say keys are created in the web UI. |
| Endpoints | `https://data.alpaca.markets/v2/...` (exact path depends on the asset). Auth headers `APCA-API-KEY-ID` and `APCA-API-SECRET-KEY`. |
| GIG use | Optional delayed/IEX equity bars and quotes. Not used by Market Charts, which serves admitted bars or the labelled synthetic fixture. |
| Credentials | `ALPACA_API_KEY`, `ALPACA_SECRET_KEY`. Also `ALPACA_BASE_URL` (default paper trading URL) and `ALPACA_DATA_URL`. |
| Free tier | Basic is documented as free / "zero cost" for Trading API users. Equities: US stocks and ETFs, real-time coverage IEX only, 30 websocket symbols, historical data "since 2016", historical API 200 calls/min. The historical-limitation column reads "latest 15 minutes". The page's footnote was not in the extracted text, so that phrase is quoted rather than interpreted. Options on Basic: indicative feed, not OPRA. |
| Card required | Not stated on the page reviewed. Do not enter payment information. |
| Paid tier | Algo Trader Plus is listed at $99/month (all US exchanges, OPRA options, higher limits). Not authorized. |
| Licensing | Not quoted beyond the subscription table. Do not redistribute. |
| Freshness | Basic real-time coverage is IEX, not the consolidated tape. |
| Monthly cost at $0 | $0 on Basic, if an account is created later. $99 is out of policy. |
| Zero-cost fallback | The synthetic chart fixture, and any bars a person admits locally with an explicit classification. |
| Priority | Later. Not required for the current terminal. |
| Status | ADAPTER (`gig.data.providers.alpaca_iex`). Broker connectivity stays off. No orders. |
| User action | None now. |

### Finnhub

| Field | Value |
| --- | --- |
| Documentation | https://finnhub.io/docs/api |
| Credentials | `FINNHUB_API_KEY` |
| Pricing and card | Not re-read in this review. Do not subscribe. |
| Status | ADAPTER present, inactive without a key. |
| Zero-cost fallback | Same as Alpaca. |
| User action | None. |

### Yahoo / yfinance

Not an official licensed market-data API. A convenience adapter exists for older research commands. Market Charts does not call it. Do not treat its output as a permitted redistribution feed. No credential. Status: ADAPTER, not used by the new workspace.

## Macroeconomic data

### FRED (St. Louis Fed)

| Field | Value |
| --- | --- |
| Documentation | https://fred.stlouisfed.org/docs/api/api_key.html (reviewed 2026-11-04) |
| Registration | A fredaccount.stlouisfed.org login, then a 32-character key. One key per application. |
| Endpoint example | `https://api.stlouisfed.org/fred/series/search?api_key=...` |
| GIG use | The existing adapter can request series such as DGS10 and CPIAUCSL. The current workspace does not call it. |
| Credentials | `FRED_API_KEY` |
| Free tier | The key page requires a free account. No price is stated. |
| Card required | Not stated. |
| Rate limits | Not stated on the key page. |
| Status | ADAPTER. Not called by this milestone. |
| User action | None until a macro workspace is switched on. |

### U.S. Treasury Fiscal Data

| Field | Value |
| --- | --- |
| Documentation | https://fiscaldata.treasury.gov/api-documentation/ (reviewed 2026-11-04) |
| Registration | None. The docs say the API is open, with no account and no token. |
| Base URL | `https://api.fiscaldata.treasury.gov/services/api/fiscal_service/` |
| Example | `v1/accounting/od/rates_of_exchange` and `v2/accounting/od/avg_interest_rates` (exact dataset versions are on each dataset page). |
| License, quoted | "The data is offered free, without restriction, and available to copy, adapt, redistribute, or otherwise use for non-commercial or commercial purposes." |
| Card | No. |
| Cost | $0. |
| Status | NOT INTEGRATED. This is the preferred zero-cost rates source when that work starts. |
| User action | None. |

### Bureau of Labor Statistics

| Field | Value |
| --- | --- |
| Documentation | https://www.bls.gov/developers/home.htm and https://www.bls.gov/developers/api_signature_v2.htm (reviewed 2026-11-04; v2 page last modified 2020-10-05 on the site) |
| Version 1 | No registration. More limited. Endpoint shape `https://api.bls.gov/publicAPI/v1/timeseries/data/`. |
| Version 2 | Registration key. Up to 50 series in one request. Optional calculations need the key. A window "up to 20 years" is stated for the optional-parameter signature. |
| Daily quota | Not stated on the pages reviewed. |
| Card | Not stated. |
| Status | NOT INTEGRATED. |
| User action | None. |

### Bureau of Economic Analysis

| Field | Value |
| --- | --- |
| Signup | https://apps.bea.gov/API/signup/index.cfm (reviewed 2026-11-04) |
| What the page lists | NIPA, GDP-by-industry, regional, international transactions, input-output, and others. |
| Price and card | Not stated on the page reviewed. |
| Status | NOT INTEGRATED. |
| User action | None. |

## Corporate filings

### SEC EDGAR

| Field | Value |
| --- | --- |
| Documentation | https://www.sec.gov/search-filings/edgar-application-programming-interfaces (reviewed 2026-11-04, page dated 2024-06-06, last reviewed note 2025-04-08) and the developer FAQ at https://www.sec.gov/os/webmaster-faq#developers |
| Authentication | data.sec.gov "do not require any authentication or API keys". |
| Required header | A declared User-Agent with a contact email. Sample shape: `Sample Company Name AdminContact@.com`. |
| Rate limit, quoted | "our current maximum access rate is 10 requests per second." |
| Endpoints | `https://data.sec.gov/submissions/CIK##########.json`, `https://data.sec.gov/api/xbrl/companyfacts/CIK##########.json`, `https://data.sec.gov/api/xbrl/companyconcept/...` |
| Freshness | Submissions update as filings are disseminated, "typical processing delay of less than a second"; XBRL "under a minute", longer at peak. |
| Credentials | `EDGAR_USER_AGENT` (a real contact email, not a secret). The default in code is a placeholder and must be replaced before any EDGAR request. |
| Card | No. |
| Cost | $0. |
| Status | ADAPTER (`gig.data.providers.edgar`). Not called by the current workspace. |
| User action | None until an EDGAR workflow is turned on. Then the action is an email in `EDGAR_USER_AGENT`, not an account. |

## Already integrated, no key

| Provider | Status | Basis |
| --- | --- | --- |
| USGS earthquakes | INTEGRATED, labelled DELAYED | Fixed HTTPS GeoJSON. No key. Terms recorded in `docs/global-intelligence.md`. |
| NASA EONET | INTEGRATED, labelled DELAYED | Fixed HTTPS. No key. Same document. |
| ECB reference rates | INTEGRATED for the earlier observed-data research | No key. Terms recorded in `docs/observed-data.md`. Not re-fetched in this pass. |

## Global Intelligence candidates, not activated

| Provider | What was verified | Status |
| --- | --- | --- |
| NWS api.weather.gov | Docs reviewed 2026-11-04 (page updated 2026-03-24). "open data, free to use for any purpose." No fees. A User-Agent is required. The rate limit "is not public information". Alerts example: `https://api.weather.gov/alerts/active?area={state}`. Station observations keep about 7 days. | NOT INTEGRATED. No key. A contact User-Agent will be required when this is built. |
| GDACS | Previously reachable. Reuse terms not reviewed. | CANDIDATE. Not ingested. |
| NASA FIRMS | Needs a free MAP_KEY. None is configured. | BLOCKED. Not requested. |
| AIS, OpenSky, Sentinel/Landsat | Blockers already shown in the terminal: keys, terms, or registration, and no positions are simulated. | BLOCKED. Not requested. |

## Explicitly not authorized

Paid Alpaca Algo Trader Plus ($99/month), any OPRA feed, any vendor that requires a card, and any provider whose current terms were not read. No brokerage orders, paper or live.
