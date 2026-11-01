# Provider ingestion and knowledge-time replay

`gig.data.ingestion.ProviderWorkspace` extends the existing durable workspace;
it does not connect to a brokerage account. Status and missing-data reads do not
create state. The HTTP service exposes its contract envelopes and the terminal
renders capability, ingestion, rate-observation and universe views.

## Sources and classification

* `alpaca_iex` is **fixture-only** in this build. There is no transport that reads
  credentials, environment keys or account configuration. Any other mode returns
  UNAVAILABLE. IEX is explicitly distinct from consolidated SIP. Every admitted
  fixture is SYNTHETIC, even if its values resemble historical prices. Implemented
  request construction does not mean a provider connection was verified.
* `us_treasury` with explicit `PUBLIC_OBSERVED` mode fetches the fixed official
  Treasury daily par-yield XML endpoint. Its payload is HISTORICAL public source
  evidence acquired now. A direct `ingest_treasury_bytes` call defaults to
  SYNTHETIC fixture evidence. Rates have observation dates and percent-per-year
  units; they are not equity bars. Historical publication time remains UNKNOWN.
* The official [Treasury yield-curve dataset catalog](https://catalog.data.gov/dataset/interest-rate-statistics-daily-treasury-yield-curve-rates)
  declares public access and CC0-1.0. Source URL, catalog license source, original
  XML, body digest and acquisition metadata are retained. This does not extend to
  third-party content or marks. Treasury's general notices are at
  <https://home.treasury.gov/subfooter/site-policies-and-notices>.
  Alpaca data licensing/entitlements remain unverified and require approval before
  any external acquisition; fixtures confer no redistribution rights.

## Contracts

Equity fixture admission requires `provider`, `mode=OFFLINE_FIXTURE`, `job_id`,
explicit `instruments`, `start`, `end`, `timeframe`, `pages` and `now`.
Each instrument supplies `instrument_id`, uppercase provider `symbol`, `calendar`
(`XNYS`) and `currency` (`USD`). Every page supplies parsed JSON `body` and an aware
`observed_at`, with optional HTTP `status` and bounded `retry_after_seconds`.
The regression fixture in `tests/test_provider_ingestion.py` provides an executable
example. Timeframes are `1Min` and `1Day`; adjustments are RAW.

Minute timestamps describe bar starts and normalize to minute closes. Daily
timestamps are New York session labels and normalize to the installed exchange
calendar's actual close, including early closes. Observation availability is the
page acquisition timestamp, preserving nanoseconds; it cannot precede a close.
Unknown/duplicate symbols, duplicate bars, invalid OHLCV, naive timestamps and
future observations fail admission. Integer volume remains exact across JSON.

Treasury HTTP admission takes only
`{"provider":"us_treasury","mode":"PUBLIC_OBSERVED","year":2025}`.
Clients cannot supply a URL, a historical acquisition timestamp or credentials.
The fixed `/resource-center/data-chart-center/interest-rates/pages/xml` path is
specified by the [official XML feed documentation](https://home.treasury.gov/treasury-daily-interest-rate-xml-feed).
The superficially similar `/resource-center-data-chart-center/` path is invalid.
The adapter refuses redirects, uses TLS with normal trust, bounds responses to
4 MiB and uses a 20-second timeout. XML DTD/entity declarations, duplicate fields,
nonfinite values, observations after acquisition and malformed content fail.
Dates remain DATE_ONLY; midnight is never invented as an observation timestamp.
The supported tenor list is explicit in capabilities and includes BC_1_5MONTH
(six weeks); display-only aliases are not additional yield observations.

`rates(as_of, max_age_seconds=..., limit=...)` requires an explicit freshness
policy to return AVAILABLE. Missing policy is UNKNOWN; expired evidence is STALE.
Provenance `as_of` remains the actual acquisition time while `query_as_of` records
the requested knowledge cutoff. No observation becomes known before acquisition.
Queries preserve revisions, verify original XML and metadata digests, and expose
`rows_total`/`rows_complete`. Mixed real and synthetic evidence fails closed as
UNKNOWN instead of blending sources. Unanchored legacy observations require
reingestion. Missing data remains UNAVAILABLE and never becomes a zero yield.

Universe admission takes `universe_id`, declared `source`, aware `effective_at`,
aware `available_at` and a complete explicit `members` list. Snapshot reads require
both `universe_id` and `as_of`; only revisions effective and known by the cutoff
are selected. A known empty membership is AVAILABLE with an empty list, distinct
from an unknown universe. This is a declared research universe, not verified
historical index membership or a provider-observed security master.

## Recovery and bounds

SQLite uses full synchronous writes. Equity jobs keep configuration identity,
cursor, attempts, retry deadline and immutable page receipts separately from the
existing point-in-time bar store. A pending receipt is durable before bar
admission; replay after a crash reuses those exact bytes and acquisition time,
without fetching again. Duplicate admission is idempotent. Conflicting job intent,
cursor cycles, receipt corruption and conflicting revisions fail explicitly.
The transport layer applies no more than four attempts, bounded retry deadlines,
ten pages, 1 MiB per equity page, 32 explicit instruments, and a 366-day window.
Retry waiting is UNKNOWN and terminal provider rejection is ERROR, with sanitized
external errors. An empty successful acquisition is UNAVAILABLE.

Treasury receipts and immutable acquisition metadata survive restarts with the
SQLite index. Reads rederive observations from original XML once per acquisition
and check indexed values against that evidence. There is no freshness laundering
through a later query timestamp. Corrupt/tampered evidence fails admission.

The unit suite uses only offline fixtures. A separate actual public acquisition
can establish that Treasury transport worked in one environment at one instant;
it does not verify an Alpaca feed, financial fitness, provider uptime, or strategy
performance. T4 remains T4_NOT_CERTIFIED.
