# Admitted quote and print microstructure research

Sprint 6 persists explicit quote and trade-print observations and reuses the
existing microstructure kernels. It does not infer trades from displayed depth,
bars or own orders. There are no provider, brokerage or entitlement connections.
Hashes establish content identity; every source/feed classification remains a
caller assertion. Authored fixtures remain SYNTHETIC.

## Contracts

`microstructure-admit` accepts exactly `source`, `classification`, `feed`,
`instrument_id`, `currency` (USD), and `events`. Each of 1–1,000 events contains
`event_id`, `kind` (QUOTE or TRADE), `event_at`, `acquired_at`, `knowledge_at`,
and exact nonnegative integer `sequence`. QUOTE additionally requires positive
`bid`, `ask` with bid <= ask, and nonnegative `bid_size`, `ask_size`. TRADE requires
positive `price`, `size`. Numerical fields are finite and bounded at 1e12.

Observation <= acquisition <= revision knowledge is mandatory. Aware nanosecond
timestamps are retained; subnanosecond strings are rejected. An event identity
cannot silently change its original clock, sequence, kind or source terms.
Later corrections append a new knowledge-time revision; conflicts at the same
knowledge time reject. Different event identities cannot occupy the same
instrument/event-time/sequence coordinate.

`microstructure-research` accepts exactly:

```json
{
  "source": "authored-sprint6-events",
  "classification": "SYNTHETIC",
  "feed": "AUTHORED_PRINT_AND_QUOTE_FIXTURE",
  "instrument_id": "SYN101",
  "start": "2026-11-02T15:00:00.123456789Z",
  "end": "2026-11-02T15:00:04.123456789Z",
  "as_of": "2026-11-02T15:00:04.123456789Z",
  "max_quote_age_ns": 5000000000,
  "lag_ns": 0,
  "bucket_volume": 2,
  "window_buckets": 2,
  "participation_rate": 0.1,
  "proposed_quantity": 1
}
```

Start <= end <= cutoff is required. Quote age and lag are bounded exact integer
nanoseconds; VPIN buckets/window and participation are explicit assumptions.
`microstructure-replay` accepts only `report_id`. The `microstructure-research`
terminal engine supports Markets, Statistical Arbitrage, Research and Execution.

## Decision-time diagnostics

Features freeze each event's earliest preserved decision vintage. For a trade,
only quotes already known by that trade's decision and preceding its event-time/
sequence coordinate can match. Quote lag and age are enforced by the existing
as-of matcher. Later sequence quotes at the same clock cannot leak backwards.
The Lee–Ready quote rule falls back to a tick rule using only prior prints known
by that decision. Signs are estimates, not exchange-observed aggressor flags;
missing evidence leaves sign UNKNOWN.

Cont-style best-level OFI measures displayed queue changes using the prior quote
vintage known at each quote decision. The first quote has no inferred previous
queue and returns UNAVAILABLE. It is neither executed order flow nor liquidity
available to this strategy. Kyle's Lambda is OLS of observed sample midpoint
changes on signed print volume over contiguous quote intervals. Unknown interval
signs/knowledge make the estimate UNKNOWN instead of dropping intervals. Too few
or unidentifiable observations return UNAVAILABLE.

VPIN uses equal-volume buckets of estimated event signs, not bulk-volume trade
classification. The kernel separately reports excluded unknown-sign volume and
unfinished final volume. Any excluded unknown volume makes the public estimate
UNKNOWN, even if a partial known-sign kernel series exists. These diagnostics do
not establish toxicity probabilities or causal market impact.

Capacity is only `participation_rate * admitted_print_sample_volume`. Executable
capacity stays UNKNOWN; the source sample is not a complete consolidated tape.
The linear Kyle price-change diagnostic is not an executable cost estimate.
Own-execution cost metrics stay UNAVAILABLE without own execution side and an
arrival benchmark; displayed quote sizes never substitute for those observations.

## Durability and limits

The fixed `microstructure-research/` workspace stores immutable source events,
snapshots and reports with FULL-synchronous SQLite indexes. Retention and reads
are bounded to 1,000 event revisions and 1,000 reports. Replays verify all source
hashes, the publication index and deterministic canonical serialization; tuple
kernel fields and JSON arrays describe the same preserved wire contract.
Future evidence is excluded and missing reads create no workspace. Report
freshness uses the original observation time, with nanosecond comparison.

Tests include independent OFI/Kyle/VPIN arithmetic, future-knowledge and equal-clock
sequence attacks, late prior ticks, unknown signs, revisions across restart,
duplicate/conflicting identities, index tampering, interrupted source publication,
missing data and nanosecond stale boundaries. These are fixture-backed numerical
and implementation checks, not vendor coverage or economic validation.
