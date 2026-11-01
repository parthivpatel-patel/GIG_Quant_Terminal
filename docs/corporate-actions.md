# Point-in-time corporate-action research

`gig.data.actions` introduces explicit caller-supplied research contracts. These
definitions are not reconstructed historical 219-section authority. No routine
contacts a provider, infers exchange identity, places an order, authenticates
source claims, or establishes complete corporate-action coverage.

## Action admission and durable evidence

`admit_actions(records)` returns immutable `ActionSnapshot` canonical bytes and
SHA-256. Every revision requires `action_id`, `instrument_id`, `source`, `kind`,
`effective_at`, and `available_at`. Timestamps must be aware and retain up to
nanosecond precision. An announcement may precede effectiveness; a correction
may become knowable later. Instrument and kind cannot change under one
source/action identity. Corrected numerical terms require a later knowledge
timestamp. Unknown fields and ambiguous action kinds fail admission.

Supported kinds are:

| Kind | Additional required evidence |
| --- | --- |
| `SPLIT` | Positive integer `ratio_new`, `ratio_old` |
| `CASH_DIVIDEND` | Exact decimal-string `cash_per_share`, `currency`, `reference_timestamp`, `reference_close` |
| `SPLIT_AND_DIVIDEND` | Split terms, exact `cash_per_pre_action_share`, and the same currency/reference fields |

A combined event's cash is explicitly per pre-action share. Separate events at
the same instrument/effective timestamp are rejected because their ordering and
units are otherwise ambiguous. Unsupported spinoffs, mergers, return-of-capital,
tax adjustments, FX conversions, and rights distributions are not substituted
with an unrelated split or cash event.

Decimal strings preserve their exact source spelling, including decimal zeros
and scientific notation. Floats, nonfinite values, negative cash, unsupported
precision, and locale-style/underscore/whitespace spellings are rejected.

`ActionStore(path).ingest(snapshot)` commits a batch atomically. Exact replay
is counted without duplication; conflicts roll back the entire batch. Each
row's canonical bytes and hash are checked on read. `snapshot(as_of, source=...)`
selects the latest knowable revision, then includes only events already
effective. Reopening the SQLite store preserves those semantics. Store creation
is explicit; callers wishing to avoid a read-side creation must first check the
path exists, as with the existing local store interfaces.

## Raw-bar adjustment

```python
adjust_bars(rows, actions, as_of=aware_timestamp,
            mode="SPLIT_ADJUSTED", source="caller-price-feed",
            classification="HISTORICAL", currency="USD")
```

Raw OHLCV rows use the existing `admit_bars` schema. Only closing observations
and revisions knowable at the cutoff participate. The latest knowable revision
of each instrument/observation is selected without overwriting original source
evidence. The output preserves original OHLC prices, integer-volume strings,
observation times, and revision knowledge times.

For a new/old split ratio `r`, bars strictly before effectiveness have price
factor `1/r` and volume factor `r`. For cash `d` per pre-action share and the
actual first visible post-effective close `q`, total-return adjustment applies
`q/(r*q+d)` (`r=1` for cash-only events). Consequently the adjusted return across
the event is exactly `(r*q+d)/p - 1`, where `p` is the prior source close.
Sequential factors multiply. Cash dividends never change volume. Split-only
mode explicitly excludes cash factors; it does not claim total-return adjustment.

This is an explicitly introduced ex-event wealth-reinvestment convention, not
an asserted vendor adjustment convention. The common backward factor `(p-d)/p`
only matches this wealth return under an exact mechanical price drop; it is not
used here. A missing knowable post-effective close prevents the calculation.
If another action lies between the reference and post-event observations, the
interval is rejected; provide an explicit combined event or an intervening source
closing observation rather than guessing share/cash ordering.

The reference must exactly match the latest visible original closing bar before
effectiveness. Missing or mismatched reference evidence, mixed action providers
for one instrument, currency disagreement, cash at least equal to the reference
close, and unknown instrument identities fail rather than assuming zero cash or
an estimated reference. The routine does not infer that adjacent rows represent
adjacent exchange sessions; the supplied reference assertion remains caller
evidence and should be joined to the explicit calendar contract when available.

All split factors, adjusted volume, and adjusted close are emitted as exact
rationals with numerator/denominator integer strings. Other adjusted OHLC fields
are finite positive binary64 estimates; the exact factor and unchanged original
values permit independent reconstruction. Fractional adjusted volume is not
silently rounded, including original volumes beyond binary64 exactness.

The selected source snapshot, selected action revisions, configuration, and
transformation have deterministic hashes. Future-effective or unknown events
and future bar corrections cannot alter an earlier transformation identity.
Timezone-equivalent inputs canonicalize to the same UTC nanosecond identity.
No visible bars returns explicit `UNAVAILABLE`.

Every result row adds `adjusted_available_at`, the latest knowledge/effectiveness
time of all evidence it uses, including the actual ex-event closing observation
and any later-known revision. The original `timestamp` and `available_at` remain
untouched raw evidence. `uses_later_observations` and
`derived_available_after_original_timestamp` identify retrospective adjustment.
Features derived from these results must use `adjusted_available_at` as their
feature-availability time; backdating them to the original observation would
create look-ahead. The purged research workflow rejects that mismatch in a
regression test. A cutoff-scoped reconstruction is not a claim that its adjusted
history was knowable at each historical row time.

## Declared bounds and coverage

One adjustment accepts at most 500 raw bar revisions and 128 action revisions.
Split integer terms are at most one billion. Monetary strings are bounded to
100 decimal digits and exponent magnitude 100. Each emitted rational term is
bounded to 128 digits, and the serialized result is bounded to 900,000 UTF-8
bytes before returning. Precision or representability overflow rejects the
calculation. There is no truncation or silent clipping. The maximum-row/action
test crosses the actual `Contract.admit` wire boundary, including its node,
depth, scalar, and byte constraints.

An action store is a bounded research evidence archive, not an unbounded vendor
event lake. A selected snapshot over 128 knowable revisions fails explicitly;
use a narrower source/time archive. `coverage` always states that only
caller-declared events were applied and independent completeness is unknown.
An empty selected event set is a calculation over no declared effective events,
not proof that no corporate event occurred in the market.

## Verification

Tests verify independent two-for-one and three-for-two split arithmetic, exact
large/fractional volume, cash and combined-event total-return references,
sequential composition, point-in-time corrections, future independence,
nanosecond/timezone identity, immutable admission, missing reference rejection,
store restart/replay, conflicting-batch rollback, and tamper detection. No
economic, broker, provider, or institutional validation is implied.
