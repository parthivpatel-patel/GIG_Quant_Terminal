# Funded equity lifecycle and dated cash

The existing isolated funded OMS remains the execution authority. New lifecycle
records never rewrite fills, acknowledgements, orders or raw OMS quantities.
Economic positions include separately disclosed split offsets; option expiry
offsets retain their existing meaning. All support is USD paid-long simulation;
`fully_funded_certified` remains false and external settlement confirmation stays
UNKNOWN. Reference FX namespaces cannot be registered as funded equity shares.

## Configure before trading

`POST /api/v1/workspace/funded-lifecycle` accepts:

```json
{"operation":"configure","now":"2026-11-02T15:00:00Z","settlement_policy":"EXPLICIT_EQUITY_DATES"}
```

Supported policies are IMMEDIATE (legacy default), EXPLICIT_EQUITY_DATES, and
XNYS_T_PLUS_ONE. A dated policy cannot be added to a traded account or changed
after orders exist. Existing immediate-settlement accounts are not reinterpreted.
With a dated policy an equity FILL submitted to `funded-account` must include an
additional top-level `settlement_at` aware timestamp. The immutable per-execution
settlement record is checked on every replay. Options remain in their existing
explicit immediate-premium/cash-expiry scope.

EXPLICIT_EQUITY_DATES admits a caller-declared settlement from trade time through
ten days later. It does not claim legal exchange settlement validation.
XNYS_T_PLUS_ONE uses the existing versioned exchange-calendar rules: trade must
occur during a regular scheduled XNYS session, and settlement equals the next
scheduled session close, including holidays/early closes. This is a stated
simulation convention, not observed bank or broker processing. The optional
exchange-calendars dependency must be installed; unavailable calendars reject.

Gross buys debit trade-date cash immediately and reserve capital. `settled_cash`
adds unsettled buy payables and subtracts unsettled sale receivables from book
cash. Available cash excludes all unsettled sale proceeds. Fees are expensed and
paid immediately. Ownership admission uses the smaller of effective long
inventory and settled trade inventory: pending buys cannot be resold, pending
sales cannot be sold twice, and conditional future buys never fund current sales.
Scheduled settlement changes balances deterministically as the explicit clock
advances. None of these modeled dates establishes external confirmation.

## Corporate-action evidence

An `admit` request includes `now` and an `action` with action_id, revision,
instrument_id, source, kind, known_at, effective_at, record_at and pay_at. SPLIT
adds integer ratio_new/ratio_old; its record/pay clocks equal effective_at.
CASH_DIVIDEND adds exact decimal amount_per_share; effective_at is the ex-time,
and ex-time <= record-time <= pay-time. Data is caller-authored evidence, not
vendor authenticity. Known time cannot exceed admission time. Original revision
is one; subsequent revisions advance both number and knowledge time while
preserving identity, source, dates and kind. Content-addressed immutable source
files and the OMS index detect conflicts and tampering. Every current-state read
checks admitted action bytes and applied lifecycle-event bytes against their
immutable evidence, as well as per-execution settlement evidence.

Apply only by the returned hash:

```json
{"operation":"apply","now":"2026-11-06T15:00:00Z","action_hash":"<64-hex admitted identity>"}
```

Application requires the latest known revision, reached effective/payment time,
and no affected working orders. Unapplied due actions make funding and financial
reconciliation UNKNOWN and block new reservations and dispatch. Future
announcements alone do not block funding. A split preserves total FIFO basis,
adjusts quantities/unit costs exactly and clears the old mark. Whole-lot delivery
and representable exact unit costs are required. Fractional delivery,
cash-in-lieu and nonterminating unit-basis precision reject without mutation.
Pre-effective trades must have settled; late split application or split revision
after intervening trades/another split is unsupported. Trades occurring at or
before an already applied split instant cannot be newly appended ambiguously.

Dividend entitlement uses trades strictly before ex-time that settle by
record-time. Later ex-date buys receive no entitlement. Delayed execution
knowledge or a late split that makes original ownership ambiguous rejects
instead of guessing zero. The first application freezes entitlement; a later
rate revision posts only the cash difference on that frozen quantity. Earlier
cash/NAV snapshots remain immutable. An unfundable debit correction rejects;
the admitted unapplied revision remains a known unresolved obligation.

Dividend receivable accrual between ex-time and explicit payment is not modeled.
Once ex-time is reached, an unapplied dividend makes complete NAV, total P&L and
financial reconciliation UNKNOWN; these values are not filled with zero. Existing
cash can remain available before payment under the conservative funding rules,
but the future dividend never supplies buying power. At payment time the existing
unresolved-obligation guard also blocks new funding until the action is applied.

## Persistence and replay

Lifecycle transition, action-application index, financial snapshot index and
existing risk/kill hooks use one OMS transaction. Faults or process termination
before commit roll back the entire transition. Synced immutable orphan files may
remain unindexed; these are not admitted account states. Duplicate action
applications create no duplicate cash or split adjustment. `state` returns the
current book; `replay` with now and snapshot_id returns the original historical
snapshot rather than approving a current account.

Snapshots expose lifecycle_policy, lifecycle_events, trade_settlements and pending
corporate actions. The account exposes dividend_income, settled_cash,
unsettled_sale_receivables, unsettled_buy_payables, settled_trade_positions and
lifecycle_position_adjustments. The old account.settled_positions field continues
to mean option-expiry offsets, not trade settlement.

Focused tests verify cash and FIFO conservation, entitlement exclusions,
compensating revisions, split inventory versus raw OMS state, unavailable sale
proceeds, unsupported fractions/precision, explicit date/tamper checks, calendar
T+1, exception injection, real process-death recovery and legacy funded behavior.
Unsupported taxation, corporate mergers, spin-offs, FX, borrowed cash, short
positions, option assignment/physical delivery and unreported cash flows remain
outside the declared scope.
