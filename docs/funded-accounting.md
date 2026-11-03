# Bounded USD funded offline accounting

`FundedWorkspace` adds an isolated account below `funded-account/oms.sqlite`.
It reuses `OfflineOMS`, its transaction/savepoint controls, existing approval,
limit-price, kill-switch and durable outbox methods. It does not modify the
legacy equity account, connect a broker, or import hypothetical research
positions as executions.

The verified scope is fully paid long USD equities and long European,
cash-settled CALL/PUT contracts. Equity quantities may be fractional; options
require whole contracts and explicit positive integer multipliers. Opening
shorts, including short puts, is rejected. Selling already owned inventory is
supported and outstanding sales reserve that inventory conservatively. Naked
liabilities, borrowing, American exercise, physical delivery, FX and unreported
cash flows are unsupported. `fully_funded_certified` remains false.

## Explicit account actions

`POST /api/v1/workspace/funded-account` requires `operation` and an aware `now`.
The OMS clock must be representable in microseconds; nanoseconds are rejected
before account mutation and never silently truncated. The operations are:

- `initialize`: exact decimal `initial_cash`, immutable after admission.
- `register`: `instruments`, with exact identity, kind, currency and multiplier;
  options also require underlying, expiry, strike, right, European style and CASH
  settlement. Conflicting identities and expired registrations reject.
- `reserve`: `client_id`, `instrument_id`, side, exact quantity/limit price,
  price observation timestamp and explicit fee budget.
- `dispatch`: sends pending intents only to the local offline counterparty.
- `event`: an explicit immutable protocol fixture with event/client IDs, kind,
  occurrence timestamp, and fill quantity/price/fee where applicable.
- `cancel`: client ID, followed by explicit dispatch of the cancellation intent.
- `mark`: instrument-to-price/timestamp mapping and explicit freshness policy.
- `settle`: settlement/instrument IDs, expiry spot, exact original expiry
  observation time, acquisition/availability time and settlement fee.
- `state`, `replay` with `snapshot_id`, and explicit `kill` with boolean engaged.

Initial intent admission, acknowledgement and a fill are separate events.
Dispatch does not invent positions or spend premiums. Manual fill fixtures are
not described as adapter-observed or broker-observed positions; external
counterparty reconciliation remains UNKNOWN. Existing OMS rules quarantine a
limit-price violation and engage the existing kill switch without applying an
invalid cash fill. Unsupported fixture liabilities reject before admission.

## Cash, reservations and lifecycle

Execution cash flow is signed quantity times quoted premium/price times the
registered multiplier, plus explicit execution fees. Fees are expensed at the
event; FIFO cost basis and realized/unrealized gross P&L use multiplier-adjusted
costs. At admitted marks, NAV equals initial capital plus realized and unrealized
gross P&L minus fees exactly in Decimal arithmetic.

All unfilled buys reserve limit-price cash times multiplier and remaining fee
budget. Conditional proceeds from working sells never fund buys. Partial fills
release only the consumed reservation; cancellation releases only the remaining
reservation. Fees cannot exceed their explicit admitted budget. All sales
reserve owned inventory, so simultaneous sales cannot open an implicit short.
Cash and reserved-cash funding admission is independent of process Decimal
precision. There is no interest, securities lending or margin debit facility.

At expiry, an unclosed option becomes an explicit pending obligation. It is not
silently marked zero and new orders are blocked until settlement. Settlement
requires the admitted spot at the exact contract expiry, known no later than the
requested clock, and terminated working orders in that contract. Automatic long
European cash exercise pays intrinsic value times quantity and multiplier; OTM
expiry pays zero. A settlement fee requiring borrowed cash rejects. Future
settlement fees remain UNKNOWN until explicitly supplied: this bounded cash
mechanism does not certify every possible future liability as funded.

Settlement is a separate immutable lifecycle event rather than an invented OMS
fill. Raw OMS quantities remain preserved; effective economic quantities subtract
settled quantities. Both are exposed with their reconciliation relationship.
Late fills after settlement require unsupported lifecycle revision and reject.

## State, atomicity and restart

Snapshots contain exact cash/reservations, current positions, original marks,
fills, orders, lifecycle events and source hashes. Missing/expired observations
produce null NAV and UNKNOWN financial reconciliation, never a zero valuation.
Stale marks retain a STALE financial state. An action's AVAILABLE receipt means
the local action completed; `account_state`, `mark_state` and reconciliation
describe the actual financial availability. Terminal account views expose
UNKNOWN/STALE when financial states are incomplete.

Order, fill, settlement, kill and snapshot-index mutations use the existing OMS
transaction domain. Immutable snapshot bytes are synced before indexing. A crash
can leave an unindexed complete file, which is not an admitted snapshot. Forced
process exits after order admission and immediately before snapshot commit verify
rollback of the order, cash, reservation and publication index.

Stable order/event/lifecycle retries return the original immutable receipt.
State, mark, dispatch and kill operations recompute the current book even when
their clock and payload repeat; they cannot return an old balance or skip newly
admitted transport work. Snapshot replay verifies canonical hashes and its
publication index and is historical evidence, not current order approval.

Configured advanced-risk guards are additional controls on reserve, transport
and post-event reconciliation; they never replace cash/inventory admission or
existing OMS controls. With no advanced policy, advanced risk is explicitly
UNKNOWN. These offline checks do not establish broker reconciliation, investment
performance, physical ARM64 acceptance or institutional certification.
