# Independent supplied offline statements

`StatementWorkspace` compares a supplied statement fixture to a published,
hash-verified funded-account snapshot. It does not connect to a broker or vendor,
import balances, change orders/positions, or release/set a kill switch. Fixture
authenticity remains UNKNOWN. The statement is supplied independently of the
engine book; tests use explicit hand-calculated cash and contract examples.

The HTTP integration exposes `statement-admit`, `statement-reconcile` and
`statement-replay`; engine `independent-statement` serves Portfolio, Risk,
Execution and Operations. The workspace methods are `ingest`, `run`, `load`,
`replay` and `section`.

## Statement admission

`ingest` accepts exactly:

- `source`, `statement_id`, `observed_at`, `available_at`, `currency: USD`.
- `balances`: `cash`, `nav`, `realized_gross_pnl`, `unrealized_pnl`, `fees`,
  `reserved_cash`, `available_cash`. Values are signed exact decimal text or
  integers; null means UNKNOWN. Floating-point financial inputs are rejected.
- `positions`: nonzero holdings with `instrument_id`, `kind`, `currency`,
  integer `multiplier`, `quantity`, `cost_basis`, `market_value`,
  `unrealized_pnl`. OPTION rows also require `underlying_id`, `expiry`, `strike`,
  `right`, `style`, `settlement`. Unexpected or differently specified holdings
  become discrepancies; they are not imported into the engine.
- `orders`: `client_id`, `instrument_id`, `side`, `quantity`, `filled`,
  `limit_price`, `state`, boolean `acknowledged`, `created_at`, `updated_at`.
- `fills`: `execution_id`, `client_id`, `instrument_id`, `side`, `quantity`,
  `price`, `fee`, `occurred_at`.
- `settlements`: option cash lifecycle records with `settlement_id`,
  `instrument_id`, `position_quantity`, `spot`, `payoff`, `fee`, `observed_at`,
  `available_at`.
- `reservations`: `client_id`, `instrument_id`, `remaining_quantity`,
  `buy_cash`, `short_put_collateral`, `fee_cash`, `state`.

Record arrays are mandatory, including an explicit empty array when no record is
claimed. Acquisition cannot precede observation. Aware timestamps preserve
nanoseconds; naive/subnanosecond strings fail admission. Source, statement ID and
acquisition clock identify an immutable revision. A correction appends a later
knowledge-time revision; conflicting values at the same key reject.

Duplicate execution/order/position/lifecycle identifiers remain in the admitted
source evidence and produce explicit discrepancy findings. They are never
silently deduplicated into a successful reconciliation. There is no hidden price,
fee or balance tolerance: numeric decimal values compare exactly, while source
and instrument identities compare as text.

### Lifecycle coverage

For a non-IMMEDIATE settlement policy, `balances` must additionally cover the
engine's `settled_cash`, `unsettled_sale_receivables` and
`unsettled_buy_payables`. `trade_settlements` requires `execution_id` and
`settlement_at`, and `settled_trade_positions` maps instrument IDs to exact
quantities. Omitting these records leaves reconciliation UNKNOWN. Under the
explicit IMMEDIATE convention, fill occurrence clocks already define immediate
cash settlement; the scope does not imply a T+1 statement reconciliation.

Applied corporate actions require `lifecycle_events`. Each event has `event_id`,
`kind`, `instrument_id`, `action_id`, `applied_at`, `effective_at`, `known_at`,
`cash_amount`, `entitlement_quantity`, plus `ratio_new`/`ratio_old` for SPLIT or
`amount_per_share` for CASH_DIVIDEND. The statement does not need to know the
engine's internal `action_hash`. Dividend events additionally require the
`dividend_income` balance. Missing enabled-lifecycle coverage cannot produce
MATCH even if cash and quantities happen to agree.

Known unapplied corporate actions, including future announcements and revised
actions awaiting application, leave corporate-action coverage UNKNOWN. The
current statement schema does not independently reconcile future entitlement
accrual; empty applied-event lists cannot establish that coverage. This does not
change the funded book's separate treatment of future announcements. An engine
snapshot whose own reconciliation state is incomplete also cannot produce MATCH.

## Comparison and public states

`run` requires `statement_hash`, `snapshot_id`, and aware `now`. Future parent
knowledge produces an UNKNOWN envelope with null data, so an earlier decision
cannot inspect a later statement. Known but nonmatching observation clocks
produce an explicit timing finding and UNKNOWN reconciliation. Statement fill
events after their claimed observation clock also prevent a match.

A completed report has an AVAILABLE envelope and a separate mandatory
`reconciliation_state`:

- `MATCH`: all required supported fields match and no coverage is unknown.
- `DISCREPANCY`: a known balance, record, term, fee, status or position differs,
  or duplicate/missing/unexpected records exist.
- `UNKNOWN`: clocks are not comparable, financial marks are incomplete, required
  lifecycle coverage is absent, or a required value cannot be established.

`fixture_match` is true only for MATCH. `external_reconciled` is **always false**,
including a matching fixture. `fully_funded_certified` and `trading_approved`
remain false. These are local evidence comparisons, not actual broker statements
or account certification. COMPLETE comparison is not synonymous with a match.

ACKNOWLEDGED/PARTIALLY_FILLED/FILLED engine protocol states establish an observed
simulator acknowledgement. A historical CANCELLED/REJECTED/EXPIRED/SUBMITTING
snapshot lacking its earlier acknowledgement sequence remains UNKNOWN for that
field, rather than inventing an ACK. Report coverage discloses this limitation.

## Durability and replay

Artifacts live under fixed `independent-statements/` paths. Canonical JSON SHA-256
identities, atomically synced immutable files and FULL-synchronous SQLite indexes
preserve sources/reports. Requests cannot select filesystem paths or URLs.
Symlinks and invalid identities fail. Retention is bounded to 1,000 statements
and 1,000 reports; record counts and reads are bounded independently.

`replay({report_id})` verifies the report, original supplied statement, original
published engine snapshot, index metadata and deterministic comparison. A newer
statement or changed current account cannot rewrite the old report or turn
historical replay into current approval. Interrupted report publication may
leave an unindexed immutable file; it is not an admitted report. Read-only
statement comparisons leave the funded OMS unchanged.

Focused tests cover independent scalar balances, contract multipliers and cash
expiry, partial-fill reservations, missing/duplicate/extra records, exact decimal
boundaries, nanosecond knowledge cutoffs, coverage gaps, data/index tampering,
failure injection, defensive copies, restarts and deterministic replay.
