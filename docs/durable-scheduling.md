# Durable offline TWAP scheduling

`DurableTWAPScheduler` stores a bounded parent plan, its deterministic lot-exact child quantities, a monotonic simulation clock and immutable advancement results in the existing OMS SQLite transaction domain. The scheduler, OMS reservations and transport outbox share one connection. Advancing a clock admits due child intents atomically; it does not dispatch, acknowledge or fill them. Broker connectivity is absent.

The workspace action `/api/v1/workspace/execution-engine` accepts three additional operations:

- `schedule-twap`: explicit `now`, `client_id`, `symbol`, `side`, exact `quantity`, `start`, `end`, `slices`, `max_notional`, `max_position_quantity`, and optional `lot_size` and `limit_price`.
- `advance-clock`: explicit `now` and `marks`, mapping each due symbol to exact `reference_price` and original `price_asof`.
- `cancel-schedule`: explicit `now` and parent `client_id`; only unscheduled children are cancelled. Existing orders require their separate cancellation lifecycle.

The earlier `twap` operation remains a plan-only calculation. Timestamps require timezone awareness and explicit microsecond precision. The workspace retains at most 100 scheduled children and 100 orders. Parent notional bounds include recorded fills and remaining parent quantity under the current explicit quote. Existing reservations and per-symbol position caps remain enforced. Missing, future or stale marks reject the entire clock tick. A persisted kill switch blocks child admission.

The public scheduler integration API is `DurableTWAPScheduler(oms, outbox, create=False).ensure_not_before(now)` for guarding other workspace operations without creating scheduler tables. The guard checks both the durable clock and existing OMS event timestamps. A later fill or kill event cannot be followed by a backdated prepare or dispatch even if the scheduler's last explicit clock is older.

Children due before the explicit parent end can be admitted together if their aggregate risk fits. At or after the end, unadmitted children expire; the scheduler does not catch up missed execution. Repeating an identical current tick returns its preserved result. Changing marks at the same tick conflicts, and moving the clock backwards is rejected. Parent replay verifies the original intent hash, child identities, quantities and admitted order evidence. Terminal schedules cannot contain pending children.

The focused scheduling, outbox and operational preflight run passed 42 tests with zero failures or skips. These include actual abrupt subprocess exit after child admission, rollback after a later child, reopen/retry, exact lot arithmetic, partial fills, reconciliation, deadlines, cancellation, reservation conflicts, quote freshness, tampering, kill persistence and cross-operation clock ordering. Counts overlap broader regression suites and must not be added to them. Scoped Ruff and mypy on the scheduler, execution workspace and container checker pass.

This is an offline research simulator. Synthetic transport acknowledgements and fills are not broker-observed evidence. No calendar participation model, live quote feed, broker account, paper order or economic execution-quality claim is established. T4 remains T4_NOT_CERTIFIED.
