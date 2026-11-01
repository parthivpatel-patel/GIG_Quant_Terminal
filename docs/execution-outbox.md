# Durable offline execution workspace

This component records **SIMULATED, NOT BROKER OBSERVED** orders and fills. It
never reads brokerage credentials, imports an SDK, creates a network client, or
submits a paper/live order. T4 remains T4_NOT_CERTIFIED. Passing these fixtures
does not establish economic validity or live trading readiness.

`ExecutionOutbox` shares the OMS SQLite transaction for intent admission. Order,
risk approval, reservation and transport intent either commit together or roll
back together. Exact decimals remain strings in the ledger. Acknowledgements
never create positions. Explicit incremental fills update positions and retained
execution evidence in one transaction. Event identity is immutable; replay of an
identical event does not duplicate a fill, and conflicting identities are rejected.
Source occurrence and local receipt time are distinct, aware timestamps. Nonzero
submicrosecond precision is rejected rather than silently truncated.

A transport claim commits before the simulated counterparty is called. A crash
after claim or an uncertain response leaves an ambiguous intent. Restart never
resends it automatically. Reconciliation queries the durable simulated
counterparty by the exact client identity and intent hash. Only explicit proven
absence permits an explicit retry, and that retry repeats freshness, kill-switch
and exposure checks. An unknown response does not prove absence. Cancellation
races preserve actual fills; a valid delayed pre-cancel fill is retained. Invalid
transitions are quarantined, engage the kill switch and return ERROR. A real
simulated fill that breaches post-trade position limits is retained, marked as a
risk breach and returns ERROR rather than disappearing from positions.

`OfflinePaperAdapter` uses a separate SQLite database to represent the durable
simulated counterparty. Fault controls exercise disconnection, timeout before
acceptance, timeout after acceptance, unknown queries and rejection. Its Alpaca
parser accepts supplied fixtures only: identities and requested quantity must
match the admitted intent; fill quantity is incremental. These fixtures establish
neither provider authenticity nor broker connectivity.

## Workspace contracts

`ExecutionWorkspace(base_dir)` owns fixed `oms.sqlite` and
`counterparty.sqlite` files. User-provided paths, clients, URLs, credentials and
adapter implementations are not accepted. Symlink paths are rejected. Missing
section/status reads do not create databases. The HTTP integration can expose
`run(payload)` and `section(section, as_of, max_age_seconds)` through the existing
sanitized contract boundary.

Every operation requires explicit `operation` and aware ISO `now`. Supported
operations are prepare, dispatch, event, fill, cancel, retry-absent, reconcile,
status, replay, kill-switch and twap. Unknown fields are rejected. The workspace
retains at most 100 admitted client identities; dispatch supports at most 100
intents per call; replay is capped at 1000 events. Full ledger evidence remains
in SQLite. Cancel and reconcile are available with an engaged kill switch; new
submissions are blocked.

Example local workflow (repeat `now` on every request):

```json
{"operation":"prepare","now":"2026-11-01T15:00:00Z","client_id":"offline-example-1","symbol":"SYN001","side":"buy","quantity":"10","reference_price":"100","price_asof":"2026-11-01T15:00:00Z","max_notional":"1000","max_position_quantity":"10","limit_price":"100"}
```

Prepare reserves the simulated intent; it does not submit it. Explicit dispatch
records a simulated acknowledgement with no position:

```json
{"operation":"dispatch","now":"2026-11-01T15:00:00Z"}
```

An explicit simulated fill can then exercise reconciliation and restart:

```json
{"operation":"fill","now":"2026-11-01T15:00:01Z","client_id":"offline-example-1","quantity":"2","price":"100"}
```

Execution, portfolio and risk sections use persisted evidence timestamps.
Unknown freshness produces UNKNOWN; evidence older than the explicit age budget
is STALE. A query cutoff before the latest execution event produces UNKNOWN;
current balances are not presented as historical point-in-time balances. Missing
NAV/P&L remain null. Error/unknown contracts do not contain successful data.

TWAP reuses the exact existing planner, supports at most 100 slices and returns
only a plan. It creates no database, transport intent or scheduled order. No
scheduled dispatcher or broker-connected TWAP execution is certified here.

## Verification and limits

`tests/test_execution_outbox.py` covers atomic admission and rollback, crash
boundaries before/after transport and before event commit, exact partial fills,
duplicate/conflicting and out-of-order events, cancellation races, explicit
absence retries, ambiguous responses, kill/freshness checks, post-trade risk,
fixture identity, restart persistence and truthful workspace sections.

SQLite FULL synchronization and WAL are enabled. Process restart and injected
transaction failures are tested locally; host power loss, disk corruption,
distributed counterparty behavior and actual broker semantics remain unverified.
No browser, ARM64 runtime, economic validation or commercial readiness claim is
made by these tests.
