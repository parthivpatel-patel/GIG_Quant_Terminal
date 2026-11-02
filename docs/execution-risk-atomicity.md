# Execution-linked risk checkpoints

The OMS accepts timezone-aware timestamps representable in whole microseconds.
The HTTP execution clock adapter and the existing OMS `_time` admission reject a
nonzero nanosecond remainder; neither rounds nor truncates. The pure research
ledger still preserves timezone-aware nanosecond evidence. Research precision
does not imply execution-clock representability.

`reconcile_with_risk_controls` validates the execution timestamp before opening
or mutating related persistence. It then acquires the existing OMS transaction,
checks the durable clock, reconstructs the simulated ledger and admits its
checkpoint. Execution-linked checkpoint identities are indexed in the OMS
database's `portfolio_checkpoints` table. An ERROR response engages the existing
`OfflineOMS.set_kill_switch` in that same transaction. The original independent
research-ledger index remains readable and is not rewritten or migrated away.

The complete immutable checkpoint file is synced before its authoritative index
is inserted. An interrupted writer may leave an unindexed immutable file; this
is not an admitted result and terminal/history reads ignore it. The same-DB
transaction commits the risk index and kill state together. Exceptions and
process death before commit roll both back. This avoids a cross-database commit
claim with SQLite WAL. The outbox is initialized before entering the transaction
because its existing schema initialization uses `executescript`.

Identical successful retries reload the same checkpoint and do not duplicate a
kill event. AVAILABLE, STALE, UNKNOWN and UNAVAILABLE responses never release an
engaged switch. Releasing the switch remains an explicit existing OMS operation;
a replayed ERROR reengages it if an operator has released it and the request
still respects the durable clock. Allocation's
accounting/risk path uses this same coordinator. It cannot bypass the existing
outbox, clock, approval or kill-switch controls.

`tests/test_risk_coordination.py` verifies precision rejection before mutation,
direct pandas timestamp admission, exact microsecond/offset boundaries,
idempotence and reopen, both transaction interruption points, real subprocess
termination, artifact and kill-write failures, legacy checkpoint recovery and
supported non-error/risk-breach transitions. The HTTP boundary tests also verify
safe error serialization and origin controls. These are offline simulation
checks, not brokerage, hardware power-loss or live execution certification.
