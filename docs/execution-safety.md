# Execution safety and offline evidence

This development batch made no broker connection and placed no paper or real
orders. `GIG_BROKER_CONNECTIVITY_ENABLED` defaults to `false`; credentials alone
do not authorize a connection. The Alpaca constructor checks this gate before
SDK import/client construction. `allow_live=True` always raises. HTTPS with the
exact `paper-api.alpaca.markets` hostname is the only supported trading endpoint;
userinfo, alternate ports, paths, query strings, fragments, and hostname
substring spoofs are rejected. The data hostname is independently constrained.
Enabling the configuration gate is a future operator action requiring separate
approval; it was not enabled by this task.

## Broker acknowledgement is not execution

`AlpacaBroker.submit` returns `OrderAcknowledgement`, containing requested
quantity, broker/client identity, status, and acknowledgement classification. It
has no executed price or filled quantity. `close_all` reports accepted, rejected,
or unknown requests for every held symbol, including absent responses. The
legacy `flatten` report only counts actual `Fill` objects as closed positions.
`PaperBroker` fills carry `SIMULATED`, and no simulator output is broker evidence.

Cancellation requires a known successful response for every returned request.
Ambiguous submission halts the remaining batch and is recorded as `UNKNOWN`;
there is no automatic submission retry. Provider exception text is excluded
from submission, cancellation, account-read, flatten, and persistence messages.
Unknown market/borrow capability cannot authorize trading. Failed/invalid
account positions cannot become an apparently empty account. The pre-trade
gate rejects nonfinite inputs, invalid policies, future/unknown data timestamps,
unknown required market state, invalid model volatility, and invalid high-water
marks. Existing limit and drawdown checks remain independent of the optimizer.

## Durable offline OMS

`gig.execution.oms.OfflineOMS` uses local SQLite WAL and full synchronous writes.
Order transitions, execution identity, positions, and audit events commit
atomically. Quantities, prices, fees, and positions use exact decimal strings
with bounded admitted precision and a local arithmetic context. Tests inject an
event-write failure and verify complete rollback of the fill and position.

An order passes `CREATED -> APPROVED -> ACKNOWLEDGED -> PARTIALLY_FILLED -> FILLED`
or a supported terminal cancellation/rejection/expiration. Admission checks
client identity, side, finite exact numbers, price freshness, notional and
position limits, and the durable kill switch. Position reservation considers
working orders without allowing opposite orders to offset each other's risk.
Execution IDs are idempotent only for the same order, quantity, price, fee, and
execution instant; conflicting replay never changes the ledger. Recovery
reports working orders and interrupted simulations and never resubmits them.
Position reconciliation compares exact values and leaves both sides unchanged.

`plan_twap` is an offline deterministic scheduler: it conserves exact lot-sized
quantity, produces stable child IDs and timezone-normalized timestamps, and
places no orders. It uses a half-open time window and declares resource bounds.

## Integrated rehearsal

`simulate_signals(scores, prices, sectors, *, oms, run_id, asof, now, ...)` runs
the existing quantile portfolio construction, trade-plan reconciliation,
pre-trade gate, durable OMS, deterministic simulated fills, positions, and exact
post-fill comparison. Only `SYNTHETIC` and `HISTORICAL` input classifications are
accepted. Fill assumptions are explicit: supplied prices and zero spread,
impact, and fees. Results are unapproved research, not economic performance.

Each admitted run stores a canonical intent SHA-256. A successful replay returns
the original evidence without new orders; changed intent under the same run ID
is rejected. An interrupted run remains marked `RUNNING`, and subsequent
scenarios are blocked for explicit reconciliation. Completed individual
executions survive; there is no promise of atomic rollback of an entire
multi-order scenario. This is deliberate crash evidence, not permission to
delete state or resume blindly. No automatic resolution of interrupted scenario
markers is supplied in this batch.

## Verification

The following commands require only installed research dependencies and use no
credentials, provider calls, or real brokerage SDK:

```sh
PYTHONPATH=src python -m unittest discover -s tests -p test_offline_oms.py -v
PYTHONPATH=src python -m unittest discover -s tests -p test_execution_safety.py -v
```

Observed result: 25 existing and 16 added tests passed, with no failures or
skips. Coverage includes injected SDK mocks, replay conflicts, cancellation and
flatten uncertainty, exact decimal arithmetic under a hostile ambient context,
two connections, rollback, restart, stale/future data, kill switch, and the full
offline signal-to-position flow. The legacy pytest fake broker now explicitly
declares assumed shortability; inherited unknown borrow is no longer approval.

## Remaining limits

The real paper adapter has not been externally validated. It has no tested live
streaming fill ingestion, durable broker-outbox submission, broker event replay,
or automated resolution of ambiguous/cancel-race outcomes. Its legacy
`run_once` DuckDB audit trail is separate from the offline OMS; persistence
failure is logged, not a verified durable broker transaction. Cancellation
acknowledgement is not independent proof that all fills have settled. These
limits prevent claiming complete operational broker reconciliation or
institutional execution certification. Real-money mode is unsupported.
