# Research organization

A local ledger and an approved backlog. It is not a background service, and it does not call a language model.

Code: `src/gig/research_org/`. The Observatory reads `GET /api/v1/research-org/status`. `POST /api/v1/research-org/tick` runs one queued job. `POST /api/v1/research-org/objective` stores an objective and does not invent a dataset. `POST /api/v1/research-org/cycle` still runs only the lagged-sign job, for the earlier contract.

Resume, on a machine that is actually on:

```bash
python -m gig.research_org --ticks 2 --state-dir "$GIG_DATA_DIR/terminal/research-org"
```

The process exits when those ticks finish. It does not stay running, and it does not promise continuous research while the machine is off.

Recompute the futures report without the ledger:

```bash
python -m gig.research_org --compute futures-tsmom-authored-excess-v1
```

## Approved jobs

1. `lagged-sign-momentum-synthetic-v1`. Sign of the previous return on an authored series. Verdict `INCONCLUSIVE`.
2. `futures-tsmom-authored-excess-v1`. The existing `FuturesTSMOM` class, with its default 252-session lookback, 60-session volatility window, and default cost model. The input is `authored-futures-excess-v1`: 600 business days from 2016-01-04, normal draws with location 0, scale 0.01, seed 23. It is not a futures contract.

On that series the recorded after-cost result is negative. Total return -0.35364662998523033. Annualized return -0.3046893380591418. Sharpe -1.9627190382410638. Sortino -3.0482694184839993. Maximum drawdown -0.402644509459651. Annualized turnover 31.346232241967975. Scored sessions 347, from 2016-12-22 through 2018-04-20. The last 63 scored sessions, 2018-01-24 through 2018-04-20, have Sharpe -6.893596899648678. The weight-1 benchmark on the same scored dates has Sharpe 0.25206848520037184 and is not a market benchmark. The verdict is `INCONCLUSIVE` because the series is synthetic. `economic_claim` is `NONE`. Sixteen shuffled placebos have median Sharpe -1.4321012563024496. The deflated Sharpe for one trial is about 2.28e-101. Changing the last return does not move an earlier weight. A 126-session lookback stays negative, and a 20 bp spread is more negative than the default 4 bp spread.

Equity long/short and the options variance-risk premium stay `REGISTERED_NOT_EXECUTED`. No performance is shown for them.

## What a tick does not do

A tick cannot rewrite code, change risk limits, promote a model, deploy a service, submit an order, spend money, or read credentials. `orders_submitted` is 0.

An objective that asks for options-flow or dark-pool evidence is stored as `BLOCKED_NO_DATASET`. No run is created for it. There is no licensed feed for those observations in this build.

A job left `RUNNING` by a dead process returns to `QUEUED` on the next tick. Three failed attempts mark the job `FAILED`. A finished configuration is not recomputed. Cancelling a queued job keeps it out of the queue. Reading status does not create the database.

## Recovery checkpoint

`development/gig-ui-redesign` contains `ef7294343d9e01e9dd8a090721272c658d5bf081`. The check recorded before this backlog change is `docs/evidence/integration-2026-11-05/CHECKPOINT.md`. Do not reset to `beb959f`.
