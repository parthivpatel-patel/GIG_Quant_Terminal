# Research organization

## Multi-dataset extension

PR #11 is merged at `a82a528f743fd61dffe095b18b386e68538f8337`.
The subsequent PR #12 extends its finite template catalog without changing the
worker's lease, resource or brokerage boundaries:

| Approved objective | Dataset | Investigation |
| --- | --- | --- |
| `treasury-reversal-robustness-v1` | Official 2023 Treasury yields | Existing negative-parent reversal robustness; unchanged |
| `ecb-fx-correlation-stability-v1` | ECB2023 USD/GBP/JPY reference levels | Purged covariance scoring versus training-only diagonal covariance |
| `boc-usdcad-variance-calibration-v1` | BoC2023 indicative USD/CAD daily averages | Causal EWMA variance QLIKE versus training-only constant variance |

Approve each objective separately through the same Director HTTP contract or
the Observatory selector. Approval never starts compute. The existing default
three-study queue is retained; approving all objectives demonstrates six total
studies across a four-job budget pause and an explicit Resume. The Director
prioritizes eligible plans using fixed code-registered priorities, checks source
identity before execution and result publication, and refuses silent resubmission
after source/method revisions, cancellation or exhausted retries. These remain
exploratory methods, not an externally preregistered confirmatory study.

The ECB and BoC reports use appropriate covariance/variance scores, not return
MSE. Both remain inconclusive. BoC's EWMA underperformed its training-only
baseline. Original Treasury and synthetic findings remain unchanged. Source
publication/revision clocks are unknown; none of these reports grants tradable
point-in-time eligibility, model promotion or portfolio authority.

Market Flow separately reads pinned CFTC2023 weekly aggregate futures positions.
It does not execute that snapshot as a strategy or infer actual trades, cash
flows, options sweeps or dealer gamma. See the source-specific permissions and
scope in `docs/integrations/CFTC_POSITIONING.md`.

A local SQLite ledger, a bounded template-based Director, and one background worker. The worker starts only after an explicit start, or on the next process when the ledger still says `RUNNING`. It is not a language model, and it does not stay up after the approved queue is empty.

Code: `src/gig/research_org/`. The Observatory reads `GET /api/v1/research-org/status`. `POST /api/v1/research-org/scheduler` takes `start`, `pause`, `resume`, or `stop`, and an optional `cadence_seconds` from 0.05 to 3600. `POST /api/v1/research-org/tick` still runs one queued job. `POST /api/v1/research-org/objective` stores an objective and does not invent a dataset.

One start drains the queue. The process does not need a POST for each job. Without Director approval, the two synthetic jobs and the historical Treasury study run once. The worker then pauses with reason `approved backlog is empty`; another start does not invent work. A separately approved Director objective permits one registered follow-up after the Treasury finding. Pause holds the queue. Stop sets `STOPPED`. A shutdown that is not a stop leaves `RUNNING` on disk; its lease is released only once the running thread exits. The next process continues unfinished jobs after lease expiry. A live renewable lease blocks a second caller. Three failed attempts exhaust a job's durable retry budget. A repeated configuration returns the stored report.

The worker dies with the server. It is not a 24-hour service, and it does not run while the machine is off.

```bash
python -m gig.research_org --worker --cadence-seconds 5 --state-dir "$GIG_DATA_DIR/terminal/research-org"
```

Recompute one report without the ledger:

```bash
python -m gig.research_org --compute lagged-sign-momentum-synthetic-v1
python -m gig.research_org --compute futures-tsmom-authored-excess-v1
python -m gig.research_org --compute treasury-10y-lagged-sign-2023
python -m gig.research_org --compute treasury-10y-reversal-robustness-2023-v1
```

## Approved jobs

1. `lagged-sign-momentum-synthetic-v1`. Sign of the previous return on an authored series. Classification `SYNTHETIC`. Verdict `INCONCLUSIVE`.
2. `futures-tsmom-authored-excess-v1`. The existing `FuturesTSMOM` class, with its default 252-session lookback, 60-session volatility window, and default cost model. The input is `authored-futures-excess-v1`: 600 business days from 2016-01-04, normal draws with location 0, scale 0.01, seed 23. It is not a futures contract. Classification `SYNTHETIC`. Verdict `INCONCLUSIVE`.
3. `treasury-10y-lagged-sign-2023`. Official U.S. Treasury daily par yields for calendar year 2023, tenors `BC_10YEAR`. The catalog entry on Data.gov listed license `http://creativecommons.org/publicdomain/zero/1.0/` when it was read at `2026-11-05T03:01:08Z` (`dateModified` 2025-02-11). The vendored XML is `src/gig/research_org/fixtures/treasury_par_yield_2023.xml`, SHA-256 `99f80b80cfeff0a93e339735d72672080d9209073b5fe814c1252e88d1428227`, 284423 bytes. The question is whether the sign of the previous daily yield change matches the next change. On this file the lagged sign matched 105 of 228 non-zero pairs, hit rate 0.4605263157894737. The sample runs from 2023-01-03 (yield 3.79) through 2023-12-29 (yield 3.88), 250 `BC_10YEAR` points. Classification `HISTORICAL`. Verdict `INCONCLUSIVE`. `economic_claim` is `NONE`. There is no Sharpe, because these are yields and no trade is observed. The publication clock is `UNKNOWN`, so the calendar date is not a decision timestamp.

On the authored futures series the recorded after-cost result is negative. Total return -0.35364662998523033. Annualized return -0.3046893380591418. Sharpe -1.9627190382410638. Sortino -3.0482694184839993. Maximum drawdown -0.402644509459651. Annualized turnover 31.346232241967975. Scored sessions 347, from 2016-12-22 through 2018-04-20. The last 63 scored sessions, 2018-01-24 through 2018-04-20, have Sharpe -6.893596899648678. The weight-1 benchmark on the same scored dates has Sharpe 0.25206848520037184 and is not a market benchmark. Sixteen shuffled placebos have median Sharpe -1.4321012563024496. The deflated Sharpe for one trial is about 2.28e-101. Changing the last return does not move an earlier weight. A 126-session lookback stays negative, and a 20 bp spread is more negative than the default 4 bp spread. That result is not evidence about a futures market.

Equity long/short and the options variance-risk premium stay `REGISTERED_NOT_EXECUTED`. No performance is shown for them.

Each stored report carries dataset, strategy version, point-in-time note, costs, assumptions, configuration, the skeptic verdict, the evidence class, a reproduce command, and where the artifact lives. The Observatory prints that block. Rejected and inconclusive verdicts stay in the ledger.

## What a tick does not do

A tick cannot rewrite code, change risk limits, promote a model, deploy a service, submit an order, spend money, or read credentials. `orders_submitted` is 0.

An objective that asks for options-flow or dark-pool evidence is stored as `BLOCKED_NO_DATASET`. No run is created for it. There is no licensed feed for those observations in this build. See `docs/integrations/MARKET_FLOW_FEASIBILITY.md`.

A job left `RUNNING` by a dead process returns to `QUEUED` on the next tick. A lease from a dead worker blocks other workers until it expires. Three failed attempts mark the job `FAILED`. A finished configuration is not recomputed. Cancelling a queued job keeps it out of the queue. Reading status does not create the database.

Budgets on the worker: four jobs in one process, a 30-second lease, 8 MB for the ledger file, 120 seconds of process CPU time, and 2,000,000 KB of maximum resident set. Crossing one pauses the worker and runs no further job.

## Recovery checkpoint

### Approved Director milestone

PR #10 was verified green on Python 3.11, Python 3.12 and frontend CI, then merged
into `development/gig-ui-redesign` at `ed8f0712d6fa3a5a5f4e01b6dee418f67b88be0a`.
The subsequent milestone is proposed separately; it does not change main.

`GET /api/v1/research-org/director` reads durable objectives, investigation plans,
role outputs and negative/inconclusive memory. Explicitly approve the registered
objective with `POST` to that path:

```json
{"action":"approve","objective_id":"treasury-reversal-robustness-v1"}
```

Approval does not start compute. Start the existing scheduler separately. After
reviewing the negative parent, the Director prioritizes and atomically queues one
nonduplicative falsification study. Objective and hypothesis identities, persisted
attempts, unique plan rows and fenced result commits survive restarts. Revocation
cancels queued follow-up work; an already running calculation is allowed to finish.
Reapproval does not silently resubmit a cancelled or exhausted hypothesis.

The observed follow-up remains **INCONCLUSIVE**: 61/115 reversal matches versus
58/115 for a training-only baseline, exact binomial p=0.5760. It records purged
chronological folds, paired bootstrap uncertainty, circular-shift diagnostics,
declared-family Holm correction, and independently implemented Decimal reference
checks. Reusing the parent archive is explicitly post-selection exploratory
research. Publication and revision clocks remain unknown. No tradable return,
allocation, profitable-alpha or model-promotion claim is supported. See
[the experiment](research/treasury-reversal-followup.md).

This milestone supports one bounded objective/template. It is not unrestricted
hypothesis generation. Six specialist workflows produce actual source, prediction,
baseline, skepticism, risk-admission and reproducibility outputs. Options and
microstructure specialists report `BLOCKED_NO_DATASET`; their labels do not imply
that unsupported research ran. The Orchestrator performs durable prioritization
and deduplication. The Director pauses when approved work is exhausted.

All ticks now acquire renewable leases. Result insertion and job completion share
one fenced transaction. Stale owners cannot publish, including after expiry or
replacement. Genuine lease loss may cause duplicate computation, but stale results
cannot commit. CPU is conservative process-wide elapsed CPU since a worker object's
first start; it is not a durable cross-process CPU quota. Job attempt limits and
the single approved follow-up remain durable.

The Observatory provides saved-finding selection, a six-stage evidence timeline,
role outputs and explicit approval/compute controls. Market Flow Intelligence adds
an honest source-access register and authored European options scenario research.
It reuses the existing pricing kernels. Model threshold alerts are not observed
institutional flow, dealer gamma, sweeps, open interest, or funded OMS positions.
There is no connected market-flow feed, durable watchlist or unattended alert
monitor. All brokerage execution remains disabled.

### Earlier checkpoint

`development/gig-ui-redesign` is `677fecce9a4542270c159351f247b8e248e237da` after PR #9. GitHub marked that pull request merged at `2026-11-05T03:14:46Z`. `main` was not moved. The check recorded before this backlog change is `docs/evidence/integration-2026-11-05/CHECKPOINT.md`.
