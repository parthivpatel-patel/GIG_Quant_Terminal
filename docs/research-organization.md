# Research organization (first slice)

A local ledger and one explicit cycle. It is not a background service, and it
does not call a language model.

Code: `src/gig/research_org/`. API: `GET /api/v1/research-org/status`,
`POST /api/v1/research-org/cycle`. The Observatory section renders that
payload. Tests: `tests/test_research_org.py`, `scripts/browser_observatory.py`.

## What a cycle does

1. The orchestrator looks for the one approved hypothesis, `lagged-sign-momentum-synthetic-v1`.
2. The quantitative workflow builds a deterministic authored return series and a position that is the sign of the previous return.
3. Performance uses the existing `performance_metrics`, with a cost of 1 bp of notional on each position change.
4. The skeptic rejects the result if the position uses the same bar, compares 16 shuffled placebos, and computes a deflated Sharpe with one trial.
5. The verdict on this series is `INCONCLUSIVE`. A positive verdict is not available for synthetic data. A same-bar position is `REJECTED_LOOKAHEAD`.
6. The run is stored in SQLite. A second cycle with the same configuration returns the stored run and writes an audit note. Restarting the process reads the same row.

The ledger file is created on the first write. Reading status does not create it.

## What it does not do

The catalog lists equity long/short, futures time-series momentum, and the
options variance-risk premium as `REGISTERED_NOT_EXECUTED`. Their modules
exist. This cycle does not run them and does not report performance for them.

Roles for machine learning, microstructure, options, and macro are defined and
shown, and are not invoked.

A cycle cannot rewrite code, change risk limits, promote a model, deploy a
service, submit an order, spend money, or read credentials. `orders_submitted`
is 0. There is no background thread. The scheduler state is `EXPLICIT_ONLY`.

## Recovery checkpoint

The merged baseline before the visual-repair stack is
`beb959f29e6d3540ed3f048dbe6475333a61e04b` on `development/gig-ui-redesign`.
Do not reset to it. The repair and chart commits are the open stack on top of
it (PRs #5, #6, and #7). This slice continues from that stack.
