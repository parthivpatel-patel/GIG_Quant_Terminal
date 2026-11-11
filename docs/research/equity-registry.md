# Approved investigation registry and retrospective equity diagnostics

This milestone extends the existing Director without arbitrary generated code,
network acquisition, broker access or model promotion. The code-owned immutable
registry validates distinct objective, hypothesis, job and sequence identities,
and rejects duplicate dataset/method/version investigations. The three legacy
objectives retain their identities; two equity questions require separate explicit
approval. Approval never starts compute.

Selection is deterministic and inspectable: declared priority, +20 for a dataset
with no saved findings, +15 for an eligible registered falsification of a parent,
minus 10 per saved same-dataset report (capped at30), minus twice estimated cost
units. Outcomes are retained regardless of sign; a positive result does not earn
priority. Each plan checkpoints its score components, source/method fingerprint
and relevant report hashes. Scores are not fitted to historical performance.
Queued jobs retain their checkpointed order; completed, cancelled and exhausted
versions do not silently resubmit. The existing lease, fenced publication,
heartbeat, retry and process-budget controls remain authoritative. Memory is
bounded by the ledger's200-report limit; no work happens with the server off.

## Equity source and permissions

The existing official [UCI Dow Jones Index dataset](https://archive.ics.uci.edu/dataset/312/dow+jones+index)
is CC BY4.0, DOI10.24432/C5788V. The explicit official license declaration was
rechecked over HTTPS on2026-11-11. The packaged fixture records the actual check
time and response hash, retains the original CSV and preserves the original
2026-11-03 acquisition clock. It does not claim a new market observation or an
archive refresh. Original official ZIP and acquisition evidence remain under
`docs/evidence/sprint8/equity/`.

Attribution: Michael Brown, *Dow Jones Index* [Dataset], UCI Machine Learning
Repository,2013; Brown, Pelosi and Dirska(2013), *Dynamic-radius Species-conserving
Genetic Algorithm for the Financial Forecasting of Dow Jones Index Stocks*,
LNAI7988,27–41. [CC BY4.0](https://creativecommons.org/licenses/by/4.0/).
GIG selects five source labels, quarantines future labels and derives diagnostics;
these modifications are GIG's, not UCI or exchange outputs.

Source ZIP SHA256: `08ad8e30915f071b46f6f5bef2b1ccdfe70aacfa31aee116117048ad2adec0bf`.
CSV SHA256: `24b8f83168153d9cd0bc6121d3979bd8c70fb7b58a94b920194135cbe003090b`.
Packaged fixture SHA256: `cf77b64c2afbb3decfeb629d42cb2096f32632e9ca8e5904c74822a0fa8f2d17`.

This is an extension of the existing permitted archive, not newly discovered
stock-price coverage. AA,IBM,KO,MSFT,XOM are fixed source labels, not permanent
security IDs or a survivorship-free historical universe. There are125 weekly
panel observations,25 source dates,2011-01-07 through2011-06-24. Historical
publication/revision clocks, corporate actions, adjustment basis and authoritative
membership remain UNKNOWN/UNAVAILABLE. Source dollar signs do not establish
currency. No dates are promoted into publication timestamps, adjusted returns,
PIT trading data or funded allocations. Future-week and dividend-label fields
remain raw evidence only and cannot enter normalized features.

## Two distinct exploratory investigations

Weekly range is `(source high-source low)/source open`; weekly volume is transformed
by natural logarithm. Each predicts the next observed source week using persistence
versus an expanding training-only median. Validation uses three chronological
five-week folds, initial training9weeks, one purged week,15held-out weeks total.
The previous completed source week is the feature; no future target is used in
fitting. This ordering is an archive diagnostic, not proof of historical tradability.

Uncertainty resamples3-week circular blocks of weekly cross-sectional mean loss
differences, preserving contemporaneous dependence among names,512draws,seed113.
A Bonferroni bound covers these two questions only. Fifteen validation weeks and
five selected labels cannot establish robust predictability. A separate Decimal
transform, Python median and loop-score implementation challenges every forecast,
score, identity and temporal boundary. This is computational cross-checking on the
same source, not independent market evidence or human review.

| Investigation | Model MAE | Training-median MAE | Finding |
| --- | ---: | ---: | --- |
| Weekly range persistence |0.019235746987289144|0.012862643483420488|Worse; INCONCLUSIVE|
| Weekly log-volume persistence |0.27283008928772|0.2260168983236369|Worse; INCONCLUSIVE|

Both negative relative results are saved without profitability or trading claims.
Risk denies allocation eligibility. The CLI, bounded scheduler, durable ledger,
HTTP status and Strategy Laboratory use the same source/configuration/result IDs.

Reproduce with `python -m gig.research_org --compute uci-equity-weekly-range-persistence-2011-v1`
and `--compute uci-equity-weekly-volume-persistence-2011-v1`.

The separate [equity-index futures positioning slice](../integrations/EQUITY_INDEX_POSITIONING.md)
adds official observed CFTC aggregate counts. It does not fill the missing cash-equity
corporate-action, publication-vintage or options-print evidence.
