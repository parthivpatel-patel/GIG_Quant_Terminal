# Funded sector, factor and observed drawdown budgets

The existing `funded-risk` request accepts three optional policy limits in
addition to its eight required cash/model-risk caps:

```json
{
  "max_sector_gross_notional": 100000,
  "max_abs_factor_exposure": {"market": 100000},
  "max_drawdown": 0.10
}
```

These fields extend `limits`; they are not a separate endpoint. Existing requests
and saved report calculations remain unchanged when the new limits are absent.
No metadata or new calculation fields are injected into those old artifacts.

## Sector and factor evidence

Requested sector/factor policies require request-level metadata keyed by
**underlying identity**, rather than by derivative contract identity:

```json
{
  "metadata": {
    "SYN101": {"sector": "Authored", "factors": {"market": 1.0}}
  },
  "metadata_source_identity": "AUTHORED_NOT_OBSERVED_FACTOR_METADATA",
  "metadata_available_at": "2026-11-02T15:49:00Z"
}
```

Metadata is caller-declared, not a verified vendor sector classification,
estimated factor model or calibrated forecast. Source text and knowledge time
are required. The knowledge timestamp must be no later than `now` and no older
than `max_age_seconds`; nanosecond comparisons are retained. Missing metadata,
unknown/null required sectors/loadings, future knowledge or stale knowledge
returns UNKNOWN with null data and does not replace an existing valid policy.
Malformed numeric/text values reject. The same unknown inputs reject configured
pre-execution admission and trigger the existing fail-closed post-event risk/kill
path, without discarding an accepted simulated fill.

For every actual position, working order and proposed candidate, the existing
asynchronous quantity bound is multiplied by absolute per-unit delta notional.
Equities use underlying spot notional; options use the admitted model delta and
contract multiplier, not option premium notional. Sector budgets sum these
delta-equivalent gross USD components by the underlying's declared sector.
Factor budgets additionally multiply each component by the absolute declared
loading, then sum. Opposite option deltas or opposite factor loadings receive no
netting credit in this conservative budget. Signed current factor exposure is
reported separately and must not be confused with the admission bound.

Every required underlying must have every requested factor loading. An absent
loading is never interpreted as beta zero. A valid declared zero loading remains
zero. Limits and loadings are finite and bounded. The output
`sector_factor_risk` records source/knowledge provenance, underlying/sector
delta-gross USD, factor budget USD, signed current factor exposure and an explicit
`calibrated_factor_model=false` flag.

## Drawdown evidence

`max_drawdown` is a loss fraction in `[0,1]`. It compares current marked funded NAV
against the maximum of fixed initial capital, current NAV and all complete
published NAV observations admitted by the requested cutoff. Capital bases must
match. Unavailable funding, marks, reconciliation or unsettled expiry obligations
do not supply peak values. At least one complete published history observation is
required; missing publication history is UNAVAILABLE, not an inferred zero.

The report freezes original snapshot IDs, timestamps, capital and NAV values.
Load/replay verifies their immutable source records and recomputes the same
historical calculation, so later marks and future peaks cannot rewrite an older
report. `unobserved_peak_state=UNKNOWN` explicitly limits the result to admitted
account observations. It is not an inferred intraday/lifetime market peak and is
separate from the stricter regular-history admission for historical VaR/ES.

## Safety and evidence

Known sector/factor/drawdown breaches use the existing risk publication and
`OfflineOMS.set_kill_switch` transaction. A failed transaction leaves neither an
admitted risk index/policy nor a changed kill control. Passing these budgets never
approves a trade, releases an engaged kill switch or certifies full funding.

`tests/test_funded_factor_risk.py` covers independent option-delta exposure sums,
no factor-netting credit, asynchronous candidate admission, metadata knowledge
boundaries, unknown-policy replacement safety, accepted-fill preservation,
historical drawdown replay, missing history and injected rollback. Existing
Sprint 6 funded-risk tests remain in the focused verification selection.
