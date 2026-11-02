# Equity and hypothetical options overlays

`OptionsOverlayWorkspace` combines an existing reconciled simulated equity
checkpoint with the explicit hypothetical option positions and full BSM scenario
grid in a saved options-research run. It reuses the accounting and option kernels;
it does not construct a second ledger, infer option fills, debit premiums, connect
brokers, submit intents, or approve trades.

## Request and source admission

`POST /api/v1/workspace/options-overlay` accepts exactly:

```json
{
  "checkpoint_id": "<published 64-lowercase-hex portfolio checkpoint>",
  "options_run_id": "<published 64-lowercase-hex options research run>",
  "now": "2026-11-02T15:49:02Z",
  "max_age_seconds": 60,
  "limits": {
    "max_abs_delta_notional": 100000,
    "max_abs_gamma_dollar": 100000,
    "max_abs_vega_per_point": 100000,
    "max_abs_scenario_loss": 10000
  }
}
```

All four research limits are mandatory, finite, nonnegative USD caps (vega cap is
USD per one percentage point), bounded at 1e15. Zero is a real zero limit. Missing
caps and null freshness are rejected. `now` is an explicit researcher clock; there
is no host-clock substitution. Source cutoff/freshness comparisons preserve
nanoseconds, and aware timestamp admission rejects subnanosecond precision.

The equity checkpoint must be hash verified, actually indexed/published, and the
latest checkpoint at the selected cutoff. It must have AVAILABLE state, USD
accounting, and fresh original marks. The current read-only OMS snapshot must
have exactly the checkpoint's preserved fills/hash and positions, with execution
occurrence, receipt, event, and TWAP knowledge no later than now. Deleted/revised
fills, changed positions, older checkpoint selection, ERROR/STALE parent state,
missing evidence, and future knowledge cannot become an AVAILABLE overlay.

The saved option run must be available at the cutoff, fresh under both its own
admitted policy and the overlay policy, and have AVAILABLE explicit portfolio
Greek/scenario evidence. Empty option positions mean UNKNOWN exposure, not zero.
An explicit zero contract quantity is allowed and produces known zero exposure.
The latest quote revision known at the overlay cutoff must still match the run's
source snapshot; a later-known correction requires new options research before a
new current overlay. Historical reconstruction never silently mixes old option
inputs with a corrected current quote.

Only a single option underlying and USD are supported. Every nonzero equity
position must have that same underlying. A held equity mark must equal the option
spot exactly as an admitted Decimal price, and its original timestamp must equal
the option observation timestamp. A mismatched mark/clock or other held underlying
returns explicit UNAVAILABLE instead of assigning missing risk zero. A flat
reconciled cash book is supported without inventing a held-equity mark. Input
source classification and vendor authenticity remain those of each parent.

## Valuation, Greeks, and scenario definitions

The exact equity quantity and accounting cash/NAV come from the immutable ledger.
Option price, signed multiplier-aware Greeks, and full scenario repricing come
from the saved options model run. Let Q be the signed equity quantity, S the
matching spot, and option Greeks already include signed contracts/multipliers:

- Combined delta units = Q + option delta.
- Delta notional = combined delta units times S, in USD.
- Gamma dollar = option gamma times S squared, in USD. A factor of one half
  belongs only in a quadratic return-scenario approximation; it is not included
  in this exposure definition. Scenarios use full option repricing instead.
- Vega per point = option vega, in USD per absolute volatility percentage point.
- Each combined scenario P&L = Q times S times relative spot shock plus the
  saved option model P&L for the same spot/absolute-volatility shock.
- Worst scenario loss is the positive part of the largest modeled loss among
  the explicit grid points. It is not an absolute profit cap, VaR, or ES.

Decimal arithmetic protects exact ledger quantities/cash and makes calculations
independent of the process Decimal context. Converting a floating-point option
model result to decimal text does not give the model extra numerical accuracy.
The artifact retains implementation provenance, both parent IDs, original
observation times, source hashes, scenario coordinates, policy limits, and units.

`modeled_overlay_value` means equity NAV plus hypothetical option model value.
It is an **unfinanced modeled asset value**, never funded account NAV. The
`overlay_financing` object explicitly contains UNKNOWN state, null cash adjustment,
and null financed NAV. Equity cash stays unchanged. Option premium financing,
liquidity, transaction costs, borrow, dividends, and funded return accounting are
outside this immediate spot/volatility overlay. Scenario results are MODEL_ESTIMATE;
equity holdings are SIMULATED and option positions are HYPOTHETICAL. A scenario
grid supplies no probability distribution, so combined historical VaR/ES remain
UNAVAILABLE rather than being invented from grid percentiles.
Outstanding orders and future-fill exposure remain explicitly UNKNOWN and are
outside this filled-position snapshot; passing these limits does not validate
potential order-book exposure.

## Risk policy, persistence, and terminal contracts

Exceeding any required cap saves an immutable ERROR artifact. The public envelope
has null data and a sanitized error; its provenance model identity is the
`overlay_id`, which can retrieve the saved evidence. This is a hypothetical
research policy rejection. It cannot set a kill switch, change OMS state, approve
orders, or convert the overlay into an executable portfolio. Passing the policy
also leaves `trading_approved` and `economically_validated` false.

Artifacts use fixed paths under `options-overlays/`, immutable atomic JSON
publication, a FULL-synchronous SQLite index, canonical SHA-256 identities, and
bounded retention/reads of 1,000 overlays. Returned contracts and `load` documents
are defensive copies. `load(overlay_id)` verifies the index and both published
parents, then recomputes the historical artifact. `POST
/api/v1/workspace/options-overlay-replay` accepts exactly `{ "overlay_id":
"<64-lowercase-hex>" }` and returns the original verified envelope.

Replay uses preserved historical parent checkpoints. New later fills or quote
corrections do not rewrite an earlier valid artifact; replay explicitly is not
current account reapproval. A new overlay run must pass current OMS and source
checks again. Source or index tampering is a validation error. Reproducibility is
within the recorded implementation/library environment, not a promise of exact
floating-point results across arbitrary library versions.

The `options-overlay` terminal engine supports Portfolio, Risk, Research and
Options. Rows carry spot_shock, volatility_shock, equity_pnl, option_model_pnl,
combined_model_pnl, and combined_model_value with explicit USD/model labels.
Terminal selection respects the requested point in time. Missing freshness
produces UNKNOWN/null data; expired observation freshness produces STALE with
the original evidence. A saved policy ERROR remains ERROR under later reads.

## Focused verification

`tests/test_options_overlay.py` verifies exact delta/gamma/scenario aggregation,
unchanged cash and logical OMS state, policy rejection without a kill switch,
restart and Decimal-context stability, later-fill/correction historical replay,
explicit zero versus absent exposure, future/nanosecond/stale cutoffs, older/error
checkpoints, synchronous mark mismatch, unsupported underlyings, invalid limits,
index tampering, changed positions, and revised executions. HTTP/browser and full
regression evidence belong to the integrated Sprint 5 checkpoint.
