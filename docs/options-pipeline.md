# Persisted historical options research

Sprint 5 adds an append-only options evidence workspace and immutable European
research experiments. It has no provider connection, brokerage adapter, order
submission, vendor authentication, or OPRA entitlement check. All source/feed
classifications are caller declarations; hashes verify content identity only.

`OptionsWorkspace(base_dir)` uses fixed paths under `options-research/`, an
SQLite index with FULL synchronization, atomically published immutable snapshot
and run JSON, at most 1,000 snapshots and 1,000 experiments, and at most 200 quotes
per synchronous chain. Missing reads do not create a workspace. Returned contracts
and loaded JSON are defensive copies. A contract ID cannot change underlying,
currency, expiry, strike, right, exercise style, or multiplier across admissions.
Adjusted/deliverable contracts outside this identity model are unsupported.

## Admission contract

`POST /api/v1/workspace/options-admit` accepts exactly these fields:

```json
{
  "source": "authored-european-chain",
  "classification": "SYNTHETIC",
  "feed": "AUTHORED_FIXTURE",
  "underlying_id": "SYN100",
  "currency": "USD",
  "event_at": "2026-11-02T15:00:00Z",
  "acquired_at": "2026-11-02T15:00:01Z",
  "knowledge_at": "2026-11-02T15:00:01Z",
  "spot": 100,
  "spot_source_identity": "authored-synchronous-spot-100",
  "spot_event_at": "2026-11-02T15:00:00Z",
  "spot_knowledge_at": "2026-11-02T15:00:01Z",
  "quotes": [{
    "contract_id": "SYN100-20271102-100-C",
    "expiry": "2027-11-02T15:00:00Z",
    "strike": 100,
    "right": "CALL",
    "multiplier": 100,
    "style": "EUROPEAN",
    "bid": 8.81,
    "ask": 8.85,
    "volume": 10,
    "open_interest": null
  }]
}
```

Classification must be one of the existing admitted REAL_TIME, DELAYED, SNAPSHOT,
HISTORICAL, SYNTHETIC states. It does not confer authenticity or coverage. Currency
is USD; only European CALL/PUT contracts with positive integer multipliers are
supported. Expiry is an exact aware timestamp, later than the observation and at
most 30 years away. A date alone is insufficient. Identity text is bounded; prices
and spot are finite; quotes must be nonnegative and not crossed. Volume/open
interest are nonnegative integer contract counts or null. Null is never zero.
Duplicate identities or economic coordinates are rejected.

Observation <= acquisition <= revision knowledge is required. Spot must have the
same observation timestamp and be known by the snapshot knowledge time. Each
source/underlying/observation/knowledge key is immutable and replay-idempotent.
A later correction must append a new knowledge-time revision. Clock declarations
are preserved as caller evidence, not fabricated proof of when a provider first
published a historical quote. Nanosecond timestamps are retained and subnanosecond
strings are rejected. There is no fallback from missing underlying marks to a
synthetic spot or modeled option quote.

## Research contract and replay

`POST /api/v1/workspace/options-research` requires source, classification,
underlying_id, observation_at, as_of, max_age_seconds, rate, dividend_yield.
Optional fields are positions, spot_shocks, volatility_shocks, surface_query:

```json
{
  "source": "authored-european-chain",
  "classification": "SYNTHETIC",
  "underlying_id": "SYN100",
  "observation_at": "2026-11-02T15:00:00Z",
  "as_of": "2026-11-02T15:00:01Z",
  "max_age_seconds": 60,
  "rate": 0.02,
  "dividend_yield": 0.01,
  "positions": [{"contract_id": "SYN100-20271102-100-C", "quantity": 2}],
  "spot_shocks": [-0.1, 0, 0.1],
  "volatility_shocks": [-0.02, 0, 0.02],
  "surface_query": {"log_forward_moneyness": 0, "ttm": 0.75}
}
```

The source observation is selected exactly, using the latest acquisition/revision
known at as_of, with exact classification. Missing eligible evidence returns
UNAVAILABLE. A later-known correction cannot change the selected bytes or the
result reconstructed at the earlier cutoff. The run hash covers the full request,
source snapshot hash, model assumptions, diagnostics, outputs, and positions.
Repeated requests and process restarts recover the same run ID. `load(run_id)`
verifies run/index/source identity. `POST .../options-replay` with `{ "run_id":
"<64 lowercase hex>" }` independently selects the old knowledge window and
recomputes its content hash. Source/index tampering and changed replay results
raise a validation error. Byte reproducibility is established for the recorded
implementation/library environment; it is not a cross-version floating-point
guarantee or evidence of historically available research.

Valuation uses the observation clock, not the later reconstruction clock. BSM
uses ACT/365F maturity, caller-supplied continuous flat annualized rates and
dividend yield, constant volatility, European exercise, and no discrete dividend
or early exercise approximation. Rates are researcher assumptions, not observed
yield-curve evidence. Midpoint, IV, Greeks, interpolation, exposure, and scenario
P&L are MODEL_ESTIMATE even for HISTORICAL input quotes. Invalid finite-volatility
no-arbitrage midpoint inversions and zero-IV smooth-Greek boundaries remain source
rows with ERROR and null models; they are never invented or silently imputed.
Positions referencing an absent/invalid model reject the request. Quotes remain
source evidence even when their midpoint cannot support a model.

Greeks reuse the validated kernel: delta per underlying unit, gamma per spot
currency unit, vega/rho per one percentage point, theta per calendar day. Signed
integer simulated contract positions multiply each exposure by its explicit
multiplier. These are researcher-declared hypothetical positions, not reconciled
OMS holdings, approved allocations, or broker observations. Scenario spot shocks
are relative decimal changes; volatility shocks are absolute decimal changes.
Full BSM repricing produces model P&L, without transaction costs or realized P&L.
The chain can be observed historically while the positions remain simulated.

## Surfaces and documented diagnostics

Only CALL coordinates form the IV surface; puts remain visible with their own
IV/Greeks. Each point is ln(K/F), ACT/365F maturity, and inverted midpoint decimal
IV. Optional interpolation is linear in total variance inside the observed
coordinate convex hull, with no extrapolation. Missing/degenerate hulls return an
explicit UNAVAILABLE query. Static call-slice monotonicity/convexity and matching
coordinate total-variance calendar checks are retained. Missing cross-expiry
coordinates remain UNKNOWN; no continuous arbitrage certification is claimed.

Skew is the endpoint secant of admitted call IV against ln(K/F) within an expiry.
It is not a fixed-delta market risk reversal. Term structure uses the nearest
admitted call to forward ATM; each actual strike/moneyness is retained. VTS is
explicitly defined here as the adjacent-expiry slope of those IVs per ACT/365F
year. Different nearest strikes can influence that slope; no universal or
proprietary VTS definition is implied. PCR volume and PCR open interest each mean
sum of admitted put counts divided by sum of admitted call counts for this
snapshot. Any missing count, absent side, or zero call denominator is UNKNOWN.
Authoritative CSM definition is UNSUPPLIED, so CSM is UNAVAILABLE.

The options-research terminal engine exposes the saved artifact in Options,
Research and Models, hypothetical Greek rows in Portfolio, and scenario rows in
Risk. Freshness is measured from the original quote observation, not when research
was computed or persisted: expired policy gives STALE, missing policy gives
UNKNOWN/null terminal data. No model output becomes an approved trading signal.

## Focused evidence

`tests/test_options_workspace.py` verifies a standard BSM numerical reference,
IV inversion and repricing, signed multiplier-aware Greeks, a zero-shock P&L
identity, constant-IV surface interpolation, late corrections across restart and
replay, exact nanosecond knowledge/freshness boundaries, missing/zero PCR,
rejected American or conflicting identities, impossible quote diagnostics,
outside-hull interpolation, invalid research requests, and index tampering.
Broader HTTP/browser, lint/type/native/wheel evidence is recorded by the integrated
Sprint 5 checkpoint; these focused tests alone do not certify the whole platform.
