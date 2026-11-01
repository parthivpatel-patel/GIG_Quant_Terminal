# Point-in-time data and reproducible research

The foundation admits caller-provided data offline. It does not contact a
provider, certify a data license, repair corporate actions, or approve trading.

## Data admission and revision storage

`gig.data.validated.admit_bars(records, source=..., classification=...,
corporate_action_policy=..., currency="USD")` returns a frozen `DataSnapshot`.
Every row requires `instrument_id`, closing-observation `timestamp`, revision
knowledge time `available_at`, positive finite OHLC prices, and nonnegative
signed-64-bit integer `volume`. Prices must satisfy the bar's high/low bounds.
An available time preceding the observation is rejected. Classification and
RAW / SPLIT_ADJUSTED / TOTAL_RETURN_ADJUSTED treatment are caller declarations;
the module performs no corporate-action adjustment or security-master mapping.

Aware timestamps normalize to UTC with nanoseconds retained. More than nine
fractional digits are rejected before pandas can truncate them. Identity-bearing
volume uses an integer string in the persisted JSON; it is never cast through
binary64. All original noncanonical provider evidence belongs in a separate
source archive. This normalized bar contract rejects extra fields explicitly.

Canonical bytes and SHA-256 identify each snapshot. `snapshot.rows` returns a
fresh decoded structure, so mutation cannot alter admitted evidence.
`snapshot.save(directory)` publishes a complete synced file atomically; identical
concurrent writers replay safely, and conflicting existing bytes are preserved
and rejected. `DataSnapshot.load(path)` verifies the content-addressed filename.
A digest establishes integrity relative to recorded bytes, not external source
authenticity. Local files alone are not a durable external backup.

`PointInTimeStore(path).ingest(snapshot)` revalidates admission and writes each
batch transactionally. Original snapshot payloads are retained as row-integrity
anchors. Same-identity corrections require a later knowledge timestamp; exact
replay returns a replay count. `snapshot(as_of, source=None,
max_age_seconds=None)` selects only observations and revisions knowable at the
cutoff and verifies stored rows and identity columns against their source anchor.
Old stores lacking an original anchor fail closed until original admitted
snapshots are reingested. No legacy rows are silently blessed during upgrade.

Freshness is assessed separately for each source, classification, adjustment,
currency, and instrument. It uses the most recent observation in each series,
not the oldest historical bar returned. No policy means UNKNOWN; no rows means
UNAVAILABLE; an exceeded explicit policy means STALE. This is an elapsed-time
policy, not a market-session calendar. Restart and idempotent replay are tested.

## Purged research workflow

`ResearchWorkflow(output_dir).run(features, target, label_end=...,
feature_available_at=..., model="ridge", min_train=60, test_size=20,
embargo=1)` provides expanding chronological folds. Features require a unique,
increasing aware DatetimeIndex and finite values. Label ends cannot precede
feature observations. A label enters training only if its end precedes the
first test timestamp. Feature availability later than its observation is
rejected; omitted availability records a caller assumption explicitly.

Feature means/scales are fitted on the purged training rows only. Supported
models are an explicitly chosen ridge baseline and actual optional LightGBM.
An unavailable requested model raises; it never becomes a different model.
Artifacts record dataset/configuration hashes, fold membership, model identities,
train-only normalization, out-of-sample predictions, MSE, and R-squared (null for
constant targets). `ModelRegistry(directory).predict(model_id, features)` checks
the model digest and exact feature names/order before inference. Registry states
use JSON or native LightGBM text, never executable pickle deserialization.

Bounds are 20,000 rows, 128 features, and 100 folds. Artifact publication is
atomic and immutable. Hashes identify exact numerical inputs/configuration;
cross-platform floating point or library-version changes can produce different
model bytes. Training residual quantiles are explicitly NOT_OOS_CALIBRATED and
are not conformal prediction intervals. No transaction-cost-adjusted returns,
economic significance, survivorship protection, or independently verified
point-in-time universe is implied by a successful run.

`drift_report(reference, current)` computes reference-quantile PSI with declared
Jeffreys pseudocounts and standardized mean shift. Constant reference variance
has UNKNOWN mean-shift state. This is a diagnostic, not an automatic decision.

`model_capabilities()` separates implemented, dependency, loaded/trained, and
trading-approved states. A local-only FinBERT adapter exists but no checkpoint
is loaded by inventory. BiLSTM has no sequence-training adapter in this
workflow and remains explicitly unavailable. This execution environment lacks
Torch, Transformers, and LightGBM; their real training/inference was not tested.

## Verification

`PYTHONPATH=src python -m unittest discover -s tests -p test_research_foundation.py -v`
covers mathematical ridge references, label-end purging, train-only scaling,
registry parity, exact timestamps/integers, defensive copies, revision replay,
transaction rollback, provenance-specific freshness, source-anchor corruption,
and atomic concurrent artifact publication. State-space research integration is
documented separately in `models-phase7.md`.
