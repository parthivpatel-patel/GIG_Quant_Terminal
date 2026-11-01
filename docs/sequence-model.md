# CPU sequence research

`gig.ml.sequence` trains an actual one-layer bidirectional PyTorch LSTM and a
linear output head. Both directions operate inside the same historical window
ending at the decision timestamp. No later observation enters that prediction.
Missing PyTorch raises an error; there is no substitute model or generated
prediction fallback. Install the `sequence` extra for the Torch dependency.
For a CPU installation, install Torch from its official CPU wheel index first:

```sh
python -m pip install 'torch>=2.2' --index-url https://download.pytorch.org/whl/cpu
python -m pip install '.[sequence]'
```

The CI workflow uses that CPU installation explicitly. No pretrained weights, external data download, account,
or broker connection is used by this pipeline.

## Admission and numerical execution

`train_sequence(payload)` requires `source`, `classification`, and 24–1,000
`rows`. Each row contains exactly `timestamp`, `available_at`, `features`,
`target`, and `label_end`. `features` is a map of 1–16 stable feature names to
finite numbers. Timestamps must include a timezone, retain nanosecond precision,
and increase strictly. Feature knowledge cannot follow the decision timestamp;
labels cannot end before it. All labels must be known at the explicit `as_of`
research cutoff, which defaults to the maximum label end. Classification is
caller supplied and preserved: `SYNTHETIC`, `HISTORICAL`, `DELAYED`, `SNAPSHOT`, or
`REAL_TIME`. Admission is not independent verification of the source.

Optional settings are `sequence_length` (2–32, default 8), `train_size` (prefix
size, default 75%), `hidden_size` (1–32, default 8), `epochs` (1–100, default 12),
`learning_rate` (0.00001–0.1, default 0.01), and `seed` (default 42). Unknown
fields, booleans in numeric fields, nonfinite values, oversized serialized
requests, and requests exceeding the recurrent work budget are rejected.
The work admission expression counts both directions and four gates:
`8 * epochs * training_windows * sequence_length * (hidden_size + features) * hidden_size`.
Its 100 million unit ceiling is a workload guard, not a measured FLOP count or
wall-clock guarantee. The implementation rejects excessive work rather than
silently reducing epochs or model size.

Training uses the chronological prefix, purging any training window whose label
end reaches the first holdout decision. Feature mean/scale use only admitted
training-window observations, and target normalization uses only admitted
training labels. Finite inputs that overflow normalization statistics are
rejected before Torch loads. Zero variance uses unit scale. Inputs, parameters,
losses, predictions, and evaluation metrics must remain finite and compatible
with float32 execution. Adam runs on one CPU thread with gradient clipping.
The shared `gig.ml.runtime.TORCH_CPU_LOCK` protects temporary Torch settings
across sequence and NLP execution. RNG state, CPU thread count, deterministic
algorithm settings (including warning mode), and MKLDNN settings are restored
on success or failure.

## Persistence and terminal contract

`SequenceRegistry` stores content-addressed JSON model tensors and run artifacts
under `models/<model_id>.json` and `runs/<run_id>.json`. Loading verifies SHA-256
identity, tensor names, shapes, finite values, and model/run agreement. No pickle
or executable model serialization is used. Model identity includes training
data/provenance, normalization, configuration, Torch version, and trained
parameters. Run identity additionally includes holdout data and results.
Volatile elapsed-time measurements do not alter either identity. Deterministic
identity is verified in the tested CPU/library environment; cross-version or
cross-hardware bitwise equality is not promised.

`SequenceWorkspace(base_dir)` exposes `train(payload)`, `predict(payload)`, and
`section(section, as_of, max_age_seconds)`. Successful training admits the final
typed contract before publishing immutable registry/response files and a
synchronous SQLite run index. Repeated identical training preserves the same
response and identities. Restarted workspaces reload and verify those files.
Read-only missing-state requests do not create directories. Callers cannot
supply filesystem paths, and workspace paths reject symbolic links.

Prediction requires exactly `model_id` and `rows`; inference rows contain
`timestamp`, `available_at`, and `features`. The service excludes decisions
before the stored training cutoff, reports the excluded count, and emits
`UNAVAILABLE` when no out-of-sample decisions remain. The lower-level registry
prediction method also supports retrospective comparisons for numerical tests.
Inference is classified `MODEL_ESTIMATE`; training retains its source's input
classification. Both keep `trading_approved=false` and
`economically_validated=false`.

Models, Signals, and Research sections expose compact IDs, metrics, training
losses, and out-of-sample rows without returning model tensor weights. Section
selection includes only runs known by `as_of`. Missing state returns no section;
unknown freshness produces `UNKNOWN` with null data; an expired explicit age
policy produces `STALE` while preserving the original observation timestamp.
Responses retain model version, dataset identity, and `RESEARCH_ONLY` readiness.
They cannot authorize trades.

## Validation and limits

The focused sequence/model workspace suite verifies real CPU optimization,
finite held-out results, label purging, training-only normalization, future
mutation isolation, deterministic retraining, reload prediction parity, global
runtime restoration, work/variance admission, restart replay, freshness,
corruption rejection, path safety, and final serialization admission. Test
fixtures are explicitly `SYNTHETIC`; their loss improvement is not evidence of
financial forecasting value. The pipeline currently provides one purged prefix
holdout, not a complete walk-forward evaluation, transaction-cost study,
cross-validation campaign, or economic validation. Its outputs remain research
model estimates, and T4 certification is unchanged.
