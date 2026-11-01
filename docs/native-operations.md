# Native numerical kernels and local operations

## C++20 rolling statistics

`gig.native.rolling_stats(values, window, ddof=1, backend="numpy")` computes
rolling means and standard deviations. The first `window-1` positions are NaN
because their sample is incomplete; this is an internal numerical array, not an
API wire payload. Input must be finite, one dimensional and bounded by 1e150.
`ddof` is explicit; `window=1` requires `ddof=0`. Inputs are defensively copied.

Run `python scripts/build_native.py` from the source checkout to build the
optional C ABI library using a C++20 compiler. No pybind11 dependency is needed
for this kernel. Then request `backend="native"`; unavailable native code raises
an explicit error. Importing the package never invokes a compiler. The library
loads only from `gig/native/_rolling.so`; an environment path cannot inject a
different shared library. A Python process already running before build must be
restarted because library availability is cached.

The kernel uses extended-precision Welford addition/removal and periodically
rebases its accumulated moments. It writes to caller-owned buffers with no heap
allocations inside the kernel. Its wrapper allocates a defensive input copy and
two result arrays. The independent NumPy two-pass window reference batches its
temporary work to approximately 16 MiB. Both paths reject unsupported inputs;
there is no silent backend replacement.

Reproduce with:

```sh
python scripts/build_native.py
PYTHONPATH=src python -m unittest discover -s tests -p test_native_operations.py -v
PYTHONPATH=src python scripts/benchmark_native.py --output work/native-benchmark.json
```

The initial measured x86_64 run used 200,000 deterministic normal samples,
window 252, and 31 repetitions including wrapper validation/copies. The maximum
absolute native/reference error was 6.94e-18. Native p50/p95/p99 latency was
4.424/5.387/6.932 ms, versus NumPy 83.940/89.175/90.598 ms. The measured p50 ratio
was 18.97 for this workload. Native throughput was 45.2 million samples/second.
Tracked peak allocations were 4.80 MB versus 20.99 MB. Whole-process RSS reached
71.65 MB. Tracemalloc does not cover every native allocation, and RSS includes
both implementations and warmup. No general platform speedup is implied. Full
raw timings are in the batch evidence, not derived from a single timing.

Only x86_64 Linux was exercised. C++20 has no architecture-specific intrinsics,
but ARM64 compilation, numerical behavior and performance remain unverified.
The pybind sector-neutralization extension requests C++20 in setup.py and its
verified repair is described below. The historical native-fit-cache issue is
absent from the available source; no historical cache verification is claimed.

A fresh resumed build and 31-repeat rerun confirmed the same maximum absolute
error, 6.94e-18, at 200,000 samples/window 252. Native p50/p95/p99 were
4.301/4.815/4.937 ms; NumPy was 77.558/84.087/89.010 ms, an 18.03 p50 ratio for
this workload. Native throughput was 46.50 million samples/second. Tracked peak
allocation remained 4.80 MB vs 20.99 MB. Raw evidence is
`native-benchmark-resumed.json`; these are local measurements, not ARM64 evidence.

## C++20 categorical sector projection

The available source uses a constant plus `drop_first=True` sector dummies.
The full design is identifiable. However, tradable masking can remove a sector
or the baseline group and make the remaining design rank deficient. Actual
compiled pre-repair evidence reproduced two failures: ten eligible observations
produced ten NumPy residuals but no native residuals due to an off-by-one sample
threshold; excluding the baseline group produced twenty NumPy residuals but no
native residuals due to singular normal equations. These are current source
reproductions, not claims about unavailable historical checkpoints.

The repaired kernel projects onto categorical group means directly. This is
exactly the OLS fitted-value subspace for an intercept plus mutually exclusive
sector dummies, including absent groups. It avoids forming or inverting normal
equations. The public native design is explicitly categorical: zero/one entries
with at most one nonzero per row. Arbitrary continuous or overlapping designs
are rejected rather than mislabeled as sector projection. The NumPy reference
continues to use `numpy.linalg.lstsq`, independently exercising the projection.

Both backends exclude NaN and infinities, use the same `n_sector_dummies + 8`
minimum eligible count, preserve labels and masks, and never mutate inputs.
Native centered long-double accumulation avoids overflowing representable group
means; residual overflow fails explicitly. Incomplete cross-sections remain NaN
internally and must not be presented as finite observed values. Explicit
`use_cpp=True` requires native availability and propagates native failures;
default automatic selection retains the documented Python fallback.

Build with an installed compiler and the project's pybind development extra:

```sh
CXX=g++ CC=gcc python setup.py build_ext --inplace --force
PYTHONPATH=src python -m unittest discover -s tests -p test_native_neutralize.py -v
```

The executor's default compiler name `clang++` was absent, while `/usr/bin/g++`
was available; the explicit compiler selection above was used. Importing the
package never builds native code. The numerical suite covers rank-deficient
masked designs, exact sample thresholds, missing sectors, finite/NaN masks,
shape/design admission, mutation, extreme finite constants and output overflow.
Native tests explicitly skip if the optional extension is not built; the
reported run built it and exercised every native test.

An x86_64 end-to-end benchmark used seed 176, 252 dates, 1,000 names, ten sector
groups, approximately 85% tradability, one BLAS thread, and 31 repetitions per
backend. It includes pandas admission and output construction. Maximum absolute
OLS/native residual error was `1.7763568394002505e-15`. Native p50/p95/p99 were
4.747/5.931/6.501 ms, versus NumPy 39.148/41.614/44.567 ms: a measured p50 ratio
of 8.247 for this workload. Python-tracked peak allocation was 4,692,671 versus
5,014,085 bytes; this excludes some native vector allocations and is not RSS.
Raw timings are preserved in `native-neutralize-benchmark.json` alongside the
pre-repair evidence and build/test logs. ARM64 remains unverified.

A minimal reproduction uses `np.random.default_rng(176)`, a `(252,1000)` normal
panel, sector labels `S{i % 10}`, and a following uniform draw `> .15` as the
tradable mask. Call `neutralize(panel,sectors,mask,use_cpp=False)` and `True`
once for warmup and 31 timed repetitions each with `OPENBLAS_NUM_THREADS=1`;
compare the complete arrays with `equal_nan=True`. Benchmarking is evidence of
this computation, not approved trading use or institutional certification.

## Local deployment recipes

The Dockerfile installs the `gig` package with web dependencies, runs as uid
10001, checks `/healthz`, and launches port 8000. Compose publishes it only on
127.0.0.1, drops Linux capabilities, sets a read-only root filesystem, mounts
separate durable data/results volumes, and keeps broker connectivity disabled.
No .env file or inherited broker keys are mounted by the recipe. A container
port bind is not authentication; do not change its audience without approval.

`deploy/gig-terminal.service` is an installation template for Ubuntu 24.04 with
a dedicated `gig` user, `/opt/gig/.venv`, and `/var/lib/gig` state. Create data
and results directories with ownership `gig:gig` before starting. Configure
dependency installation in the isolated virtual environment first. Its startup
diagnostic rejects a broker-enabled environment or missing core dependencies.
It binds loopback and restarts failed processes. No service was installed or
started, no container was built/run, and no production deployment was performed.
These recipes are implemented, not operationally validated on Ubuntu or A1.

Run diagnostics without touching account credentials:

```sh
PYTHONPATH=src python -m gig.ops.maintenance check
```

The report distinguishes dependency readiness from `data_readiness=UNKNOWN`.
Passing it does not establish market-data freshness, economic performance,
broker authorization, or trading readiness. `/healthz` is a liveness probe.

## Backup and restore

Stop all writers first. The tool cannot independently prove quiescence. For
SQLite or DuckDB state, include the entire stopped service state directory so
any required journals are retained. Do not back up a live database by copying
only its main file. Do not run ingestion during this process.

```sh
PYTHONPATH=src python -m gig.ops.maintenance backup /var/lib/gig /tmp/gig-state.tar.gz --quiesced
PYTHONPATH=src python -m gig.ops.maintenance verify /tmp/gig-state.tar.gz
PYTHONPATH=src python -m gig.ops.maintenance restore /tmp/gig-state.tar.gz /tmp/gig-restored
```

Archives include a per-file size/SHA-256 manifest and are verified after
creation. Backup rejects symlinks, limits total uncompressed data to 512 MiB and
10,000 state files, and excludes dotfiles and names containing secret,
credential, token, or private_key. Naming filters are defense in depth, not a
content classifier: keep credentials outside state directories. Restore checks
all archive paths and digests before creating a new destination; absolute
paths, parent traversal, duplicate names and nonregular members are rejected.
It never overwrites existing directories. Run it only with exclusive access to
the destination's parent directory (no untrusted concurrent filesystem actors).

Copy the archive and separately record its reported SHA-256 at an authorized
independent destination, then recompute the checksum there. A local archive,
even verified, is not a durable external backup. Restore tests cover roundtrip,
secret-name exclusion, refusal to overwrite, corrupt digests, traversal, and
symlink rejection. Hardware failure and actual production recovery are not
validated by those tests.
