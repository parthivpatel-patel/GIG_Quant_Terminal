"""Reproducible local C++/NumPy comparison, including validation and copies.

tracemalloc covers Python/NumPy tracked allocations, not every native allocator.
ru_maxrss is the process high-water mark, not a per-kernel delta. C++ rolling
kernel itself makes no heap allocations; the wrapper owns input/output arrays.
"""
import argparse
import json
import platform
import resource
import time
import tracemalloc
from pathlib import Path

import numpy as np

from gig.native import rolling_stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--samples", type=int, default=200000)
    parser.add_argument("--repeats", type=int, default=31)
    args = parser.parse_args()
    if args.samples < 252 or args.repeats < 3:
        parser.error("require samples >=252 and repeats >=3")
    data = np.random.default_rng(107).normal(0, 0.02, args.samples)
    reference = rolling_stats(data, 252)
    actual = rolling_stats(data, 252, backend="native")
    max_error = max(float(np.nanmax(np.abs(a-b))) for a, b in zip(reference, actual, strict=True))
    for a, b in zip(reference, actual, strict=True):
        np.testing.assert_allclose(a, b, rtol=1e-10, atol=1e-12, equal_nan=True)
    report = {"seed": 107, "samples": args.samples, "window": 252,
              "repeats": args.repeats, "architecture": platform.machine(),
              "platform": platform.platform(), "numpy": np.__version__,
              "max_absolute_parity_error": max_error, "backends": {},
              "limitations": ["ARM64 not exercised on this host",
                 "allocation tracing excludes untracked native allocator activity",
                 "RSS high-water mark includes both backends and warmup",
                 "single-thread CPU benchmark; no economic-performance inference"]}
    for backend in ["numpy", "native"]:
        rolling_stats(data, 252, backend=backend)
        timings = []
        for _ in range(args.repeats):
            start = time.perf_counter_ns()
            rolling_stats(data, 252, backend=backend)
            timings.append((time.perf_counter_ns()-start)/1e6)
        tracemalloc.start()
        rolling_stats(data, 252, backend=backend)
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        report["backends"][backend] = {
            "p50_ms": float(np.percentile(timings, 50)),
            "p95_ms": float(np.percentile(timings, 95)),
            "p99_ms": float(np.percentile(timings, 99)),
            "throughput_samples_per_second": args.samples / (np.median(timings)/1000),
            "tracked_peak_bytes": peak,
            "kernel_heap_allocations": 0 if backend == "native" else "unmeasured",
            "raw_latency_ms": timings}
    report["process_peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    report["p50_speed_ratio_numpy_over_native"] = (
        report["backends"]["numpy"]["p50_ms"] / report["backends"]["native"]["p50_ms"])
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
