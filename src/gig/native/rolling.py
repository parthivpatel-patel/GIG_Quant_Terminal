"""Rolling sample statistics with a tested NumPy reference and optional C ABI.

NaN marks the first ``window-1`` outputs (insufficient observations). Inputs
must be finite; missing observations are never silently dropped. The native
library is loaded only from this package's ``_rolling.so`` path.
"""

from __future__ import annotations

import ctypes
from functools import lru_cache
from pathlib import Path

import numpy as np


@lru_cache(maxsize=1)
def _load():
    path = Path(__file__).with_name("_rolling.so")
    if not path.exists():
        return None
    lib = ctypes.CDLL(str(path))
    double_ptr = ctypes.POINTER(ctypes.c_double)
    lib.gig_rolling_stats.argtypes = [double_ptr, ctypes.c_size_t, ctypes.c_size_t,
                                    ctypes.c_size_t, double_ptr, double_ptr]
    lib.gig_rolling_stats.restype = ctypes.c_int
    return lib


def available() -> bool:
    try:
        return _load() is not None
    except (OSError, AttributeError):
        return False


def rolling_stats(values, window: int, *, ddof: int = 1, backend: str = "numpy"):
    """Return mean/std arrays of input length; explicit backend prevents substitution.

    Native mode uses extended precision Welford updates/removals and a stable
    rebase every 1024 observations. NumPy mode is the independent two-pass
    reference, using bounded batches to cap temporary memory at ~16 MiB.
    """
    data = np.array(values, dtype=np.float64, order="C", copy=True)
    if data.ndim != 1 or not len(data):
        raise ValueError("values must be a nonempty one-dimensional vector")
    if type(window) is not int or not 1 <= window <= len(data):
        raise ValueError("window must be an integer between 1 and input length")
    if type(ddof) is not int or not 0 <= ddof < window:
        raise ValueError("ddof must be an integer in [0, window)")
    if not np.isfinite(data).all() or np.max(np.abs(data)) > 1e150:
        raise ValueError("values must be finite and have magnitude <= 1e150")
    means = np.full(len(data), np.nan)
    deviations = np.full(len(data), np.nan)
    if backend == "native":
        lib = _load()
        if lib is None:
            raise RuntimeError("C++20 rolling library unavailable; run scripts/build_native.py")
        ptr = ctypes.POINTER(ctypes.c_double)
        result = lib.gig_rolling_stats(data.ctypes.data_as(ptr), len(data), window, ddof,
                                      means.ctypes.data_as(ptr), deviations.ctypes.data_as(ptr))
        if result:
            raise ValueError(f"native rolling statistics failed (code {result})")
    elif backend == "numpy":
        windows = np.lib.stride_tricks.sliding_window_view(data, window)
        batch = max(1, 2_000_000 // window)
        for start in range(0, len(windows), batch):
            rows = windows[start:start+batch]
            target = slice(start+window-1, start+window-1+len(rows))
            means[target] = rows.mean(axis=1)
            deviations[target] = rows.std(axis=1, ddof=ddof)
    else:
        raise ValueError("backend must be numpy or native")
    return means, deviations
