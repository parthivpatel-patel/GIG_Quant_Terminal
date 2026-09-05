"""Cross-sectional neutralization: residualize scores vs market and sector dummies."""

from __future__ import annotations

import numpy as np
import pandas as pd

# Prefer the C++ kernel when it is importable. The Python path stays as the
# reference and as the fallback when no extension was built.
_CPP_NEUTRALIZE = None
try:
    from gig._speed import neutralize_cs as _CPP_NEUTRALIZE
except Exception:
    _CPP_NEUTRALIZE = None


def neutralize(
    factor: pd.DataFrame,
    sectors: pd.Series,
    tradable: pd.DataFrame | None = None,
    demean: bool = True,
    *,
    use_cpp: bool | None = None,
) -> pd.DataFrame:
    """
    For each date, OLS-residualize the factor on a constant + sector dummies.

    ``use_cpp`` defaults to True when ``gig._speed`` is available. Pass False to
    force the NumPy path (tests that compare against the reference).
    """
    sector_codes = sectors.reindex(factor.columns).fillna("unknown")
    dummy = pd.get_dummies(sector_codes, drop_first=True).astype(float)
    dummy_vals = np.ascontiguousarray(dummy.to_numpy(dtype=float))
    values = np.ascontiguousarray(factor.to_numpy(dtype=float, copy=True))
    mask_all = ~np.isnan(values)
    if tradable is not None:
        mask_all &= tradable.reindex_like(factor).fillna(False).to_numpy()
    # Must stay boolean for the Python path. NumPy reads a uint8 array as
    # integer indices, which silently residualizes columns 0/1 only.
    mask_all = np.ascontiguousarray(mask_all, dtype=bool)

    want_cpp = (_CPP_NEUTRALIZE is not None) if use_cpp is None else bool(use_cpp)
    if want_cpp and _CPP_NEUTRALIZE is not None:
        try:
            arr = _CPP_NEUTRALIZE(
                values,
                dummy_vals,
                np.ascontiguousarray(mask_all.astype(np.uint8)),
                demean,
            )
            return pd.DataFrame(arr, index=factor.index, columns=factor.columns)
        except Exception:
            pass
    return _python_neutralize(values, dummy_vals, mask_all, demean, factor.index, factor.columns)


def _python_neutralize(values, dummy_vals, mask_all, demean, index, columns) -> pd.DataFrame:
    """
    NumPy reference. Writes into a contiguous array and wraps once — assigning
    through ``DataFrame.iloc`` per row was the dominant cost on a wide tape.
    """
    n_dates, n_names = values.shape
    n_sec = dummy_vals.shape[1]
    out = np.full((n_dates, n_names), np.nan, dtype=float)
    ones_cache: dict[int, np.ndarray] = {}

    for i in range(n_dates):
        mask = mask_all[i]
        n_eligible = int(mask.sum())
        if n_eligible < n_sec + 8:
            continue
        y = values[i, mask]
        ones = ones_cache.get(n_eligible)
        if ones is None:
            ones = np.ones(n_eligible, dtype=float)
            ones_cache[n_eligible] = ones
        x = np.column_stack((ones, dummy_vals[mask]))
        try:
            # lstsq is the right default: a rare degenerate sector dummy should
            # not abort the whole cross-section.
            coef, *_ = np.linalg.lstsq(x, y, rcond=None)
        except np.linalg.LinAlgError:
            continue
        resid = y - x @ coef
        if demean:
            resid -= resid.mean()
        out[i, mask] = resid

    return pd.DataFrame(out, index=index, columns=columns)


def cross_sectional_zscore(factor: pd.DataFrame, tradable: pd.DataFrame | None = None) -> pd.DataFrame:
    work = factor.copy()
    if tradable is not None:
        work = work.where(tradable.reindex_like(work).fillna(False))
    mu = work.mean(axis=1)
    sd = work.std(axis=1, ddof=1).replace(0, np.nan)
    return work.sub(mu, axis=0).div(sd, axis=0)
