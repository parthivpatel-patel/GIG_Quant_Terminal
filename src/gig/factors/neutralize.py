"""Cross-sectional neutralization: residualize scores vs market and sector dummies."""

from __future__ import annotations

import numpy as np
import pandas as pd


def neutralize(
    factor: pd.DataFrame,
    sectors: pd.Series,
    tradable: pd.DataFrame | None = None,
    demean: bool = True,
    *,
    use_cpp: bool = False,
) -> pd.DataFrame:
    """
    For each date, OLS-residualize the factor on a constant + sector dummies.

    Uses `gig._speed` (C++) when the extension is built; otherwise NumPy.
    """
    sector_codes = sectors.reindex(factor.columns).fillna("unknown")
    dummy = pd.get_dummies(sector_codes, drop_first=True).astype(float)
    dummy_vals = np.ascontiguousarray(dummy.to_numpy(dtype=float))
    values = np.ascontiguousarray(factor.to_numpy(dtype=float, copy=True))
    mask_all = ~np.isnan(values)
    if tradable is not None:
        mask_all &= tradable.reindex_like(factor).fillna(False).to_numpy()
    mask_all = np.ascontiguousarray(mask_all.astype(np.uint8))

    if use_cpp:
        try:
            from gig._speed import neutralize_cs

            arr = neutralize_cs(values, dummy_vals, mask_all, demean)
            return pd.DataFrame(arr, index=factor.index, columns=factor.columns)
        except Exception:
            pass
    return _python_neutralize(values, dummy_vals, mask_all, demean, factor.index, factor.columns)


def _python_neutralize(values, dummy_vals, mask_all, demean, index, columns) -> pd.DataFrame:
    n_sec = dummy_vals.shape[1]
    out = pd.DataFrame(np.nan, index=index, columns=columns)
    for i in range(values.shape[0]):
        mask = mask_all[i]
        if mask.sum() < n_sec + 8:
            continue
        y = values[i, mask]
        x = np.column_stack([np.ones(mask.sum()), dummy_vals[mask]])
        try:
            coef, *_ = np.linalg.lstsq(x, y, rcond=None)
        except np.linalg.LinAlgError:
            continue
        resid = y - x @ coef
        if demean:
            resid = resid - resid.mean()
        row = out.iloc[i].to_numpy(copy=True)
        row[mask] = resid
        out.iloc[i] = row
    return out


def cross_sectional_zscore(factor: pd.DataFrame, tradable: pd.DataFrame | None = None) -> pd.DataFrame:
    work = factor.copy()
    if tradable is not None:
        work = work.where(tradable.reindex_like(work).fillna(False))
    mu = work.mean(axis=1)
    sd = work.std(axis=1, ddof=1).replace(0, np.nan)
    return work.sub(mu, axis=0).div(sd, axis=0)
