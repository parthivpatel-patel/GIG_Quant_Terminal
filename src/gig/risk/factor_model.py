"""
Statistical (PCA) factor risk model.

This is the structure every institutional risk system uses:

    Sigma  =  B F B'  +  D

where ``B`` are asset exposures to a small set of common factors, ``F`` is the
factor covariance, and ``D`` is diagonal specific (idiosyncratic) variance. A
commercial model (Barra, Axioma) names its factors from fundamentals and is
licensed. Here the factors are the leading principal components of the return
covariance, which is the standard unlicensed alternative — asymptotic principal
components in the sense of Connor & Korajczyk (1986).

Rotating to orthonormal factors makes ``F`` the identity, so ``B`` carries the
factor volatilities and ``B B'`` is the systematic part of the covariance.

Why this matters more than another alpha signal: a naive equal-weight book can
look diversified by name count while being one concentrated bet in risk space.
Ex-ante vol and the risk decomposition are what surface that.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

TRADING_DAYS = 252


@dataclass(slots=True)
class RiskModel:
    """Fitted decomposition. Exposures are in daily return units."""

    exposures: pd.DataFrame       # N x k, already scaled by factor vol
    specific_var: pd.Series       # N, daily variance
    n_obs: int
    n_factors: int
    explained: float              # share of cross-sectional variance in the factors
    eigenvalues: np.ndarray | None = None  # daily factor variances, length k

    @property
    def symbols(self) -> list[str]:
        return list(self.exposures.index)

    def covariance(self) -> pd.DataFrame:
        """Full model covariance B B' + D, daily units."""
        b = self.exposures.to_numpy(dtype=float)
        cov = b @ b.T
        cov[np.diag_indices_from(cov)] += self.specific_var.to_numpy(dtype=float)
        return pd.DataFrame(cov, index=self.exposures.index, columns=self.exposures.index)

    def _align(self, weights: pd.Series) -> np.ndarray:
        return weights.reindex(self.exposures.index).fillna(0.0).to_numpy(dtype=float)

    def variance(self, weights: pd.Series) -> float:
        """Daily portfolio variance under the model."""
        w = self._align(weights)
        b = self.exposures.to_numpy(dtype=float)
        factor_part = float(np.square(b.T @ w).sum())
        specific_part = float(np.square(w) @ self.specific_var.to_numpy(dtype=float))
        return factor_part + specific_part

    def volatility(self, weights: pd.Series, annualized: bool = True) -> float:
        var = self.variance(weights)
        vol = float(np.sqrt(max(var, 0.0)))
        return vol * np.sqrt(TRADING_DAYS) if annualized else vol

    def decompose(self, weights: pd.Series) -> dict[str, float]:
        """Split portfolio variance into common-factor and specific parts."""
        w = self._align(weights)
        b = self.exposures.to_numpy(dtype=float)
        factor_part = float(np.square(b.T @ w).sum())
        specific_part = float(np.square(w) @ self.specific_var.to_numpy(dtype=float))
        total = factor_part + specific_part
        if total <= 0:
            return {
                "ex_ante_vol": 0.0,
                "systematic_share": float("nan"),
                "specific_share": float("nan"),
            }
        return {
            "ex_ante_vol": float(np.sqrt(total) * np.sqrt(TRADING_DAYS)),
            "systematic_share": factor_part / total,
            "specific_share": specific_part / total,
        }

    def factor_exposure(self, weights: pd.Series) -> pd.Series:
        """Portfolio loading on each principal factor, daily vol units."""
        w = self._align(weights)
        b = self.exposures.to_numpy(dtype=float)
        return pd.Series(b.T @ w, index=self.exposures.columns)

    def marginal_contributions(self, weights: pd.Series) -> pd.Series:
        """
        Contribution of each name to portfolio volatility.

        MCTR_i = w_i * (Sigma w)_i / sigma, which sums exactly to sigma. Names
        with a large share here are the real concentrations, regardless of how
        modest their weight looks.
        """
        w = self._align(weights)
        sigma = np.sqrt(max(self.variance(weights), 0.0))
        if sigma <= 0:
            return pd.Series(0.0, index=self.exposures.index)
        b = self.exposures.to_numpy(dtype=float)
        cov_w = b @ (b.T @ w) + self.specific_var.to_numpy(dtype=float) * w
        return pd.Series(w * cov_w / sigma, index=self.exposures.index)


def fit_statistical_risk_model(
    returns: pd.DataFrame,
    n_factors: int = 5,
    min_obs: int = 120,
    specific_floor: float = 1e-8,
) -> RiskModel | None:
    """
    Fit ``Sigma = B B' + D`` from a date x symbol return panel.

    Names without `min_obs` observations are dropped rather than imputed, since
    filling a short history with zeros would understate their risk.
    """
    if returns is None or returns.empty:
        return None

    clean = returns.replace([np.inf, -np.inf], np.nan)
    keep = clean.notna().sum() >= min_obs
    clean = clean.loc[:, keep[keep].index]
    if clean.shape[1] < 2 or clean.shape[0] < min_obs:
        return None

    # Demean per name, then zero-fill the remaining gaps. After demeaning a gap
    # contributes no deviation, which is the neutral choice for a covariance.
    demeaned = clean - clean.mean()
    x = demeaned.fillna(0.0).to_numpy(dtype=float)
    t, n = x.shape
    if t < 2:
        return None

    k = int(max(1, min(n_factors, n - 1, t - 1)))

    # Thin SVD of the T x N return matrix rather than an eigendecomposition of
    # the N x N covariance. The sample covariance has rank at most T, so on a
    # wide panel -- 252 sessions against several thousand US names, the normal
    # case -- forming that matrix costs an N^2 allocation and diagonalizing it
    # costs O(N^3) to recover a basis of rank T. The singular vectors span the
    # same space; this is the standard route to asymptotic principal
    # components and it turns minutes per refit into about a second.
    _u, s, vt = np.linalg.svd(x, full_matrices=False)
    eigvals = np.clip(np.square(s[:k]) / (t - 1), 0.0, None)

    # Orthonormal factors: fold the factor vol into the exposures so F = I.
    b = vt[:k].T * np.sqrt(eigvals)
    systematic_var = np.square(b).sum(axis=1)
    total_var = np.square(x).sum(axis=0) / (t - 1)
    specific = np.clip(total_var - systematic_var, specific_floor, None)

    trace = float(total_var.sum())
    explained = float(eigvals.sum() / trace) if trace > 0 else float("nan")

    columns = [f"pc{i + 1}" for i in range(k)]
    return RiskModel(
        exposures=pd.DataFrame(b, index=clean.columns, columns=columns),
        specific_var=pd.Series(specific, index=clean.columns),
        n_obs=int(t),
        n_factors=k,
        explained=explained,
        eigenvalues=eigvals,
    )


def risk_report(
    weights: pd.Series,
    returns: pd.DataFrame,
    sectors: pd.Series | None = None,
    n_factors: int = 5,
    top_n: int = 8,
) -> dict:
    """
    Ex-ante risk summary for a book: predicted vol, systematic vs specific
    split, the largest volatility contributors, and realized vol of the same
    weights over the estimation window for comparison.
    """
    model = fit_statistical_risk_model(returns, n_factors=n_factors)
    if model is None:
        return {"available": False}

    held = weights[weights.abs() > 0]
    decomposition = model.decompose(weights)
    mctr = model.marginal_contributions(weights)
    sigma_daily = np.sqrt(max(model.variance(weights), 0.0))

    contributors = []
    if sigma_daily > 0:
        share = (mctr / sigma_daily).reindex(held.index).dropna()
        for symbol, value in share.abs().sort_values(ascending=False).head(top_n).items():
            contributors.append(
                {
                    "symbol": str(symbol),
                    "share": float(share.loc[symbol]),
                    "weight": float(weights.get(symbol, 0.0)),
                    "sector": str(sectors.get(symbol, "unknown")) if sectors is not None else "unknown",
                    "_abs": float(value),
                }
            )
        contributors = [{k: v for k, v in c.items() if k != "_abs"} for c in contributors]

    aligned = returns.reindex(columns=weights.index).fillna(0.0)
    book = aligned.mul(weights.reindex(aligned.columns).fillna(0.0), axis=1).sum(axis=1)
    realized = float(book.std(ddof=1) * np.sqrt(TRADING_DAYS)) if len(book) > 2 else float("nan")

    eigs = (
        [float(x) for x in np.asarray(model.eigenvalues).tolist()]
        if model.eigenvalues is not None
        else []
    )
    eig_sum = float(sum(eigs)) or 1.0
    book_exposure = model.factor_exposure(weights)

    return {
        "available": True,
        "n_obs": model.n_obs,
        "n_factors": model.n_factors,
        "explained": model.explained,
        "ex_ante_vol": decomposition["ex_ante_vol"],
        "realized_vol": realized,
        "systematic_share": decomposition["systematic_share"],
        "specific_share": decomposition["specific_share"],
        "n_held": int(len(held)),
        "effective_names": float(
            1.0 / np.square(held.abs() / held.abs().sum()).sum()
        )
        if len(held)
        else 0.0,
        "top_contributors": contributors,
        "eigenvalues": eigs,
        "eigenvalue_share": [float(x / eig_sum) for x in eigs],
        "factor_exposure": {str(k): float(v) for k, v in book_exposure.items()},
        "equation": "Σ = BB′ + D",
    }
