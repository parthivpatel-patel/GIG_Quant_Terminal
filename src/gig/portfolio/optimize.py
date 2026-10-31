"""
Risk-constrained portfolio construction.

The equal-weight quantile book in ``construct.py`` is dollar-neutral by *count*,
not by *risk*. Measured against a fitted factor model it carried ~94% of its
variance in common-factor exposure: one leveraged macro bet wearing a
market-neutral label, which is how a book with 40 names on each side still ran
at 40% annualized vol. This module replaces the weighting rule and targets a
stated ex-ante volatility instead of a stated gross.

Given alpha ``a``, model covariance ``Sigma = B B' + D``, and a constraint
matrix ``C`` whose columns are the exposures the book refuses to bet on (cash,
sectors, and the statistical factors), the problem is

    max_w   a'w - (lambda / 2) w' Sigma w      s.t.   C'w = 0

On the constraint set ``w' Sigma w == w' D w`` whenever ``C`` spans the columns
of ``B``, so the first-order conditions have a closed form:

    w  =  (1 / lambda) * D^-1 ( a - C (C' D^-1 C)^-1 C' D^-1 a )

That inner term is the GLS residual of alpha on the unwanted exposures; the
``D^-1`` in front is the inverse-specific-variance tilt that mean-variance
implies. Two properties matter operationally: neutrality holds to machine
precision rather than to a solver tolerance, and there is no iteration to fail
to converge on a bad cross-section mid-backtest.

``lambda`` is not a free parameter. Scaling ``w`` moves ex-ante vol
proportionally, so it is pinned by the volatility target and then capped by
gross leverage. Box limits (``|w_i| <= max_name``) admit no closed form, so they
are imposed by alternating clip and re-projection: each projection is a linear
operator onto the null space of ``C'`` and each clip is a contraction, which
settles in a handful of passes.

References: Grinold & Kahn, *Active Portfolio Management*, ch. 5 (the
characteristic portfolio and its constrained form); Clarke, de Silva & Thorley
(2002) on the portfolio implications of a linear factor risk model.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from gig.risk.factor_model import TRADING_DAYS, RiskModel

_VAR_FLOOR = 1e-10


@dataclass(frozen=True, slots=True)
class OptimizerConfig:
    """
    Risk policy for book construction.

    ``target_vol`` is annualized ex-ante volatility under the fitted model, not
    a realized figure and not a promise. It is a *ceiling reached from below*:
    when the gross or name caps bind first the book runs quieter than target,
    which is reported rather than corrected by adding leverage.
    """

    target_vol: float = 0.10
    max_gross: float = 2.0
    max_name: float = 0.05
    neutralize_cash: bool = True
    neutralize_sectors: bool = True
    neutralize_factors: bool = True
    alpha_clip: float = 3.0
    # Floor on specific variance used for weighting, as a fraction of the
    # cross-sectional median. Relative to the median rather than a percentile
    # of the distribution: a percentile floor fails exactly when it is needed,
    # because if more than `p%` of names are degenerate the percentile itself
    # is a degenerate value.
    specific_var_floor_ratio: float = 0.05
    max_iter: int = 24
    tol: float = 1e-9

    def as_dict(self) -> dict[str, float | bool]:
        return {
            "target_vol": self.target_vol,
            "max_gross": self.max_gross,
            "max_name": self.max_name,
            "neutralize_cash": self.neutralize_cash,
            "neutralize_sectors": self.neutralize_sectors,
            "neutralize_factors": self.neutralize_factors,
        }


@dataclass(slots=True)
class OptimizedBook:
    """Weights plus the ex-ante numbers that justify them."""

    weights: pd.Series
    ex_ante_vol: float
    systematic_share: float
    specific_share: float
    gross: float
    net: float
    max_name: float
    n_held: int
    effective_names: float
    factor_exposure: pd.Series
    iterations: int
    binding: str
    available: bool = True

    def diagnostics(self) -> dict[str, float | str | bool]:
        return {
            "available": self.available,
            "ex_ante_vol": self.ex_ante_vol,
            "systematic_share": self.systematic_share,
            "specific_share": self.specific_share,
            "gross": self.gross,
            "net": self.net,
            "max_name": self.max_name,
            "n_held": self.n_held,
            "effective_names": self.effective_names,
            "worst_factor_exposure": (
                float(self.factor_exposure.abs().max()) if len(self.factor_exposure) else 0.0
            ),
            "iterations": self.iterations,
            "binding": self.binding,
        }


def _empty_book(index: pd.Index) -> OptimizedBook:
    return OptimizedBook(
        weights=pd.Series(0.0, index=index),
        ex_ante_vol=0.0,
        systematic_share=float("nan"),
        specific_share=float("nan"),
        gross=0.0,
        net=0.0,
        max_name=0.0,
        n_held=0,
        effective_names=0.0,
        factor_exposure=pd.Series(dtype=float),
        iterations=0,
        binding="infeasible",
        available=False,
    )


def _neutralization_matrix(
    symbols: list[str],
    model: RiskModel,
    sectors: pd.Series | None,
    config: OptimizerConfig,
) -> tuple[np.ndarray | None, list[str]]:
    """
    Columns of exposure the book is not allowed to carry.

    The cash column and a full set of sector dummies are collinear, so one
    sector is dropped when cash is neutralized; the remaining columns still span
    the same space. A sector holding a single name yields a dummy that forces
    that name to zero, which is the correct outcome -- a one-name sector cannot
    be held sector-neutral.
    """
    n = len(symbols)
    columns: list[np.ndarray] = []
    names: list[str] = []

    if config.neutralize_cash:
        columns.append(np.ones(n))
        names.append("cash")

    if config.neutralize_sectors and sectors is not None and len(sectors):
        aligned = sectors.reindex(symbols).fillna("unknown").astype(str).to_numpy()
        groups = sorted(set(aligned.tolist()))
        if config.neutralize_cash and len(groups) > 1:
            groups = groups[1:]
        for group in groups:
            columns.append((aligned == group).astype(float))
            names.append(f"sector:{group}")

    if config.neutralize_factors:
        exposures = model.exposures.reindex(symbols).fillna(0.0)
        for column in exposures.columns:
            columns.append(exposures[column].to_numpy(dtype=float))
            names.append(str(column))

    if not columns:
        return None, []
    return np.column_stack(columns), names


def _solve(gram: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """Least-squares solve, tolerant of a rank-deficient constraint set."""
    try:
        return np.linalg.solve(gram, rhs)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(gram, rhs, rcond=None)[0]


def _gls_residual(alpha: np.ndarray, c: np.ndarray, d_inv: np.ndarray) -> np.ndarray:
    """Residual of alpha on the constraint columns, in the D^-1 metric."""
    ct_dinv = c.T * d_inv
    coef = _solve(ct_dinv @ c, ct_dinv @ alpha)
    return alpha - c @ coef


def _project(w: np.ndarray, c: np.ndarray, d_inv: np.ndarray) -> np.ndarray:
    """
    Nearest point to ``w`` satisfying ``C'w = 0``, measured in the D metric.

    Minimizing ``(w - x)' D (w - x)`` subject to ``C'x = 0`` gives
    ``x = w - D^-1 C (C' D^-1 C)^-1 C' w``. Using the D metric rather than the
    Euclidean one keeps the correction concentrated in names the risk model says
    are cheap to trade against, instead of spreading it evenly.
    """
    ct_dinv = c.T * d_inv
    mu = _solve(ct_dinv @ c, c.T @ w)
    return w - d_inv * (c @ mu)


def _make_feasible(
    w: np.ndarray,
    c: np.ndarray | None,
    d_inv: np.ndarray,
    max_name: float,
    rounds: int = 30,
    tol: float = 1e-12,
) -> np.ndarray:
    """
    Enforce the box and the neutrality constraints together.

    Both sets are convex and their intersection is non-empty (it contains the
    zero book), so alternating projection converges to a point satisfying both
    -- von Neumann's result. That matters because the naive order, clip last,
    silently reintroduces the factor exposure the whole module exists to
    remove: a clip is a trade, and an unhedged trade is an exposure.
    """
    if c is None:
        return np.clip(w, -max_name, max_name)
    for _ in range(rounds):
        nxt = _project(np.clip(w, -max_name, max_name), c, d_inv)
        moved = float(np.max(np.abs(nxt - w))) if len(w) else 0.0
        w = nxt
        if moved < tol:
            break
    return np.clip(w, -max_name, max_name)


def optimize_book(
    alpha: pd.Series,
    model: RiskModel,
    sectors: pd.Series | None = None,
    config: OptimizerConfig | None = None,
    min_names: int = 20,
) -> OptimizedBook:
    """
    Factor-neutral, vol-targeted long/short book for one cross-section.

    ``alpha`` is a cross-sectional score, not a return forecast in percent; only
    its shape matters because the level is absorbed by the volatility target.
    Names missing from the risk model are dropped rather than assigned a
    default exposure, since an unmeasured name is not a low-risk name.
    """
    config = config or OptimizerConfig()
    index = alpha.index

    scores = alpha.replace([np.inf, -np.inf], np.nan).dropna()
    symbols = [str(s) for s in scores.index if s in model.exposures.index]
    if len(symbols) < min_names:
        return _empty_book(index)

    a = scores.loc[symbols].to_numpy(dtype=float)
    sd = float(np.std(a))
    if not np.isfinite(sd) or sd <= 0:
        return _empty_book(index)
    # Clip in sd units so one broken print cannot dominate the cross-section.
    a = np.clip((a - float(np.mean(a))) / sd, -config.alpha_clip, config.alpha_clip)

    # Two specific-variance vectors, deliberately.
    #
    # `d_risk` is the model's estimate and is what every reported risk number
    # and the volatility target are measured against, so `ex_ante_vol` equals
    # `model.volatility(weights)` exactly.
    #
    # `d_tilt` is that estimate with a floor, and is used only as the weighting
    # metric. The D^-1 tilt is unbounded as specific variance goes to zero, and
    # a stale, halted, or duplicated price series looks *calm* to a covariance
    # estimate rather than dangerous, so without a floor those names attract
    # the largest positions in the book. Flooring the metric bounds that;
    # flooring the risk estimate would instead mean reporting a number the
    # model never produced.
    d_risk = np.clip(model.specific_var.reindex(symbols).to_numpy(dtype=float), _VAR_FLOOR, None)
    d_risk = np.where(np.isfinite(d_risk), d_risk, _VAR_FLOOR)
    d_tilt = d_risk
    if config.specific_var_floor_ratio > 0:
        median = float(np.median(d_risk))
        if np.isfinite(median) and median > 0:
            floor = max(median * config.specific_var_floor_ratio, _VAR_FLOOR)
            d_tilt = np.maximum(d_risk, floor)
    d_inv = 1.0 / d_tilt
    b = model.exposures.reindex(symbols).fillna(0.0).to_numpy(dtype=float)

    c, constraint_names = _neutralization_matrix(symbols, model, sectors, config)
    if c is None:
        raw = d_inv * a
    else:
        raw = d_inv * _gls_residual(a, c, d_inv)

    gross = float(np.abs(raw).sum())
    if not np.isfinite(gross) or gross <= 0:
        return _empty_book(index)
    w = raw / gross

    def annual_vol(vec: np.ndarray) -> float:
        systematic = float(np.square(b.T @ vec).sum())
        specific = float(np.square(vec) @ d_risk)
        return float(np.sqrt(max(systematic + specific, 0.0)) * np.sqrt(TRADING_DAYS))

    binding = "vol_target"
    iterations = 0
    for step in range(config.max_iter):
        iterations = step + 1
        vol = annual_vol(w)
        current_gross = float(np.abs(w).sum())
        by_vol = config.target_vol / vol if vol > 0 else np.inf
        by_gross = config.max_gross / current_gross if current_gross > 0 else np.inf
        binding = "gross" if by_gross <= by_vol else "vol_target"
        scale = min(by_vol, by_gross)
        if not np.isfinite(scale):
            return _empty_book(index)
        w = w * scale

        if float(np.abs(w).max()) <= config.max_name + config.tol:
            break
        binding = "name_cap"
        w = _make_feasible(w, c, d_inv, config.max_name)

    w = _make_feasible(w, c, d_inv, config.max_name)
    # Scaling down by a factor below one preserves both the box and neutrality,
    # so the gross cap is the last thing applied and cannot be undone.
    final_gross = float(np.abs(w).sum())
    if final_gross > config.max_gross > 0:
        w = w * (config.max_gross / final_gross)
        binding = "gross"

    weights = pd.Series(w, index=symbols).reindex(index).fillna(0.0)
    systematic = float(np.square(b.T @ w).sum())
    specific = float(np.square(w) @ d_risk)
    total = systematic + specific
    held = weights[weights.abs() > 0]
    share = held.abs() / held.abs().sum() if len(held) else pd.Series(dtype=float)

    return OptimizedBook(
        weights=weights,
        ex_ante_vol=float(np.sqrt(max(total, 0.0)) * np.sqrt(TRADING_DAYS)),
        systematic_share=systematic / total if total > 0 else float("nan"),
        specific_share=specific / total if total > 0 else float("nan"),
        gross=float(weights.abs().sum()),
        net=float(weights.sum()),
        max_name=float(weights.abs().max()) if len(weights) else 0.0,
        n_held=int(len(held)),
        effective_names=float(1.0 / np.square(share).sum()) if len(share) else 0.0,
        factor_exposure=pd.Series(b.T @ w, index=model.exposures.columns),
        iterations=iterations,
        binding=binding,
    )


def constraint_summary(
    weights: pd.Series,
    model: RiskModel,
    sectors: pd.Series | None = None,
) -> dict[str, float]:
    """
    Post-hoc check that a book actually is what the optimizer claims.

    Useful on books built elsewhere (the quantile rule, or positions read back
    from a broker) where nothing guarantees the constraints hold.
    """
    aligned = weights.reindex(model.exposures.index).fillna(0.0)
    exposure = model.exposures.T @ aligned
    out = {
        "gross": float(weights.abs().sum()),
        "net": float(weights.sum()),
        "max_name": float(weights.abs().max()) if len(weights) else 0.0,
        "worst_factor_exposure": float(exposure.abs().max()) if len(exposure) else 0.0,
        **model.decompose(weights),
    }
    if sectors is not None and len(sectors):
        by_sector = weights.groupby(sectors.reindex(weights.index).fillna("unknown")).sum()
        out["worst_sector_net"] = float(by_sector.abs().max()) if len(by_sector) else 0.0
    return out
