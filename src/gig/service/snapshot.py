"""
JSON payload builders for the terminal UI.

Nothing here imports a web framework, so the payloads are unit-testable and the
HTTP layer stays a thin adapter. Every builder is read-only: it reads DuckDB and
`results/`, computes factors in memory, and returns plain Python types.

NaN is converted to ``None`` because strict JSON parsers reject bare NaN.
"""

from __future__ import annotations

import json
import math
import time
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd

from gig import __version__
from gig.config import get_settings

_CACHE: dict[str, tuple[float, Any]] = {}


def _cached(key: str, ttl: float, build: Callable[[], Any]) -> Any:
    now = time.time()
    hit = _CACHE.get(key)
    if hit is not None and now - hit[0] < ttl:
        return hit[1]
    value = build()
    _CACHE[key] = (now, value)
    return value


def clear_cache() -> None:
    _CACHE.clear()


def _f(value: Any) -> float | None:
    """Finite float or None. Keeps the wire format valid JSON."""
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _open_store():
    from gig.data.lake import open_store

    # Terminal is read-only so the paper loop can keep the write lock.
    return open_store(read_only=True)


def service_status() -> dict[str, Any]:
    """Install, data source, key availability, and DuckDB row counts."""
    settings = get_settings()
    payload: dict[str, Any] = {
        "version": __version__,
        "source": settings.data_source,
        "db_path": str(settings.db_path.resolve()),
        "universe_file": str(settings.universe_file),
        "keys": settings.keys_status(),
        "limits": {
            "max_gross_leverage": settings.max_gross_leverage,
            "max_net_exposure": settings.max_net_exposure,
            "max_name_weight": settings.max_name_weight,
            "max_sector_net": settings.max_sector_net,
        },
        "costs": {
            "spread_bps": settings.spread_bps,
            "commission_bps": settings.commission_bps,
            "impact_eta": settings.impact_eta,
        },
        "tables": {},
        "asof": None,
        "db_ready": False,
        "construction": "risk_constrained" if settings.use_optimizer else "quantile_equal_weight",
        "target_vol": settings.target_vol,
        "ml": {},
        "ollama": {},
    }
    try:
        from gig.ml.ranker import available_backends

        payload["ml"] = available_backends()
    except Exception:
        payload["ml"] = {"sklearn": False, "lightgbm": False}
    try:
        from gig.nlp.ollama import ollama_status

        payload["ollama"] = ollama_status()
    except Exception as exc:
        payload["ollama"] = {"available": False, "error": f"{type(exc).__name__}: {exc}"}
    try:
        from gig.ops.status import ops_snapshot

        payload["ops"] = ops_snapshot()
    except Exception as exc:
        payload["ops"] = {"available": False, "error": f"{type(exc).__name__}: {exc}"}
    try:
        store = _open_store()
        payload["tables"] = store.stats()
        last = store.con.execute("SELECT MAX(dt) FROM bars").fetchone()
        if last and last[0] is not None:
            payload["asof"] = str(pd.Timestamp(last[0]).date())
        payload["db_ready"] = bool(payload["tables"].get("bars", 0))
    except Exception as exc:
        payload["error"] = f"{type(exc).__name__}: {exc}"
    return payload


def research_snapshot() -> dict[str, Any]:
    """Metrics and per-factor IC from the last persisted backtest."""
    settings = get_settings()
    path = settings.results_dir / "equity_ls_last.json"
    if not path.exists():
        return {"available": False, "hint": "python -m gig backtest --source yahoo --no-ml"}
    try:
        raw = json.loads(path.read_text(encoding="utf-8").replace("NaN", "null"))
    except Exception as exc:
        return {"available": False, "error": f"{type(exc).__name__}: {exc}"}

    metrics = {k: _f(v) for k, v in (raw.get("metrics") or {}).items()}
    ic = {k: _f(v) for k, v in (raw.get("factor_ic") or {}).items()}
    ic_n = {k: _f(v) for k, v in (raw.get("factor_ic_n") or {}).items()}
    return {
        "available": True,
        "source": raw.get("source"),
        "config_hash": raw.get("config_hash"),
        "metrics": metrics,
        "factor_ic": ic,
        "factor_ic_n": ic_n,
        "modified": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(path.stat().st_mtime)),
    }


def experiment_history(limit: int = 40) -> dict[str, Any]:
    """Recent rows from results/experiments.jsonl, newest last."""
    settings = get_settings()
    path = settings.results_dir / "experiments.jsonl"
    if not path.exists():
        return {"available": False, "runs": []}
    runs: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line.replace("NaN", "null"))
        except Exception:
            continue
        metrics = row.get("metrics") or {}
        runs.append(
            {
                "ts": row.get("ts"),
                "name": row.get("name"),
                "config_hash": row.get("config_hash"),
                "config": row.get("config") or {},
                "sharpe": _f(metrics.get("sharpe")),
                "ann_return": _f(metrics.get("ann_return")),
                "ann_vol": _f(metrics.get("ann_vol")),
                "max_drawdown": _f(metrics.get("max_drawdown")),
                "ann_turnover": _f(metrics.get("ann_turnover")),
                "dsr": _f(metrics.get("dsr")),
            }
        )
    return {"available": True, "runs": runs[-limit:]}


def _zscore(row: pd.Series, clip: float = 3.0) -> pd.Series:
    x = row.replace([np.inf, -np.inf], np.nan).astype(float)
    mu = x.mean()
    sd = x.std(ddof=0)
    if not np.isfinite(sd) or sd == 0:
        return pd.Series(0.0, index=row.index)
    return ((x - mu) / sd).clip(-clip, clip)


def _rank_percentile(row: pd.Series) -> pd.Series:
    """Cross-sectional rank in [0, 1]. NaNs stay NaN."""
    x = row.replace([np.inf, -np.inf], np.nan).astype(float)
    return x.rank(pct=True)


def _rank_normal(row: pd.Series, clip: float = 2.5) -> pd.Series:
    """
    Rank then map through the inverse normal CDF.

    Raw factor cross-sections are fat-tailed and pile up near the mean, so a
    plain z-score plots as a dense blob with a few far outliers. Gaussianizing
    the rank is a monotone transform — the ordering the alpha cares about is
    untouched — and it spreads the cross-section evenly enough to read.
    """
    from scipy import stats

    pct = _rank_percentile(row)
    n = int(pct.notna().sum())
    if n < 3:
        return pd.Series(0.0, index=row.index)
    # Squeeze off the endpoints so the extremes stay finite.
    squeezed = (pct * n - 0.5) / n
    out = pd.Series(stats.norm.ppf(squeezed.to_numpy(dtype=float)), index=row.index)
    return out.clip(-clip, clip)


def _clean(value: Any) -> Any:
    """Recursively replace non-finite floats with None so the payload is valid JSON."""
    if isinstance(value, dict):
        return {k: _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if isinstance(value, float):
        return _f(value)
    if isinstance(value, (np.floating, np.integer)):
        return _f(value)
    return value


def _load_cloud_panel(limit: int, lookback_days: int):
    """Panel of the `limit` most liquid names, long enough for 12-1 momentum."""
    from gig.data.lake import load_liquid_panel

    return load_liquid_panel(limit=limit, lookback_days=lookback_days, store=_open_store())


def factor_cloud(
    limit: int = 220,
    lookback_days: int = 640,
    min_adv_usd: float = 1_000_000.0,
) -> dict[str, Any]:
    """
    One point per name in neutralized factor space, plus the book the strategy
    would hold on the latest cross-section.

    Axes are cross-sectional z-scores of the *neutralized* factors, so the cloud
    shows what the alpha actually sees, not raw price levels. Scores come from
    the strategy itself so the UI cannot drift from the research path.

    The $1M ADV floor is a live-tape rule. On a small or synthetic store it would
    reject every name and render an empty book with no explanation, so the floor
    is dropped when too few names could clear it and the applied value is
    reported back as ``adv_floor``.
    """
    from gig.data.lake import db_lock

    with db_lock():
        return _factor_cloud_impl(limit, lookback_days, min_adv_usd)


def _factor_cloud_impl(
    limit: int,
    lookback_days: int,
    min_adv_usd: float,
) -> dict[str, Any]:
    from gig.factors.momentum import Momentum12m1, ShortTermReversal
    from gig.factors.neutralize import neutralize
    from gig.factors.volatility import IdiosyncraticVol
    from gig.pipeline import optimizer_config
    from gig.portfolio.construct import dollar_neutral_quantiles, sector_net_exposure
    from gig.portfolio.optimize import optimize_book
    from gig.risk.factor_model import fit_statistical_risk_model
    from gig.risk.limits import check_limits
    from gig.strategies.equity_ls import EquityLongShort

    settings = get_settings()
    panel, _store = _load_cloud_panel(limit, lookback_days)
    if panel is None or panel.close.empty:
        return {
            "available": False,
            "hint": "python -m gig universe refresh && python -m gig ingest",
            "points": [],
        }

    n_names = len(panel.symbols())
    n_side = max(5, min(50, n_names // 6))

    adv21 = (
        panel.dollar_volume.tail(21).mean()
        if panel.dollar_volume is not None and not panel.dollar_volume.empty
        else None
    )
    liquid = int((adv21 >= min_adv_usd).sum()) if adv21 is not None else 0
    adv_floor = min_adv_usd if liquid >= 2 * n_side + 2 else 0.0

    strategy = EquityLongShort(
        n_long=n_side,
        n_short=n_side,
        min_adv_usd=adv_floor,
        use_ml=False,
        use_news=False,
    )
    combo, tradable, ic_table = strategy.scores(panel)
    if combo.empty:
        return {"available": False, "hint": "not enough history in DuckDB", "points": []}

    asof = pd.Timestamp(combo.index[-1])
    raw = {
        "momentum": Momentum12m1().compute(panel),
        "reversal": ShortTermReversal().compute(panel),
        "ivol": IdiosyncraticVol().compute(panel),
    }
    axes = {
        key: _rank_normal(neutralize(df, panel.sectors, tradable=tradable).loc[asof])
        for key, df in raw.items()
    }

    score = combo.loc[asof]
    # Same construction as `trade plan` / the backtest: factor-neutral vol target
    # when the model fits, equal-weight quantiles only as a documented fallback.
    construction = "quantile_equal_weight"
    returns_panel = panel.close.pct_change().tail(settings.risk_lookback)
    model = fit_statistical_risk_model(returns_panel, n_factors=settings.risk_factors)
    opt = optimizer_config(settings)
    if opt is not None and model is not None:
        book = optimize_book(score, model, sectors=panel.sectors, config=opt)
        if book.available:
            weights = book.weights
            construction = "risk_constrained"
        else:
            weights = dollar_neutral_quantiles(
                score,
                n_long=n_side,
                n_short=n_side,
                max_name_weight=settings.max_name_weight,
                gross_leverage=settings.max_gross_leverage,
            )
    else:
        weights = dollar_neutral_quantiles(
            score,
            n_long=n_side,
            n_short=n_side,
            max_name_weight=settings.max_name_weight,
            gross_leverage=settings.max_gross_leverage,
        )
    score_z = _zscore(score)
    score_pct = _rank_percentile(score)
    adv = adv21 if adv21 is not None else pd.Series(dtype=float)
    close = panel.close
    chg = close.pct_change().loc[asof] if len(close) > 1 else pd.Series(dtype=float)

    # Rank-normalize the first three PCA axes so the risk-space view matches
    # the factor cloud's visual scale.
    pc_axes: dict[str, pd.Series] = {}
    if model is not None and model.exposures.shape[1] >= 1:
        for i, col in enumerate(model.exposures.columns[:3]):
            pc_axes[f"pc{i + 1}"] = _rank_normal(model.exposures[col])

    points: list[dict[str, Any]] = []
    for symbol in panel.symbols():
        w = _f(weights.get(symbol, 0.0)) or 0.0
        points.append(
            {
                "symbol": symbol,
                "sector": str(panel.sectors.get(symbol, "unknown")),
                "momentum": _f(axes["momentum"].get(symbol)),
                "reversal": _f(axes["reversal"].get(symbol)),
                "ivol": _f(axes["ivol"].get(symbol)),
                "pc1": _f(pc_axes["pc1"].get(symbol)) if "pc1" in pc_axes else None,
                "pc2": _f(pc_axes["pc2"].get(symbol)) if "pc2" in pc_axes else None,
                "pc3": _f(pc_axes["pc3"].get(symbol)) if "pc3" in pc_axes else None,
                "score": _f(score_z.get(symbol)),
                "score_pct": _f(score_pct.get(symbol)),
                "weight": w,
                "side": "long" if w > 0 else ("short" if w < 0 else "flat"),
                "adv": _f(adv.get(symbol)) if len(adv) else None,
                "close": _f(close.loc[asof].get(symbol)),
                "chg": _f(chg.get(symbol)) if len(chg) else None,
            }
        )

    breaches = check_limits(
        weights,
        panel.sectors,
        max_gross=settings.max_gross_leverage,
        max_net=settings.max_net_exposure,
        max_name=settings.max_name_weight,
        max_sector_net=settings.max_sector_net,
    )
    sector_net = sector_net_exposure(weights, panel.sectors)

    from gig.risk.factor_model import risk_report
    from gig.ml.ranker import available_backends

    risk = risk_report(
        weights,
        returns_panel.tail(252),
        sectors=panel.sectors,
        n_factors=settings.risk_factors,
    )
    backends = available_backends()
    ranker = "lightgbm lambdarank" if backends.get("lightgbm") else (
        "sklearn HistGBDT" if backends.get("sklearn") else "factors only"
    )
    pipeline = [
        {
            "id": "factors",
            "label": "Factors",
            "detail": "12-1 mom · ST reversal · idiosyncratic vol",
        },
        {
            "id": "neutralize",
            "label": "Neutralize",
            "detail": "sector residualize · tradable mask",
        },
        {
            "id": "ranker",
            "label": "Ranker",
            "detail": ranker + " · walk-forward OOS only",
        },
        {
            "id": "risk",
            "label": "Σ = BB′ + D",
            "detail": (
                f"{risk.get('n_factors', settings.risk_factors)} PCs · "
                f"{100 * float(risk.get('explained') or 0):.0f}% systematic variance"
                if risk.get("available")
                else "risk model unavailable"
            ),
        },
        {
            "id": "optimize",
            "label": "Book",
            "detail": construction.replace("_", " ") + " · vol-targeted",
        },
    ]

    return _clean({
        "available": True,
        "asof": str(asof.date()),
        "names": n_names,
        "n_side": n_side,
        "adv_floor": adv_floor,
        "construction": construction,
        "pipeline": pipeline,
        "points": points,
        "book": {
            "gross": _f(weights.abs().sum()),
            "net": _f(weights.sum()),
            "max_name": _f(weights.abs().max()),
            "longs": int((weights > 0).sum()),
            "shorts": int((weights < 0).sum()),
        },
        "sector_net": {str(k): _f(v) for k, v in sector_net.items() if _f(v)},
        "breaches": [
            {"name": b.name, "value": _f(b.value), "limit": _f(b.limit)} for b in breaches
        ],
        "risk": risk,
        "factor_ic": {
            key: {
                "ic_mean": _f(stats.get("ic_mean")),
                "ic_tstat_nw": _f(stats.get("ic_tstat_nw")),
                "ic_ir": _f(stats.get("ic_ir")),
                "n_obs": _f(stats.get("n_obs")),
            }
            for key, stats in ic_table.items()
        },
    })


def book_snapshot(limit: int = 220) -> dict[str, Any]:
    """Long and short legs of the current cross-section, sorted by conviction."""
    cloud = factor_cloud(limit=limit)
    if not cloud.get("available"):
        return {"available": False, "longs": [], "shorts": []}
    held = [p for p in cloud["points"] if p["side"] != "flat"]
    held.sort(key=lambda p: (p["score"] if p["score"] is not None else 0.0), reverse=True)
    return {
        "available": True,
        "asof": cloud["asof"],
        "book": cloud["book"],
        "longs": [p for p in held if p["side"] == "long"],
        "shorts": [p for p in held if p["side"] == "short"][::-1],
    }


def live_tape(limit: int = 40) -> dict[str, Any]:
    """
    Latest quotes, with the true source labelled.

    Alpaca IEX and Finnhub are snapshot endpoints, not a consolidated feed. When
    no vendor key is configured this falls back to the last stored close and says
    so, because showing a stale close as "live" is the kind of thing that makes a
    research stack untrustworthy.
    """
    from gig.pipeline import live_quotes

    ts = time.time()
    try:
        quotes = live_quotes(limit=limit)
    except Exception as exc:
        quotes = pd.DataFrame()
        error = f"{type(exc).__name__}: {exc}"
    else:
        error = None

    rows: list[dict[str, Any]] = []
    if quotes is not None and not quotes.empty:
        for _, row in quotes.iterrows():
            rows.append(
                {
                    "symbol": str(row.get("symbol", "")),
                    "bid": _f(row.get("bid")),
                    "ask": _f(row.get("ask")),
                    "last": _f(row.get("last")),
                    "source": str(row.get("source", "vendor")),
                }
            )
        return {
            "available": True,
            "delayed": False,
            "source": rows[0]["source"] if rows else "vendor",
            "polled_at": ts,
            "quotes": rows,
            "error": error,
        }

    fallback = _closes_as_quotes(limit)
    return {
        "available": bool(fallback),
        "delayed": True,
        "source": "duckdb_last_close",
        "polled_at": ts,
        "quotes": fallback,
        "error": error,
        "hint": "add ALPACA_API_KEY/ALPACA_SECRET_KEY or FINNHUB_API_KEY to .env for snapshots",
    }


def _closes_as_quotes(limit: int) -> list[dict[str, Any]]:
    try:
        store = _open_store()
        rows = store.con.execute(
            f"""
            SELECT symbol, close, dollar_volume
            FROM bars
            WHERE dt = (SELECT MAX(dt) FROM bars)
            ORDER BY dollar_volume DESC NULLS LAST
            LIMIT {int(limit)}
            """
        ).df()
    except Exception:
        return []
    if rows.empty:
        return []
    return [
        {
            "symbol": str(r["symbol"]),
            "bid": None,
            "ask": None,
            "last": _f(r["close"]),
            "source": "duckdb_last_close",
        }
        for _, r in rows.iterrows()
    ]


def terminal_payload(limit: int = 220) -> dict[str, Any]:
    """Everything the UI needs on first paint. Cached briefly; the cloud is heavy."""
    return {
        "status": service_status(),
        "research": research_snapshot(),
        "cloud": _cached(f"cloud:{limit}", 90.0, lambda: factor_cloud(limit=limit)),
        "experiments": experiment_history(),
    }
