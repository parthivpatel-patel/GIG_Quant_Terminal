"""
PnL / signal attribution for the research terminal.

This is research-grade attribution, not a full Brinson book:
  - signal share from Newey–West factor IC magnitudes
  - risk split from the live Σ = BB′ + D report
  - paper NAV trail from audited trade_runs
"""

from __future__ import annotations

import math
from typing import Any

import pandas as pd


def _f(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def signal_attribution(factor_ic: dict[str, Any] | None) -> list[dict[str, Any]]:
    """Normalize |IC| into contribution shares across factors."""
    if not factor_ic:
        return []
    rows: list[tuple[str, float]] = []
    for name, payload in factor_ic.items():
        if isinstance(payload, dict):
            ic = payload.get("ic_mean", payload.get("mean"))
        else:
            ic = payload
        val = _f(ic)
        if val is None:
            continue
        rows.append((str(name), abs(val)))
    total = sum(v for _, v in rows) or 1.0
    out = [
        {
            "name": name,
            "ic": _f((factor_ic.get(name) or {}).get("ic_mean") if isinstance(factor_ic.get(name), dict) else factor_ic.get(name)),
            "share": v / total,
        }
        for name, v in sorted(rows, key=lambda x: x[1], reverse=True)
    ]
    # Re-attach signed IC cleanly
    for row in out:
        raw = factor_ic.get(row["name"])
        if isinstance(raw, dict):
            row["ic"] = _f(raw.get("ic_mean", raw.get("mean")))
            row["tstat"] = _f(raw.get("ic_tstat_nw", raw.get("tstat")))
        else:
            row["ic"] = _f(raw)
            row["tstat"] = None
    return out


def risk_attribution(risk: dict[str, Any] | None) -> dict[str, Any]:
    if not risk or not risk.get("available"):
        return {"available": False}
    gap = None
    pred = _f(risk.get("ex_ante_vol"))
    real = _f(risk.get("realized_vol"))
    if pred is not None and real is not None and real > 0:
        gap = pred / real
    exposure = risk.get("factor_exposure") or {}
    heat = [
        {"name": str(k), "value": _f(v)}
        for k, v in sorted(
            exposure.items(),
            key=lambda kv: abs(float(kv[1]) if kv[1] is not None else 0.0),
            reverse=True,
        )
    ]
    return {
        "available": True,
        "ex_ante_vol": pred,
        "realized_vol": real,
        "pred_over_real": gap,
        "systematic_share": _f(risk.get("systematic_share")),
        "specific_share": _f(risk.get("specific_share")),
        "factor_heat": heat,
        "top_contributors": risk.get("top_contributors") or [],
        "eigenvalue_share": risk.get("eigenvalue_share") or [],
        "equation": risk.get("equation") or "Σ = BB′ + D",
    }


def paper_nav_trail(limit: int = 40) -> dict[str, Any]:
    """NAV path from live (non-dry) trade runs when available."""
    try:
        from gig.data.lake import db_lock, open_store

        with db_lock():
            store = open_store(read_only=True)
            df = store.load_trade_runs(limit=max(limit, 5))
    except Exception as exc:
        return {"available": False, "error": f"{type(exc).__name__}: {exc}"}

    if df is None or df.empty:
        return {"available": False, "hint": "no trade_runs yet — run trade plan/run"}

    live = df[df["dry_run"] == False] if "dry_run" in df.columns else df  # noqa: E712
    use = live if not live.empty else df
    use = use.sort_values("ts")
    points = []
    prev = None
    for _, row in use.tail(limit).iterrows():
        nav = _f(row.get("nav"))
        chg = None if prev is None or nav is None or prev <= 0 else (nav / prev) - 1.0
        points.append(
            {
                "ts": str(row.get("ts") or ""),
                "nav": nav,
                "chg": chg,
                "blocked": bool(row.get("blocked")),
                "dry_run": bool(row.get("dry_run")),
            }
        )
        if nav is not None:
            prev = nav

    if not points:
        return {"available": False}

    first = next((p["nav"] for p in points if p["nav"]), None)
    last = next((p["nav"] for p in reversed(points) if p["nav"]), None)
    total = None if first is None or last is None or first <= 0 else (last / first) - 1.0
    return {
        "available": True,
        "n": len(points),
        "nav_first": first,
        "nav_last": last,
        "total_return": total,
        "points": points,
    }


def research_metrics_attribution() -> dict[str, Any]:
    """Pull last backtest metrics + IC for the attribution panel."""
    from gig.service.snapshot import research_snapshot

    research = research_snapshot()
    if not research.get("available"):
        return {"available": False, "hint": research.get("hint")}
    metrics = research.get("metrics") or {}
    factor_ic = research.get("factor_ic") or {}
    # research_snapshot may store factor_ic as name->float; normalize
    ic_payload: dict[str, Any] = {}
    for k, v in factor_ic.items():
        if isinstance(v, dict):
            ic_payload[k] = v
        else:
            ic_payload[k] = {"ic_mean": v}
    return {
        "available": True,
        "source": research.get("source"),
        "construction": research.get("construction"),
        "metrics": {
            "sharpe": _f(metrics.get("sharpe")),
            "ann_return": _f(metrics.get("ann_return")),
            "ann_vol": _f(metrics.get("ann_vol")),
            "max_drawdown": _f(metrics.get("max_drawdown")),
            "ex_ante_vol_mean": _f(metrics.get("ex_ante_vol_mean")),
            "systematic_share_mean": _f(metrics.get("systematic_share_mean")),
        },
        "signal": signal_attribution(ic_payload),
    }


def blotter_snapshot(limit: int = 120, drift_n: int = 20) -> dict[str, Any]:
    """
    Planned vs held blotter from the latest target + account drift.

    Falls back to the last audited target_book when the broker is offline.
    """
    from gig.config import get_settings

    settings = get_settings()
    if not settings.keys_status().get("alpaca"):
        return {
            "available": False,
            "hint": "Set ALPACA paper keys to populate held vs target",
            **_blotter_from_audit(limit=drift_n),
        }

    try:
        from gig.execution.trader import account_status

        status = account_status(limit=limit)
    except Exception as exc:
        return {
            "available": False,
            "error": f"{type(exc).__name__}: {exc}",
            **_blotter_from_audit(limit=drift_n),
        }

    target = status.get("target") or {}
    drift = status.get("drift") or []
    rows = []
    for d in drift[:drift_n]:
        tgt = _f(d.get("target_weight"))
        held = _f(d.get("current_weight"))
        rows.append(
            {
                "symbol": str(d.get("symbol") or ""),
                "held": held,
                "target": tgt,
                "drift": _f(d.get("drift")),
                "side": "long" if (tgt or 0) > 0 else "short" if (tgt or 0) < 0 else "flat",
            }
        )

    return {
        "available": True,
        "nav": _f(status.get("nav")),
        "market_open": status.get("market_open"),
        "n_positions": status.get("n_positions"),
        "worst_drift": _f(status.get("worst_drift")),
        "target_asof": target.get("asof"),
        "target_gross": _f(target.get("gross")),
        "target_net": _f(target.get("net")),
        "construction": target.get("construction"),
        "rows": rows,
        "account": status.get("account"),
    }


def _blotter_from_audit(limit: int = 20) -> dict[str, Any]:
    """Last target_book weights when broker is unavailable."""
    try:
        from gig.data.lake import open_store

        store = open_store(read_only=True)
        runs = store.load_trade_runs(limit=1)
        if runs is None or runs.empty:
            return {"audit": False}
        run_id = str(runs.iloc[0]["run_id"])
        df = store.con.execute(
            "SELECT symbol, weight, sector FROM target_book WHERE run_id = ? ORDER BY abs(weight) DESC LIMIT ?",
            [run_id, int(limit)],
        ).df()
        rows = [
            {
                "symbol": str(r.symbol),
                "held": None,
                "target": _f(r.weight),
                "drift": None,
                "side": "long" if float(r.weight) > 0 else "short",
                "sector": str(r.sector),
            }
            for r in df.itertuples()
        ]
        return {
            "audit": True,
            "run_id": run_id,
            "rows": rows,
            "hint": "broker offline — showing last audited target only",
        }
    except Exception:
        return {"audit": False}


def attribution_snapshot(risk: dict[str, Any] | None = None, factor_ic: dict[str, Any] | None = None) -> dict[str, Any]:
    """Combined payload for `/api/attribution`."""
    research = research_metrics_attribution()
    signal = signal_attribution(factor_ic) if factor_ic else (research.get("signal") or [])
    return {
        "available": True,
        "research": research,
        "signal": signal,
        "risk": risk_attribution(risk),
        "paper": paper_nav_trail(),
    }
