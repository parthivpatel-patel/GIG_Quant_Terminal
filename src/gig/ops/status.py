"""
Ops snapshot for the terminal and CLI.

Aggregates lake freshness, kill-switch state, last trade run, paper-loop
heartbeat, and actionable alerts. Read-only — never submits.
"""

from __future__ import annotations

import json
import math
from datetime import UTC, date, datetime
from typing import Any

from gig.ops.killswitch import kill_switch

# Soft / hard staleness thresholds for the equity tape (calendar days).
FRESH_WARN_DAYS = 3
FRESH_BLOCK_DAYS = 5

_HEARTBEAT: dict[str, Any] = {
    "ts": None,
    "ok": False,
    "detail": "paper loop not started in this process",
    "run_id": None,
}


def touch_heartbeat(*, ok: bool, detail: str = "", run_id: str | None = None) -> None:
    """Called by the in-process paper loop after each pass."""
    _HEARTBEAT["ts"] = datetime.now(UTC).isoformat()
    _HEARTBEAT["ok"] = bool(ok)
    _HEARTBEAT["detail"] = detail
    _HEARTBEAT["run_id"] = run_id


def heartbeat() -> dict[str, Any]:
    return dict(_HEARTBEAT)


def _bar_asof() -> date | None:
    try:
        from gig.data.lake import last_bar_date, open_store

        return last_bar_date(open_store(read_only=True))
    except Exception:
        return None


def _freshness(asof: date | None) -> dict[str, Any]:
    today = datetime.now(UTC).date()
    if asof is None:
        return {
            "asof": None,
            "age_days": None,
            "level": "bad",
            "label": "no bars",
        }
    age = (today - asof).days
    if age >= FRESH_BLOCK_DAYS:
        level = "bad"
    elif age >= FRESH_WARN_DAYS:
        level = "warn"
    else:
        level = "ok"
    return {
        "asof": str(asof),
        "age_days": int(age),
        "level": level,
        "label": f"{age}d stale" if age else "fresh",
    }


def _last_trade_run() -> dict[str, Any] | None:
    try:
        from gig.data.lake import open_store

        store = open_store(read_only=True)
        df = store.load_trade_runs(limit=1)
        if df is None or df.empty:
            return None
        row = df.iloc[0]
        report: dict[str, Any] = {}
        raw = row.get("report")
        if isinstance(raw, str) and raw:
            try:
                report = json.loads(raw)
            except Exception:
                report = {}
        pretrade = report.get("pretrade") or {}
        findings = pretrade.get("findings") or []
        return {
            "run_id": str(row.get("run_id") or ""),
            "ts": str(row.get("ts") or ""),
            "asof": str(row.get("asof_date") or row.get("asof") or "") or None,
            "dry_run": bool(row.get("dry_run")),
            "blocked": bool(row.get("blocked")),
            "nav": _f(row.get("nav")),
            "gross": _f(row.get("gross")),
            "net": _f(row.get("net")),
            "turnover": _f(row.get("turnover")),
            "n_orders": int(row.get("n_orders") or 0),
            "construction": str(row.get("construction") or ""),
            "findings": findings[:8],
        }
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def _f(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _alerts(
    *,
    kill: dict[str, Any],
    fresh: dict[str, Any],
    last: dict[str, Any] | None,
    hb: dict[str, Any],
) -> list[dict[str, str]]:
    alerts: list[dict[str, str]] = []
    if kill.get("engaged"):
        alerts.append(
            {
                "level": "bad",
                "code": "kill_switch",
                "detail": str(kill.get("reason") or "trading halted"),
            }
        )
    if fresh.get("level") == "bad":
        alerts.append(
            {
                "level": "bad",
                "code": "stale_bars",
                "detail": f"last bar {fresh.get('asof')} ({fresh.get('age_days')}d) — refresh lake",
            }
        )
    elif fresh.get("level") == "warn":
        alerts.append(
            {
                "level": "warn",
                "code": "aging_bars",
                "detail": f"last bar {fresh.get('asof')} ({fresh.get('age_days')}d)",
            }
        )
    if last and last.get("blocked"):
        alerts.append(
            {
                "level": "warn",
                "code": "last_run_blocked",
                "detail": "most recent trade run was blocked by the pre-trade gate",
            }
        )
    if hb.get("ts") is None:
        alerts.append(
            {
                "level": "warn",
                "code": "no_heartbeat",
                "detail": "no in-process paper loop heartbeat (start serve --trade-every)",
            }
        )
    elif hb.get("ok") is False:
        alerts.append(
            {
                "level": "warn",
                "code": "heartbeat_fail",
                "detail": str(hb.get("detail") or "last paper loop pass failed"),
            }
        )
    return alerts


def ops_snapshot() -> dict[str, Any]:
    """Full ops payload for `/api/ops` and the terminal strip."""
    kill = kill_switch().status()
    fresh = _freshness(_bar_asof())
    last = _last_trade_run()
    hb = heartbeat()
    alerts = _alerts(kill=kill, fresh=fresh, last=last, hb=hb)
    level = "ok"
    if any(a["level"] == "bad" for a in alerts):
        level = "bad"
    elif alerts:
        level = "warn"
    return {
        "available": True,
        "level": level,
        "kill_switch": kill,
        "freshness": fresh,
        "heartbeat": hb,
        "last_run": last,
        "alerts": alerts,
        "hint": (
            "python -m gig ops halt|resume  ·  "
            "python -m gig trade flatten --yes  ·  "
            "python -m gig ingest"
        ),
    }
