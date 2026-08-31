"""
compliance_audit_engine.py
=============================================================================
Institutional compliance + audit trail engine.
Provides immutable append-only event logging, trade reconstruction, and
basic regulatory reporting summaries (SEC/MiFID style operational metadata).
=============================================================================
"""

from __future__ import annotations

import json
import os
import threading
import uuid
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional


class ComplianceAuditEngine:
    def __init__(self, log_dir: str = "logs") -> None:
        self._log_dir = log_dir
        self._audit_file = os.path.join(self._log_dir, "compliance_audit.jsonl")
        self._lock = threading.Lock()
        os.makedirs(self._log_dir, exist_ok=True)
        if not os.path.exists(self._audit_file):
            with open(self._audit_file, "a", encoding="utf-8"):
                pass

    @staticmethod
    def _now_iso() -> str:
        return datetime.now(timezone.utc).isoformat()

    def log_event(self, event_type: str, payload: Dict) -> Dict:
        event = {
            "event_id": str(uuid.uuid4()),
            "event_type": event_type,
            "timestamp_utc": self._now_iso(),
            "payload": payload or {},
        }
        line = json.dumps(event, ensure_ascii=True)
        with self._lock:
            with open(self._audit_file, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        return event

    def _read_all_events(self) -> List[Dict]:
        with self._lock:
            with open(self._audit_file, "r", encoding="utf-8") as f:
                rows = [x.strip() for x in f if x.strip()]
        out = []
        for row in rows:
            try:
                out.append(json.loads(row))
            except Exception:
                continue
        return out

    def get_recent_events(self, limit: int = 200, event_type: Optional[str] = None) -> List[Dict]:
        events = self._read_all_events()
        if event_type:
            events = [e for e in events if e.get("event_type") == event_type]
        return events[-max(1, int(limit)):]

    def reconstruct_trade(self, symbol: str, lookback_days: int = 30) -> Dict:
        symbol_u = symbol.upper()
        cutoff = datetime.now(timezone.utc) - timedelta(days=max(1, lookback_days))
        timeline = []
        for e in self._read_all_events():
            p = e.get("payload", {})
            ts = e.get("timestamp_utc", "")
            try:
                ts_dt = datetime.fromisoformat(ts)
            except Exception:
                continue
            if ts_dt < cutoff:
                continue
            if str(p.get("symbol", "")).upper() == symbol_u:
                timeline.append(e)
        timeline.sort(key=lambda x: x.get("timestamp_utc", ""))
        status = "NOT_FOUND"
        if timeline:
            last_type = timeline[-1].get("event_type")
            status = "OPEN" if last_type in ("order_submitted", "order_executed") else "CLOSED"
        return {"symbol": symbol_u, "status": status, "events": timeline, "event_count": len(timeline)}

    def compliance_report(self, days: int = 7) -> Dict:
        cutoff = datetime.now(timezone.utc) - timedelta(days=max(1, days))
        events = []
        for e in self._read_all_events():
            ts = e.get("timestamp_utc", "")
            try:
                ts_dt = datetime.fromisoformat(ts)
            except Exception:
                continue
            if ts_dt >= cutoff:
                events.append(e)

        counts = {}
        symbols = set()
        for e in events:
            et = e.get("event_type", "unknown")
            counts[et] = counts.get(et, 0) + 1
            sym = str(e.get("payload", {}).get("symbol", "")).upper().strip()
            if sym:
                symbols.add(sym)

        return {
            "window_days": days,
            "generated_at_utc": self._now_iso(),
            "events_total": len(events),
            "events_by_type": counts,
            "symbols_touched": len(symbols),
            "reporting_notes": [
                "Append-only audit log for trade lifecycle reconstruction",
                "Store this file in immutable retention storage for production compliance",
                "Map event types to SEC/MiFID internal reporting templates as needed",
            ],
        }


_engine: Optional[ComplianceAuditEngine] = None


def get_compliance_engine() -> ComplianceAuditEngine:
    global _engine
    if _engine is None:
        _engine = ComplianceAuditEngine()
    return _engine

