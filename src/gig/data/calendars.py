"""Bounded exchange-calendar estimates with explicit identity and UTC semantics.

Calendar rules describe scheduled sessions. They establish neither provider
freshness nor an actual exchange/broker operating state. No network is used.
"""
from __future__ import annotations

import hashlib
import importlib
import json
import math
import re
from dataclasses import dataclass
from datetime import date
from typing import Any

import pandas as pd

from gig.timestamps import aware_timestamp

MAX_COVERAGE_DAYS = 3660
MAX_SCHEDULE_DAYS = 366
MAX_DAILY_OBSERVATIONS = 2000


def _date(value: Any) -> date:
    if not isinstance(value, str) or re.fullmatch(r"\d{4}-\d{2}-\d{2}", value) is None:
        raise ValueError("session dates must use YYYY-MM-DD")
    return date.fromisoformat(value)


def _integer(value: Any, low: int, high: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
        raise ValueError("integer parameter is outside its declared bounds")
    return value


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False,
                                     separators=(",", ":")).encode()).hexdigest()


def _iso(ns: int) -> str:
    return pd.Timestamp(ns, tz="UTC").isoformat()


@dataclass(frozen=True, slots=True)
class TradingSession:
    session: str
    open_ns: int
    close_ns: int
    break_start_ns: int | None
    break_end_ns: int | None
    early_close: bool

    def as_dict(self) -> dict[str, Any]:
        break_ns = 0
        if self.break_start_ns is not None:
            if self.break_end_ns is None:
                raise ValueError("session break requires both endpoints")
            break_ns = self.break_end_ns - self.break_start_ns
        return {"session": self.session, "open": _iso(self.open_ns),
                "close": _iso(self.close_ns),
                "break_start": None if self.break_start_ns is None else _iso(self.break_start_ns),
                "break_end": None if self.break_end_ns is None else _iso(self.break_end_ns),
                "session_span_seconds": (self.close_ns - self.open_ns) / 1e9,
                "trading_seconds": (self.close_ns - self.open_ns - break_ns) / 1e9,
                "early_close": self.early_close}


class ExchangeCalendar:
    """A detached schedule snapshot from a named installed calendar version.

    Coverage is explicit and bounded. Returned dictionaries are independent of
    this snapshot and the third-party package's shared calendar cache.
    """

    def __init__(self, calendar_id: str, *, coverage_start: str = "2020-01-01",
                 coverage_end: str = "2030-01-01") -> None:
        if not isinstance(calendar_id, str) or not 1 <= len(calendar_id) <= 32:
            raise ValueError("explicit bounded calendar identity required")
        start, end = _date(coverage_start), _date(coverage_end)
        if not 32 <= (end - start).days <= MAX_COVERAGE_DAYS:
            raise ValueError("calendar coverage must span 32 to 3660 days")
        try:
            xc = importlib.import_module("exchange_calendars")
        except ImportError:
            raise ImportError("Install the optional calendar dependency exchange-calendars") from None
        if calendar_id not in xc.get_calendar_names():
            raise ValueError("unknown calendar identity")
        try:
            calendar = xc.get_calendar(calendar_id, start=coverage_start, end=coverage_end)
        except (ValueError, TypeError, OverflowError):
            raise ValueError("calendar cannot supply the requested coverage") from None
        schedule = calendar.schedule.copy(deep=True)
        early = set(calendar.early_closes)
        sessions = []
        for label, row in schedule.iterrows():
            opening, closing = aware_timestamp(row["open"]), aware_timestamp(row["close"])
            left = None if pd.isna(row["break_start"]) else aware_timestamp(row["break_start"]).value
            right = None if pd.isna(row["break_end"]) else aware_timestamp(row["break_end"]).value
            if opening >= closing or ((left is None) != (right is None)):
                raise ValueError("calendar emitted an invalid session")
            if left is not None and not opening.value < left < right < closing.value:
                raise ValueError("calendar emitted an invalid session break")
            sessions.append(TradingSession(str(label.date()), opening.value, closing.value,
                                           left, right, label in early))
        self.calendar_id = calendar_id
        self.canonical_id = str(calendar.name)
        self.version = str(xc.__version__)
        self.timezone = str(calendar.tz)
        self.coverage_start, self.coverage_end = start, end
        self._sessions = tuple(sessions)
        self._close_indices = {session.close_ns: i for i, session in enumerate(self._sessions)}
        self.identity = _digest({"calendar": self.canonical_id, "version": self.version,
                                 "coverage_start": coverage_start, "coverage_end": coverage_end,
                                 "sessions": [session.as_dict() for session in self._sessions]})

    def provenance(self) -> dict[str, Any]:
        return {"requested_calendar": self.calendar_id, "calendar": self.canonical_id,
                "calendar_package": "exchange_calendars", "calendar_version": self.version,
                "timezone": self.timezone, "coverage_start": self.coverage_start.isoformat(),
                "coverage_end": self.coverage_end.isoformat(), "calendar_sha256": self.identity,
                "classification": "CALENDAR_ESTIMATE", "evidence_state": "VERSIONED_RULES",
                "trading_approved": False}

    def _cutoff(self, as_of: Any) -> pd.Timestamp:
        value = aware_timestamp(as_of)
        local_date = value.tz_convert(self.timezone).date()
        if not self.coverage_start <= local_date <= self.coverage_end:
            raise ValueError("timestamp is outside declared calendar coverage")
        return value

    def schedule(self, start: str, end: str) -> dict[str, Any]:
        left, right = _date(start), _date(end)
        if not self.coverage_start <= left <= right <= self.coverage_end:
            raise ValueError("schedule range is outside declared coverage or reversed")
        if (right - left).days + 1 > MAX_SCHEDULE_DAYS:
            raise ValueError("schedule response exceeds 366 calendar days")
        rows = [session.as_dict() for session in self._sessions
                if left.isoformat() <= session.session <= right.isoformat()]
        return {"state": "AVAILABLE", "title": "Scheduled market sessions",
                "metrics": {"session_count": len(rows)}, "rows": rows,
                "range_start": start, "range_end": end, "provenance": self.provenance()}

    def status(self, as_of: Any) -> dict[str, Any]:
        cutoff = self._cutoff(as_of)
        now = cutoff.value
        active = next((session for session in self._sessions
                       if session.open_ns <= now < session.close_ns), None)
        on_break = bool(active is not None and active.break_start_ns is not None
                        and active.break_start_ns <= now < active.break_end_ns)
        completed = [session for session in self._sessions if session.close_ns <= now]
        return {"state": "AVAILABLE", "as_of": cutoff.isoformat(),
                "scheduled_state": "BREAK" if on_break else "OPEN" if active else "CLOSED",
                "active_session": None if active is None else active.session,
                "last_completed_session": completed[-1].session if completed else None,
                "last_completed_close": _iso(completed[-1].close_ns) if completed else None,
                "provenance": self.provenance()}

    def validate_daily_closes(self, timestamps: Any, *, require_contiguous: bool = False) -> dict[str, Any]:
        if not isinstance(require_contiguous, bool):
            raise ValueError("require_contiguous must be boolean")
        if not isinstance(timestamps, (list, tuple)) or not 1 <= len(timestamps) <= MAX_DAILY_OBSERVATIONS:
            raise ValueError("daily close validation requires one to 2000 observations")
        normalized = [self._cutoff(value) for value in timestamps]
        indices = []
        for stamp in normalized:
            if stamp.value not in self._close_indices:
                raise ValueError("daily bar timestamp must equal a scheduled session close")
            indices.append(self._close_indices[stamp.value])
        if any(right <= left for left, right in zip(indices, indices[1:], strict=False)):
            raise ValueError("daily bar timestamps must increase without duplicate instants")
        expected = set(range(indices[0], indices[-1] + 1))
        missing = [self._sessions[i].session for i in sorted(expected - set(indices))]
        if require_contiguous and missing:
            raise ValueError("daily bars omit scheduled sessions")
        return {"state": "AVAILABLE", "observations": len(indices),
                "session_labels": [self._sessions[i].session for i in indices],
                "normalized_timestamps": [stamp.isoformat() for stamp in normalized],
                "missing_sessions": missing, "contiguous": not missing,
                "provenance": self.provenance()}

    def freshness(self, last_bar_close: Any, as_of: Any, *, max_missed_sessions: int = 0,
                  grace_seconds: float = 0) -> dict[str, Any]:
        maximum = _integer(max_missed_sessions, 0, 30)
        if isinstance(grace_seconds, bool) or not isinstance(grace_seconds, (int, float)):
            raise ValueError("grace_seconds must be finite numeric seconds")
        if not math.isfinite(grace_seconds) or not 0 <= grace_seconds <= 3600:
            raise ValueError("grace_seconds must lie between zero and 3600")
        observed, cutoff = self._cutoff(last_bar_close), self._cutoff(as_of)
        if observed > cutoff:
            raise ValueError("observation timestamp is in the future")
        if observed.value not in self._close_indices:
            raise ValueError("last bar must identify an actual scheduled session close")
        deadline = cutoff.value - int(grace_seconds * 1e9)
        expected = [session for session in self._sessions if session.close_ns <= deadline]
        missed = [session.session for session in expected if session.close_ns > observed.value]
        return {"state": "UNKNOWN" if not expected else "STALE" if len(missed) > maximum else "AVAILABLE",
                "as_of": cutoff.isoformat(), "observed_close": observed.isoformat(),
                "expected_session": expected[-1].session if expected else None,
                "expected_close": _iso(expected[-1].close_ns) if expected else None,
                "missed_session_count": len(missed), "max_missed_sessions": maximum,
                "missing_sessions": missed[:64], "missing_sessions_complete": len(missed) <= 64,
                "missing_sessions_sha256": _digest(missed), "grace_seconds": grace_seconds,
                "provenance": self.provenance()}


def calendar_analytics(payload: dict[str, Any]) -> dict[str, Any]:
    """Bounded adapter for the terminal's local analytical workbench."""
    allowed = {"calendar", "start", "end", "as_of", "last_bar_close", "max_missed_sessions",
               "grace_seconds", "daily_closes", "require_contiguous"}
    if not isinstance(payload, dict) or set(payload) - allowed or not {"calendar", "start", "end"} <= set(payload):
        raise ValueError("calendar request contains unknown or missing fields")
    calendar = ExchangeCalendar(payload["calendar"])
    result = calendar.schedule(payload["start"], payload["end"])
    if "as_of" in payload:
        result["scheduled_market_status"] = calendar.status(payload["as_of"])
    if "last_bar_close" in payload:
        if "as_of" not in payload:
            raise ValueError("freshness requires an explicit as_of timestamp")
        result["freshness"] = calendar.freshness(payload["last_bar_close"], payload["as_of"],
                                                max_missed_sessions=payload.get("max_missed_sessions", 0),
                                                grace_seconds=payload.get("grace_seconds", 0))
    elif "max_missed_sessions" in payload or "grace_seconds" in payload:
        raise ValueError("freshness parameters require last_bar_close")
    if "daily_closes" in payload:
        result["daily_close_validation"] = calendar.validate_daily_closes(
            payload["daily_closes"], require_contiguous=payload.get("require_contiguous", False))
    elif "require_contiguous" in payload:
        raise ValueError("contiguity policy requires daily_closes")
    return result
