"""Immutable weekly public-equity archive evidence, separate from trading bars."""
from __future__ import annotations

import sqlite3
from pathlib import Path

from gig.service.contracts import Contract, State
from gig.service.observed_workspace import ObservedWorkspace
from gig.service.serialization import public_error
from gig.service.workspace import _age, _fields
from gig.timestamps import aware_timestamp


class ObservedEquityWorkspace(ObservedWorkspace):
    """Reuse checked artifact boundaries without coercing weekly dates to clocks."""

    def _path(self, name):
        return self.workspace._path("observed-equity/"+name)

    @staticmethod
    def _missing():
        return Contract.admit(State.UNAVAILABLE, None, source="public-equity-archive", classification="UNAVAILABLE",
                              scope="workspace.observed-equity", error=public_error("DATA_UNAVAILABLE"))

    @staticmethod
    def _contract(document, *, state=State.AVAILABLE, research=False):
        return Contract.admit(state, {"title": "Historical weekly equity archive evidence", **document}
            if state in {State.AVAILABLE, State.STALE} else None,
            source="UCI_DOW_JONES_WEEKLY_2011" if document["classification"] == "HISTORICAL" else "authored-equity-archive-fixture",
            classification=document["classification"], as_of=document["acquired_at"], scope="workspace.observed-equity",
            source_identity=document["snapshot_id"], evidence_state=document["source_authenticity"], readiness="RESEARCH_ONLY")

    def run(self, payload):
        if not isinstance(payload, dict) or payload.get("operation") not in {"fetch", "admit", "replay"}:
            raise ValueError("explicit supported public-equity evidence operation required")
        operation = payload["operation"]
        request = {k: v for k, v in payload.items() if k != "operation"}
        if operation == "fetch":
            from gig.data.public_equity import fetch_public_equity
            return self._admit(*fetch_public_equity(request))
        if operation == "admit":
            from gig.data.public_equity import admit_equity_fixture
            return self._admit(*admit_equity_fixture(request))
        _fields(request, {"snapshot_id"}, {"snapshot_id"})
        try:
            return self._contract(self.snapshot(request["snapshot_id"]))
        except FileNotFoundError:
            return self._missing()

    def section(self, section, as_of, max_age_seconds):
        if section not in {"markets", "research", "operations"}:
            return None
        cutoff, maximum = _age(as_of, max_age_seconds)
        path = self._path("reference.sqlite")
        if not path.is_file():
            return None
        with sqlite3.connect(Path(path).as_uri()+"?mode=ro", uri=True) as connection:
            entries = connection.execute("SELECT identity,as_of FROM entries WHERE kind='snapshot' LIMIT 1001").fetchall()
        if len(entries) > 1000:
            raise ValueError("public-equity snapshot history exceeds bound")
        eligible = [(aware_timestamp(stamp), identity) for identity, stamp in entries if aware_timestamp(stamp) <= cutoff]
        if not eligible:
            return None
        observed, identity = max(eligible)
        state = State.UNKNOWN if maximum is None else State.STALE if (cutoff.value-observed.value)/1e9 > maximum else State.AVAILABLE
        return self._contract(self.snapshot(identity), state=state)
