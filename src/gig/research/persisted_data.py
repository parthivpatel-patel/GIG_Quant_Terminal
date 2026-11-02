"""Causal research windows from immutable, locally admitted observation evidence.

No provider is contacted. Source classifications and action coverage remain
caller assertions; a digest establishes integrity rather than authenticity.
"""

from __future__ import annotations

import hashlib
import math
import sqlite3
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from gig.data.actions import adjust_bars, admit_actions
from gig.data.ingestion import Instrument
from gig.data.validated import CLASSIFICATIONS, DataSnapshot, admit_bars
from gig.service.serialization import dumps, loads
from gig.timestamps import aware_timestamp

IMPLEMENTATION = "gig.persisted-pit-dataset.v1"
MAX_ARCHIVE_ROWS = 20_000
MAX_PANEL_ROWS = 1000
MAX_DATASET_ROWS = 1000


class DatasetUnavailable(ValueError):
    """The requested archive cannot supply an admissible research dataset."""


def digest(value: Any) -> str:
    return hashlib.sha256(dumps(value).encode()).hexdigest()


def fixed_path(base: Path, relative: str) -> Path:
    path = Path(base).absolute() / relative
    if any(part.is_symlink() for part in (path, *path.parents)):
        raise ValueError("persisted research paths cannot follow symbolic links")
    return path


def _text(value: Any, bound: int = 512) -> str:
    if (not isinstance(value, str) or not value.strip() or len(value) > bound
            or any(ord(char) < 32 or 0xD800 <= ord(char) <= 0xDFFF for char in value)):
        raise ValueError("bounded nonempty evidence text required")
    return value


def _integer(value: Any, low: int, high: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
        raise ValueError("bounded integer configuration required")
    return value


def admit_request(payload: Any) -> dict[str, Any]:
    fields = {"source", "classification", "currency", "universe_id", "start", "end", "as_of",
              "decision_delay_seconds", "horizon", "lookback", "adjustment"}
    if not isinstance(payload, dict) or set(payload) != fields:
        raise ValueError("persisted dataset requires its explicit source, universe and window contract")
    value = loads(dumps(payload))
    value["source"] = _text(value["source"])
    value["universe_id"] = _text(value["universe_id"], 128)
    if value["classification"] not in CLASSIFICATIONS:
        raise ValueError("explicit admitted source classification required")
    currency = value["currency"]
    if not isinstance(currency, str) or len(currency) != 3 or not currency.isascii() or not currency.isalpha():
        raise ValueError("three-letter source currency required")
    value["currency"] = currency.upper()
    if value["adjustment"] not in {"RAW", "SPLIT_ADJUSTED", "TOTAL_RETURN_ADJUSTED"}:
        raise ValueError("unsupported explicit adjustment policy")
    for name in ("start", "end", "as_of"):
        value[name] = aware_timestamp(value[name]).isoformat()
    start, end, cutoff = (aware_timestamp(value[k]) for k in ("start", "end", "as_of"))
    if not start < end <= cutoff or end - start > pd.Timedelta(days=3660):
        raise ValueError("a bounded historical window ending by research cutoff is required")
    value["decision_delay_seconds"] = _integer(value["decision_delay_seconds"], 0, 86400)
    value["horizon"] = _integer(value["horizon"], 1, 30)
    value["lookback"] = _integer(value["lookback"], 2, 128)
    return value


def _read_bars(base: Path, request: dict[str, Any]) -> list[dict[str, Any]]:
    path = fixed_path(base, "bars.sqlite")
    if not path.is_file():
        raise DatasetUnavailable("no admitted observations")
    rows: list[dict[str, Any]] = []
    anchors: dict[str, tuple[dict[str, Any], set[bytes]]] = {}
    evidence_bytes = 0
    cutoff = aware_timestamp(request["as_of"])
    start, end = aware_timestamp(request["start"]), aware_timestamp(request["end"])
    with sqlite3.connect(path.as_uri() + "?mode=ro", uri=True) as connection:
        query = ("SELECT row_json,snapshot_hash,instrument,event,available FROM bar_revisions "
                 "WHERE source=? AND classification=? AND adjustment='RAW' AND currency=? LIMIT ?")
        records = connection.execute(query, (request["source"], request["classification"],
                                            request["currency"], MAX_ARCHIVE_ROWS + 1)).fetchall()
        if len(records) > MAX_ARCHIVE_ROWS:
            raise ValueError("source archive exceeds declared research scan bound")
        for encoded, identity, instrument, event, available in records:
            if not isinstance(identity, str) or len(identity) != 64:
                raise ValueError("invalid source snapshot identity")
            if identity not in anchors:
                found = connection.execute("SELECT payload FROM bar_snapshot_evidence WHERE snapshot_hash=?",
                                           (identity,)).fetchone()
                if found is None or not isinstance(found[0], bytes):
                    raise ValueError("source snapshot evidence missing")
                evidence_bytes += len(found[0])
                if evidence_bytes > 32 * 1024 * 1024 or len(anchors) >= 64:
                    raise ValueError("source evidence exceeds research verification bound")
                original = DataSnapshot(found[0], identity).as_dict()
                checked = admit_bars([{**r, "volume": int(r["volume"])} for r in original["rows"]],
                                     source=original["source"], classification=original["classification"],
                                     corporate_action_policy=original["corporate_action_policy"],
                                     currency=original["currency"])
                if checked.payload != found[0]:
                    raise ValueError("noncanonical source snapshot evidence")
                expected = (request["source"], request["classification"], "RAW", request["currency"])
                actual = tuple(original[k] for k in ("source", "classification", "corporate_action_policy", "currency"))
                if actual != expected:
                    raise ValueError("source snapshot metadata differs from selected archive")
                anchors[identity] = (original, {dumps(r).encode() for r in original["rows"]})
            row = loads(encoded)
            if not isinstance(row, dict) or dumps(row).encode() not in anchors[identity][1]:
                raise ValueError("stored revision differs from original evidence")
            if (instrument, event, available) != (row["instrument_id"], row["timestamp"], row["available_at"]):
                raise ValueError("stored revision index differs from evidence")
            at, known = aware_timestamp(event), aware_timestamp(available)
            if start <= at <= end and known <= cutoff:
                rows.append({**row, "source_sha256": identity})
    if not rows:
        raise DatasetUnavailable("no observations visible under the requested source and cutoff")
    return rows


def _read_universes(base: Path, request: dict[str, Any]) -> list[dict[str, Any]]:
    path = fixed_path(base, "universe.sqlite")
    if not path.is_file():
        raise DatasetUnavailable("point-in-time universe has not been admitted")
    rows = []
    with sqlite3.connect(path.as_uri() + "?mode=ro", uri=True) as connection:
        records = connection.execute("SELECT effective,available,digest,payload FROM universe_revisions WHERE universe=? LIMIT 1001",
                                     (request["universe_id"],)).fetchall()
        if len(records) > 1000:
            raise ValueError("universe archive exceeds research bound")
        for effective, available, sha, encoded in records:
            if not isinstance(encoded, bytes) or hashlib.sha256(encoded).hexdigest() != sha:
                raise ValueError("universe source evidence failed integrity verification")
            row = loads(encoded)
            if not isinstance(row, dict) or set(row) != {"universe_id", "source", "effective_at", "available_at", "members"}:
                raise ValueError("invalid universe source contract")
            if (effective, available, request["universe_id"]) != (row["effective_at"], row["available_at"], row["universe_id"]):
                raise ValueError("universe index differs from evidence")
            if not isinstance(row["members"], list) or len(row["members"]) > 32:
                raise ValueError("research universes support at most 32 members")
            members = [asdict(Instrument(**member)) for member in row["members"]]
            if len({m["instrument_id"] for m in members}) != len(members) or len({m["symbol"] for m in members}) != len(members):
                raise ValueError("duplicate point-in-time universe identity")
            _text(row["source"])
            if aware_timestamp(available) <= aware_timestamp(request["as_of"]):
                rows.append({**row, "source_sha256": sha})
    if not rows:
        raise DatasetUnavailable("no known universe revision")
    return rows


def _read_actions(base: Path, request: dict[str, Any]) -> list[dict[str, Any]]:
    if request["adjustment"] == "RAW":
        return []
    path = fixed_path(base, "actions.sqlite")
    if not path.is_file():
        raise DatasetUnavailable("adjustment requires an explicitly admitted corporate-action archive")
    rows = []
    with sqlite3.connect(path.as_uri() + "?mode=ro", uri=True) as connection:
        records = connection.execute("SELECT source,action_id,available_at,payload,sha256 FROM action_revisions LIMIT 129").fetchall()
        if len(records) > 128:
            raise ValueError("action revision research bound exceeded")
        for source, action_id, available, encoded, sha in records:
            if not isinstance(encoded, bytes) or hashlib.sha256(encoded).hexdigest() != sha:
                raise ValueError("corporate-action source integrity failure")
            row = loads(encoded)
            admitted = admit_actions([row]).as_dict()["rows"][0]
            if (source, action_id, available) != (admitted["source"], admitted["action_id"], admitted["available_at"]):
                raise ValueError("corporate-action index differs from evidence")
            if aware_timestamp(available) <= aware_timestamp(request["as_of"]):
                rows.append(admitted)
    return rows


def _visible(actions: list[dict[str, Any]], instrument: str, start: pd.Timestamp,
             end: pd.Timestamp, cutoff: pd.Timestamp) -> list[dict[str, Any]]:
    latest: dict[tuple[str, str], dict[str, Any]] = {}
    for row in actions:
        if row["instrument_id"] != instrument or aware_timestamp(row["available_at"]) > cutoff:
            continue
        identity = row["source"], row["action_id"]
        if identity not in latest or aware_timestamp(row["available_at"]) > aware_timestamp(latest[identity]["available_at"]):
            latest[identity] = row
    return sorted([r for r in latest.values() if start < aware_timestamp(r["effective_at"]) <= end],
                  key=lambda r: (aware_timestamp(r["effective_at"]), r["source"], r["action_id"]))


def _prices(rows: list[dict[str, Any]], actions: list[dict[str, Any]], cutoff: pd.Timestamp,
            request: dict[str, Any]) -> tuple[list[float], pd.Timestamp, str]:
    known = max(aware_timestamp(r["available_at"]) for r in rows)
    if request["adjustment"] == "RAW":
        return [r["close"] for r in rows], known, digest([])
    relevant = _visible(actions, rows[0]["instrument_id"], aware_timestamp(rows[0]["timestamp"]),
                        aware_timestamp(rows[-1]["timestamp"]), cutoff)
    raw = [{k: v for k, v in r.items() if k != "source_sha256"} for r in rows]
    raw = [{**r, "volume": int(r["volume"])} for r in raw]
    result = adjust_bars(raw, relevant, as_of=cutoff, mode=request["adjustment"], source=request["source"],
                         classification=request["classification"], currency=request["currency"])
    if result["state"] != "AVAILABLE" or len(result["rows"]) != len(rows):
        raise ValueError("corporate-action transformation lacks the requested observations")
    known = max(aware_timestamp(r["adjusted_available_at"]) for r in result["rows"])
    return [r["adjusted_close"] for r in result["rows"]], known, digest(relevant)


def build_document(base_dir: Path, payload: Any) -> dict[str, Any]:
    """Select a decision's own evidence, then separately admit later outcome labels."""
    request = admit_request(payload)
    base = Path(base_dir).absolute()
    bars, universes, actions = _read_bars(base, request), _read_universes(base, request), _read_actions(base, request)
    cutoff = aware_timestamp(request["as_of"])
    revisions: dict[tuple[str, pd.Timestamp], list[dict[str, Any]]] = {}
    for row in bars:
        revisions.setdefault((row["instrument_id"], aware_timestamp(row["timestamp"])), []).append(row)
    for versions in revisions.values():
        versions.sort(key=lambda row: aware_timestamp(row["available_at"]))
    grid = sorted({aware_timestamp(row["timestamp"]) for row in bars})
    if len(grid) > 1000:
        raise ValueError("research event grid exceeds 1000 observations")

    def latest(instrument: str, event: pd.Timestamp, known: pd.Timestamp) -> dict[str, Any] | None:
        versions = revisions.get((instrument, event), [])
        return next((r for r in reversed(versions) if aware_timestamp(r["available_at"]) <= known), None)

    def outcome_vintages(instrument, event, current, index):
        # The horizon follows the source event grid as it was learned. A late
        # discovered event may revise a label, but cannot erase its older vintage.
        future_grid = grid[index + 1:]
        if len(future_grid) < request["horizon"]:
            return []
        first_clock = sorted(grid_knowledge[t] for t in future_grid)[request["horizon"] - 1]
        initially_known = [t for t in future_grid if grid_knowledge[t] <= first_clock]
        last_relevant = initially_known[request["horizon"] - 1]
        relevant_grid = [t for t in future_grid if t <= last_relevant]
        clocks = {grid_knowledge[t] for t in relevant_grid}
        clocks.update(aware_timestamp(r["available_at"]) for t in relevant_grid
                      for r in revisions.get((instrument, t), []))
        clocks.update(aware_timestamp(r["available_at"]) for r in actions
                      if r["instrument_id"] == instrument and event < aware_timestamp(r["effective_at"]) <= last_relevant)
        if len(clocks) > 128:
            raise ValueError("label vintage work bound exceeded")
        vintages, previous = [], None
        for known in sorted(t for t in clocks if t <= cutoff):
            selected_grid = [t for t in relevant_grid if grid_knowledge[t] <= known][:request["horizon"]]
            if len(selected_grid) != request["horizon"]:
                continue
            future = [latest(instrument, stamp, known) for stamp in selected_grid]
            if any(r is None for r in future):
                continue
            label_rows = [current, *(r for r in future if r is not None)]
            relevant_actions = _visible(actions, instrument, event, selected_grid[-1], known)
            key = digest([label_rows, relevant_actions])
            if key == previous:
                continue
            label_prices, label_end, action_hash = _prices(label_rows, actions, known, request)
            label_end = max(label_end, selected_grid[-1])
            target = label_prices[-1] / label_prices[0] - 1
            if not math.isfinite(target):
                raise ValueError("target arithmetic exceeds finite numerical range")
            vintages.append({"target": target, "label_end": label_end.isoformat(),
                             "knowledge_at": max(known, label_end).isoformat(),
                             "label_action_sha256": action_hash, "label_source_sha256": digest(label_rows)})
            source_hashes.update(r["source_sha256"] for r in label_rows)
            action_hashes.add(action_hash)
            previous = key
        return vintages

    grid_knowledge = {event: min(aware_timestamp(r["available_at"]) for r in bars
                                 if aware_timestamp(r["timestamp"]) == event) for event in grid}

    diagnostics: dict[str, Any] = {"source_revisions": len(bars), "event_grid_observations": len(grid),
                                   "unknown_universe_decisions": 0, "missing_or_late_current_bars": 0,
                                   "incomplete_feature_windows": 0, "warmup_decisions": 0,
                                   "missing_future_labels": 0, "labels_not_future": 0,
                                   "decision_after_research_cutoff": 0, "observed_zero_volume_bars": 0}
    rows: list[dict[str, Any]] = []
    panel: list[dict[str, Any]] = []
    source_hashes: set[str] = set()
    universe_hashes: set[str] = set()
    action_hashes: set[str] = set()
    for index, event in enumerate(grid):
        decision = event + pd.Timedelta(seconds=request["decision_delay_seconds"])
        if decision > cutoff:
            diagnostics["decision_after_research_cutoff"] += 1
            continue
        valid_universes = [r for r in universes if aware_timestamp(r["effective_at"]) <= decision
                           and aware_timestamp(r["available_at"]) <= decision]
        if not valid_universes:
            diagnostics["unknown_universe_decisions"] += 1
            continue
        universe = max(valid_universes, key=lambda r: (aware_timestamp(r["effective_at"]), aware_timestamp(r["available_at"])))
        universe_hashes.add(universe["source_sha256"])
        history_grid = [stamp for stamp in grid[:index+1] if grid_knowledge[stamp] <= decision]
        for member in universe["members"]:
            if member["currency"] != request["currency"]:
                raise ValueError("universe currency differs from selected source")
            instrument = member["instrument_id"]
            current = latest(instrument, event, decision)
            if current is None:
                diagnostics["missing_or_late_current_bars"] += 1
                continue
            if len(panel) >= MAX_PANEL_ROWS:
                raise ValueError("research panel exceeds 1000 observations")
            source_hashes.add(current["source_sha256"])
            diagnostics["observed_zero_volume_bars"] += int(int(current["volume"]) == 0)
            previous = latest(instrument, history_grid[-2], decision) if len(history_grid) > 1 else None
            pair = [previous, current] if previous is not None else [current]
            pair_prices, _, _ = _prices(pair, actions, decision, request)
            previous_return = pair_prices[-1] / pair_prices[0] - 1 if len(pair_prices) == 2 else None
            panel.append({"instrument_id": instrument, "timestamp": decision.isoformat(),
                          "event_timestamp": event.isoformat(), "available_at": current["available_at"],
                          "close": current["close"], "raw_close": current["close"],
                          "volume": current["volume"], "return_1": previous_return,
                          "source_sha256": current["source_sha256"], "universe_sha256": universe["source_sha256"]})
            if len(history_grid) <= request["lookback"]:
                diagnostics["warmup_decisions"] += 1
                continue
            window = [latest(instrument, stamp, decision) for stamp in history_grid[-request["lookback"]-1:]]
            if any(r is None for r in window):
                diagnostics["incomplete_feature_windows"] += 1
                continue
            visible = [r for r in window if r is not None]
            prices, available, action_hash = _prices(visible, actions, decision, request)
            available = max(available, aware_timestamp(universe["available_at"]))
            if available > decision:
                raise ValueError("feature window contains future knowledge")
            values = np.asarray(prices, dtype=float)
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                returns = values[1:] / values[:-1] - 1
                features = {"momentum": float(values[-1] / values[0] - 1),
                            "reversal": float(-returns[-1]),
                            "volatility": float(np.std(returns, ddof=1)),
                            "log_volume": math.log1p(int(current["volume"]))}
            if not all(math.isfinite(x) for x in features.values()):
                raise ValueError("feature arithmetic exceeds finite numerical range")
            vintages = outcome_vintages(instrument, event, current, index)
            if not vintages:
                diagnostics["missing_future_labels"] += 1
                continue
            label = vintages[-1]
            label_end = aware_timestamp(label["label_end"])
            if label_end <= decision:
                diagnostics["labels_not_future"] += 1
                continue
            source_hashes.update(r["source_sha256"] for r in visible)
            action_hashes.add(action_hash)
            if len(rows) >= MAX_DATASET_ROWS:
                raise ValueError("research dataset exceeds 1000 examples")
            rows.append({"instrument_id": instrument, "timestamp": decision.isoformat(),
                         "available_at": available.isoformat(), "features": features,
                         "target": label["target"], "label_end": label_end.isoformat(), "label_vintages": vintages,
                         "feature_event_timestamp": event.isoformat(),
                         "feature_source_sha256": digest(visible), "universe_sha256": universe["source_sha256"],
                         "action_sha256": action_hash, "label_action_sha256": label["label_action_sha256"],
                         "label_source_sha256": label["label_source_sha256"]})
    rows.sort(key=lambda r: (aware_timestamp(r["timestamp"]), r["instrument_id"]))
    panel.sort(key=lambda r: (aware_timestamp(r["timestamp"]), r["instrument_id"]))
    counts = {instrument: sum(r["instrument_id"] == instrument for r in rows)
              for instrument in sorted({r["instrument_id"] for r in panel})}
    diagnostics.update({"training_rows_by_instrument": counts, "training_rows": len(rows),
                        "panel_rows": len(panel), "missing_values_imputed": 0,
                        "source_authenticity": "CALLER_ASSERTED_NOT_VENDOR_VERIFIED",
                        "action_coverage": "RAW_NO_ACTION_TRANSFORMATION" if request["adjustment"] == "RAW"
                        else "CALLER_DECLARED_EVENTS_ONLY_NOT_INDEPENDENTLY_COMPLETE"})
    if not counts or max(counts.values()) < 24:
        raise DatasetUnavailable("at least one instrument needs 24 complete causal training examples")
    body = {"schema_version": IMPLEMENTATION, "request": request, "source": request["source"],
            "classification": request["classification"], "as_of": request["as_of"], "rows": rows,
            "panel_rows": panel, "diagnostics": diagnostics,
            "provenance": {"bar_snapshot_hashes": sorted(source_hashes),
                           "universe_snapshot_hashes": sorted(universe_hashes),
                           "action_transformation_hashes": sorted(action_hashes),
                           "feature_implementation": "causal_simple_returns.v1",
                           "horizon_basis": "SOURCE_EVENT_GRID_NOT_INFERRED_EXCHANGE_SESSIONS",
                           "revision_policy": "LATEST_KNOWN_AT_EACH_DECISION",
                           "label_revision_policy": "VINTAGES_KNOWN_AT_EACH_TRAINING_CUTOFF",
                           "feature_adjustment_policy": request["adjustment"],
                           "stored_corporate_action_policy": "RAW",
                           "close_semantics": "Raw observed quote; adjusted return_1 is computed within each decision window",
                           "publication_time": "UNKNOWN_UNLESS_DECLARED_BY_ADMITTED_SOURCE",
                           "acquisition_time_semantics": "Original source available_at; never backdated"},
            "trading_approved": False, "economically_validated": False}
    return {"dataset_id": digest(body), **body}
