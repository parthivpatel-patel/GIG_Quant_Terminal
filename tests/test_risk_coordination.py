"""Atomic risk/kill publication, process termination, exact clock boundaries."""
from __future__ import annotations

import os
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from datetime import timedelta
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from gig.execution.oms import OfflineOMS, OrderError
from gig.service.execution_workspace import _stamp
from gig.service.portfolio_workspace import PortfolioWorkspace
from gig.service.risk_coordination import reconcile_with_risk_controls
from gig.service.serialization import dumps
from test_portfolio_ledger import NOW, populate, request


class AtomicRiskTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.base = Path(self.temporary.name) / "state"
        populate(self.base)

    def tearDown(self):
        self.temporary.cleanup()

    def oms(self):
        return OfflineOMS(self.base / "execution-engine/oms.sqlite")

    def assert_unpublished(self):
        with self.oms() as oms:
            self.assertFalse(oms.kill_switch)
            self.assertFalse(any(row["state"] == "KILL_SWITCH" for row in oms.events()))
        ledger = PortfolioWorkspace(self.base)
        self.assertEqual(ledger._history(), [])
        self.assertIsNone(ledger.section("risk", NOW + timedelta(minutes=3), 60))

    def breach(self):
        return request(limits={"max_gross": .001})

    def test_nanosecond_preflight_precedes_any_ledger_or_control_mutation(self):
        for value in ("2026-11-02T14:00:00.000000001Z", "2026-11-02T14:00:00.999999999Z",
                      "2026-11-02T14:00:00.1234567890Z", "NaT", "2026-11-02T14:00:00"):
            payload = self.breach()
            payload["now"] = value
            with self.subTest(timestamp=value), self.assertRaises(ValueError):
                reconcile_with_risk_controls(self.base, payload)
            self.assertFalse((self.base / "portfolio-ledger").exists())
            self.assert_unpublished()

    def test_oms_direct_pandas_nanoseconds_cannot_bypass_precision_contract(self):
        with self.oms() as oms:
            before = oms.events()
            with self.assertRaises(OrderError):
                oms.set_kill_switch(True, now=pd.Timestamp("2026-11-02T14:00:00.000000001Z"))
            with self.assertRaises(OrderError):
                oms.create("invalid-clock", "A", "buy", "1", now=pd.Timestamp("2026-11-02T14:00:00.999999999Z"))
            self.assertEqual(oms.events(), before)
            self.assertFalse(oms.kill_switch)

    def test_exact_microsecond_edges_offsets_and_repeated_restart_are_idempotent(self):
        for stamp in ("2026-11-02T14:00:00.000000000Z", "2026-11-02T14:00:00.999999000Z",
                      "2026-11-02T09:00:00.999999-05:00", "1677-09-21T00:12:43.145225Z",
                      "2262-04-11T23:47:16.854775Z"):
            self.assertEqual(_stamp(stamp).isoformat(), pd.Timestamp(stamp).tz_convert("UTC").isoformat())
        payload = self.breach()
        payload["now"] = "2026-11-02T09:00:00.999999-05:00"
        result = reconcile_with_risk_controls(self.base, payload).to_dict()
        self.assertEqual(result["state"], "ERROR")
        self.assertEqual(result, reconcile_with_risk_controls(self.base, payload).to_dict())
        with self.oms() as oms:
            self.assertTrue(oms.kill_switch)
            self.assertEqual(sum(row["state"] == "KILL_SWITCH" for row in oms.events()), 1)
            self.assertEqual(oms.connection.execute("SELECT COUNT(*) FROM portfolio_checkpoints").fetchone()[0], 1)
        self.assertEqual(len(PortfolioWorkspace(self.base)._history()), 1)

    def test_explicit_release_does_not_suppress_repeated_error_controls(self):
        first = reconcile_with_risk_controls(self.base, self.breach()).wire
        with self.oms() as oms:
            oms.set_kill_switch(False, now=NOW)
            self.assertFalse(oms.kill_switch)
        self.assertEqual(first, reconcile_with_risk_controls(self.base, self.breach()).wire)
        with self.oms() as oms:
            self.assertTrue(oms.kill_switch)
            self.assertEqual(oms.connection.execute("SELECT COUNT(*) FROM portfolio_checkpoints").fetchone()[0], 1)
        reconcile_with_risk_controls(self.base, request(now=NOW + timedelta(seconds=1)))
        with self.oms() as oms:
            self.assertTrue(oms.kill_switch)

    def test_mismatched_positions_nonpositive_nav_and_missing_factor_are_errors(self):
        for kind in ("mismatch", "nonpositive_nav", "missing_factor", "drawdown"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as directory:
                base = Path(directory)
                populate(base)
                payload = request()
                if kind == "mismatch":
                    with sqlite3.connect(base / "execution-engine/oms.sqlite") as connection:
                        connection.execute("UPDATE positions SET quantity='9'")
                elif kind == "nonpositive_nav":
                    populate(base, client="short", symbol="B", side="sell", quantity="100", price="50")
                    payload["marks"]["B"]["price"] = "1000"
                elif kind == "missing_factor":
                    payload.update(metadata={}, limits={"max_factor_exposure": {"market": .5}})
                else:
                    reconcile_with_risk_controls(base, request())
                    payload = request(now=NOW + timedelta(seconds=1), limits={"max_drawdown": .001})
                    payload["marks"]["A"]["price"] = "10"
                result = reconcile_with_risk_controls(base, payload).to_dict()
                self.assertEqual(result["state"], "ERROR")
                with OfflineOMS(base / "execution-engine/oms.sqlite") as oms:
                    self.assertTrue(oms.kill_switch)

    def test_failure_after_either_sql_step_rolls_back_both_and_retry_recovers(self):
        for phase in ("checkpoint_indexed", "risk_controls_applied"):
            def fail(name, target=phase):
                if name == target:
                    raise RuntimeError("injected interruption")
            with self.subTest(phase=phase), self.assertRaises(RuntimeError):
                reconcile_with_risk_controls(self.base, self.breach(), fault_hook=fail)
            # Synced immutable orphan bytes may exist; they are not indexed or
            # exposed as an admitted risk result after a failed transaction.
            self.assert_unpublished()
        result = reconcile_with_risk_controls(self.base, self.breach()).to_dict()
        self.assertEqual(result["state"], "ERROR")
        with self.oms() as oms:
            self.assertTrue(oms.kill_switch)
        self.assertEqual(len(PortfolioWorkspace(self.base)._history()), 1)

    def test_process_death_before_commit_recovers_exactly(self):
        code = (
            "import os,sys;from pathlib import Path;"
            "from gig.service.risk_coordination import reconcile_with_risk_controls;"
            "from gig.service.serialization import loads;"
            "reconcile_with_risk_controls(Path(sys.argv[1]),loads(sys.argv[3]),"
            "fault_hook=lambda phase:os._exit(67) if phase==sys.argv[2] else None)"
        )
        for phase in ("checkpoint_indexed", "risk_controls_applied"):
            with self.subTest(phase=phase):
                completed = subprocess.run([sys.executable, "-c", code, str(self.base), phase,
                    dumps(self.breach())], env=os.environ.copy(), capture_output=True, timeout=30)
                self.assertEqual(completed.returncode, 67, completed.stderr.decode())
                self.assert_unpublished()
        first = reconcile_with_risk_controls(self.base, self.breach()).wire
        self.assertEqual(first, reconcile_with_risk_controls(self.base, self.breach()).wire)

    def test_artifact_failure_never_publishes_risk_or_kill(self):
        with patch("gig.service.portfolio_workspace.write_immutable", side_effect=OSError("disk fault")), self.assertRaises(OSError):
            reconcile_with_risk_controls(self.base, self.breach())
        self.assert_unpublished()

    def test_kill_method_failure_rolls_back_index_and_preserves_previous_history(self):
        good = reconcile_with_risk_controls(self.base, request()).wire
        with patch.object(OfflineOMS, "set_kill_switch", side_effect=sqlite3.OperationalError("write fault")), self.assertRaises(sqlite3.OperationalError):
            reconcile_with_risk_controls(self.base, self.breach())
        history = PortfolioWorkspace(self.base)._history()
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]["state"], "AVAILABLE")
        with self.oms() as oms:
            self.assertFalse(oms.kill_switch)
        self.assertEqual(good, reconcile_with_risk_controls(self.base, request()).wire)

    def test_non_error_states_never_engage_or_release_switch(self):
        cases = ((request(), "AVAILABLE"),
                 (request(now=NOW + timedelta(seconds=61), marks=request()["marks"]), "STALE"),
                 (request(now=NOW + timedelta(seconds=62), max_age_seconds=None), "UNKNOWN"),
                 (request(now=NOW + timedelta(seconds=63), marks={}), "UNAVAILABLE"))
        for payload, expected in cases:
            with self.subTest(state=expected):
                self.assertEqual(reconcile_with_risk_controls(self.base, payload).to_dict()["state"], expected)
                with self.oms() as oms:
                    self.assertFalse(oms.kill_switch)
        reconcile_with_risk_controls(self.base, request(now=NOW + timedelta(seconds=64), limits={"max_gross": .001}))
        self.assertEqual(reconcile_with_risk_controls(self.base, request(now=NOW + timedelta(seconds=65))).to_dict()["state"], "AVAILABLE")
        with self.oms() as oms:
            self.assertTrue(oms.kill_switch)

    def test_supported_risk_breaches_engage_existing_oms_switch(self):
        for changes in ({"limits": {"max_gross": .001}}, {"limits": {"max_name_weight": .001}},
                        {"limits": {"min_cash": "10000"}},
                        {"limits": {"max_factor_exposure": {"market": 0}}},
                        {"metadata": {}, "limits": {"max_sector_gross": .5}}):
            with self.subTest(changes=changes), tempfile.TemporaryDirectory() as directory:
                base = Path(directory)
                populate(base)
                result = reconcile_with_risk_controls(base, request(**changes)).to_dict()
                self.assertEqual(result["state"], "ERROR")
                with OfflineOMS(base / "execution-engine/oms.sqlite") as oms:
                    self.assertTrue(oms.kill_switch)

    def test_research_ledger_preserves_nanoseconds_and_legacy_checkpoints(self):
        ledger = PortfolioWorkspace(self.base)
        legacy = ledger.run(request()).to_dict()
        payload = self.breach()
        payload["now"] = "2026-11-02T14:00:00.000000001Z"
        for mark in payload["marks"].values():
            mark["as_of"] = payload["now"]
        research = ledger.run(payload).to_dict()
        saved = ledger.load(research["provenance"]["source_identity"])
        self.assertIn("000000001", saved["as_of"])
        later = request(now=NOW + timedelta(seconds=1), limits={"max_gross": .001})
        reconcile_with_risk_controls(self.base, later)
        self.assertEqual(len(PortfolioWorkspace(self.base)._history()), 3)
        self.assertEqual(ledger.load(legacy["data"]["checkpoint_id"])["state"], "AVAILABLE")

    def test_clock_reversal_rejects_before_checkpoint_publication(self):
        reconcile_with_risk_controls(self.base, request(now=NOW + timedelta(seconds=1)))
        with self.assertRaisesRegex(ValueError, "clock"):
            reconcile_with_risk_controls(self.base, self.breach())
        self.assertEqual(len(PortfolioWorkspace(self.base)._history()), 1)
