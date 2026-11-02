"""OOS label-vintage scoring through actual allocation and offline accounting."""
from __future__ import annotations

import sqlite3
import tempfile
import unittest
from copy import deepcopy
from decimal import Decimal
from pathlib import Path

import pandas as pd

from gig.execution.oms import OfflineOMS
from gig.service.allocation_workspace import ResearchAllocationWorkspace
from gig.service.contracts import EnvelopeSchema
from gig.service.dataset_workspace import PersistedDatasetWorkspace
from gig.service.execution_workspace import ExecutionWorkspace
from gig.service.persisted_research import PersistedResearchWorkspace
from gig.service.portfolio_workspace import PortfolioWorkspace
from gig.service.signal_workspace import SignalWorkspace, score_runs
from test_persisted_flow_http import allocation_request, seed_flow


def scoring_fixture():
    """Reference values for the scoring formula, not fitted-model evidence."""
    times = pd.date_range("2026-01-01T00:00:00Z", periods=10)
    rows, labels = [], []
    for stamp in times:
        for i, symbol in enumerate(("A", "B", "C")):
            rows.append({"instrument_id": symbol, "timestamp": stamp.isoformat(), "prediction": float(i), "model_id": "fixture-model"})
            labels.append({"instrument_id": symbol, "timestamp": stamp.isoformat(), "target": float(i)/100,
                "label_end": (stamp+pd.Timedelta(hours=1)).isoformat(),
                "label_vintages": [{"target": float(i)/100,
                    "label_end": (stamp+pd.Timedelta(hours=1)).isoformat(),
                    "knowledge_at": (stamp+pd.Timedelta(hours=1)).isoformat()}]})
    dataset = {"dataset_id": "fixture-dataset", "source": "authored-scoring-reference", "classification": "SYNTHETIC", "rows": labels}
    runs = [{"run_id": "fixture-run", "rows": rows, "model_ids": ["fixture-model"], "model": "reference",
             "prediction_kind": "AUTHORED_NUMERICAL_REFERENCE"}]
    request = {"now": times[-1].isoformat(), "run_ids": ["fixture-run"], "lookback": 20, "min_ic_observations": 5}
    return runs, dataset, request


class SignalScoringReferenceTests(unittest.TestCase):
    def test_positive_ic_reference_and_unknown_calibration(self):
        runs, dataset, request = scoring_fixture()
        result = score_runs(runs, dataset, request)
        self.assertEqual(result["scoring_state"], "AVAILABLE")
        self.assertEqual(result["diagnostics"][0]["mean_oos_rank_ic"], 1.)
        self.assertAlmostEqual(result["diagnostics"][0]["raw_adaptive_weight"], 9/29)
        self.assertEqual([r["prediction"] for r in result["rows"]], [-.5, 0., .5])
        self.assertEqual(result["diagnostics"][0]["confidence"]["state"], "UNKNOWN")
        self.assertFalse(result["trading_approved"])

    def test_later_revisions_and_current_outcome_do_not_change_earlier_signal(self):
        runs, dataset, request = scoring_fixture()
        expected = score_runs(runs, dataset, request)
        changed = deepcopy(dataset)
        cutoff = request["now"]
        for row in changed["rows"]:
            # Both equality and future availability are excluded by strict '<'.
            row["label_vintages"].append({"target": -999., "label_end": cutoff, "knowledge_at": cutoff})
            row["label_vintages"].append({"target": 999., "label_end": cutoff,
                "knowledge_at": (pd.Timestamp(cutoff)+pd.Timedelta(nanoseconds=1)).isoformat()})
            row["target"] = 1234.
        self.assertEqual(score_runs(runs, changed, request), expected)

    def test_negative_ic_never_flips_sign_and_insufficient_is_unknown(self):
        runs, dataset, request = scoring_fixture()
        for row in runs[0]["rows"]:
            row["prediction"] *= -1
        result = score_runs(runs, dataset, request)
        self.assertFalse(result["usable_for_portfolio"])
        self.assertEqual(result["rows"], [])
        self.assertEqual(result["diagnostics"][0]["raw_adaptive_weight"], 0)
        request["min_ic_observations"] = 20
        self.assertEqual(score_runs(runs, dataset, request)["diagnostics"][0]["ic_state"], "UNKNOWN")


class SignalWorkspaceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.base = Path(self.temporary.name)
        self.request = seed_flow(self.base)
        self.now = self.request["as_of"]
        self.dataset = PersistedDatasetWorkspace(self.base).build(self.request).to_dict()["data"]
        self.research = PersistedResearchWorkspace(self.base)
        self.run = self.research.run({"dataset_id": self.dataset["dataset_id"], "model": "lightgbm",
            "min_train": 30, "test_size": 12, "embargo": 1, "seed": 17}).to_dict()["data"]
        self.signals = SignalWorkspace(self.base)
        self.payload = {"run_ids": [self.run["run_id"]], "now": self.now, "lookback": 20, "min_ic_observations": 5}

    def tearDown(self):
        self.temporary.cleanup()

    def signal(self):
        result = self.signals.run(self.payload).to_dict()
        EnvelopeSchema.model_validate(result)
        self.assertEqual(result["state"], "AVAILABLE")
        self.assertTrue(result["data"]["usable_for_portfolio"])
        return result

    def test_actual_lightgbm_signal_restart_and_deterministic_replay(self):
        result = self.signal()
        self.assertEqual(self.signals.run(self.payload).to_dict(), result)
        reopened = SignalWorkspace(self.base)
        self.assertEqual(reopened.replay({"signal_id": result["data"]["signal_id"]}).to_dict(), result)
        self.assertEqual(reopened.section("signals", self.now, 60).to_dict(), result)
        self.assertEqual(reopened.section("signals", self.now, None).to_dict()["state"], "UNKNOWN")
        self.assertEqual(reopened.section("signals", pd.Timestamp(self.now)+pd.Timedelta(seconds=61), 60).to_dict()["state"], "STALE")
        self.assertIsNone(reopened.section("signals", pd.Timestamp(self.now)-pd.Timedelta(nanoseconds=1), 60))

    def test_actual_signal_to_allocation_partial_fills_cash_and_restart(self):
        signal = self.signal()["data"]
        request = allocation_request("", self.now)
        del request["run_id"]
        request.update(signal_id=signal["signal_id"], max_turnover=.3, max_transaction_cost="100")
        allocations = ResearchAllocationWorkspace(self.base)
        allocation = allocations.build(request).to_dict()
        self.assertEqual(allocation["state"], "AVAILABLE")
        data = allocation["data"]
        self.assertEqual(data["source_kind"], "signal_id")
        self.assertIsNone(data["run_id"])
        self.assertGreater(data["metrics"]["turnover_nav"], 0)
        self.assertFalse((self.base/"execution-engine/oms.sqlite").exists())
        prepared_request = {"allocation_id": data["allocation_id"], "now": self.now}
        prepared = allocations.prepare(prepared_request).to_dict()
        self.assertEqual(ResearchAllocationWorkspace(self.base).prepare(prepared_request).to_dict(), prepared)
        execution = ExecutionWorkspace(self.base/"execution-engine")
        execution.run({"operation": "dispatch", "now": self.now})
        for client, row in zip(prepared["data"]["prepared_client_ids"], prepared["data"]["rows"], strict=True):
            execution.run({"operation": "fill", "client_id": client, "quantity": "1", "price": row["reference_price"], "fee": "0.1", "now": self.now})
        execution.run({"operation": "reconcile", "now": self.now})
        account_request = {"operation": "reconcile", "initial_cash": "100000", "now": self.now,
            "marks": {r["instrument_id"]: {"price": r["reference_price"], "as_of": self.now} for r in data["rows"]},
            "max_age_seconds": 60, "metadata": request["metadata"], "limits": request["limits"]}
        account = PortfolioWorkspace(self.base).run(account_request).to_dict()
        self.assertEqual(account["state"], "AVAILABLE")
        self.assertEqual(Decimal(account["data"]["metrics"]["nav"]), Decimal("99999.8"))
        self.assertEqual(PortfolioWorkspace(self.base).run(account_request).to_dict(), account)

    def test_admitted_late_label_correction_cannot_rewrite_prior_ensemble_score(self):
        from gig.data.validated import PointInTimeStore, admit_bars
        from test_persisted_data import fixture_bars
        # Build both experiments with the same final acquisition cutoff. The
        # correction arrives after every decision in the observation window.
        request = {**self.request, "as_of": (pd.Timestamp(self.now)+pd.Timedelta(days=3)).isoformat()}
        dataset_workspace = PersistedDatasetWorkspace(self.base)
        before_dataset = dataset_workspace.build(request).to_dict()["data"]
        config = {"model": "lightgbm", "min_train": 30, "test_size": 12, "embargo": 1, "seed": 17}
        before_run = self.research.run({**config, "dataset_id": before_dataset["dataset_id"]}).to_dict()["data"]
        payload = {"run_ids": [before_run["run_id"]], "now": request["as_of"]}
        before = self.signals.run(payload).to_dict()["data"]
        row = deepcopy(fixture_bars()[40])
        row["timestamp"] = pd.Timestamp(row["timestamp"]).floor("us").isoformat()
        row["available_at"] = (pd.Timestamp(self.now)+pd.Timedelta(days=1)).isoformat()
        for key in ("open", "high", "low", "close"):
            row[key] *= 1.5
        PointInTimeStore(self.base/"bars.sqlite").ingest(admit_bars([row], source=self.request["source"],
            classification="SYNTHETIC", corporate_action_policy="RAW", currency="USD"))
        after_dataset = dataset_workspace.build(request).to_dict()["data"]
        self.assertNotEqual(before_dataset["dataset_id"], after_dataset["dataset_id"])
        after_run = self.research.run({**config, "dataset_id": after_dataset["dataset_id"]}).to_dict()["data"]
        after = self.signals.run({**payload, "run_ids": [after_run["run_id"]]}).to_dict()["data"]
        self.assertEqual(before["rows"], after["rows"])
        for a, b in zip(before["diagnostics"], after["diagnostics"], strict=True):
            self.assertEqual(a["selected_label_evidence_hash"], b["selected_label_evidence_hash"])
            self.assertEqual(a["mean_oos_rank_ic"], b["mean_oos_rank_ic"])
            self.assertEqual(a["adaptive_weight"], b["adaptive_weight"])
        # Reopening the original evidence remains exact after the correction.
        self.assertEqual(SignalWorkspace(self.base).load(before["signal_id"])["rows"], before["rows"])

    def test_turnover_and_cost_limits_reject_before_offline_intent(self):
        signal = self.signal()["data"]
        request = allocation_request("", self.now)
        del request["run_id"]
        request["signal_id"] = signal["signal_id"]
        for cap in [{"max_turnover": 0}, {"max_transaction_cost": "0"}]:
            self.assertEqual(ResearchAllocationWorkspace(self.base).build({**request, **cap}).to_dict()["state"], "ERROR")
            self.assertFalse((self.base/"execution-engine").exists())
        with self.assertRaises(ValueError):
            ResearchAllocationWorkspace(self.base).build({**request, "run_id": self.run["run_id"]})

    def test_unknown_ic_not_eligible_for_allocation_and_bad_models_reject(self):
        request = {**self.payload, "min_ic_observations": 100, "lookback": 100}
        result = self.signals.run(request).to_dict()
        self.assertEqual(result["data"]["scoring_state"], "UNKNOWN")
        allocation = allocation_request("", self.now)
        del allocation["run_id"]
        allocation["signal_id"] = result["data"]["signal_id"]
        self.assertEqual(ResearchAllocationWorkspace(self.base).build(allocation).to_dict()["state"], "UNAVAILABLE")
        for extra in [{"run_ids": ["../x"]}, {"run_ids": [self.run["run_id"]]*2},
                      {"now": (pd.Timestamp(self.now)-pd.Timedelta(nanoseconds=1)).isoformat()},
                      {"lookback": True}]:
            with self.assertRaises(ValueError):
                self.signals.run({**self.payload, **extra})

    def test_tamper_defensive_copy_and_readonly_missing(self):
        result = self.signal()
        identity = result["data"]["signal_id"]
        loaded = self.signals.load(identity)
        loaded["rows"][0]["prediction"] = 999
        self.assertNotEqual(self.signals.load(identity)["rows"], loaded["rows"])
        with sqlite3.connect(self.base/"research-signals/signals.sqlite") as connection:
            connection.execute("UPDATE signals SET as_of='2025-01-01T00:00:00+00:00'")
        with self.assertRaises(ValueError):
            self.signals.load(identity)
        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory)/"not-created"
            workspace = SignalWorkspace(missing)
            self.assertIsNone(workspace.section("signals", self.now, 60))
            self.assertEqual(workspace.run({"run_ids": ["0"*64], "now": self.now}).to_dict()["state"], "UNAVAILABLE")
            self.assertFalse(missing.exists())


    def test_actual_pair_hedge_proposal_preserves_common_scaling_and_prepares(self):
        from gig.service.pairs_workspace import PairsResearchWorkspace
        from test_persisted_data import seed_persisted_fixture
        from test_persisted_pairs import pair_bars
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            bars = pair_bars()
            for row in bars:
                row["timestamp"] = pd.Timestamp(row["timestamp"]).floor("us").isoformat()
                row["available_at"] = row["timestamp"]
            dataset_request = seed_persisted_fixture(base, periods=120, rows=bars)
            dataset_request.update(start=bars[0]["timestamp"], end=bars[-1]["timestamp"], as_of=bars[-1]["timestamp"])
            dataset = PersistedDatasetWorkspace(base).build(dataset_request).to_dict()["data"]
            run = PairsResearchWorkspace(base).run({"dataset_id": dataset["dataset_id"],
                "training_window": 60, "test_size": 15, "entry_z": 0.}).to_dict()["data"]
            self.assertTrue(run["usable_for_portfolio"])
            request = allocation_request("", dataset_request["as_of"])
            del request["run_id"]
            request["pair_run_id"] = run["run_id"]
            allocations = ResearchAllocationWorkspace(base)
            allocation = allocations.build(request).to_dict()
            self.assertEqual(allocation["state"], "AVAILABLE")
            data = allocation["data"]
            self.assertEqual(data["source_kind"], "pair_run_id")
            proposal = run["proposals"][0]["weights"]
            scale = min(1., request["gross_leverage"]/sum(abs(v) for v in proposal.values()),
                        request["max_name_weight"]/max(abs(v) for v in proposal.values()))
            for row in data["rows"]:
                reference = proposal[row["instrument_id"]]*scale*100000/float(row["reference_price"])
                self.assertLess(abs(float(row["target_quantity"])-reference), 1.)
                self.assertEqual(float(row["target_quantity"]) > 0, proposal[row["instrument_id"]] > 0)
            prepared = allocations.prepare({"allocation_id": data["allocation_id"], "now": dataset_request["as_of"]}).to_dict()
            self.assertEqual(prepared["state"], "AVAILABLE")
            with OfflineOMS(base/"execution-engine/oms.sqlite") as oms:
                self.assertEqual(oms.positions(), {})

    def test_nanosecond_execution_clock_is_preflighted_before_accounting(self):
        signal = self.signal()["data"]
        with OfflineOMS(self.base/"execution-engine/oms.sqlite"):
            pass
        request = allocation_request("", (pd.Timestamp(self.now)+pd.Timedelta(nanoseconds=1)).isoformat())
        del request["run_id"]
        request["signal_id"] = signal["signal_id"]
        with self.assertRaises(ValueError):
            ResearchAllocationWorkspace(self.base).build(request)
        self.assertFalse((self.base/"portfolio-ledger").exists())
