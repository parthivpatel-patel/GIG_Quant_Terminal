"""Actual admitted-run economics, causal exits, statistical and replay references."""
from __future__ import annotations

import copy
import sqlite3
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from gig.research.economic_validation import configuration, evaluate_runs, uncertainty
from gig.research.persisted_data import _read_universes
from gig.service.contracts import EnvelopeSchema
from gig.service.dataset_workspace import PersistedDatasetWorkspace
from gig.service.economic_workspace import EconomicResearchWorkspace
from gig.service.pairs_workspace import PairsResearchWorkspace
from gig.service.persisted_research import PersistedResearchWorkspace
from test_persisted_data import (
    fixture_bars,
    fixture_times,
    ingest_fixture_bars,
    seed_persisted_fixture,
)
from test_persisted_pairs import pair_bars


def economic_request(run_id, now, **changes):
    return {"kind": "CPU", "run_ids": [run_id], "now": now, "annualization": 252,
        "cost_bps_grid": [0, 10], "slippage_bps_grid": [0], "notional": 1000,
        "participation_limit": .1, "declared_trials": 2, "bootstrap_repetitions": 100,
        "block_size": 3, "seed": 17, "benchmark_instrument_id": "US:AAA",
        "factor_instrument_ids": {"admitted_BBB_proxy": "US:BBB"}, **changes}


class StatisticalReferenceTests(unittest.TestCase):
    def test_hac_reference_and_seeded_block_interval(self):
        x = np.random.default_rng(13).normal(.002, .02, 60)
        result = uncertainty(x.tolist(), block_size=3, repetitions=100, seed=17)
        centered = x-x.mean()
        variance = sum(centered**2)/len(x)
        for lag in (1, 2):
            variance += 2*(1-lag/3)*sum(centered[lag:]*centered[:-lag])/len(x)
        expected = np.sqrt(variance/len(x))
        self.assertAlmostEqual(result["mean_standard_error"], expected, places=14)
        self.assertAlmostEqual(result["positive_mean_pvalue"], stats.t.sf(x.mean()/expected, 59), places=14)
        self.assertEqual(result, uncertainty(x.tolist(), block_size=3, repetitions=100, seed=17))
        changed = uncertainty(x.tolist(), block_size=3, repetitions=100, seed=18)
        self.assertNotEqual(result["mean_return_ci95"], changed["mean_return_ci95"])
        self.assertEqual(uncertainty([.01]*40, block_size=3, repetitions=100, seed=17)["state"], "UNKNOWN")
        self.assertEqual(uncertainty([1., 2.], block_size=3, repetitions=100, seed=17)["state"], "UNAVAILABLE")


class EconomicWorkspaceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.base = Path(self.temporary.name)/"state"
        self.source_request = seed_persisted_fixture(self.base)
        self.dataset = PersistedDatasetWorkspace(self.base).build(self.source_request).to_dict()["data"]
        self.run = PersistedResearchWorkspace(self.base).run({"dataset_id": self.dataset["dataset_id"],
            "model": "ridge", "min_train": 25, "test_size": 10, "embargo": 1, "seed": 17}).to_dict()["data"]
        self.workspace = EconomicResearchWorkspace(self.base)
        self.request = economic_request(self.run["run_id"], self.source_request["as_of"])

    def tearDown(self):
        self.temporary.cleanup()

    def evaluate(self, dataset=None, run=None, request=None):
        universes = _read_universes(self.base, self.source_request)
        return evaluate_runs([run or self.run], dataset or self.dataset, universes,
                             configuration(request or self.request))

    def test_actual_saved_cpu_costs_capacity_attribution_and_family(self):
        result = self.workspace.run(self.request).to_dict()
        EnvelopeSchema.model_validate(result)
        self.assertEqual(result["state"], "AVAILABLE")
        data = result["data"]
        self.assertEqual(data["input_classification"], "SYNTHETIC")
        self.assertEqual(data["evaluation_state"], "AVAILABLE")
        low, high = data["rows"]
        self.assertLess(high["performance"]["compounded_return"], low["performance"]["compounded_return"])
        periods = data["experiments"][0]["periods"]
        expected_cost = sum(p["turnover"]*.001 for p in periods)
        self.assertAlmostEqual(high["cost_return_sum"], expected_cost, places=14)
        self.assertAlmostEqual(high["gross_return_sum"]-high["cost_return_sum"], high["net_return_sum"], places=14)
        for period in periods:
            self.assertAlmostEqual(period["gross_return"], sum(period["instrument_attribution"].values()), places=14)
            self.assertLess(pd.Timestamp(period["timestamp"]), pd.Timestamp(period["outcome_at"]))
        self.assertGreater(periods[-1]["terminal_liquidation_turnover"], 0)
        if high["uncertainty"]["positive_mean_pvalue"] is not None:
            self.assertEqual(high["bonferroni_positive_mean_pvalue"], min(1., high["uncertainty"]["positive_mean_pvalue"]*2))
        self.assertEqual(low["benchmark"]["state"], "AVAILABLE")
        self.assertEqual(low["factor_comparison"]["state"], "AVAILABLE")
        self.assertEqual(low["degradation"]["training_to_oos"]["state"], "UNKNOWN")
        self.assertFalse(data["economically_validated"])
        self.assertFalse(data["production_suitable"])
        self.assertEqual(data["family"]["declared_trials"], 2)
        self.assertFalse(data["family"]["family_completeness_verified"])

    def test_reopen_exact_replay_detachment_and_index_tamper(self):
        original = self.workspace.run(self.request).to_dict()
        identity = original["data"]["report_id"]
        reopened = EconomicResearchWorkspace(self.base)
        self.assertEqual(reopened.run(self.request).to_dict(), original)
        self.assertEqual(reopened.replay({"report_id": identity}).to_dict(), original)
        loaded = reopened.load(identity)
        loaded["rows"][0]["cost_bps"] = 999
        self.assertEqual(reopened.load(identity)["rows"][0]["cost_bps"], 0)
        self.assertEqual(reopened.section("models", self.request["now"], None).to_dict()["state"], "UNKNOWN")
        later = (pd.Timestamp(self.request["now"])+pd.Timedelta(seconds=61)).isoformat()
        self.assertEqual(reopened.section("research", later, 60).to_dict()["state"], "STALE")
        self.assertIsNone(reopened.section("risk", later, 60))
        with sqlite3.connect(self.base/"economic-research/reports.sqlite") as connection:
            connection.execute("UPDATE reports SET as_of='2026-01-01T00:00:00+00:00'")
        with self.assertRaisesRegex(ValueError, "index"):
            reopened.load(identity)

    def test_missing_held_security_is_unknown_not_dropped_or_zero(self):
        before = self.evaluate()
        chosen = before["experiments"][0]["periods"][5]
        held = next(s for s, w in chosen["weights"].items() if w)
        changed = copy.deepcopy(self.dataset)
        changed["panel_rows"] = [r for r in changed["panel_rows"]
            if not (r["instrument_id"] == held and r["timestamp"] == chosen["outcome_at"])]
        after = self.evaluate(dataset=changed)
        self.assertEqual(after["evaluation_state"], "UNKNOWN")
        self.assertIsNone(after["rows"][0]["performance"]["compounded_return"])
        self.assertIsNone(after["rows"][0]["net_return_sum"])
        self.assertEqual(after["experiments"][0]["survivorship"]["dropped_unknown_periods"], 0)
        self.assertGreater(after["experiments"][0]["unknown_periods"], 0)
        self.assertEqual(len(before["experiments"][0]["periods"]), len(after["experiments"][0]["periods"]))

    def test_missing_eligible_prediction_and_future_training_reject(self):
        changed = copy.deepcopy(self.run)
        changed["rows"].pop(12)
        self.assertEqual(self.evaluate(run=changed)["evaluation_state"], "UNKNOWN")
        changed = copy.deepcopy(self.run)
        changed["folds"][0]["max_training_label_end"] = changed["folds"][0]["test_start"]
        with self.assertRaisesRegex(ValueError, "overlap"):
            self.evaluate(run=changed)
        changed = copy.deepcopy(self.run)
        changed["rows"][0]["timestamp"] = self.source_request["start"]
        with self.assertRaisesRegex(ValueError, "outside"):
            self.evaluate(run=changed)

    def test_entire_missing_oos_cross_section_remains_an_unknown_period(self):
        before = self.evaluate()
        stamp = before["experiments"][0]["periods"][5]["timestamp"]
        changed = copy.deepcopy(self.run)
        changed["rows"] = [row for row in changed["rows"] if row["timestamp"] != stamp]
        after = self.evaluate(run=changed)
        self.assertEqual(len(after["experiments"][0]["periods"]), len(before["experiments"][0]["periods"]))
        missing = next(period for period in after["experiments"][0]["periods"] if period["timestamp"] == stamp)
        self.assertEqual(missing["state"], "UNKNOWN")
        self.assertIn("MISSING_ELIGIBLE_SECURITY_PREDICTION", missing["unknown_reasons"])
        self.assertEqual(after["evaluation_state"], "UNKNOWN")
        self.assertIsNone(after["rows"][0]["performance"]["compounded_return"])

    def test_capacity_failure_zero_volume_and_incomplete_trial_family(self):
        large = self.evaluate(request={**self.request, "notional": 1e10})
        self.assertEqual(large["rows"][0]["capacity_state"], "BREACH")
        changed = copy.deepcopy(self.dataset)
        for row in changed["panel_rows"]:
            row["volume"] = 0
        zero = self.evaluate(dataset=changed)
        self.assertEqual(zero["rows"][0]["capacity_state"], "BREACH")
        self.assertEqual(zero["rows"][0]["minimum_capacity_notional"], 0)
        with self.assertRaisesRegex(ValueError, "family"):
            self.workspace.run({**self.request, "declared_trials": 1})
        larger_family = self.evaluate(request={**self.request, "declared_trials": 100})
        self.assertEqual(larger_family["family"]["unobserved_trials"], 98)
        self.assertFalse(larger_family["economically_validated"])

    def test_later_acquisition_correction_preserves_saved_report_and_old_decisions(self):
        original = self.workspace.run(self.request).to_dict()
        revision = copy.deepcopy(fixture_bars()[60])
        revision["available_at"] = (fixture_times()[-1]+pd.Timedelta(days=2)).isoformat()
        for name in ("open", "high", "low", "close"):
            revision[name] *= 2
        ingest_fixture_bars(self.base, [revision])
        self.assertEqual(EconomicResearchWorkspace(self.base).replay({"report_id": original["data"]["report_id"]}).to_dict(), original)
        self.assertEqual(EconomicResearchWorkspace(self.base).run(self.request).to_dict(), original)
        future_request = {**self.source_request, "as_of": revision["available_at"]}
        later_dataset = PersistedDatasetWorkspace(self.base).build(future_request).to_dict()["data"]
        # The causal panel and saved decisions remain unchanged even though
        # later training labels can acquire a correction vintage.
        later = self.evaluate(dataset=later_dataset, request={**self.request, "now": revision["available_at"]})
        self.assertEqual(later["experiments"][0]["periods"], original["data"]["experiments"][0]["periods"])

    def test_actual_pairs_experiment_family_and_missing_reads(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            request = seed_persisted_fixture(base, periods=120, rows=pair_bars())
            dataset = PersistedDatasetWorkspace(base).build(request).to_dict()["data"]
            pairs = PairsResearchWorkspace(base).run({"dataset_id": dataset["dataset_id"], "entry_z": 0.,
                "training_window": 60, "test_size": 15, "embargo": 1}).to_dict()["data"]
            payload = economic_request(pairs["run_id"], request["as_of"], kind="PAIRS")
            result = EconomicResearchWorkspace(base).run(payload).to_dict()
            self.assertEqual(result["state"], "AVAILABLE")
            self.assertEqual(result["data"]["experiments"][0]["model"], "PAIRS")
            # Final pair proposal has no future observation; its unknown exit
            # is not discarded to present a deceptively complete backtest.
            self.assertEqual(result["data"]["evaluation_state"], "UNKNOWN")
        missing = EconomicResearchWorkspace(self.base/"missing")
        self.assertEqual(missing.replay({"report_id": "0"*64}).to_dict()["state"], "UNAVAILABLE")
        self.assertEqual(missing.run(self.request).to_dict()["state"], "UNAVAILABLE")
        self.assertIsNone(missing.section("research", self.request["now"], 60))
        self.assertFalse((self.base/"missing").exists())

    def test_unsupported_horizon_garch_source_and_invalid_parameters(self):
        changed = copy.deepcopy(self.dataset)
        changed["request"]["horizon"] = 2
        with self.assertRaisesRegex(ValueError, "one-step"):
            self.evaluate(dataset=changed)
        changed = copy.deepcopy(self.run)
        changed["usable_for_portfolio"] = False
        with self.assertRaisesRegex(ValueError, "not economic"):
            self.evaluate(run=changed)
        for update in ({"notional": True}, {"participation_limit": 2}, {"cost_bps_grid": [0, 0]},
                       {"bootstrap_repetitions": 2}, {"run_ids": ["not-a-sha"]}, {"extra": 1}):
            with self.subTest(update=update), self.assertRaises(ValueError):
                self.workspace.run({**self.request, **update})
