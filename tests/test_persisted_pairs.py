"""Library reference, causal revisions, costs and immutable pairs replay."""
from __future__ import annotations

import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd

from gig.data.actions import ActionStore, admit_actions
from gig.research.persisted_pairs import _panel, configuration, fit_pairs
from gig.service.contracts import EnvelopeSchema
from gig.service.dataset_workspace import PersistedDatasetWorkspace
from gig.service.pairs_workspace import PairsResearchWorkspace
from test_persisted_data import (
    fixture_times,
    ingest_fixture_bars,
    seed_persisted_fixture,
)


def pair_bars(periods=120):
    """Authored cointegrated log prices, never provider-observed observations."""
    rng = np.random.default_rng(19)
    log_x = 4.6 + np.cumsum(rng.normal(.001, .025, periods))
    residual = rng.normal(0, .008, periods)
    log_y = .2 + 1.04*log_x + residual
    rows = []
    for index, stamp in enumerate(fixture_times(periods)):
        for identity, value in (("US:AAA", np.exp(log_y[index])), ("US:BBB", np.exp(log_x[index]))):
            value = float(value)
            rows.append({"instrument_id": identity, "timestamp": stamp.isoformat(),
                         "available_at": stamp.isoformat(), "open": value, "high": value,
                         "low": value, "close": value, "volume": 1000})
    return rows


class PersistedPairTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.base = Path(self.temporary.name)/"workspace"
        request = seed_persisted_fixture(self.base, periods=120, rows=pair_bars())
        self.dataset = PersistedDatasetWorkspace(self.base).build(request).to_dict()["data"]
        self.request = request
        self.workspace = PairsResearchWorkspace(self.base)
        self.config = {"dataset_id": self.dataset["dataset_id"], "training_window": 60,
                       "test_size": 15, "embargo": 1, "entry_z": 0., "cost_bps": 12., "borrow_bps": 50.}

    def tearDown(self):
        self.temporary.cleanup()

    def test_actual_engle_granger_reference_causal_models_and_proposals(self):
        contract = self.workspace.run(self.config)
        result = contract.to_dict()
        EnvelopeSchema.model_validate(result)
        artifact = result["data"]
        self.assertEqual(result["state"], "AVAILABLE")
        self.assertEqual(artifact["input_classification"], "SYNTHETIC")
        self.assertGreater(artifact["metrics"]["selected_pair_folds"], 0)
        self.assertTrue(artifact["model_ids"])
        self.assertTrue(all(artifact["library_versions"].get(key) for key in ("python", "numpy", "scipy", "statsmodels")))
        first = artifact["folds"][0]
        stamps = fixture_times(120)[:60]
        prices = {(row["instrument_id"], pd.Timestamp(row["timestamp"])): row["raw_close"]
                  for row in self.dataset["panel_rows"]}
        from statsmodels.tsa.stattools import coint
        reference = coint(np.log([prices["US:AAA", stamp] for stamp in stamps]),
                          np.log([prices["US:BBB", stamp] for stamp in stamps]), trend="c", autolag="aic")
        self.assertAlmostEqual(first["screening"][0]["pvalue"], float(reference[1]), places=14)
        self.assertLess(pd.Timestamp(first["training_end"]), pd.Timestamp(first["training_cutoff"]))
        for row in artifact["rows"]:
            self.assertFalse(row["trading_approved"])
            self.assertLessEqual(sum(abs(v) for v in row["weights"].values()), 1.+1e-12)
            self.assertEqual(row["regime_method"], "DESCRIPTIVE_TRAILING_VOLATILITY_NOT_HMM")
            if row["net_return"] is not None:
                self.assertAlmostEqual(row["net_return"], row["gross_return"]-
                                       row["transaction_cost_return"]-row["borrow_cost_return"], places=14)
        self.assertTrue(artifact["usable_for_portfolio"])
        self.assertFalse(artifact["economically_validated"])
        self.assertEqual(artifact["proposals"][0]["state"], "AVAILABLE")

    def test_durable_replay_reopen_hashes_and_caller_isolation(self):
        first = self.workspace.run(self.config).to_dict()
        identity = first["data"]["run_id"]
        reopened = PairsResearchWorkspace(self.base)
        self.assertEqual(reopened.replay({"run_id": identity}).to_dict(), first)
        self.assertEqual(reopened.run(self.config).to_dict(), first)
        modified = reopened.load(identity)
        modified["config"]["entry_z"] = 99
        self.assertEqual(reopened.load(identity)["config"]["entry_z"], 0.)
        section = reopened.section("stat-arb", self.request["as_of"], 60).to_dict()
        self.assertEqual(section["data"]["run_id"], identity)
        for section_name in ("models", "signals", "portfolio", "risk"):
            grouped = reopened.section(section_name, self.request["as_of"], 60).to_dict()["data"]
            self.assertEqual(grouped["rows_total"], len(grouped["rows"]))
            self.assertTrue(grouped["rows_complete"])
        self.assertEqual(reopened.section("stat-arb", self.request["as_of"], None).to_dict()["state"], "UNKNOWN")
        later = (pd.Timestamp(self.request["as_of"])+pd.Timedelta(seconds=61)).isoformat()
        self.assertEqual(reopened.section("stat-arb", later, 60).to_dict()["state"], "STALE")
        self.assertIsNone(reopened.section("stat-arb", self.request["start"], 60))
        self.assertIsNone(reopened.section("news", later, 60))
        model = first["data"]["model_ids"][0]
        path = self.base/"pairs-research"/"models"/(model+".json")
        path.write_text("{}")
        with self.assertRaisesRegex(ValueError, "integrity"):
            reopened.replay({"run_id": identity})

    def test_future_revisions_do_not_change_earlier_models_signals_or_backtest(self):
        before = self.workspace.run(self.config).to_dict()["data"]
        correction = deepcopy(pair_bars()[190])
        correction.update(available_at=fixture_times(120)[96].isoformat())
        for key in ("open", "high", "low", "close"):
            correction[key] *= 1.1
        ingest_fixture_bars(self.base, [correction])
        revised = PersistedDatasetWorkspace(self.base).build(self.request).to_dict()["data"]
        after = self.workspace.run({**self.config, "dataset_id": revised["dataset_id"]}).to_dict()["data"]
        cutoff = fixture_times(120)[96]
        self.assertEqual([r for r in before["rows"] if pd.Timestamp(r["timestamp"]) < cutoff],
                         [r for r in after["rows"] if pd.Timestamp(r["timestamp"]) < cutoff])
        self.assertEqual([fold for fold in before["folds"] if pd.Timestamp(fold["training_cutoff"]) < cutoff],
                         [fold for fold in after["folds"] if pd.Timestamp(fold["training_cutoff"]) < cutoff])
        # The immutable earlier run is still reproducible after source changes.
        self.assertEqual(self.workspace.load(before["run_id"])["rows"], before["rows"])

    def test_prefix_is_causal_and_costs_never_improve_same_weight_path(self):
        config = configuration(self.config)
        result, _ = fit_pairs(self.dataset, config)
        costless, _ = fit_pairs(self.dataset, {**config, "cost_bps": 0., "borrow_bps": 0.})
        self.assertEqual([r["weights"] for r in result["rows"]], [r["weights"] for r in costless["rows"]])
        self.assertGreater(result["metrics"]["transaction_cost_return_sum"], 0.)
        self.assertLessEqual(result["metrics"]["hypothetical_compounded_net_return"],
                             costless["metrics"]["hypothetical_compounded_net_return"])
        future = deepcopy(self.dataset)
        cutoff = fixture_times(120)[90]
        for row in future["panel_rows"]:
            if pd.Timestamp(row["timestamp"]) >= cutoff:
                row["raw_close"] *= 2. if row["instrument_id"] == "US:AAA" else 3.
        changed, _ = fit_pairs(future, config)
        # Future outcomes may change; decisions and already-known outcomes do not.
        decision_fields = ("timestamp", "model_id", "innovation_zscore", "weights", "regime")
        def decisions(artifact):
            return [{key: row[key] for key in decision_fields} for row in artifact["rows"]
                    if pd.Timestamp(row["timestamp"]) < cutoff]
        self.assertEqual(decisions(result), decisions(changed))
        self.assertEqual([r for r in result["rows"] if r["outcome_at"] and pd.Timestamp(r["outcome_at"]) < cutoff],
                         [r for r in changed["rows"] if r["outcome_at"] and pd.Timestamp(r["outcome_at"]) < cutoff])

    def test_missing_membership_rejects_model_and_never_imputes_quotes(self):
        dataset = deepcopy(self.dataset)
        dataset["panel_rows"] = [r for r in dataset["panel_rows"]
                                 if r["instrument_id"] != "US:AAA" or pd.Timestamp(r["timestamp"]) >= fixture_times(120)[60]]
        result, _ = fit_pairs(dataset, configuration(self.config))
        self.assertEqual(result["metrics"]["selected_pair_folds"], 0)
        self.assertEqual(result["proposals"], [])
        self.assertIsNone(result["metrics"]["hypothetical_compounded_net_return"])
        self.assertFalse(result["usable_for_portfolio"])
        self.assertGreater(result["metrics"]["missing_training_memberships"], 0)

    def test_split_adjustment_uses_causal_return_index_instead_of_raw_price_jump(self):
        base = Path(self.temporary.name)/"adjusted"
        bars = pair_bars()
        split = fixture_times(120)[65]
        for row in bars:
            if pd.Timestamp(row["timestamp"]) >= split:
                for field in ("open", "high", "low", "close"):
                    row[field] /= 2
        request = seed_persisted_fixture(base, periods=120, rows=bars)
        actions = [{"action_id": "split-"+identity, "instrument_id": identity, "source": "authored-actions",
                    "kind": "SPLIT", "effective_at": split.isoformat(),
                    "available_at": fixture_times(120)[64].isoformat(), "ratio_new": 2, "ratio_old": 1}
                   for identity in ("US:AAA", "US:BBB")]
        ActionStore(base/"actions.sqlite").ingest(admit_actions(actions))
        dataset = PersistedDatasetWorkspace(base).build({**request, "adjustment": "SPLIT_ADJUSTED"}).to_dict()["data"]
        _, panel = _panel(dataset)
        for identity in panel:
            current = panel[identity][split.isoformat()]
            previous = panel[identity][fixture_times(120)[64].isoformat()]
            self.assertLess(current["raw_close"]/previous["raw_close"], .6)
            self.assertLess(abs(current["pair_price"]/previous["pair_price"]-1), .1)
        artifact = PairsResearchWorkspace(base).run({**self.config, "dataset_id": dataset["dataset_id"]}).to_dict()["data"]
        self.assertEqual(artifact["provenance"]["dataset_policy"]["feature_adjustment_policy"], "SPLIT_ADJUSTED")
        self.assertTrue(artifact["rows"])
        self.assertLess(max(abs(row["gross_return"]) for row in artifact["rows"]), .2)

    def test_sparse_gap_does_not_reconnect_a_fitted_path_or_invent_an_outcome(self):
        dataset = deepcopy(self.dataset)
        gap = fixture_times(120)[75]
        dataset["panel_rows"] = [row for row in dataset["panel_rows"]
                                 if row["instrument_id"] != "US:AAA" or pd.Timestamp(row["timestamp"]) != gap]
        artifact, _ = fit_pairs(dataset, configuration(self.config))
        held = [row for row in artifact["rows"] if row["outcome_at"] == gap.isoformat()]
        self.assertTrue(any(any(row["weights"].values()) for row in held))
        self.assertTrue(all(row["net_return"] is None for row in held if any(row["weights"].values())))
        self.assertGreater(artifact["metrics"]["missing_held_outcomes"], 0)
        self.assertIsNone(artifact["metrics"]["hypothetical_compounded_net_return"])
        self.assertFalse(artifact["proposals"])
        for row in artifact["rows"]:
            if pd.Timestamp(row["timestamp"]) >= gap:
                self.assertFalse(any(row["weights"].values()))
                self.assertEqual(row["state"], "UNAVAILABLE")

    def test_invalid_requests_missing_runs_and_insufficient_history_are_explicit(self):
        for changes in ({"alpha": float("nan")}, {"embargo": 0}, {"training_window": True},
                        {"observation_variance": 0}, {"cost_bps": -1}, {"live": True}):
            with self.assertRaises(ValueError):
                self.workspace.run({**self.config, **changes})
        with self.assertRaises(ValueError):
            self.workspace.run({**self.config, "training_window": 120})
        for payload in ({"run_id": "../x"}, {"run_id": "0"*64, "refit": True}):
            with self.assertRaises(ValueError):
                self.workspace.replay(payload)
        self.assertEqual(self.workspace.replay({"run_id": "0"*64}).to_dict()["state"], "UNAVAILABLE")
        self.assertEqual(self.workspace.run({**self.config, "dataset_id": "0"*64}).to_dict()["state"], "UNAVAILABLE")


if __name__ == "__main__":
    unittest.main()
