"""Saved calculator runs replay from the pinned inputs and reject a changed record."""
from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path

import httpx

from gig.service.app import create_app
from gig.service.calculator_journal import (
    MAX_RECORDS,
    CalculatorJournal,
    CalculatorJournalWorkspace,
    aligned_returns,
    panel_series,
)
from gig.service.dataset_workspace import PersistedDatasetWorkspace
from gig.service.quant_analytics import ANALYSES
from test_persisted_data import dataset_request, seed_persisted_fixture


class JournalTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)/"state"
        self.journal = CalculatorJournal(self.root)

    def test_every_registered_calculator_replays_its_example(self):
        self.assertGreaterEqual(len(ANALYSES), 32)
        for kind, analysis in ANALYSES.items():
            with self.subTest(kind):
                saved = self.journal.save(kind, analysis.example)
                again = self.journal.save(kind, analysis.example)
                self.assertEqual(again["record_id"], saved["record_id"])
                replayed = CalculatorJournal(self.root).replay(saved["record_id"])
                self.assertEqual(replayed["replay_state"], "REPRODUCED_FROM_PINNED_INPUTS")
                self.assertEqual(replayed["result"], saved["result"])
                self.assertFalse(replayed["trading_approved"])

    def test_a_changed_record_or_a_missing_one_is_refused(self):
        saved = self.journal.save("track-record-length", {"sharpe": .1, "benchmark": 0, "skew": -.5, "kurtosis": 5, "confidence": .95})
        with sqlite3.connect(self.journal.path) as db:
            db.execute("UPDATE records SET body=replace(body,'0.1','0.2') WHERE id=?", (saved["record_id"],))
        with self.assertRaisesRegex(ValueError, "integrity"):
            self.journal.replay(saved["record_id"])
        fresh = CalculatorJournal(self.root/"empty")
        with self.assertRaises(FileNotFoundError):
            fresh.replay(saved["record_id"])
        self.assertFalse(fresh.path.exists())
        with self.assertRaises(ValueError):
            self.journal.save("not-a-calculator", {})
        with self.assertRaises(ValueError):
            self.journal.replay("not-a-hash")

    def test_the_journal_stops_at_its_stated_capacity(self):
        for index in range(MAX_RECORDS):
            self.journal.save("track-record-length", {"sharpe": .05+index/10000})
        with self.assertRaisesRegex(ValueError, "capacity"):
            self.journal.save("track-record-length", {"sharpe": .9})
        self.assertEqual(sqlite3.connect(self.journal.path).execute("SELECT COUNT(*) FROM records").fetchone()[0], MAX_RECORDS)


class HTTPTests(unittest.IsolatedAsyncioTestCase):
    async def test_save_and_replay_over_the_local_workspace(self):
        with tempfile.TemporaryDirectory() as directory:
            transport = httpx.ASGITransport(app=create_app(state_dir=Path(directory)))
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                endpoint = "/api/v1/workspace/calculator-journal"
                saved = await client.post(endpoint, json={"operation": "save", "kind": "roll-spread",
                                                          "inputs": ANALYSES["roll-spread"].example})
                self.assertEqual(saved.status_code, 200, saved.text)
                body = saved.json()
                self.assertEqual(body["provenance"]["classification"], "MODEL_ESTIMATE")
                self.assertEqual(body["provenance"]["readiness"], "RESEARCH_ONLY")
                self.assertEqual(body["data"]["result"]["spread_state"], "IDENTIFIED")
                replay = await client.post(endpoint, json={"operation": "replay", "record_id": body["data"]["record_id"]})
                self.assertEqual(replay.json()["data"]["replay_state"], "REPRODUCED_FROM_PINNED_INPUTS")
                missing = await client.post(endpoint, json={"operation": "replay", "record_id": "0"*64})
                self.assertEqual(missing.json()["state"], "UNAVAILABLE")
                for bad in ({"operation": "save", "kind": "roll-spread"},
                            {"operation": "save", "kind": "roll-spread", "inputs": {"prices": [1, 2]}, "broker": "live"},
                            {"operation": "replay"},
                            {"operation": "trade", "kind": "roll-spread", "inputs": {}}):
                    self.assertEqual((await client.post(endpoint, json=bad)).status_code, 422)
                hostile = await client.post(endpoint, json={"operation": "replay", "record_id": body["data"]["record_id"]},
                                            headers={"Origin": "https://hostile.invalid"})
                self.assertEqual(hostile.status_code, 403)

    async def test_a_saved_dataset_series_is_calculated_and_replayed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            seed_persisted_fixture(root)
            dataset = PersistedDatasetWorkspace(root).build(dataset_request()).to_dict()["data"]
            transport = httpx.ASGITransport(app=create_app(state_dir=root))
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                endpoint = "/api/v1/workspace/calculator-journal"
                saved = await client.post(endpoint, json={
                    "operation": "from-dataset", "dataset_id": dataset["dataset_id"],
                    "kind": "cornish-fisher-var", "instrument_id": "US:AAA", "field": "return_1",
                    "inputs": {"confidence": 0.95}})
                self.assertEqual(saved.status_code, 200, saved.text)
                body = saved.json()
                self.assertEqual(body["data"]["source"]["dataset_classification"], "SYNTHETIC")
                self.assertFalse(body["data"]["trading_approved"])
                self.assertGreater(body["data"]["source"]["observations"], 30)
                replay = await client.post(endpoint, json={"operation": "replay", "record_id": body["data"]["record_id"]})
                self.assertEqual(replay.json()["data"]["result"], body["data"]["result"])
                refused = await client.post(endpoint, json={
                    "operation": "from-dataset", "dataset_id": dataset["dataset_id"],
                    "kind": "roll-spread", "instrument_id": "US:AAA", "field": "return_1", "inputs": {}})
                self.assertEqual(refused.status_code, 422)


class DatasetBindingTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name) / "state"
        seed_persisted_fixture(self.root)
        built = PersistedDatasetWorkspace(self.root).build(dataset_request()).to_dict()
        self.dataset = built["data"]
        self.workspace = CalculatorJournalWorkspace(self.root)

    def test_only_a_leading_return_warmup_may_be_absent(self):
        rows = [{"instrument_id": "US:AAA", "return_1": None},
                {"instrument_id": "US:AAA", "return_1": 0.01},
                {"instrument_id": "US:AAA", "return_1": None}]
        with self.assertRaisesRegex(ValueError, "leading warmup"):
            panel_series(rows, "US:AAA", "return_1")
        rows[2]["return_1"] = True
        with self.assertRaisesRegex(ValueError, "finite"):
            panel_series(rows, "US:AAA", "return_1")
        with self.assertRaisesRegex(ValueError, "missing observation"):
            panel_series([{"instrument_id": "US:AAA", "close": 1.0}, {"instrument_id": "US:AAA", "close": None}],
                         "US:AAA", "close")
        partial = [{"instrument_id": "US:AAA", "timestamp": "t1", "return_1": 0.01},
                   {"instrument_id": "US:BBB", "timestamp": "t1", "return_1": None},
                   {"instrument_id": "US:AAA", "timestamp": "t2", "return_1": 0.02},
                   {"instrument_id": "US:BBB", "timestamp": "t2", "return_1": -0.01}]
        with self.assertRaisesRegex(ValueError, "partial"):
            aligned_returns(partial, ["US:AAA", "US:BBB"], "return_1")

    def test_the_extracted_series_is_pinned_after_the_dataset_is_removed(self):
        saved = self.workspace.run({
            "operation": "from-dataset", "dataset_id": self.dataset["dataset_id"],
            "kind": "stationary-bootstrap", "instrument_id": "US:AAA", "field": "return_1",
            "inputs": {"mean_block": 5, "replications": 100, "seed": 1, "confidence": 0.95},
        }).to_dict()
        data = saved["data"]
        series = [row["return_1"] for row in self.dataset["panel_rows"]
                  if row["instrument_id"] == "US:AAA" and row["return_1"] is not None]
        self.assertEqual(data["inputs"]["series"], series)
        self.assertEqual(data["source"]["extraction"], "PANEL_ROWS_IN_STORED_ORDER_NO_IMPUTATION")
        self.assertEqual(data["source"]["leading_missing_observations"], 1)
        self.assertFalse(data["trading_approved"])
        manual = CalculatorJournal(self.root).save("stationary-bootstrap", data["inputs"])
        self.assertNotEqual(manual["record_id"], data["record_id"])
        self.assertEqual(manual["result"], data["result"])
        spread = self.workspace.run({
            "operation": "from-dataset", "dataset_id": self.dataset["dataset_id"],
            "kind": "roll-spread", "instrument_id": "US:BBB", "field": "close", "inputs": {},
        }).to_dict()["data"]
        closes = [row["close"] for row in self.dataset["panel_rows"] if row["instrument_id"] == "US:BBB"]
        self.assertEqual(spread["inputs"]["prices"], closes)
        self.assertEqual(spread["source"]["leading_missing_observations"], 0)
        panel = self.workspace.run({
            "operation": "from-dataset", "dataset_id": self.dataset["dataset_id"],
            "kind": "ledoit-wolf", "instruments": ["US:BBB", "US:AAA"], "field": "return_1", "inputs": {},
        }).to_dict()["data"]
        self.assertEqual(panel["source"]["assets"], 2)
        self.assertEqual(panel["source"]["skipped_all_missing_timestamps"], 1)
        self.assertEqual(panel["source"]["extraction"], "ALIGNED_RETURN_PANEL_NO_IMPUTATION")
        columns = list(zip(*panel["inputs"]["returns"], strict=True))
        other = [row["return_1"] for row in self.dataset["panel_rows"]
                 if row["instrument_id"] == "US:BBB" and row["return_1"] is not None]
        self.assertEqual(list(columns[0]), other)
        self.assertEqual(list(columns[1]), series)
        allocation = self.workspace.run({
            "operation": "from-dataset", "dataset_id": self.dataset["dataset_id"],
            "kind": "minimum-cvar", "instruments": ["US:AAA", "US:BBB"], "field": "return_1",
            "inputs": {"confidence": 0.95, "max_weight": 1},
        }).to_dict()["data"]
        self.assertEqual(len(allocation["result"]["weights"]), 2)
        self.assertFalse(allocation["trading_approved"])
        self.assertAlmostEqual(sum(allocation["result"]["weights"]), 1)
        (self.root / "research-datasets" / "datasets" / f"{self.dataset['dataset_id']}.json").unlink()
        replayed = self.workspace.run({"operation": "replay", "record_id": data["record_id"]}).to_dict()
        self.assertEqual(replayed["data"]["replay_state"], "REPRODUCED_FROM_PINNED_INPUTS")
        self.assertEqual(replayed["data"]["result"], data["result"])
        covariance = self.workspace.run({"operation": "replay", "record_id": panel["record_id"]}).to_dict()
        self.assertEqual(covariance["data"]["result"], panel["result"])
        missing = self.workspace.run({
            "operation": "from-dataset", "dataset_id": self.dataset["dataset_id"],
            "kind": "roll-spread", "instrument_id": "US:BBB", "field": "close", "inputs": {},
        }).to_dict()
        self.assertEqual(missing["state"], "UNAVAILABLE")

    def test_unbound_or_mismatched_fields_are_refused(self):
        base = {"operation": "from-dataset", "dataset_id": self.dataset["dataset_id"], "instrument_id": "US:AAA"}
        for payload in (
            {**base, "kind": "cornish-fisher-var", "field": "close", "inputs": {}},
            {**base, "kind": "stationary-bootstrap", "field": "return_1",
             "inputs": {"series": [0.01, -0.02], "mean_block": 2}},
            {**base, "kind": "ledoit-wolf", "field": "return_1", "inputs": {}},
            {**base, "kind": "roll-spread", "field": "close", "inputs": {}, "broker": "live"},
            {**base, "kind": "roll-spread", "field": "close"},
        ):
            with self.assertRaises(ValueError):
                self.workspace.run(payload)

    def test_a_saved_covariance_allocates_without_rereading_the_dataset(self):
        covariance = self.workspace.run({
            "operation": "from-dataset", "dataset_id": self.dataset["dataset_id"],
            "kind": "ledoit-wolf", "instruments": ["US:AAA", "US:BBB"], "field": "return_1", "inputs": {},
        }).to_dict()["data"]
        allocated = self.workspace.run({
            "operation": "from-record", "record_id": covariance["record_id"],
            "kind": "hierarchical-risk-parity", "inputs": {},
        }).to_dict()["data"]
        self.assertEqual(allocated["source"]["parent_record_id"], covariance["record_id"])
        self.assertEqual(allocated["source"]["dataset_classification"], "SYNTHETIC")
        self.assertEqual(len(allocated["result"]["weights"]), 2)
        self.assertFalse(allocated["trading_approved"])
        diversified = self.workspace.run({
            "operation": "from-record", "record_id": covariance["record_id"],
            "kind": "maximum-diversification", "inputs": {"max_weight": 1},
        }).to_dict()["data"]
        self.assertAlmostEqual(sum(diversified["result"]["weights"]), 1)
        (self.root / "research-datasets" / "datasets" / f"{self.dataset['dataset_id']}.json").unlink()
        replayed = self.workspace.run({"operation": "replay", "record_id": allocated["record_id"]}).to_dict()
        self.assertEqual(replayed["data"]["result"]["weights"], allocated["result"]["weights"])
        manual = self.workspace.run({
            "operation": "save", "kind": "roll-spread", "inputs": ANALYSES["roll-spread"].example,
        }).to_dict()["data"]
        with self.assertRaisesRegex(ValueError, "Ledoit-Wolf"):
            self.workspace.run({"operation": "from-record", "record_id": manual["record_id"],
                                "kind": "hierarchical-risk-parity", "inputs": {}})
        with self.assertRaisesRegex(ValueError, "unbound"):
            self.workspace.run({"operation": "from-record", "record_id": covariance["record_id"],
                                "kind": "hierarchical-risk-parity", "inputs": {"covariance": [[1, 0], [0, 1]]}})

    def test_from_screen_binds_black_litterman_and_risk_parity(self):
        from gig.service.persisted_research import PersistedResearchWorkspace
        from gig.service.research_screen import ResearchScreenWorkspace
        from gig.service.signal_workspace import SignalWorkspace
        from test_persisted_flow_http import seed_flow

        request = seed_flow(self.root)
        dataset = PersistedDatasetWorkspace(self.root).build(request).to_dict()["data"]
        run = PersistedResearchWorkspace(self.root).run({
            "dataset_id": dataset["dataset_id"], "model": "lightgbm",
            "min_train": 30, "test_size": 12, "embargo": 1, "seed": 17,
        }).to_dict()["data"]
        signal = SignalWorkspace(self.root).run({
            "run_ids": [run["run_id"]], "now": request["as_of"],
            "lookback": 20, "min_ic_observations": 5,
        }).to_dict()["data"]
        screen = ResearchScreenWorkspace(self.root).run({
            "signal_id": signal["signal_id"], "buy_threshold": 0.0, "sell_threshold": -0.01,
        }).to_dict()["data"]
        instruments = [row["instrument_id"] for row in screen["rows"][:2]]
        self.assertEqual(len(instruments), 2)
        covariance = self.workspace.run({
            "operation": "from-dataset", "dataset_id": dataset["dataset_id"],
            "kind": "ledoit-wolf", "instruments": instruments, "field": "return_1", "inputs": {},
        }).to_dict()["data"]
        litterman = self.workspace.run({
            "operation": "from-screen", "screen_id": screen["screen_id"],
            "record_id": covariance["record_id"], "kind": "black-litterman",
            "inputs": {"risk_aversion": 2.5, "tau": 0.05},
        }).to_dict()
        self.assertEqual(litterman["state"], "AVAILABLE", litterman)
        self.assertFalse(litterman["data"]["trading_approved"])
        self.assertEqual(len(litterman["data"]["result"]["expected_excess_returns"]), 2)
        self.assertGreaterEqual(litterman["data"]["source"]["view_count"], 1)
        parity = self.workspace.run({
            "operation": "from-screen", "screen_id": screen["screen_id"],
            "record_id": covariance["record_id"], "kind": "risk-parity", "inputs": {},
        }).to_dict()
        self.assertEqual(parity["state"], "AVAILABLE", parity)
        self.assertAlmostEqual(sum(parity["data"]["result"]["weights"]), 1.0, places=6)
        self.assertFalse(parity["data"]["trading_approved"])

    def test_from_options_fits_svi_on_admitted_call_slice(self):
        from gig.options.analytics import black_scholes
        from gig.service.options_workspace import OptionsWorkspace
        from test_options_workspace import option_fixture, option_request

        options = OptionsWorkspace(self.root)
        fixture = option_fixture(underlying_id="US:OPT", source="authored-svi-bind")
        expiry = next(quote["expiry"] for quote in fixture["quotes"] if quote["right"] == "CALL")
        event = fixture["event_at"]
        from gig.timestamps import aware_timestamp
        ttm = (aware_timestamp(expiry).value - aware_timestamp(event).value) / (365 * 86400 * 1_000_000_000)
        for strike in (80, 95, 105, 120):
            mid = float(black_scholes(100, strike, .02, .01, .2, ttm, True)["price"])
            fixture["quotes"].append({
                "contract_id": f"SYN100-extra-{strike}-CALL", "expiry": expiry,
                "strike": strike, "right": "CALL", "multiplier": 100, "style": "EUROPEAN",
                "bid": mid - .01, "ask": mid + .01, "volume": 10, "open_interest": 100,
            })
        calls = [quote for quote in fixture["quotes"]
                 if quote["right"] == "CALL" and quote["expiry"] == expiry]
        self.assertGreaterEqual(len(calls), 5)
        request = option_request(
            underlying_id="US:OPT", source=fixture["source"],
            positions=[{"contract_id": calls[0]["contract_id"], "quantity": 1}],
        )
        options.ingest(fixture)
        run = options.run(request).to_dict()
        self.assertEqual(run["state"], "AVAILABLE", run)
        fitted = self.workspace.run({
            "operation": "from-options", "options_run_id": run["data"]["run_id"],
            "kind": "svi-smile", "expiry": expiry, "inputs": {},
        }).to_dict()
        self.assertEqual(fitted["state"], "AVAILABLE", fitted)
        self.assertFalse(fitted["data"]["trading_approved"])
        self.assertIn("parameters", fitted["data"]["result"])
        self.assertEqual(fitted["data"]["source"]["extraction"],
                         "ADMITTED_CALL_IV_TOTAL_VARIANCE_ONE_EXPIRY")
        self.assertGreaterEqual(fitted["data"]["source"]["quotes"], 5)
        greeks = self.workspace.run({
            "operation": "from-options", "options_run_id": run["data"]["run_id"],
            "kind": "option-greeks-advanced", "contract_id": calls[0]["contract_id"],
            "inputs": {},
        }).to_dict()
        self.assertEqual(greeks["state"], "AVAILABLE", greeks)
        self.assertFalse(greeks["data"]["trading_approved"])
        self.assertIn("vanna", greeks["data"]["result"])
        self.assertEqual(greeks["data"]["source"]["extraction"],
                         "ADMITTED_OPTION_IV_HIGHER_ORDER_GREEKS")

    def test_from_size_record_runs_conditional_stress(self):
        from gig.service.persisted_research import PersistedResearchWorkspace
        from gig.service.research_screen import ResearchScreenWorkspace
        from gig.service.research_strategy import ResearchStrategyWorkspace
        from gig.service.signal_workspace import SignalWorkspace
        from test_persisted_flow_http import seed_flow

        request = seed_flow(self.root)
        dataset = PersistedDatasetWorkspace(self.root).build(request).to_dict()["data"]
        run = PersistedResearchWorkspace(self.root).run({
            "dataset_id": dataset["dataset_id"], "model": "lightgbm",
            "min_train": 30, "test_size": 12, "embargo": 1, "seed": 17,
        }).to_dict()["data"]
        signal = SignalWorkspace(self.root).run({
            "run_ids": [run["run_id"]], "now": request["as_of"],
            "lookback": 20, "min_ic_observations": 5,
        }).to_dict()["data"]
        screen = ResearchScreenWorkspace(self.root).run({
            "signal_id": signal["signal_id"], "buy_threshold": 0.0, "sell_threshold": -0.01,
        }).to_dict()["data"]
        sized = ResearchStrategyWorkspace(self.root).size({
            "screen_id": screen["screen_id"], "now": request["as_of"],
            "portfolio_notional": "100000", "max_name_notional": "8000",
            "lot_size": 1, "vol_scale": "1",
        }).to_dict()
        self.assertEqual(sized["state"], "AVAILABLE", sized)
        instruments = [row["instrument_id"] for row in sized["data"]["rows"][:2]]
        self.assertEqual(len(instruments), 2)
        covariance = self.workspace.run({
            "operation": "from-dataset", "dataset_id": dataset["dataset_id"],
            "kind": "ledoit-wolf", "instruments": instruments, "field": "return_1", "inputs": {},
        }).to_dict()["data"]
        stressed = self.workspace.run({
            "operation": "from-size-record", "size_id": sized["data"]["size_id"],
            "record_id": covariance["record_id"], "kind": "conditional-stress",
            "inputs": {"shocked_assets": [0], "shocks": [-0.05]},
        }).to_dict()
        self.assertEqual(stressed["state"], "AVAILABLE", stressed)
        self.assertFalse(stressed["data"]["trading_approved"])
        self.assertIn("portfolio_return", stressed["data"]["result"])
        self.assertEqual(stressed["data"]["source"]["extraction"],
                         "RESEARCH_SIZE_WEIGHTS_PINNED_LEDOIT_WOLF_COVARIANCE")


if __name__ == "__main__":
    unittest.main()
