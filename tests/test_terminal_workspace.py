"""Durable local workflows using caller evidence; no network or broker fixtures."""
import math
import sqlite3
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from gig.service.contracts import EnvelopeSchema
from gig.service.serialization import SerializationError, loads, normalize
from gig.service.workspace import EVIDENCE, TerminalWorkspace

NOW = pd.Timestamp("2026-06-01T12:00:00.000000123Z")
MICRO_NOW = NOW.floor("us")


def bar_request(count=40):
    rows = []
    for i in range(count):
        stamp = NOW - pd.Timedelta(days=count-1-i)
        for name, level, slope in (("ID:AAA", 100., .3), ("ID:BBB", 90., .2)):
            close = round(level+slope*i+.1*math.sin(i), 2)
            rows.append({"instrument_id": name, "timestamp": stamp.isoformat(),
                         "available_at": stamp.isoformat(), "open": close, "high": close+1,
                         "low": close-1, "close": close, "volume": 2**53+17+i})
    return {"rows": rows, "source": "caller-feed", "classification": "SYNTHETIC",
            "corporate_action_policy": "RAW", "currency": "USD"}


def research_request(count=35):
    rows = []
    for i in range(count):
        stamp = NOW-pd.Timedelta(days=count-i)
        rows.append({"timestamp": stamp.isoformat(), "available_at": stamp.isoformat(),
                     "label_end": (stamp+pd.Timedelta(hours=1)).isoformat(),
                     "features": {"trend": float(i), "oscillation": math.sin(i)},
                     "target": .1*i+.2*math.sin(i+1)})
    return {"rows": rows, "source": "caller-feature-set", "classification": "SYNTHETIC",
            "model": "ridge", "min_train": 10, "test_size": 5, "embargo": 1,
            "ridge_alpha": 1., "seed": 42, "as_of": NOW.isoformat()}


def simulation_request():
    latest = bar_request()["rows"][-2:]
    return {"scores": {"ID:AAA": 1., "ID:BBB": -1.},
            "prices": {row["instrument_id"]: row["close"] for row in latest},
            "sectors": {"ID:AAA": "same-sector", "ID:BBB": "same-sector"},
            "run_id": "caller-rehearsal-1", "asof": MICRO_NOW.isoformat(), "now": MICRO_NOW.isoformat(),
            "source_id": "caller-feed", "classification": "SYNTHETIC", "currency": "USD"}


def news_request():
    return {"as_of": NOW.isoformat(), "aliases": {"Acme Inc": "ID:AAA"},
            "articles": [{"source": "caller-news", "article_id": "article-1", "url": "https://example.invalid/1",
                          "license": "caller asserts permission", "title": "Acme Inc earnings",
                          "body": "  Acme Inc profit growth; NaN Labs source text.\n",
                          "published_at": (NOW-pd.Timedelta(minutes=2)).isoformat(),
                          "available_at": (NOW-pd.Timedelta(minutes=1)).isoformat()}]}


class WorkspaceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.base = Path(self.temporary.name)/"local-workspace"
        self.workspace = TerminalWorkspace(self.base)

    def tearDown(self):
        self.temporary.cleanup()

    def payload(self, contract):
        result = contract.to_dict()
        EnvelopeSchema.model_validate(result)
        self.assertEqual(result["provenance"]["evidence_state"], EVIDENCE)
        return result

    def test_missing_reads_do_not_create_any_state(self):
        for section in ("command-center", "markets", "signals", "portfolio", "risk", "stat-arb", "news", "research", "execution"):
            with self.subTest(section=section):
                self.assertEqual(self.payload(self.workspace.section(section, NOW, 60))["state"], "UNAVAILABLE")
        self.assertIsNone(self.workspace.section("options", NOW, 60))
        self.assertFalse(self.base.exists())

    def test_bar_receipt_immutable_identity_and_restart(self):
        request = bar_request()
        first = self.payload(self.workspace.ingest_bars(request))
        receipt = first["data"]["source_receipt_sha256"]
        immutable = (self.base/"receipts"/f"{receipt}.json").read_bytes()
        request["rows"][-1]["close"] = -1
        self.assertEqual((self.base/"receipts"/f"{receipt}.json").read_bytes(), immutable)
        replay = self.payload(self.workspace.ingest_bars(bar_request()))
        self.assertEqual(replay["data"]["source_receipt_sha256"], receipt)
        self.assertEqual(replay["data"]["metrics"]["replayed"], 80)
        reopened = TerminalWorkspace(self.base)
        markets = self.payload(reopened.section("markets", NOW, 0))
        self.assertEqual(markets["state"], "AVAILABLE")
        self.assertEqual(markets["provenance"]["classification"], "SYNTHETIC")
        self.assertEqual(markets["provenance"]["as_of"], NOW.isoformat())
        self.assertEqual(markets["data"]["rows"][0]["volume"], str(2**53+17+39))

    def test_point_in_time_correction_never_rewrites_prior_knowledge(self):
        request = bar_request()
        self.workspace.ingest_bars(request)
        correction = deepcopy(request)
        correction["rows"] = [deepcopy(request["rows"][-2])]
        correction["rows"][0].update(close=120., open=120., high=121., low=119.,
                                      available_at=(NOW+pd.Timedelta(seconds=30)).isoformat())
        self.workspace.ingest_bars(correction)
        before = self.payload(self.workspace.section("markets", NOW, 60))
        after = self.payload(self.workspace.section("markets", NOW+pd.Timedelta(seconds=30), 60))
        self.assertNotEqual(before["data"]["rows"][0]["close"], 120.)
        self.assertEqual(after["data"]["rows"][0]["close"], 120.)

    def test_staleness_original_ns_timestamp_and_unknown_freshness(self):
        self.workspace.ingest_bars(bar_request())
        stale = self.payload(self.workspace.section("markets", NOW+pd.Timedelta(seconds=61), 60))
        self.assertEqual(stale["state"], "STALE")
        self.assertEqual(stale["provenance"]["as_of"], NOW.isoformat())
        unknown = self.payload(self.workspace.section("markets", NOW, None))
        self.assertEqual(unknown["state"], "UNKNOWN")
        self.assertIsNone(unknown["data"])
        for maximum in (True, float("nan"), -1, float("inf")):
            with self.subTest(maximum=maximum), self.assertRaises(ValueError):
                self.workspace.section("markets", NOW, maximum)

    def test_mixed_currency_source_or_adjustment_never_combines_prices(self):
        for field, value in (("currency", "EUR"), ("source", "second-feed"),
                             ("classification", "HISTORICAL"), ("corporate_action_policy", "SPLIT_ADJUSTED")):
            workspace = TerminalWorkspace(self.base/field)
            workspace.ingest_bars(bar_request())
            request = bar_request()
            request[field] = value
            workspace.ingest_bars(request)
            for section in ("markets", "signals", "risk", "stat-arb"):
                with self.subTest(field=field, section=section):
                    result = self.payload(workspace.section(section, NOW, 60))
                    self.assertEqual(result["state"], "UNKNOWN")
                    self.assertIsNone(result["data"])

    def test_aligned_signals_and_pair_research_use_actual_admitted_prices(self):
        self.workspace.ingest_bars(bar_request())
        signals = self.payload(self.workspace.section("signals", NOW, 60))
        self.assertEqual(signals["state"], "AVAILABLE")
        self.assertEqual(len(signals["data"]["chart"]["points"]), 2)
        prices = [row["close"] for row in bar_request()["rows"] if row["instrument_id"] == "ID:AAA"]
        self.assertAlmostEqual(signals["data"]["rows"][0]["momentum_20"], prices[-1]/prices[-21]-1)
        pairs = self.payload(self.workspace.section("stat-arb", NOW, 60))
        self.assertEqual(pairs["state"], "AVAILABLE")
        self.assertFalse(pairs["data"]["metrics"]["stationarity_tested"])

    def test_incomplete_aligned_panel_is_unknown_without_forward_filling(self):
        request = bar_request()
        request["rows"].pop(3)
        self.workspace.ingest_bars(request)
        self.assertEqual(self.payload(self.workspace.section("signals", NOW, 60))["state"], "UNKNOWN")

    def test_news_attribution_original_text_duplicates_and_alias_restart(self):
        request = news_request()
        first = self.payload(self.workspace.ingest_news(request))
        self.assertEqual(first["data"]["rows"][0]["body"], request["articles"][0]["body"])
        self.assertEqual(first["data"]["rows"][0]["sentiment"]["method"], "LEXICON")
        replay = self.payload(self.workspace.ingest_news(request))
        self.assertEqual(replay["data"]["metrics"]["replayed"], 1)
        visible = self.payload(TerminalWorkspace(self.base).section("news", NOW, 120))
        self.assertIn("ID:AAA", visible["data"]["rows"][0]["linked_symbols"])
        stale = self.payload(self.workspace.section("news", NOW+pd.Timedelta(hours=1), 120))
        self.assertEqual(stale["state"], "STALE")

    def test_research_runs_are_purged_hash_verified_and_durable(self):
        request = research_request()
        first = self.payload(self.workspace.run_research(request))
        self.assertEqual(first["state"], "AVAILABLE")
        self.assertEqual(first["data"]["metrics"]["fold_count"], 4)
        for fold in first["data"]["folds"]:
            self.assertLess(pd.Timestamp(fold["max_training_label_end"]), pd.Timestamp(fold["test_start"]))
        replay = self.payload(self.workspace.run_research(request))
        self.assertEqual(replay["data"]["run_id"], first["data"]["run_id"])
        reopened = self.payload(TerminalWorkspace(self.base).section("research", NOW, 60))
        self.assertEqual(reopened["data"]["model_ids"], first["data"]["model_ids"])
        self.assertEqual(reopened["data"]["rows"], first["data"]["rows"])

    def test_research_corrupt_model_or_receipt_is_rejected(self):
        result = self.payload(self.workspace.run_research(research_request()))
        model_id = result["data"]["model_ids"][0]
        (self.base/"research"/"models"/f"{model_id}.json").write_text("{}")
        with self.assertRaisesRegex(ValueError, "model failed"):
            self.workspace.section("research", NOW, 60)

    def test_research_cannot_leak_future_features_or_exceed_fold_budget(self):
        request = research_request()
        request["rows"][0]["available_at"] = NOW.isoformat()
        with self.assertRaisesRegex(ValueError, "look-ahead"):
            self.workspace.run_research(request)
        request = research_request(100)
        request["test_size"] = 1
        with self.assertRaisesRegex(ValueError, "20 complete folds"):
            self.workspace.run_research(request)
        self.assertFalse(self.base.exists())

    def test_simulated_fills_positions_replay_and_reopen_valuation(self):
        self.workspace.ingest_bars(bar_request())
        request = simulation_request()
        result = self.payload(self.workspace.simulate(request))
        self.assertEqual(result["state"], "AVAILABLE")
        self.assertFalse(result["data"]["result"]["broker_connected"])
        self.assertEqual(result["data"]["result"]["execution_classification"], "SIMULATED")
        replay = self.payload(self.workspace.simulate(request))
        self.assertTrue(replay["data"]["result"]["replayed"])
        self.assertEqual(replay["data"]["rows"], [])
        reopened = TerminalWorkspace(self.base)
        portfolio = self.payload(reopened.section("portfolio", NOW, 60))
        self.assertEqual(portfolio["state"], "AVAILABLE")
        self.assertTrue(all(row["execution_classification"] == "SIMULATED" for row in portfolio["data"]["rows"]))
        risk = self.payload(reopened.section("risk", NOW, 60))
        self.assertEqual(risk["state"], "AVAILABLE")
        self.assertEqual(risk["data"]["risk"]["annualization"], 1)

    def test_simulation_missing_currency_does_not_infer_usd_valuation(self):
        self.workspace.ingest_bars(bar_request())
        request = simulation_request()
        del request["currency"]
        self.workspace.simulate(request)
        self.assertEqual(self.payload(self.workspace.section("portfolio", NOW, 60))["state"], "UNKNOWN")

    def test_simulation_changed_run_intent_never_replays_new_orders(self):
        request = simulation_request()
        self.workspace.simulate(request)
        request["scores"] = {"ID:AAA": -1., "ID:BBB": 1.}
        with self.assertRaisesRegex(ValueError, "run_id conflicts"):
            self.workspace.simulate(request)

    def test_simulation_refuses_nanosecond_to_microsecond_truncation(self):
        request = simulation_request()
        request["asof"] = NOW.isoformat()
        with self.assertRaisesRegex(ValueError, "microseconds"):
            self.workspace.simulate(request)
        self.assertFalse(self.base.exists())

    def test_stale_simulation_cannot_be_promoted_by_different_ui_freshness_policy(self):
        request = simulation_request()
        request["asof"] = (MICRO_NOW-pd.Timedelta(seconds=61)).isoformat()
        first = self.payload(self.workspace.simulate(request))
        self.assertEqual(first["state"], "STALE")
        later = self.payload(self.workspace.section("execution", NOW, 3600))
        self.assertEqual(later["state"], "STALE")
        self.assertEqual(later["provenance"]["as_of"], request["asof"])

    def test_blocked_simulation_remains_error_and_creates_no_orders(self):
        request = simulation_request()
        request["sectors"] = {"ID:AAA": "technology", "ID:BBB": "energy"}
        # Default sector limit .10 allows this portfolio; an invalid tiny
        # NAV is rejected before state creation instead of weakening the gate.
        request["nav"] = 1.
        with self.assertRaises(ValueError):
            self.workspace.simulate(request)
        self.assertFalse(self.base.exists())

    def test_kill_switch_failure_is_outer_error_and_remains_error_on_restart(self):
        from gig.execution.oms import OfflineOMS
        request = simulation_request()
        self.workspace.simulate(request)
        with OfflineOMS(self.base/"execution.sqlite") as oms:
            oms.set_kill_switch(True, now=MICRO_NOW.to_pydatetime())
            previous_events = len(oms.events())
        request["run_id"] = "new-blocked-rehearsal"
        blocked = self.payload(self.workspace.simulate(request))
        self.assertEqual(blocked["state"], "ERROR")
        self.assertIsNone(blocked["data"])
        visible = self.payload(TerminalWorkspace(self.base).section("execution", NOW, 60))
        self.assertEqual(visible["state"], "ERROR")
        with sqlite3.connect(self.base/"execution.sqlite") as connection:
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM events").fetchone()[0], previous_events)

    def test_unknown_request_paths_and_symlinks_cannot_redirect_state(self):
        request = bar_request()
        request["path"] = "/tmp/unrequested-target"
        with self.assertRaises(ValueError):
            self.workspace.ingest_bars(request)
        target = Path(self.temporary.name)/"other"
        target.mkdir()
        self.base.symlink_to(target, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "symbolic"):
            self.workspace.ingest_bars(bar_request())
        self.assertEqual(list(target.iterdir()), [])

    def test_expanded_news_analysis_is_admitted_before_any_state_mutation(self):
        request = news_request()
        request["articles"][0]["body"] = "profit " * 8000
        with self.assertRaisesRegex(SerializationError, "TOO_MANY_NODES"):
            self.workspace.ingest_news(request)
        self.assertFalse(self.base.exists())

    def test_future_news_cannot_bypass_serialization_admission(self):
        request = news_request()
        request["articles"][0]["body"] = "profit " * 8000
        request["articles"][0]["available_at"] = (NOW + pd.Timedelta(hours=1)).isoformat()
        with self.assertRaises(SerializationError):
            self.workspace.ingest_news(request)
        self.assertFalse(self.base.exists())

    def test_rejected_news_preserves_existing_source_and_manifest(self):
        self.workspace.ingest_news(news_request())
        receipts = set((self.base / "receipts").glob("*.json"))
        manifests = set((self.base / "manifests").glob("*.json"))
        request = news_request()
        request["articles"][0].update(article_id="oversized-analysis", body="profit " * 8000)
        with self.assertRaises(SerializationError):
            self.workspace.ingest_news(request)
        self.assertEqual(set((self.base / "receipts").glob("*.json")), receipts)
        self.assertEqual(set((self.base / "manifests").glob("*.json")), manifests)
        with sqlite3.connect(self.base / "news.sqlite") as connection:
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM articles").fetchone()[0], 1)
        visible = self.payload(self.workspace.section("news", NOW, 120))
        self.assertEqual(visible["data"]["rows"][0]["body"], news_request()["articles"][0]["body"])

    def test_registered_models_and_research_preserve_actual_training_identity(self):
        self.assertIsNone(self.workspace.section("models", NOW, 60))
        result = self.payload(self.workspace.run_research(research_request()))
        research = self.payload(self.workspace.section("research", NOW, 60))
        models = self.payload(self.workspace.section("models", NOW, 60))
        self.assertEqual(research["provenance"]["model_identity"], result["data"]["run_id"])
        self.assertEqual(models["provenance"]["model_identity"], result["data"]["run_id"])
        self.assertEqual({row["model_id"] for row in models["data"]["rows"]}, set(result["data"]["model_ids"]))
        self.assertTrue(all(row["trained"] and not row["trading_approved"] for row in models["data"]["rows"]))
        stale = self.payload(self.workspace.section("models", NOW + pd.Timedelta(seconds=61), 60))
        self.assertEqual(stale["state"], "STALE")

    def test_large_provenance_summary_retains_full_identity_without_oversized_wire(self):
        self.workspace.ingest_bars(bar_request())
        snapshot = self.workspace._bars(NOW, 60)
        snapshot["provenance_hashes"] = [f"{i:064x}" for i in range(1, 10_002)]
        with patch.object(self.workspace, "_bars", return_value=snapshot):
            first = self.payload(self.workspace.section("markets", NOW, 60))
        self.assertEqual(first["state"], "AVAILABLE")
        self.assertEqual(first["data"]["source_snapshot_hash_count"], 10_001)
        self.assertFalse(first["data"]["source_snapshot_hashes_complete"])
        self.assertEqual(len(first["data"]["source_snapshot_hashes"]), 256)
        snapshot["provenance_hashes"][-1] = "f" * 64
        with patch.object(self.workspace, "_bars", return_value=snapshot):
            second = self.payload(self.workspace.section("markets", NOW, 60))
        self.assertEqual(first["data"]["source_snapshot_hashes"], second["data"]["source_snapshot_hashes"])
        self.assertNotEqual(first["provenance"]["source_identity"], second["provenance"]["source_identity"])

    def test_pair_view_explicitly_refuses_oversized_analytical_window(self):
        request = bar_request(2001)
        for offset in range(0, len(request["rows"]), 2000):
            self.workspace.ingest_bars({**request, "rows": request["rows"][offset:offset + 2000]})
        self.assertEqual(self.payload(self.workspace.section("markets", NOW, 60))["state"], "AVAILABLE")
        result = self.payload(self.workspace.section("stat-arb", NOW, 60))
        self.assertEqual(result["state"], "UNKNOWN")
        self.assertIsNone(result["data"])

    def test_mutation_after_validation_cannot_change_receipt_or_stored_identity(self):
        request = bar_request()
        original = deepcopy(request)
        ensure = self.workspace._ensure

        def mutate_original():
            request["source"] = "changed-by-caller-after-admission"
            request["rows"][-1]["close"] = -1
            ensure()

        with patch.object(self.workspace, "_ensure", side_effect=mutate_original):
            admitted = self.payload(self.workspace.ingest_bars(request))
        self.assertEqual(admitted["provenance"]["source"], original["source"])
        receipt = loads((self.base / "receipts" / f"{admitted['data']['source_receipt_sha256']}.json").read_bytes())
        self.assertEqual(receipt["payload"], normalize(original))
        visible = self.payload(self.workspace.section("markets", NOW, 60))
        self.assertEqual(visible["data"]["rows"][-1]["close"], original["rows"][-1]["close"])


if __name__ == "__main__":
    unittest.main()
