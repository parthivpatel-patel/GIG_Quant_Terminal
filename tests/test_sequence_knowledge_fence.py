"""Reproduced BiLSTM label-knowledge fallback: reject before fit and on replay."""
from __future__ import annotations

import asyncio
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

import httpx

from gig.research.model_pipeline import configuration, fit_dataset, identity
from gig.service.allocation_workspace import ResearchAllocationWorkspace
from gig.service.app import create_app
from gig.service.dataset_workspace import PersistedDatasetWorkspace
from gig.service.persisted_research import PersistedResearchWorkspace
from test_persisted_data import seed_persisted_fixture
from test_persisted_flow_http import allocation_request, seed_flow


class SequenceKnowledgeTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)
        request = seed_persisted_fixture(self.base)
        self.dataset = PersistedDatasetWorkspace(self.base).build(request).to_dict()["data"]
        self.request = {"dataset_id": self.dataset["dataset_id"], "model": "bilstm", "min_train": 30,
                        "test_size": 12, "embargo": 1, "sequence_length": 4, "hidden_size": 2, "epochs": 1, "seed": 17}

    def late_labels(self):
        dataset = deepcopy(self.dataset)
        for row in dataset["rows"]:
            row["label_vintages"] = [{"target": row["target"], "label_end": row["label_end"], "knowledge_at": dataset["as_of"]}]
        dataset["dataset_id"] = identity({k: v for k, v in dataset.items() if k != "dataset_id"})
        return dataset

    def test_unavailable_supervision_rejected_before_actual_fitter_or_persistence(self):
        late = self.late_labels()
        workspace = PersistedResearchWorkspace(self.base)
        with (patch.object(workspace, "_dataset", return_value=late),
              patch("gig.research.model_pipeline.train_sequence", side_effect=AssertionError("must not fit")),
              self.assertRaisesRegex(ValueError, "knowledge")):
            workspace.run({**self.request, "dataset_id": late["dataset_id"]})
        self.assertFalse((self.base/"persisted-research").exists())

    def test_unknown_warmup_targets_do_not_change_models_or_predictions(self):
        original, _ = fit_dataset(self.dataset, configuration(self.request))
        changed = deepcopy(self.dataset)
        seen = set()
        for row in changed["rows"]:
            if row["instrument_id"] not in seen:
                seen.add(row["instrument_id"])
                row["target"] += .4
                row["label_vintages"] = [{"target": row["target"], "label_end": row["label_end"], "knowledge_at": changed["as_of"]}]
        changed["dataset_id"] = identity({k: v for k, v in changed.items() if k != "dataset_id"})
        result, _ = fit_dataset(changed, configuration(self.request))
        self.assertEqual(result["model_ids"], original["model_ids"])
        self.assertEqual(result["rows"], original["rows"])

    def test_unsafe_legacy_saved_run_fails_on_reopen_without_rewriting_files(self):
        late = self.late_labels()
        workspace = PersistedResearchWorkspace(self.base)
        # Reproduce former adapter behavior: select raw targets regardless of
        # their declared revision knowledge. Actual CPU training still occurs.
        with patch.object(workspace, "_dataset", return_value=late):
            with patch("gig.research.model_pipeline._training_vintage", side_effect=lambda row, _: row):
                result = workspace.run({**self.request, "dataset_id": late["dataset_id"]}).to_dict()
            files = {p: p.read_bytes() for p in (self.base/"persisted-research").rglob("*.json")}
            with self.assertRaisesRegex(ValueError, "knowledge"):
                workspace.load(result["data"]["run_id"])
            self.assertEqual(files, {p: p.read_bytes() for p in files})


    def test_legacy_allocation_rechecks_source_before_view_or_oms_preparation(self):
        # A separate authored microsecond fixture supports the OMS precision
        # boundary; no source timestamp is silently truncated by the service.
        request = seed_flow(self.base)
        dataset = PersistedDatasetWorkspace(self.base).build(request).to_dict()["data"]
        for row in dataset["rows"]:
            row["label_vintages"] = [{"target": row["target"], "label_end": row["label_end"], "knowledge_at": dataset["as_of"]}]
        dataset["dataset_id"] = identity({k: v for k, v in dataset.items() if k != "dataset_id"})
        with patch.object(PersistedResearchWorkspace, "_dataset", return_value=dataset):
            with patch("gig.research.model_pipeline._training_vintage", side_effect=lambda row, _: row):
                run = PersistedResearchWorkspace(self.base).run({**self.request, "dataset_id": dataset["dataset_id"]}).to_dict()["data"]
                with patch.object(PersistedDatasetWorkspace, "load", return_value=dataset):
                    allocation = ResearchAllocationWorkspace(self.base).build(allocation_request(run["run_id"], dataset["as_of"])).to_dict()
            self.assertEqual(allocation["state"], "AVAILABLE")
            allocation_id = allocation["data"]["allocation_id"]
            preserved = {p: p.read_bytes() for p in (self.base/"allocations").rglob("*.json")}
            reopened = ResearchAllocationWorkspace(self.base)
            for operation in (lambda: reopened.load(allocation_id),
                              lambda: reopened.section("risk", dataset["as_of"], 60),
                              lambda: reopened.prepare({"allocation_id": allocation_id, "now": dataset["as_of"]})):
                with self.subTest(operation=operation), self.assertRaisesRegex(ValueError, "knowledge"):
                    operation()
            self.assertFalse((self.base/"execution-engine").exists())
            self.assertEqual(preserved, {p: p.read_bytes() for p in preserved})


class SequenceKnowledgeHTTPTests(unittest.IsolatedAsyncioTestCase):
    async def test_http_rejects_unavailable_supervision_with_safe_error_and_no_run(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            request = seed_persisted_fixture(base)
            dataset = PersistedDatasetWorkspace(base).build(request).to_dict()["data"]
            for row in dataset["rows"]:
                row["label_vintages"] = [{"target": row["target"], "label_end": row["label_end"], "knowledge_at": dataset["as_of"]}]
            async def clock():
                while True:
                    await asyncio.sleep(.01)
            heartbeat = asyncio.create_task(clock())
            try:
                with patch("gig.service.persisted_research.PersistedResearchWorkspace._dataset", return_value=dataset):
                    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=create_app(state_dir=base)), base_url="http://test") as client:
                        response = await client.post("/api/v1/workspace/research-model", json={"dataset_id": dataset["dataset_id"], "model": "bilstm", "min_train": 30, "test_size": 12, "epochs": 1})
                        self.assertEqual(response.status_code, 422)
                        self.assertEqual(response.json()["state"], "ERROR")
                        self.assertEqual(response.json()["error"]["code"], "INVALID_REQUEST")
                        self.assertFalse((base/"persisted-research").exists())
                        self.assertFalse((base/"execution-engine").exists())
            finally:
                heartbeat.cancel()
