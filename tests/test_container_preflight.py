"""Prevent storage exhaustion and accidental weight/credential build inclusion."""

from __future__ import annotations

import hashlib
import importlib.util
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location(
    "container_recovery", Path(__file__).parents[1] / "scripts/check_container_recovery.py"
)
assert SPEC is not None and SPEC.loader is not None
CHECKER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHECKER)


class ContainerPreflightTests(unittest.TestCase):
    def test_storage_gate_requires_declared_free_budget(self):
        low = CHECKER.preflight(CHECKER.MINIMUM_FREE - 1, "vfs")
        self.assertEqual(low["state"], "BLOCKED_STORAGE")
        self.assertEqual(CHECKER.preflight(CHECKER.MINIMUM_FREE, "vfs")["state"],
                         "READY_FOR_BOUNDED_BUILD")
        for value in (True, -1, 3.5):
            with self.subTest(value=value), self.assertRaises(ValueError):
                CHECKER.preflight(value, "vfs")

    def test_source_context_excludes_weights_binaries_credentials_and_recovery(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "source"
            target = Path(directory) / "frozen"
            root.mkdir()
            for name in CHECKER.SOURCE_FILES:
                (root / name).write_text("required source\n")
            allowed = {"src/gig/module.py": b"original source\n", "src/cpp/native.cpp": b"C++ source",
                       "configs/simulation.yaml": b"offline: true"}
            forbidden = ["src/gig/weights.safetensors", "src/gig/pytorch_model.bin",
                         "src/gig/_speed.so", "src/gig/.env", "src/gig/credentials.json",
                         "src/gig/model-assets/config.json", "tests/private.py", "recovery/checkpoint.bundle"]
            for name, data in {**allowed, **dict.fromkeys(forbidden, b"must stay outside image")}.items():
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(data)
            report = CHECKER.freeze_source(root, target)
            copied = {row["path"] for row in report["files"]}
            self.assertEqual(copied, set(CHECKER.SOURCE_FILES) | set(allowed))
            for name, data in allowed.items():
                self.assertEqual((target / name).read_bytes(), data)
                record = next(row for row in report["files"] if row["path"] == name)
                self.assertEqual(record["sha256"], hashlib.sha256(data).hexdigest())
            self.assertFalse(report["restricted_weights_included"])
            self.assertFalse(report["credentials_included"])

    def test_source_symlink_fails_before_reading_external_content(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "source"
            root.mkdir()
            for name in CHECKER.SOURCE_FILES:
                (root / name).write_text("source")
            (root / "src").mkdir()
            (root / "configs").mkdir()
            outside = Path(directory) / "outside"
            outside.write_text("external content")
            (root / "configs/redirect.json").symlink_to(outside)
            with self.assertRaisesRegex(ValueError, "symbolic"):
                CHECKER.admitted_source(root)

    def test_running_build_is_cancelled_before_exhausting_reserve(self):
        class Process:
            returncode = None
            terminated = False

            def poll(self):
                return self.returncode

            def terminate(self):
                self.terminated = True
                self.returncode = -15

            def wait(self, timeout):
                return self.returncode

        process = Process()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            with (patch.object(CHECKER.subprocess, "Popen", return_value=process),
                  patch.object(CHECKER.shutil, "disk_usage", return_value=SimpleNamespace(free=1))):
                report = CHECKER.monitored_build(path, "task-image", path, None)
            self.assertTrue(process.terminated)
            self.assertEqual(report["state"], "BLOCKED_STORAGE_RESERVE")


if __name__ == "__main__":
    unittest.main()
