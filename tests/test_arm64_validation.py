"""Evidence classification and platform identity admission are deterministic."""
from __future__ import annotations

import importlib.util
import os
import struct
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("arm64_check", ROOT / "scripts/check_arm64.py")
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class ARM64EvidenceTests(unittest.TestCase):
    def test_actual_compose_has_explicit_safe_local_resource_policy(self):
        report = MODULE.compose_policy(ROOT / "docker-compose.yml")
        self.assertTrue(report["passed"])
        self.assertEqual(report["level"], "STATIC_CONFIGURATION_ONLY")
        self.assertFalse(report["arm64_runtime_validated"])

    def test_missing_tools_do_not_claim_cross_or_hardware_validation(self):
        report = MODULE.validate(ROOT, None, None)
        self.assertEqual(report["state"], "BLOCKED")
        self.assertEqual(report["hardware_validation"], "HARDWARE_NOT_VALIDATED")
        self.assertEqual(report["ubuntu_24_04_runtime"], "NOT_VALIDATED")
        self.assertNotIn("execution", report)

    def test_elf_parser_distinguishes_target_and_rejects_malformed_input(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixture"
            for machine in (62, 183):
                header = bytearray(20)
                header[:6] = b"\x7fELF\x02\x01"
                header[18:20] = struct.pack("<H", machine)
                path.write_bytes(header)
                self.assertEqual(MODULE.elf_machine(path), machine)
            path.write_bytes(b"not an executable")
            with self.assertRaises(ValueError):
                MODULE.elf_machine(path)

    def test_tool_runner_forwards_only_explicit_public_tool_configuration(self):
        with tempfile.TemporaryDirectory() as directory:
            library = Path(directory)
            completed = subprocess.CompletedProcess(["compiler"], 0, "ok", "")
            with patch.dict(os.environ, {"ALPACA_API_KEY": "FAKE_DO_NOT_FORWARD"}), \
                    patch.object(MODULE.subprocess, "run", return_value=completed) as execute:
                self.assertEqual(MODULE._run(["compiler"], tool_library_dir=library).stdout, "ok")
            environment = execute.call_args.kwargs["env"]
            self.assertNotIn("ALPACA_API_KEY", environment)
            self.assertEqual(environment["LD_LIBRARY_PATH"], str(library.resolve()))
            self.assertLessEqual(set(environment), {"PATH", "LANG", "LC_ALL", "TMPDIR", "LD_LIBRARY_PATH"})
            with self.assertRaisesRegex(ValueError, "does not exist"):
                MODULE._run(["compiler"], tool_library_dir=library / "missing")


if __name__ == "__main__":
    unittest.main()
