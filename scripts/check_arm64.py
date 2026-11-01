"""Explicit ARM64 cross/emulation evidence; never claims physical hardware QA.

Tool installation is external and opt-in. This script does not install binfmt,
invoke a privileged container, change Docker, or connect to a data provider.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import struct
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import yaml


def elf_machine(path: Path) -> int:
    """Read ELF identity directly so host compilation cannot be mislabeled."""
    with path.open("rb") as stream:
        header = stream.read(20)
    if len(header) != 20 or header[:4] != b"\x7fELF" or header[4] != 2 or header[5] not in (1, 2):
        raise ValueError("expected a 64-bit ELF executable")
    return struct.unpack("<H" if header[5] == 1 else ">H", header[18:20])[0]


def compose_policy(path: Path) -> dict[str, Any]:
    """Static local deployment policy; not an executed ARM64 Compose test."""
    raw = yaml.safe_load(path.read_text())
    service = raw["services"]["gig"]
    checks = {
        "loopback_port_only": service.get("ports") == ["127.0.0.1:8000:8000"],
        "broker_disabled": str(service.get("environment", {}).get("GIG_BROKER_CONNECTIVITY_ENABLED")).lower() == "false",
        "read_only_root": service.get("read_only") is True,
        "capabilities_dropped": "ALL" in service.get("cap_drop", []),
        "no_new_privileges": "no-new-privileges:true" in service.get("security_opt", []),
        "persistent_state": set(service.get("volumes", [])) >= {"gig-data:/app/data", "gig-results:/app/results"},
        "memory_limit_declared": service.get("mem_limit") == "4g",
        "restart_declared": service.get("restart") == "unless-stopped",
        "graceful_shutdown_declared": service.get("stop_grace_period") == "30s",
    }
    return {"level": "STATIC_CONFIGURATION_ONLY", "checks": checks,
            "passed": all(checks.values()), "arm64_runtime_validated": False}


def _run(args: list[str], *, timeout: int = 60,
         tool_library_dir: Path | None = None) -> subprocess.CompletedProcess[str]:
    # No brokerage or other service credentials are forwarded to tool processes.
    child_env = {key: os.environ[key] for key in ("PATH", "LANG", "LC_ALL", "TMPDIR") if key in os.environ}
    if tool_library_dir is not None:
        if not tool_library_dir.is_dir():
            raise ValueError("explicit tool library directory does not exist")
        child_env["LD_LIBRARY_PATH"] = str(tool_library_dir.resolve())
    return subprocess.run(args, text=True, capture_output=True, timeout=timeout,
                          check=True, env=child_env)


def validate(root: Path, compiler: str | None, emulator: str | None,
             *, compiler_flags: list[str] | None = None,
             tool_library_dir: Path | None = None) -> dict[str, Any]:
    report: dict[str, Any] = {
        "host_architecture": platform.machine(),
        "target_architecture": "aarch64",
        "hardware_validation": "HARDWARE_NOT_VALIDATED",
        "ubuntu_24_04_runtime": "NOT_VALIDATED",
        "oracle_ampere_a1_runtime": "NOT_VALIDATED",
        "docker_arm64_image_build": "NOT_VALIDATED",
        "broker_connections": 0,
        "privileged_or_binfmt_changes": False,
        "compose": compose_policy(root / "docker-compose.yml"),
    }
    if compiler is None or emulator is None:
        report.update(state="BLOCKED", reason="CROSS_COMPILER_OR_USER_EMULATOR_UNAVAILABLE")
        return report
    report["compiler_version"] = _run([compiler, "--version"], tool_library_dir=tool_library_dir).stdout.splitlines()[0]
    report["emulator_version"] = _run([emulator, "--version"], tool_library_dir=tool_library_dir).stdout.splitlines()[0]
    report["compiler_flags"] = list(compiler_flags or [])
    report["private_tool_library_directory"] = None if tool_library_dir is None else str(tool_library_dir.resolve())
    kernel = root / "src/cpp/rolling.cpp"
    fixture = root / "deploy/arm64/rolling_reference.cpp"
    report["sources"] = {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
                         for path in (kernel, fixture)}
    with tempfile.TemporaryDirectory(prefix="gig-arm64-check-") as scratch:
        binary = Path(scratch) / "rolling-reference-aarch64"
        command = [compiler, "-std=c++20", "-O2", "-static", *(compiler_flags or []),
                   str(kernel), str(fixture), "-o", str(binary)]
        _run(command, tool_library_dir=tool_library_dir)
        if elf_machine(binary) != 183:
            raise ValueError("compiler output is not EM_AARCH64")
        report["cross_compilation"] = "CROSS_COMPILED"
        report["elf_machine"] = "EM_AARCH64"
        report["binary_sha256"] = hashlib.sha256(binary.read_bytes()).hexdigest()
        result = json.loads(_run([emulator, str(binary)], tool_library_dir=tool_library_dir).stdout)
        if result.get("architecture") != "aarch64" or result.get("result") != "PASS":
            raise ValueError("emulated reference did not report successful ARM64 execution")
        report["execution"] = "EMULATED_QEMU_USER"
        report["reference"] = result
        report["state"] = "PASS" if report["compose"]["passed"] else "ERROR"
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compiler", default=shutil.which("aarch64-linux-gnu-g++"))
    parser.add_argument("--emulator", default=shutil.which("qemu-aarch64-static") or shutil.which("qemu-aarch64"))
    parser.add_argument("--compiler-flag", action="append", default=[])
    parser.add_argument("--tool-library-dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = validate(Path(__file__).resolve().parents[1], args.compiler, args.emulator,
                      compiler_flags=args.compiler_flag, tool_library_dir=args.tool_library_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if report["state"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
