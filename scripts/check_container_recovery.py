"""Bounded fresh-image and restart evidence using the managed local Docker socket.

Default invocation performs preflight only. ``--build`` freezes a small source
context, builds the repository Dockerfile, and verifies native/import/HTTP and
offline execution recovery. It never prunes, changes Docker, exposes host ports,
passes broker credentials, or claims ARM64 hardware verification.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
GIB = 1024**3
MINIMUM_FREE = 3 * GIB
ABORT_RESERVE = 512 * 1024**2
SOURCE_FILES = ("Dockerfile", ".dockerignore", "pyproject.toml", "setup.py", "README.md")
SOURCE_SUFFIXES = {".py", ".cpp", ".h", ".hpp", ".sql", ".typed", ".html", ".css", ".js", ".yaml", ".yml", ".json", ".toml"}
DOCKER_PREFIX = ["docker", "--host=unix:///var/run/docker.sock"]


def docker_environment() -> dict[str, str]:
    """Keep proxy/registry selectors; prevent redirection to another daemon."""
    excluded = {"DOCKER_HOST", "DOCKER_CONTEXT", "DOCKER_TLS", "DOCKER_TLS_VERIFY", "DOCKER_CERT_PATH"}
    result = {key: value for key, value in os.environ.items() if key not in excluded}
    result["BUILDX_CONFIG"] = str(ROOT.parent / "next-operations-buildx")
    return result


def docker(*arguments: str) -> str:
    result = subprocess.run(DOCKER_PREFIX + list(arguments), env=docker_environment(),
                            check=True, capture_output=True, text=True)
    return result.stdout.strip()


def admitted_source(root: Path) -> list[Path]:
    """Copy source-only inputs, never credentials, weights or host binaries."""
    paths = [root / name for name in SOURCE_FILES]
    for name in ("src", "configs"):
        directory = root / name
        if directory.is_symlink():
            raise ValueError("source inputs cannot follow symbolic links")
        for path in sorted(directory.rglob("*")):
            if path.is_symlink():
                raise ValueError("source inputs cannot follow symbolic links")
            if path.is_file() and path.suffix in SOURCE_SUFFIXES:
                if any(part.startswith(".") or part in {"model-assets", "huggingface-cache"}
                       for part in path.relative_to(root).parts):
                    continue
                if path.name.startswith(("credentials", "pytorch_model")):
                    continue
                paths.append(path)
    if any(not path.is_file() or path.is_symlink() for path in paths):
        raise ValueError("required source input is absent or redirected")
    return paths


def freeze_source(root: Path, destination: Path) -> dict[str, Any]:
    files = []
    for path in admitted_source(root):
        relative = path.relative_to(root)
        data = path.read_bytes()
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        files.append({"path": str(relative), "bytes": len(data),
                      "sha256": hashlib.sha256(data).hexdigest()})
    manifest = json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
    return {"files": files, "manifest_sha256": hashlib.sha256(manifest).hexdigest(),
            "compiled_host_artifacts_included": False, "credentials_included": False,
            "restricted_weights_included": False}


def preflight(free_bytes: int, driver: str) -> dict[str, Any]:
    if isinstance(free_bytes, bool) or not isinstance(free_bytes, int) or free_bytes < 0:
        raise ValueError("free space must be a nonnegative byte count")
    return {"driver": driver, "free_bytes": free_bytes, "minimum_free_bytes": MINIMUM_FREE,
            "abort_reserve_bytes": ABORT_RESERVE,
            "state": "READY_FOR_BOUNDED_BUILD" if free_bytes >= MINIMUM_FREE else "BLOCKED_STORAGE",
            "estimate": "Reduced VFS layers: ~0.7 GiB builder, ~0.7 GiB runtime, ~0.7 GiB test container, plus transient work and >=0.5 GiB reserve"}


def monitored_build(context: Path, tag: str, evidence: Path, python_image: str | None) -> dict[str, Any]:
    command = DOCKER_PREFIX + ["build", "--pull=false", "--progress=plain", "-t", tag]
    certificate = os.environ.get("CODEX_PROXY_CERT")
    if certificate:
        command += ["--secret", f"id=proxy_ca,src={certificate}"]
    if python_image:
        command += ["--build-arg", f"PYTHON_IMAGE={python_image}"]
    command.append(str(context))
    samples = []
    with (evidence / "build.txt").open("w") as output:
        process = subprocess.Popen(command, env=docker_environment(), stdout=output,
                                   stderr=subprocess.STDOUT)
        deadline = time.monotonic() + 900
        blocked = None
        while process.poll() is None:
            free = shutil.disk_usage(ROOT).free
            samples.append(free)
            if free < ABORT_RESERVE or time.monotonic() > deadline:
                blocked = "BLOCKED_STORAGE_RESERVE" if free < ABORT_RESERVE else "BLOCKED_BUILD_DEADLINE"
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=10)
                break
            time.sleep(.5)
    return {"state": blocked or ("BUILT" if process.returncode == 0 else "BUILD_FAILED"),
            "returncode": process.returncode, "free_bytes_minimum_observed": min(samples, default=shutil.disk_usage(ROOT).free),
            "source_bind_to_runtime": False}


HTTP_CODE = """import json,urllib.request,sys
path,payload=json.loads(sys.argv[1]);data=None if payload is None else json.dumps(payload).encode()
request=urllib.request.Request('http://127.0.0.1:8000'+path,data=data,headers={'Content-Type':'application/json'})
with urllib.request.urlopen(request,timeout=10) as response:print(response.read().decode())
"""
NATIVE_CODE = """import importlib.util,json,numpy as np,pandas as pd
import duckdb,lightgbm,statsmodels,exchange_calendars
from gig.factors.neutralize import neutralize
import gig._speed
columns=[str(i) for i in range(60)]
x=pd.DataFrame(np.random.default_rng(19).normal(size=(8,60)),columns=columns)
sectors=pd.Series(['A']*20+['B']*20+['C']*20,index=columns)
a=neutralize(x,sectors,use_cpp=True).to_numpy();b=neutralize(x,sectors,use_cpp=False).to_numpy()
np.testing.assert_allclose(a,b,atol=2e-12,rtol=2e-11)
print(json.dumps({'native_parity':'PASS','comparisons':int(a.size),'max_abs_error':float(np.max(np.abs(a-b))),
'libraries':{name:bool(importlib.util.find_spec(name)) for name in ['duckdb','lightgbm','statsmodels','exchange_calendars','torch','transformers']},
'imported_library_versions':{module.__name__:module.__version__ for module in [duckdb,lightgbm,statsmodels,exchange_calendars]}}))
"""


def runtime_recovery(tag: str, evidence: Path) -> dict[str, Any]:
    suffix = uuid.uuid4().hex[:12]
    volume = "gig-bounded-state-" + suffix
    names = ["gig-bounded-write-" + suffix, "gig-bounded-read-" + suffix]
    docker("volume", "create", "--label", "gig.task=accelerated-2026", volume)
    docker("run", "--rm", "--network", "none", "--user", "0:0", "--cap-drop", "ALL",
           "--cap-add", "CHOWN", "--mount", f"type=volume,src={volume},dst=/app/data",
           tag, "python", "-c", "import os;os.chown('/app/data',10001,10001)")

    def request(name: str, path: str, payload: Any = None) -> dict[str, Any]:
        return json.loads(docker("exec", name, "python", "-c", HTTP_CODE, json.dumps([path, payload])))

    def start(name: str) -> None:
        if shutil.disk_usage(ROOT).free < GIB + ABORT_RESERVE:
            raise RuntimeError("runtime storage reserve would be exceeded")
        docker("run", "-d", "--name", name, "--label", "gig.task=accelerated-2026",
               "--network", "none", "--read-only", "--tmpfs", "/tmp:rw,noexec,nosuid,size=64m",
               "--memory", "1g", "--cpus", "1", "--pids-limit", "128", "--cap-drop", "ALL",
               "--security-opt", "no-new-privileges", "--mount", f"type=volume,src={volume},dst=/app/data",
               "-e", "OPENBLAS_NUM_THREADS=1", "-e", "OMP_NUM_THREADS=1",
               "-e", "GIG_DATA_DIR=/app/data", "-e", "GIG_BROKER_CONNECTIVITY_ENABLED=false", tag)
        for _ in range(60):
            try:
                if request(name, "/healthz")["ok"]:
                    return
            except (subprocess.CalledProcessError, KeyError):
                time.sleep(.2)
        raise RuntimeError("container readiness deadline exceeded")

    def stop_remove(name: str) -> None:
        try:
            docker("stop", "--time", "10", name)
            docker("rm", name)
        except subprocess.CalledProcessError:
            pass

    now = "2026-11-01T15:00:00Z"
    query = "?mode=local&engine=execution-engine&as_of=2026-11-01T15%3A00%3A00Z&max_age_seconds=60"
    try:
        start(names[0])
        native = json.loads(docker("exec", names[0], "python", "-c", NATIVE_CODE))
        assert docker("exec", names[0], "id", "-u") == "10001"

        def operation(name: str, **values: Any) -> dict[str, Any]:
            result = request(names[0], "/api/v1/workspace/execution-engine",
                             {"operation": name, "now": now, **values})
            if result["state"] != "AVAILABLE":
                raise RuntimeError("offline execution operation was not admitted")
            return result

        operation("prepare", client_id="bounded-one", symbol="SYN001", side="buy", quantity="10",
                  reference_price="100", price_asof=now, max_notional="1000", max_position_quantity="10")
        acknowledgement = operation("dispatch")
        assert acknowledgement["data"]["positions"] == {}
        fill = operation("fill", client_id="bounded-one", quantity="2", price="100")
        assert fill["data"]["positions"]["SYN001"] == "2"
        operation("reconcile")
        operation("schedule-twap", client_id="bounded-parent", symbol="SYN002", side="buy",
                  quantity="2", start=now, end="2026-11-01T15:02:00Z", slices=2,
                  max_notional="200", max_position_quantity="2")
        advanced = operation("advance-clock", marks={"SYN002": {"reference_price": "100", "price_asof": now}})
        assert advanced["data"]["operation_result"]["admitted_children"] == ["bounded-parent:1"]
        assert advanced["data"]["positions"] == {"SYN001": "2"}
        operation("kill-switch", engaged=True)
        before = {name: request(names[0], "/api/v1/terminal/" + name + query)
                  for name in ("execution", "portfolio", "risk")}
        assert all(value["state"] == "AVAILABLE" for value in before.values())
        stop_remove(names[0])
        start(names[1])
        after = {name: request(names[1], "/api/v1/terminal/" + name + query) for name in before}
        assert before == after
        assert after["risk"]["data"]["metrics"]["kill_switch"]
        assert after["execution"]["data"]["schedules"][0]["children"][0]["state"] == "ADMITTED"
        assert after["execution"]["data"]["schedules"][0]["children"][1]["state"] == "PENDING"
        report = {"state": "PASS", "native": native, "uid": 10001, "network": "none",
                  "read_only_root": True, "host_ports": [], "source_binds": [],
                  "persistent_volume": volume, "volume_retained": True,
                  "content_hashes_before_after": {key: value["content_sha256"] for key, value in after.items()},
                  "twap_pending_and_admitted_children_persisted": True,
                  "clock_advance_dispatched_or_filled": False,
                  "broker_connections": 0, "public_deployment": False}
        (evidence / "runtime.json").write_text(json.dumps(report, indent=2) + "\n")
        return report
    finally:
        for name in names:
            stop_remove(name)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-dir", type=Path, required=True)
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--python-image", help="Optional immutable official Python base selector")
    args = parser.parse_args()
    evidence = args.evidence_dir.absolute()
    evidence.mkdir(parents=True, exist_ok=False)
    info = json.loads(docker("info", "--format", '{{json .}}'))
    report = preflight(shutil.disk_usage(ROOT).free, info["Driver"])
    report.update({"host_architecture": platform.machine(), "validation_level": "PREFLIGHT_ONLY",
                   "physical_ampere_validated": False, "arm64_runtime_validated": False,
                   "docker_version": info["ServerVersion"]})
    if args.build and report["state"] == "READY_FOR_BOUNDED_BUILD":
        tag = "gig-bounded-source-" + uuid.uuid4().hex[:12]
        with tempfile.TemporaryDirectory(prefix="gig-bounded-context-") as temporary:
            context = Path(temporary)
            source = freeze_source(ROOT, context)
            (evidence / "source.json").write_text(json.dumps(source, indent=2) + "\n")
            build = monitored_build(context, tag, evidence, args.python_image)
            report["build"] = build
        if build["state"] == "BUILT":
            report["image_id"] = docker("image", "inspect", tag, "--format", '{{.Id}}')
            report["image_tag"] = tag
            try:
                report["runtime"] = runtime_recovery(tag, evidence)
            except (RuntimeError, subprocess.CalledProcessError, AssertionError, ValueError) as exc:
                report["state"] = "RUNTIME_FAILED"
                report["runtime_error_type"] = type(exc).__name__
                report["validation_level"] = "FRESH_DOCKERFILE_BUILD_ONLY"
            else:
                report["state"] = "PASS"
                report["validation_level"] = "FRESH_DOCKERFILE_BUILD_AND_X86_RESTART"
        else:
            report["state"] = build["state"]
    (evidence / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if report["state"] in {"READY_FOR_BOUNDED_BUILD", "PASS"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
