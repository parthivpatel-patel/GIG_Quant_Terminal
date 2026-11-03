"""Read-only Docker capacity admission; never creates, builds or deletes resources.

VFS copies image contents into a new container layer even with a read-only root
filesystem. Healthy scratch storage on another device does not fund that copy.
An AVAILABLE result admits a bounded capacity plan only; it is not runtime proof.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from collections import defaultdict
from pathlib import Path

MIB = 1024**2
GIB = 1024**3
SELECTORS = ("DOCKER_HOST", "DOCKER_CONTEXT", "DOCKER_TLS", "DOCKER_TLS_VERIFY", "DOCKER_CERT_PATH")
IMAGE = re.compile(r"(?:sha256:[a-f0-9]{64}|[A-Za-z0-9][A-Za-z0-9._/@:+-]{0,255})\Z")


def filesystem_observation(path, *, stat=os.stat, statvfs=os.statvfs):
    """Observe available-to-caller bytes, never privileged/reserved free bytes."""
    try:
        resolved = Path(path).resolve(strict=True)
        volume = statvfs(resolved)
        device = stat(resolved).st_dev
        if volume.f_bavail < 0 or volume.f_frsize <= 0:
            raise ValueError("invalid filesystem observation")
        return {"state": "AVAILABLE", "path": str(resolved), "device": str(device),
                "available_bytes": volume.f_bavail*volume.f_frsize,
                "free_including_reserved_bytes": volume.f_bfree*volume.f_frsize}
    except (OSError, ValueError, TypeError):
        return {"state": "UNKNOWN", "path": str(path), "device": None,
                "available_bytes": None, "error_code": "FILESYSTEM_OBSERVATION_UNAVAILABLE"}


def docker_observation(image, *, runner=subprocess.run, environment=None):
    if (not isinstance(image, str) or IMAGE.fullmatch(image) is None or "://" in image
            or ("@" in image and re.search(r"@sha256:[a-f0-9]{64}\Z", image) is None)):
        raise ValueError("bounded Docker image identity required")
    env = dict(os.environ if environment is None else environment)
    for name in SELECTORS:
        env.pop(name, None)
    command = ["docker", "--host=unix:///var/run/docker.sock"]
    try:
        info = runner([*command, "info", "--format", '{"driver":{{json .Driver}},"root":{{json .DockerRootDir}}}'],
                      capture_output=True, text=True, timeout=10, check=True, env=env)
        image_result = runner([*command, "image", "inspect", image, "--format",
                               '{"id":{{json .Id}},"size":{{json .Size}},"architecture":{{json .Architecture}}}'],
                              capture_output=True, text=True, timeout=10, check=True, env=env)
        daemon, observed = json.loads(info.stdout), json.loads(image_result.stdout)
        size = observed["size"]
        if not isinstance(daemon["driver"], str) or not isinstance(daemon["root"], str) or not Path(daemon["root"]).is_absolute():
            raise ValueError("invalid managed daemon observation")
        if isinstance(size, bool) or not isinstance(size, int) or not 0 < size <= 128*GIB:
            raise ValueError("bounded observed image size required")
        return {"state": "AVAILABLE", "driver": daemon["driver"], "root": daemon["root"],
                "image": image, "image_id": observed["id"], "image_bytes": size,
                "architecture": observed["architecture"], "managed_local_socket": True}
    except (OSError, subprocess.SubprocessError, ValueError, KeyError, TypeError):
        return {"state": "UNKNOWN", "image": image, "driver": None, "root": None,
                "image_bytes": None, "error_code": "DOCKER_OBSERVATION_UNAVAILABLE"}


def assess_capacity(daemon, filesystems, *, operation="container", build_budget_bytes=None):
    if operation not in {"container", "build"}:
        raise ValueError("explicit bounded operation required")
    if build_budget_bytes is not None and (isinstance(build_budget_bytes, bool)
            or not isinstance(build_budget_bytes, int) or not 0 < build_budget_bytes <= 128*GIB):
        raise ValueError("bounded positive build write budget required")
    reasons = []
    required = {"source": 128*MIB, "scratch": 256*MIB}
    if daemon.get("state") != "AVAILABLE":
        reasons.append("DOCKER_OBSERVATION_UNAVAILABLE")
    elif daemon.get("driver") not in {"vfs", "overlay2", "overlayfs"}:
        reasons.append("STORAGE_DRIVER_COPY_BUDGET_UNKNOWN")
    else:
        size = daemon.get("image_bytes")
        if isinstance(size, bool) or not isinstance(size, int) or not 0 < size <= 128*GIB:
            raise ValueError("observed image byte bound required")
        # Conservative policy, not an inferred measurement of future allocations.
        # VFS requires complete copies; overlay still needs an explicit write reserve.
        copy_allowance = 2*size if daemon["driver"] == "vfs" else size
        required["docker"] = max(3*GIB, copy_allowance+512*MIB+GIB)
        if operation == "build":
            if build_budget_bytes is None:
                reasons.append("ADDITIONAL_BUILD_WRITE_BUDGET_REQUIRED")
            else:
                required["docker"] = max(6*GIB, 2*copy_allowance+build_budget_bytes+GIB)
    grouped = defaultdict(lambda: {"roles": [], "required_bytes": 0, "available_bytes": None})
    for role, needed in required.items():
        observation = filesystems.get(role, {})
        available = observation.get("available_bytes")
        if (observation.get("state") != "AVAILABLE" or observation.get("device") is None
                or isinstance(available, bool) or not isinstance(available, int) or available < 0):
            reasons.append(role.upper()+"_FILESYSTEM_UNKNOWN")
            continue
        group = grouped[str(observation["device"])]
        group["roles"].append(role)
        group["required_bytes"] += needed
        # Probe drift/concurrent writers cannot improve the smaller observation.
        prior = group["available_bytes"]
        group["available_bytes"] = available if prior is None else min(prior, available)
    volumes = []
    for device, group in sorted(grouped.items()):
        item = {"device": device, **group}
        item["capacity_sufficient"] = item["available_bytes"] >= item["required_bytes"]
        if not item["capacity_sufficient"]:
            reasons.append("INSUFFICIENT_AVAILABLE_BYTES:"+",".join(item["roles"]))
        volumes.append(item)
    known_shortfall = any(reason.startswith("INSUFFICIENT_AVAILABLE_BYTES:") for reason in reasons)
    state = "UNAVAILABLE" if known_shortfall else "UNKNOWN" if reasons else "AVAILABLE"
    return {"schema_version": "bounded-operations-capacity-v1", "state": state,
            "capacity_admitted": not reasons, "operation": operation, "reasons": reasons,
            "daemon": daemon, "filesystems": filesystems, "required_bytes_by_role": required,
            "aggregated_filesystems": volumes, "additional_build_write_budget_bytes": build_budget_bytes,
            "budget_scope": "Conservative admission policy; concurrent writers or larger workloads can still exhaust capacity.",
            "container_create_attempted": False, "container_start_verified": False,
            "container_restart_verified": False, "fresh_image_built": False,
            "physical_arm64_verified": False, "resources_deleted": False,
            "broker_connections": 0, "paid_services": False,
            "required_runtime_boundaries": {"network": "none", "root_filesystem": "read-only",
                                            "cpus": 1, "memory_bytes": 768*MIB,
                                            "pids_limit": 128, "tmpfs_bytes": 64*MIB}}


def observe(image, source, scratch, *, operation="container", build_budget_bytes=None,
            docker_probe=docker_observation, filesystem_probe=filesystem_observation):
    daemon = docker_probe(image)
    filesystems = {"source": filesystem_probe(source), "scratch": filesystem_probe(scratch)}
    if daemon.get("root"):
        filesystems["docker"] = filesystem_probe(daemon["root"])
    return assess_capacity(daemon, filesystems, operation=operation, build_budget_bytes=build_budget_bytes)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", default="gig-bounded-source-11dcdc87a8c5:latest")
    parser.add_argument("--source", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--scratch", type=Path, default=Path("/tmp"))
    parser.add_argument("--operation", choices=("container", "build"), default="container")
    parser.add_argument("--build-budget-bytes", type=int)
    parser.add_argument("--output", type=Path, help="Write a new evidence file; existing evidence is never overwritten.")
    args = parser.parse_args(argv)
    report = observe(args.image, args.source, args.scratch, operation=args.operation,
                     build_budget_bytes=args.build_budget_bytes)
    wire = json.dumps(report, indent=2, sort_keys=True)+"\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as stream:
            stream.write(wire)
    print(wire, end="")
    return 0 if report["capacity_admitted"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
