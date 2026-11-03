"""Read-only operational admission prevents the reproduced VFS disk failure."""
from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from types import SimpleNamespace

from scripts.sprint8_operations_check import (
    GIB,
    MIB,
    SELECTORS,
    assess_capacity,
    docker_observation,
    filesystem_observation,
    observe,
)


def volume(device, available):
    return {"state": "AVAILABLE", "device": device, "available_bytes": available}


def daemon(driver="vfs"):
    return {"state": "AVAILABLE", "driver": driver, "root": "/var/lib/docker", "image_bytes": 697300556}


class OperationsCapacityTests(unittest.TestCase):
    def test_full_docker_volume_blocks_even_with_healthy_independent_scratch(self):
        result = assess_capacity(daemon(), {"docker": volume("root", 0),
                                           "source": volume("tmp", 2*GIB), "scratch": volume("tmp", 2*GIB)})
        self.assertEqual(result["state"], "UNAVAILABLE")
        self.assertFalse(result["capacity_admitted"])
        self.assertEqual(result["required_bytes_by_role"]["docker"], 3*GIB)
        self.assertIn("INSUFFICIENT_AVAILABLE_BYTES:docker", result["reasons"])
        self.assertFalse(result["container_create_attempted"])
        self.assertFalse(result["container_restart_verified"])
        self.assertFalse(result["resources_deleted"])

    def test_shared_filesystem_reservations_are_added_before_admission(self):
        # Three GiB alone covers the image plan but not source/scratch writes on it.
        result = assess_capacity(daemon(), {role: volume("same", 3*GIB) for role in ("docker", "source", "scratch")})
        self.assertFalse(result["capacity_admitted"])
        self.assertEqual(len(result["aggregated_filesystems"]), 1)
        self.assertEqual(result["aggregated_filesystems"][0]["required_bytes"], 3*GIB+384*MIB)

    def test_boundary_passes_only_with_all_required_bytes_and_is_not_runtime_proof(self):
        needed = 3*GIB+384*MIB
        result = assess_capacity(daemon(), {role: volume("same", needed) for role in ("docker", "source", "scratch")})
        self.assertTrue(result["capacity_admitted"])
        self.assertEqual(result["state"], "AVAILABLE")
        self.assertFalse(result["fresh_image_built"])
        self.assertFalse(result["container_start_verified"])
        result = assess_capacity(daemon(), {role: volume("same", needed-1) for role in ("docker", "source", "scratch")})
        self.assertFalse(result["capacity_admitted"])

    def test_probe_drift_uses_smaller_available_observation(self):
        result = assess_capacity(daemon(), {"docker": volume("root", 8*GIB),
                                           "source": volume("tmp", GIB), "scratch": volume("tmp", 300*MIB)})
        self.assertFalse(result["capacity_admitted"])
        tmp = next(row for row in result["aggregated_filesystems"] if row["device"] == "tmp")
        self.assertEqual(tmp["available_bytes"], 300*MIB)

    def test_unknown_driver_or_missing_probe_never_authorizes_execution(self):
        filesystems = {role: volume("device", 20*GIB) for role in ("docker", "source", "scratch")}
        for item in ({"state": "UNKNOWN"}, daemon("unmodeled-driver")):
            result = assess_capacity(item, filesystems)
            self.assertEqual(result["state"], "UNKNOWN")
            self.assertFalse(result["capacity_admitted"])
        del filesystems["docker"]
        result = assess_capacity(daemon(), filesystems)
        self.assertEqual(result["state"], "UNKNOWN")
        self.assertFalse(result["capacity_admitted"])

    def test_build_requires_explicit_additional_write_bound(self):
        filesystems = {role: volume(role, 20*GIB) for role in ("docker", "source", "scratch")}
        result = assess_capacity(daemon(), filesystems, operation="build")
        self.assertFalse(result["capacity_admitted"])
        self.assertIn("ADDITIONAL_BUILD_WRITE_BUDGET_REQUIRED", result["reasons"])
        result = assess_capacity(daemon(), filesystems, operation="build", build_budget_bytes=GIB)
        self.assertTrue(result["capacity_admitted"])
        self.assertGreaterEqual(result["required_bytes_by_role"]["docker"], 6*GIB)
        for invalid in (True, 0, -1, .1, 129*GIB):
            with self.assertRaises(ValueError):
                assess_capacity(daemon(), filesystems, operation="build", build_budget_bytes=invalid)

    def test_filesystem_uses_caller_available_not_reserved_free_bytes(self):
        with tempfile.TemporaryDirectory() as temporary:
            result = filesystem_observation(temporary, stat=lambda _: SimpleNamespace(st_dev=7),
                statvfs=lambda _: SimpleNamespace(f_bavail=0, f_bfree=100000, f_frsize=4096))
        self.assertEqual(result["available_bytes"], 0)
        self.assertEqual(result["free_including_reserved_bytes"], 409600000)
        self.assertEqual(filesystem_observation("/path/that/does/not/exist")["state"], "UNKNOWN")

    def test_docker_probe_uses_only_managed_local_read_commands_and_preserves_credentials(self):
        calls = []
        environment = {name: "untrusted-selector" for name in SELECTORS}
        environment.update(DOCKER_CONFIG="/existing/config", HTTPS_PROXY="configured-proxy", SAFE="unchanged")
        original = environment.copy()

        def runner(command, **kwargs):
            calls.append((command, kwargs))
            body = {"driver": "vfs", "root": "/var/lib/docker"} if "info" in command else {
                "id": "sha256:"+"a"*64, "size": 697300556, "architecture": "amd64"}
            return SimpleNamespace(stdout=json.dumps(body))

        result = docker_observation("gig-bounded-source-11dcdc87a8c5:latest", runner=runner, environment=environment)
        self.assertEqual(result["state"], "AVAILABLE")
        self.assertEqual(environment, original)
        self.assertEqual(len(calls), 2)
        for command, kwargs in calls:
            self.assertEqual(command[:2], ["docker", "--host=unix:///var/run/docker.sock"])
            self.assertNotIn("run", command)
            self.assertNotIn("build", command)
            self.assertNotIn("prune", command)
            self.assertTrue(all(name not in kwargs["env"] for name in SELECTORS))
            self.assertEqual(kwargs["env"]["DOCKER_CONFIG"], "/existing/config")
            self.assertEqual(kwargs["env"]["HTTPS_PROXY"], "configured-proxy")

    def test_daemon_failure_is_sanitized_and_invalid_image_cannot_be_a_cli_flag(self):
        def runner(command, **kwargs):
            raise subprocess.CalledProcessError(1, command, stderr="private-credential-path")
        result = docker_observation("existing:latest", runner=runner, environment={})
        self.assertEqual(result["state"], "UNKNOWN")
        self.assertNotIn("private-credential-path", json.dumps(result))
        for invalid in ("--help", "image; echo secret", "", "https://user:password@host/image"):
            with self.assertRaises(ValueError):
                docker_observation(invalid, runner=runner)

    def test_orchestrator_observes_docker_store_separately_and_never_uses_scratch_as_substitute(self):
        paths = []
        def filesystem_probe(path):
            paths.append(str(path))
            return volume("root", 0) if str(path) == "/var/lib/docker" else volume("tmp", 2*GIB)
        result = observe("existing:latest", "/tmp/source", "/tmp", docker_probe=lambda _: daemon(), filesystem_probe=filesystem_probe)
        self.assertEqual(paths, ["/tmp/source", "/tmp", "/var/lib/docker"])
        self.assertEqual(result["state"], "UNAVAILABLE")
        self.assertFalse(result["capacity_admitted"])
