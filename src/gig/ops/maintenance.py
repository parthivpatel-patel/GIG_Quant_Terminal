"""Offline diagnostics and verified state backups. Stop writers before backup.

Backups remain local artifacts until independently copied and verified. Restore
creates a new directory and never overwrites state. No credentials are read.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import io
import json
import os
import platform
import tarfile
from pathlib import Path, PurePosixPath

MAX_BYTES = 512 * 1024 * 1024
MAX_FILES = 10000


def startup_report() -> dict:
    dependencies = {name: importlib.util.find_spec(name) is not None
                    for name in ("numpy", "pandas", "scipy", "duckdb", "fastapi", "uvicorn")}
    gate = os.environ.get("GIG_BROKER_CONNECTIVITY_ENABLED", "false").lower()
    safe = gate in {"false", "0", "off", "no"}
    return {"state": "AVAILABLE" if all(dependencies.values()) and safe else "UNAVAILABLE",
            "dependencies": dependencies, "architecture": platform.machine(),
            "python": platform.python_version(), "broker_connectivity_disabled": safe,
            "trading_authorized": False, "data_readiness": "UNKNOWN"}


def _safe_name(name: str) -> bool:
    path = PurePosixPath(name)
    return (bool(name) and name != "." and path.as_posix() == name
            and not path.is_absolute() and ".." not in path.parts and "\\" not in name)


def _secret_name(path: Path) -> bool:
    return any(part.startswith(".") or any(token in part.lower() for token in
               ("secret", "credential", "token", "private_key")) for part in path.parts)


def create_backup(source: Path, archive: Path, *, quiesced: bool = False) -> dict:
    if not quiesced:
        raise ValueError("Stop all state writers and pass quiesced=True")
    source = source.resolve(strict=True)
    archive = archive.absolute()
    if not source.is_dir() or archive.exists() or archive.is_symlink():
        raise ValueError("source must be a directory and archive must be new")
    if source == archive or source in archive.parents:
        raise ValueError("archive must be outside source")
    payloads: list[tuple[str, bytes]] = []
    total = 0
    for path in sorted(source.rglob("*")):
        relative = path.relative_to(source)
        if path.is_symlink():
            raise ValueError("state backups reject symbolic links")
        if not path.is_file() or _secret_name(relative):
            continue
        if relative.as_posix() == "manifest.json":
            raise ValueError("reserved state filename manifest.json")
        if path.stat().st_size > MAX_BYTES:
            raise ValueError("state file exceeds backup bound")
        content = path.read_bytes()
        total += len(content)
        if total > MAX_BYTES or len(payloads) >= MAX_FILES:
            raise ValueError("state exceeds backup bounds")
        payloads.append((relative.as_posix(), content))
    manifest = {"schema_version": 1, "quiesced": True, "files": [
        {"path": name, "size": len(content), "sha256": hashlib.sha256(content).hexdigest()}
        for name, content in payloads]}
    archive.parent.mkdir(parents=True, exist_ok=True)
    with archive.open("xb") as stream, tarfile.open(fileobj=stream, mode="w:gz") as tar:
        for name, content in payloads + [("manifest.json", json.dumps(manifest, sort_keys=True).encode())]:
            info = tarfile.TarInfo(name)
            info.size = len(content)
            info.mode = 0o600
            tar.addfile(info, io.BytesIO(content))
    verified = verify_backup(archive)
    return {"sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
            "file_count": len(verified["files"]), "durable_external_backup": False}


def _read_backup(archive: Path):
    payloads: dict[str, bytes] = {}
    total = 0
    with tarfile.open(archive, "r:gz") as tar:
        for member in tar:
            if not member.isfile() or not _safe_name(member.name) or member.name in payloads:
                raise ValueError("unsafe or duplicate archive member")
            total += member.size
            if member.size < 0 or total > MAX_BYTES or len(payloads) > MAX_FILES:
                raise ValueError("archive exceeds bounds")
            stream = tar.extractfile(member)
            if stream is None:
                raise ValueError("unreadable archive member")
            payloads[member.name] = stream.read()
    raw = payloads.pop("manifest.json", None)
    if raw is None:
        raise ValueError("manifest absent")
    manifest = json.loads(raw)
    if manifest.get("schema_version") != 1 or manifest.get("quiesced") is not True:
        raise ValueError("unsupported or non-quiesced backup")
    records = manifest.get("files")
    if not isinstance(records, list) or len(records) != len(payloads):
        raise ValueError("manifest does not describe archive")
    names = set()
    for record in records:
        name = record["path"]
        if name in names or name not in payloads:
            raise ValueError("duplicate or absent manifest file")
        names.add(name)
        data = payloads[name]
        if len(data) != record["size"] or hashlib.sha256(data).hexdigest() != record["sha256"]:
            raise ValueError("backup digest mismatch")
    return manifest, payloads


def verify_backup(archive: Path) -> dict:
    return _read_backup(archive)[0]


def restore_backup(archive: Path, destination: Path) -> dict:
    manifest, payloads = _read_backup(archive)
    if destination.exists() or destination.is_symlink():
        raise ValueError("restore requires a new destination")
    destination.mkdir(parents=True)
    for name, data in payloads.items():
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as stream:
            stream.write(data)
        target.chmod(0o600)
    return manifest


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("check")
    backup = sub.add_parser("backup")
    backup.add_argument("source", type=Path)
    backup.add_argument("archive", type=Path)
    backup.add_argument("--quiesced", action="store_true")
    verify = sub.add_parser("verify")
    verify.add_argument("archive", type=Path)
    restore = sub.add_parser("restore")
    restore.add_argument("archive", type=Path)
    restore.add_argument("destination", type=Path)
    args = parser.parse_args(argv)
    if args.command == "check":
        report = startup_report()
        print(json.dumps(report, sort_keys=True))
        return 0 if report["state"] == "AVAILABLE" else 1
    if args.command == "backup":
        result = create_backup(args.source, args.archive, quiesced=args.quiesced)
    elif args.command == "verify":
        result = verify_backup(args.archive)
    else:
        result = restore_backup(args.archive, args.destination)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
