"""Hash configs and persist experiment metrics as JSON lines."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def config_hash(config: dict[str, Any]) -> str:
    blob = json.dumps(config, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


class ExperimentLog:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def record(self, name: str, config: dict[str, Any], metrics: dict[str, float]) -> str:
        digest = config_hash(config)
        row = {
            "ts": datetime.now(UTC).isoformat(),
            "name": name,
            "config_hash": digest,
            "config": config,
            "metrics": metrics,
        }
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, default=str) + "\n")
        return digest
