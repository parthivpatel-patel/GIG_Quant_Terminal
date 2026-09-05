"""
File-based kill switch for the paper / live trading loop.

When engaged, ``run_once`` refuses to submit and the embedded paper loop
idles. Flatten remains available so you can still exit. The flag lives on disk
so a restarted process inherits the halt without depending on process memory.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def _path() -> Path:
    from gig.config import get_settings

    settings = get_settings()
    settings.ensure_dirs()
    return settings.data_dir / "KILL_SWITCH.json"


@dataclass(slots=True)
class KillSwitch:
    """Persist halt state under ``data/KILL_SWITCH.json``."""

    path: Path | None = None

    def _file(self) -> Path:
        return self.path or _path()

    def status(self) -> dict[str, Any]:
        path = self._file()
        if not path.exists():
            return {"engaged": False, "path": str(path)}
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            return {
                "engaged": True,
                "path": str(path),
                "error": f"{type(exc).__name__}: {exc}",
                "reason": "unreadable kill-switch file — treating as engaged",
            }
        return {
            "engaged": bool(raw.get("engaged", True)),
            "reason": str(raw.get("reason") or ""),
            "ts": raw.get("ts"),
            "by": str(raw.get("by") or ""),
            "path": str(path),
        }

    @property
    def engaged(self) -> bool:
        return bool(self.status().get("engaged"))

    def halt(self, reason: str = "manual halt", by: str = "cli") -> dict[str, Any]:
        path = self._file()
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "engaged": True,
            "reason": reason,
            "by": by,
            "ts": datetime.now(UTC).isoformat(),
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return self.status()

    def resume(self, by: str = "cli") -> dict[str, Any]:
        path = self._file()
        if path.exists():
            path.unlink()
        return {"engaged": False, "path": str(path), "resumed_by": by}


def kill_switch() -> KillSwitch:
    return KillSwitch()
