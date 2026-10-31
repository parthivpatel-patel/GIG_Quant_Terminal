"""Atomic immutable artifact admission. Digests establish integrity, not origin."""
from __future__ import annotations

import os
import tempfile
from pathlib import Path


def write_immutable(path: str | Path, payload: bytes) -> Path:
    """Publish complete bytes atomically without replacing an existing artifact.

    The temporary file is synced before a same-directory hard link publishes it.
    Concurrent identical writers replay safely; conflicts fail and preserve the
    original. A killed writer can leave an unreferenced temporary file, never a
    partially published model. State backup procedures may omit ``.admit-*``.
    """
    target = Path(path)
    if not isinstance(payload, bytes):
        raise TypeError("immutable artifact payload must be bytes")
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".admit-", dir=target.parent)
    staged = Path(temporary)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(staged, target)
        except FileExistsError:
            if target.is_symlink() or target.read_bytes() != payload:
                raise ValueError("existing immutable artifact was modified") from None
        if hasattr(os, "O_DIRECTORY"):
            directory = os.open(target.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
    finally:
        staged.unlink(missing_ok=True)
    return target
