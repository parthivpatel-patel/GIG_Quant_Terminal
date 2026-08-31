"""Structured logging used by CLI and library code."""

from __future__ import annotations

import logging
import sys


def configure_logging(level: str = "INFO") -> logging.Logger:
    root = logging.getLogger("gig")
    if root.handlers:
        root.setLevel(level.upper())
        return root
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s - %(message)s", "%Y-%m-%d %H:%M:%S")
    )
    root.addHandler(handler)
    root.setLevel(level.upper())
    root.propagate = False
    return root
