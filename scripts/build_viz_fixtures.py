"""Regenerate the stored deterministic outputs of the six synthetic 3D modules."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gig.service import viz  # noqa: E402


def main() -> int:
    payload = {"source": "deterministic-synthetic-v1", "classification": "SYNTHETIC",
               "generator": "scripts/build_viz_fixtures.py",
               "modules": {name: viz.BUILDERS[name]() for name in viz.MODULES}}
    viz.PRECOMPUTED.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n")
    print(f"wrote {viz.PRECOMPUTED} ({viz.PRECOMPUTED.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
