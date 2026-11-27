"""Write the bounded-family registry into the static terminal shell.

The terminal ships static assets with no build step, so the workbench table
and options are generated from the Python registry. ``--check`` fails when
the committed assets differ from the registry.
"""
from __future__ import annotations

import argparse
import html
import json
import sys
from pathlib import Path

from gig.ml.bounded import FAMILIES

STATIC = Path(__file__).resolve().parents[1]/"src/gig/service/static"
PREFIX = "workspace-research-family-"


def blocks() -> dict[str, tuple[str, str, str]]:
    rows = [[f.key, f.label, f.scope, max(40, f.min_train)] for f in FAMILIES.values()]
    table = "  const boundedFamilies = [\n" + "".join(f"    {json.dumps(r, ensure_ascii=False)},\n" for r in rows) + "  ];\n"
    options = "".join(f'<option value="{PREFIX}{f.key}">{html.escape(f.label)}</option>' for f in FAMILIES.values())
    return {"app.js": ("  // BEGIN GENERATED BOUNDED FAMILIES\n", "  // END GENERATED BOUNDED FAMILIES", table),
            "index.html": ("<!-- BEGIN BOUNDED FAMILIES -->", "<!-- END BOUNDED FAMILIES -->", options)}


def render(name: str) -> str:
    start, end, body = blocks()[name]
    text = (STATIC/name).read_text(encoding="utf-8")
    if text.count(start) != 1 or text.count(end) != 1:
        raise SystemExit(f"{name} must contain exactly one generated bounded-family block")
    head, rest = text.split(start)
    return head + start + body + end + rest.split(end)[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    check = parser.parse_args().check
    stale = [name for name in blocks() if render(name) != (STATIC/name).read_text(encoding="utf-8")]
    if not check:
        for name in stale:
            (STATIC/name).write_text(render(name), encoding="utf-8")
    print(("STALE: " if check else "UPDATED: ") + ", ".join(stale) if stale else "bounded family assets match the registry")
    return 1 if check and stale else 0


if __name__ == "__main__":
    sys.exit(main())
