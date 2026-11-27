"""Write the Python registries into the static terminal shell.

The terminal ships static assets with no build step, so the bounded-family
table and the stateless calculator table, with their options, are generated
from the Python registries. ``--check`` fails when the committed assets
differ from them.
"""
from __future__ import annotations

import argparse
import html
import json
import sys
from pathlib import Path

from gig.ml.bounded import FAMILIES
from gig.service.quant_analytics import ANALYSES

STATIC = Path(__file__).resolve().parents[1]/"src/gig/service/static"
PREFIX = "workspace-research-family-"


def _table(name: str, rows: list[list]) -> str:
    return f"  const {name} = [\n" + "".join(f"    {json.dumps(r, ensure_ascii=False)},\n" for r in rows) + "  ];\n"


def blocks() -> dict[str, list[tuple[str, str, str]]]:
    families = [[f.key, f.label, f.scope, max(40, f.min_train)] for f in FAMILIES.values()]
    analyses = [[a.key, a.label, a.scope, a.example] for a in ANALYSES.values()]
    return {
        "app.js": [("  // BEGIN GENERATED BOUNDED FAMILIES\n", "  // END GENERATED BOUNDED FAMILIES",
                    _table("boundedFamilies", families)),
                   ("  // BEGIN GENERATED QUANT ANALYTICS\n", "  // END GENERATED QUANT ANALYTICS",
                    _table("quantAnalytics", analyses))],
        "index.html": [("<!-- BEGIN BOUNDED FAMILIES -->", "<!-- END BOUNDED FAMILIES -->",
                        "".join(f'<option value="{PREFIX}{f.key}">{html.escape(f.label)}</option>' for f in FAMILIES.values())),
                       ("<!-- BEGIN QUANT ANALYTICS -->", "<!-- END QUANT ANALYTICS -->",
                        "".join(f'<option value="{a.key}">{html.escape(a.label)}</option>' for a in ANALYSES.values()))],
    }


def render(name: str) -> str:
    text = (STATIC/name).read_text(encoding="utf-8")
    for start, end, body in blocks()[name]:
        if text.count(start) != 1 or text.count(end) != 1:
            raise SystemExit(f"{name} must contain exactly one block from {start.strip()} to {end.strip()}")
        head, rest = text.split(start)
        text = head + start + body + end + rest.split(end)[1]
    return text


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    check = parser.parse_args().check
    stale = [name for name in blocks() if render(name) != (STATIC/name).read_text(encoding="utf-8")]
    if not check:
        for name in stale:
            (STATIC/name).write_text(render(name), encoding="utf-8")
    print(("STALE: " if check else "UPDATED: ") + ", ".join(stale) if stale else "generated terminal assets match the registries")
    return 1 if check and stale else 0


if __name__ == "__main__":
    sys.exit(main())
