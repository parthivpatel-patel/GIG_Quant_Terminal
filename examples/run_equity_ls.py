"""Run the flagship equity long/short.

Live Yahoo is the default. Pass source='synthetic' for a deterministic CI run.
"""

from __future__ import annotations

import json
import sys

from gig.pipeline import run_equity_research


if __name__ == "__main__":
    source = sys.argv[1] if len(sys.argv) > 1 else "yahoo"
    result = run_equity_research(source=source, persist=True)
    print(json.dumps({"metrics": result.metrics, "factor_ic": result.factor_ic}, indent=2, default=float))
