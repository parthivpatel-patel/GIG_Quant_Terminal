"""Run approved research jobs, then exit.

``--ticks`` runs that many manual steps. ``--worker`` starts the bounded
scheduler and waits until it pauses or stops. Neither mode keeps running
after the approved queue is empty, and neither submits an order.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from gig.research_org.experiment import run_seeded
from gig.research_org.futures_experiment import run_futures
from gig.research_org.ledger import ResearchLedger
from gig.research_org.scheduler import tick
from gig.research_org.service import ResearchWorker
from gig.research_org.treasury_experiment import run_treasury

_COMPUTE = {
    "futures-tsmom-authored-excess-v1": run_futures,
    "lagged-sign-momentum-synthetic-v1": run_seeded,
    "treasury-10y-lagged-sign-2023": run_treasury,
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-dir", type=Path)
    parser.add_argument("--ticks", type=int, default=0)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--cadence-seconds", type=float, default=0.05)
    parser.add_argument("--compute", choices=sorted(_COMPUTE))
    args = parser.parse_args(argv)
    if args.compute:
        _, report = _COMPUTE[args.compute]()
        print(json.dumps({"verdict": report["verdict"], "hypothesis_id": report["hypothesis_id"],
                          "classification": report["classification"],
                          "total_return": report.get("total_return"),
                          "sharpe_after_cost": report.get("sharpe_after_cost"),
                          "economic_claim": report["economic_claim"]}, sort_keys=True))
        return 0
    if args.worker:
        if args.state_dir is None:
            parser.error("--worker requires --state-dir")
        worker = ResearchWorker(ResearchLedger(args.state_dir))
        worker.start(args.cadence_seconds)
        worker.wait()
        print(worker.ledger.scheduler_state()["reason"] or worker.ledger.scheduler_state()["desired"])
        return 0
    if args.state_dir is None or args.ticks < 1:
        parser.error("pass --compute, --worker, or --state-dir and --ticks greater than zero")
    ledger = ResearchLedger(args.state_dir)
    for _ in range(args.ticks):
        result = tick(ledger)
        step = result["tick"]
        last = result.get("last") or {}
        print(f"{step['action']} {step.get('job_id') or '-'} {last.get('verdict') or '-'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
