"""Explicit permitted ECB acquisition and real CPU research, never brokerage."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from gig.service.observed_workspace import ObservedWorkspace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--evidence-dir", type=Path, required=True)
    parser.add_argument("--models", nargs="+", default=["ridge", "lightgbm", "hmm", "garch", "kalman", "bilstm"])
    args = parser.parse_args()
    args.evidence_dir.mkdir(parents=True, exist_ok=True)
    workspace = ObservedWorkspace(args.state_dir)
    source = workspace.run({"operation": "fetch", "provider": "ECB_REFERENCE_FX", "currencies": ["USD", "GBP", "JPY"],
                            "start": "2025-01-01", "end": "2025-12-31"}).to_dict()
    if source["state"] != "AVAILABLE":
        raise ValueError("public observed source not admitted")
    snapshot = source["data"]
    proof = {"source": {key: snapshot[key] for key in ("snapshot_id", "classification", "acquired_at", "source_url", "source_authenticity", "raw_sha256", "csv_sha256", "diagnostics", "attribution", "license_url")},
             "models": {}, "broker_connections": 0, "paid_services": False, "historical_point_in_time_validated": False,
             "funded_allocation_eligible": False, "economic_validation_established": False}
    for model in args.models:
        result = workspace.run({"operation": "research", "snapshot_id": snapshot["snapshot_id"], "model": model,
            "training_observations": 60, "test_observations": 20, "embargo": 1, "seed": 17}).to_dict()
        if result["state"] != "AVAILABLE":
            raise ValueError("observed reference research failed")
        document = result["data"]
        replay = workspace.run({"operation": "replay", "report_id": document["report_id"]}).to_dict()
        if replay != result:
            raise ValueError("observed reference replay mismatch")
        proof["models"][model] = {key: document[key] for key in ("report_id", "model_ids", "prediction_kind", "metrics", "input_classification", "research_mode")}
        proof["models"][model]["fit_states"] = sorted({row["fit_state"] for row in document["folds"]})
        proof["models"][model]["replay_verified"] = True
        (args.evidence_dir/("observed-"+model+".json")).write_text(json.dumps(result, indent=2)+"\n")
    (args.evidence_dir/"observed-source.json").write_text(json.dumps(source, indent=2)+"\n")
    (args.evidence_dir/"observed-verification.json").write_text(json.dumps(proof, indent=2)+"\n")
    print(json.dumps(proof, indent=2))


if __name__ == "__main__":
    main()
