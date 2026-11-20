"""Local authored study and two-process immutable-state replay acceptance."""
import hashlib
import json
import sys
from pathlib import Path
from unittest.mock import patch

from gig.service.dataset_workspace import PersistedDatasetWorkspace
from gig.service.persisted_research import PersistedResearchWorkspace
from test_persisted_data import seed_persisted_fixture

base = Path(sys.argv[2])
output = Path(sys.argv[3])
workspace = PersistedResearchWorkspace(base)
if sys.argv[1] == "fit":
    dataset = PersistedDatasetWorkspace(base).build(seed_persisted_fixture(base)).to_dict()["data"]
    response = workspace.run({"dataset_id": dataset["dataset_id"], "model": "knn", "min_train": 30,
                              "test_size": 12, "embargo": 1, "seed": 17}).to_dict()
    data = response["data"]
    payload = {"dataset_id": dataset["dataset_id"], "run_id": data["run_id"]}
    predicted = workspace.predict(payload).to_dict()
    (output / "replay-request.json").write_text(json.dumps(payload))
    (output / "expected-response.json").write_text(json.dumps(predicted, sort_keys=True))
    report = {"source_commit": "012a7ab49cdbd1e57b780b1b9d51b569f7a8d00f",
              "classification": "SYNTHETIC_AUTHORED_FIXTURE_NOT_OBSERVED_MARKET_DATA",
              **payload, "model": "knn", "model_count": len(data["model_ids"]),
              "oos_rows": len(data["rows"]), "metrics": data["metrics"], "folds": data["folds"],
              "underperforming_folds": [f for f in data["folds"] if f["mse"] >= f["training_mean_baseline_mse"]],
              "usable_for_portfolio": data["usable_for_portfolio"], "economically_validated": False}
    (output / "research-result.json").write_text(json.dumps(report, indent=2) + "\n")
else:
    payload = json.loads((output / "replay-request.json").read_text())
    expected = json.loads((output / "expected-response.json").read_text())
    with patch("sklearn.neighbors.KNeighborsRegressor.fit", side_effect=AssertionError("no refit")), \
         patch("sklearn.neighbors.KNeighborsRegressor.kneighbors", side_effect=AssertionError("persisted JSON only")):
        actual = workspace.predict(payload).to_dict()
    assert expected == actual
    assert actual["state"] == "AVAILABLE"
    assert actual["data"]["metrics"]["training_performed"] is False
    import gig
    import gig._speed
    assert "/tmp/gig-knn-installed/" in gig.__file__
    assert "/tmp/gig-knn-installed/" in gig._speed.__file__
    digest = hashlib.sha256(json.dumps(actual, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    files = {str(p.relative_to(base)): hashlib.sha256(p.read_bytes()).hexdigest()
             for p in base.rglob("*.json")}
    report = {"state": "PASS", "fresh_installed_process": True, "no_refit": True,
              "response_sha256": digest, "run_id": payload["run_id"], "dataset_id": payload["dataset_id"],
              "package": gig.__file__, "native": gig._speed.__file__, "immutable_json_files": files,
              "runtime_backup_external": False, "study_classification": "SYNTHETIC"}
    (output / "process-replay.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "immutable_json_files"}, indent=2))
