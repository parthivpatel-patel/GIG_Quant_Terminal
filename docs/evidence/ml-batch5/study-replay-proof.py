"""Authored persisted Bayesian study and independent installed-process replay."""
import hashlib
import json
import sys
import time
from pathlib import Path
from unittest.mock import patch

from gig.service.dataset_workspace import PersistedDatasetWorkspace
from gig.service.persisted_research import PersistedResearchWorkspace
from test_persisted_data import seed_persisted_fixture

mode, output = sys.argv[1], Path(sys.argv[2])
base = output/"state"
workspace = PersistedResearchWorkspace(base)
if mode == "fit":
    output.mkdir(exist_ok=False)
    dataset = PersistedDatasetWorkspace(base).build(seed_persisted_fixture(base)).to_dict()["data"]
    started, cpu = time.perf_counter(), time.process_time()
    data = workspace.run({"dataset_id": dataset["dataset_id"], "model": "bayesian_ridge", "min_train": 30,
                          "test_size": 12, "embargo": 1, "seed": 17}).to_dict()["data"]
    elapsed, cpu = time.perf_counter()-started, time.process_time()-cpu
    payload = {"dataset_id": dataset["dataset_id"], "run_id": data["run_id"]}
    expected = workspace.predict(payload).to_dict()
    (output/"request.json").write_text(json.dumps(payload))
    (output/"expected.json").write_text(json.dumps(expected, sort_keys=True))
    folds = data["folds"]
    count = sum(f["test_count"] for f in folds)
    average = lambda key: sum(f[key]*f["test_count"] for f in folds)/count
    report = {"source_commit": "37e3550cf166462b4553d4449708a86b5493277b",
              "classification": "AUTHORED_SYNTHETIC_NOT_OBSERVED_MARKET_DATA", **payload,
              "model_count": len(data["model_ids"]), "oos_rows": len(data["rows"]),
              "training_wall_seconds": elapsed, "training_process_cpu_seconds": cpu,
              "metrics": data["metrics"], "training_mean_baseline_mse": average("training_mean_baseline_mse"),
              "gaussian_nll": average("gaussian_nll"),
              "training_gaussian_baseline_nll": average("training_gaussian_baseline_nll"),
              "nominal95_interval_coverage": average("nominal95_interval_coverage"),
              "underperforming_mse_folds": [f for f in folds if f["mse"] >= f["training_mean_baseline_mse"]],
              "folds": folds, "economically_validated": False, "interval_calibrated": False,
              "usable_for_portfolio": data["usable_for_portfolio"]}
    (output/"research-result.json").write_text(json.dumps(report, indent=2)+"\n")
else:
    import gig
    import gig._speed
    assert "/tmp/gig-bayes-installed/" in gig.__file__ and "/tmp/gig-bayes-installed/" in gig._speed.__file__
    payload = json.loads((output/"request.json").read_text())
    expected = json.loads((output/"expected.json").read_text())
    with patch("sklearn.linear_model.BayesianRidge.fit", side_effect=AssertionError("saved replay cannot fit")):
        actual = workspace.predict(payload).to_dict()
    assert actual == expected and actual["state"] == "AVAILABLE"
    assert actual["data"]["usable_for_portfolio"] is False
    assert actual["data"]["metrics"]["training_performed"] is False
    report = {"state": "PASS", **payload, "separate_installed_process": True, "no_refit": True,
              "response_sha256": hashlib.sha256(json.dumps(actual, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
              "runtime_backup_external": False, "package_import": gig.__file__, "native_import": gig._speed.__file__}
    (output/"process-replay.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report, indent=2))
