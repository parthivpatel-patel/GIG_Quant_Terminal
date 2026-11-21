"""Actual authored directional study and separate installed-process JSON replay."""
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
workspace = PersistedResearchWorkspace(output/"state")
if mode == "fit":
    output.mkdir(exist_ok=False)
    dataset = PersistedDatasetWorkspace(output/"state").build(seed_persisted_fixture(output/"state")).to_dict()["data"]
    start, cpu = time.perf_counter(), time.process_time()
    data = workspace.run({"dataset_id": dataset["dataset_id"], "model": "binomial_glm", "min_train": 30,
                          "test_size": 12, "embargo": 1, "seed": 17}).to_dict()["data"]
    elapsed, cpu = time.perf_counter()-start, time.process_time()-cpu
    payload = {"dataset_id": dataset["dataset_id"], "run_id": data["run_id"]}
    expected = workspace.predict(payload).to_dict()
    (output/"request.json").write_text(json.dumps(payload))
    (output/"expected.json").write_text(json.dumps(expected, sort_keys=True))
    folds = data["folds"]
    count = sum(f["test_count"] for f in folds)
    average = lambda key: sum(f[key]*f["test_count"] for f in folds)/count
    report = {"source_commit": "5322a17c796487e647d0b42ffdf618d1b56439ea",
              "classification": "AUTHORED_SYNTHETIC_NOT_OBSERVED_MARKET_DATA", **payload,
              "model_count": len(data["model_ids"]), "oos_rows": len(data["rows"]),
              "training_wall_seconds": elapsed, "training_process_cpu_seconds": cpu,
              "metrics": data["metrics"], "training_rate_baseline_brier": average("training_rate_baseline_brier"),
              "training_rate_baseline_log_loss": average("training_rate_baseline_log_loss"),
              "underperforming_brier_folds": [f for f in folds if f["brier_score"] >= f["training_rate_baseline_brier"]],
              "underperforming_log_loss_folds": [f for f in folds if f["log_loss"] >= f["training_rate_baseline_log_loss"]],
              "folds": folds, "probability_calibrated": False, "economically_validated": False,
              "usable_for_portfolio": data["usable_for_portfolio"]}
    (output/"research-result.json").write_text(json.dumps(report, indent=2)+"\n")
else:
    import gig
    import gig._speed
    assert "/tmp/gig-glm-installed/" in gig.__file__ and "/tmp/gig-glm-installed/" in gig._speed.__file__
    payload = json.loads((output/"request.json").read_text())
    expected = json.loads((output/"expected.json").read_text())
    with patch("sklearn.linear_model.LogisticRegression.fit", side_effect=AssertionError("no fitting")):
        actual = workspace.predict(payload).to_dict()
    assert actual == expected and actual["state"] == "AVAILABLE"
    assert actual["data"]["metrics"]["training_performed"] is False
    assert actual["data"]["usable_for_portfolio"] is False
    assert actual["data"]["prediction_kind"] == "FORWARD_DIRECTION_PROBABILITY"
    report = {"state": "PASS", **payload, "separate_installed_process": True, "no_refit": True,
              "response_sha256": hashlib.sha256(json.dumps(actual, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
              "runtime_backup_external": False, "package_import": gig.__file__, "native_import": gig._speed.__file__}
    (output/"process-replay.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report, indent=2))
