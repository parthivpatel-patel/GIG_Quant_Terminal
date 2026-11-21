import hashlib
import json
import sqlite3
import sys
from pathlib import Path
from unittest.mock import patch

from gig.financial_intelligence.store import FinancialCorpus
from test_financial_nlp_profiles import GOLDEN_TEXT, T0, authored

output = Path(sys.argv[2])
if sys.argv[1] == "save":
    output.mkdir(exist_ok=False)
    corpus = FinancialCorpus(output/"source", clock=lambda: T0)
    corpus.admit_authored(authored(GOLDEN_TEXT))
    results = [corpus.search("Revenue", T0, extract_facts=True),
               corpus.search("Revenue", T0, extract_facts=True, nlp_profile="v2",
                             retrieval_mode="hybrid", latent_dimensions=2)]
    (output/"expected.json").write_text(json.dumps(results, sort_keys=True))
    restored = FinancialCorpus(output/"restored")
    restored.path.parent.mkdir()
    with sqlite3.connect(corpus.path) as source, sqlite3.connect(restored.path) as target:
        source.backup(target)
else:
    import gig
    assert "/tmp/gig-nlp-installed/" in gig.__file__
    expected = json.loads((output/"expected.json").read_text())
    corpus = FinancialCorpus(output/"restored")
    with patch("gig.financial_intelligence.latent.fit", side_effect=AssertionError("no refit")):
        actual = [corpus.replay(r["snapshot_id"]) for r in expected]
    assert actual == expected
    with sqlite3.connect(corpus.path) as db:
        assert db.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    report = {"state": "PASS", "separate_installed_process": True,
              "sqlite_backup_restored": True, "no_refit": True, "runtime_backup_external": False,
              "snapshots": [{"id": r["snapshot_id"], "profile": r.get("nlp_profile", "v1"),
                             "response_sha256": hashlib.sha256(json.dumps(r, sort_keys=True).encode()).hexdigest()}
                            for r in actual]}
    (output/"process-restore.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report, indent=2))
