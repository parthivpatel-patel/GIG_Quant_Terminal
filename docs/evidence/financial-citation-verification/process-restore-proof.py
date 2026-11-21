"""Source-to-installed-process SQLite restoration, without fitting or writes."""
import hashlib
import json
import sqlite3
import sys
from pathlib import Path
from unittest.mock import patch

from gig.financial_intelligence.citations import verify_citation
from gig.financial_intelligence.store import FinancialCorpus
from gig.service.serialization import dumps

mode, output = sys.argv[1], Path(sys.argv[2])
if mode == "save":
    output.mkdir(exist_ok=False)
    corpus = FinancialCorpus(output/"source", clock=lambda: "2026-11-13T21:00:00.000000001Z")
    corpus.admit_authored({"source_id": "independent-citation-protocol", "title": "GIG-authored citation restoration protocol",
        "license": "GIG-AUTHORED", "text": "Café revenue USD 7 million. Original NaN and NYSE:BRK.B remain source mentions."})
    snapshot = corpus.search("revenue", "2026-11-13T21:00:00.000000001Z", retrieval_mode="hybrid", latent_dimensions=2,
                             extract_facts=True, nlp_profile="v2")
    payload = {"snapshot_id": snapshot["snapshot_id"], "passage_id": snapshot["rows"][0]["passage_id"]}
    proof = verify_citation(corpus, **payload)
    (output/"request.json").write_text(json.dumps(payload))
    (output/"expected.json").write_text(json.dumps(proof, ensure_ascii=False, sort_keys=True))
    restored = FinancialCorpus(output/"restored")
    restored.directory.mkdir()
    with sqlite3.connect(corpus.path) as source, sqlite3.connect(restored.path) as target:
        source.backup(target)
else:
    import gig
    import gig._speed
    assert "/tmp/gig-citations-installed/" in gig.__file__ and "/tmp/gig-citations-installed/" in gig._speed.__file__
    corpus = FinancialCorpus(output/"restored")
    payload = json.loads((output/"request.json").read_text())
    expected = json.loads((output/"expected.json").read_text())
    with sqlite3.connect(corpus.path) as db:
        before = {table: db.execute("SELECT COUNT(*) FROM "+table).fetchone()[0]
                  for table in ["documents", "passages", "latent_models", "snapshots"]}
    with patch("gig.financial_intelligence.latent.fit", side_effect=AssertionError("no fit")):
        actual = verify_citation(corpus, **payload)
    assert actual == expected
    with sqlite3.connect(corpus.path) as db:
        assert db.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        after = {table: db.execute("SELECT COUNT(*) FROM "+table).fetchone()[0] for table in before}
    assert before == after
    report = {"state": "PASS", "separate_installed_process": True, "sqlite_backup_restored": True,
              "no_refit": True, "unchanged_counts": after, "integrity": "ok",
              "verification_id": actual["verification_id"], "verifier_code_sha256": actual["verifier_code_sha256"],
              "response_sha256": hashlib.sha256(dumps(actual).encode()).hexdigest(),
              "hash_encoding": "gig.service.serialization.dumps UTF-8", "runtime_backup_external": False,
              "package_import": gig.__file__, "native_import": gig._speed.__file__}
    (output/"process-restore.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report, indent=2))
