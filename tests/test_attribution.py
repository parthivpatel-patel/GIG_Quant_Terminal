"""Attribution helpers stay finite and JSON-serializable."""

import json

from gig.research.attribution import attribution_snapshot, risk_attribution, signal_attribution


def test_signal_attribution_shares_sum_to_one():
    rows = signal_attribution(
        {
            "momentum_12_1": {"ic_mean": 0.02},
            "ivol_63d": {"ic_mean": -0.04},
            "strev_21d": {"ic_mean": 0.01},
        }
    )
    assert abs(sum(r["share"] for r in rows) - 1.0) < 1e-9
    assert rows[0]["name"] == "ivol_63d"


def test_risk_attribution_empty():
    assert risk_attribution(None)["available"] is False
    assert risk_attribution({"available": False})["available"] is False


def test_attribution_snapshot_serializable(tmp_path, monkeypatch):
    from gig.config import get_settings
    from gig.service import snapshot

    monkeypatch.setenv("GIG_DB_PATH", str(tmp_path / "a.duckdb"))
    monkeypatch.setenv("GIG_RESULTS_DIR", str(tmp_path / "results"))
    monkeypatch.setenv("GIG_DATA_DIR", str(tmp_path / "data"))
    get_settings.cache_clear()
    snapshot.clear_cache()
    payload = attribution_snapshot(
        risk={
            "available": True,
            "ex_ante_vol": 0.1,
            "realized_vol": 0.2,
            "systematic_share": 0.3,
            "specific_share": 0.7,
            "factor_exposure": {"pc1": 0.01},
            "eigenvalue_share": [0.5, 0.3, 0.2],
            "top_contributors": [],
            "equation": "Σ = BB′ + D",
            "shrinkage": 0.15,
        },
        factor_ic={"momentum_12_1": {"ic_mean": 0.02, "ic_tstat_nw": 2.1}},
    )
    json.dumps(payload)
    assert payload["available"] is True
    get_settings.cache_clear()
