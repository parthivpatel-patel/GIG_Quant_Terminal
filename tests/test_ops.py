"""Ops kill-switch and snapshot stay local and JSON-safe."""

import json

from gig.ops.killswitch import KillSwitch
from gig.ops.status import FRESH_BLOCK_DAYS, _freshness, ops_snapshot, touch_heartbeat
from datetime import date, timedelta


def test_kill_switch_roundtrip(tmp_path):
    ks = KillSwitch(path=tmp_path / "KILL_SWITCH.json")
    assert ks.engaged is False
    out = ks.halt(reason="unit test", by="pytest")
    assert out["engaged"] is True
    assert out["reason"] == "unit test"
    assert ks.engaged is True
    cleared = ks.resume(by="pytest")
    assert cleared["engaged"] is False
    assert ks.engaged is False


def test_freshness_levels():
    today = date.today()
    assert _freshness(None)["level"] == "bad"
    assert _freshness(today)["level"] == "ok"
    assert _freshness(today - timedelta(days=FRESH_BLOCK_DAYS))["level"] == "bad"


def test_ops_snapshot_json_safe(tmp_path, monkeypatch):
    from gig.config import get_settings

    monkeypatch.setenv("GIG_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("GIG_DB_PATH", str(tmp_path / "empty.duckdb"))
    get_settings.cache_clear()
    touch_heartbeat(ok=True, detail="test", run_id="abc")
    payload = ops_snapshot()
    assert payload["available"] is True
    assert "kill_switch" in payload
    assert "alerts" in payload
    json.dumps(payload)  # must not raise
    get_settings.cache_clear()
