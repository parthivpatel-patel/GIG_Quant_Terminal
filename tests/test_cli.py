from gig.cli import main


def test_doctor_exit_zero():
    assert main(["doctor"]) == 0


def test_backtest_cli_no_persist():
    assert main(["backtest", "--source", "synthetic", "--no-persist", "--seed", "42", "--no-news"]) == 0


def test_db_init(tmp_path, monkeypatch):
    monkeypatch.setenv("GIG_DB_PATH", str(tmp_path / "t.duckdb"))
    from gig.config import get_settings

    get_settings.cache_clear()
    assert main(["db", "init"]) == 0
    get_settings.cache_clear()

