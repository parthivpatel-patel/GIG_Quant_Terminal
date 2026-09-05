"""Typed configuration. Secrets come from the environment, never source."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None  # type: ignore[assignment]


def _load_env_file() -> None:
    """Put FRED_API_KEY / ALPACA_* from .env into os.environ (no GIG_ prefix required)."""
    if load_dotenv is None:
        return
    here = Path(__file__).resolve()
    for folder in (Path.cwd(), here.parents[2], here.parents[1]):
        candidate = folder / ".env"
        if candidate.exists():
            load_dotenv(candidate, override=False)
            break
    for key, val in list(os.environ.items()):
        if key.startswith("QALPHA_"):
            os.environ.setdefault("GIG_" + key[7:], val)


_load_env_file()


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="GIG_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    data_dir: Path = Field(default=Path("data"))
    results_dir: Path = Field(default=Path("results"))
    log_level: str = "INFO"
    seed: int = 42
    data_source: str = "yahoo"
    db_path: Path = Field(default=Path("data/gig.duckdb"))
    universe_file: Path = Field(default=Path("configs/universe.yaml"))

    max_gross_leverage: float = 2.0
    max_net_exposure: float = 0.10
    max_name_weight: float = 0.05
    max_sector_net: float = 0.10
    max_drawdown: float = 0.12
    var_confidence: float = 0.99

    # Risk-constrained construction. `target_vol` is annualized ex-ante vol
    # under the fitted factor model and binds only when the gross and name caps
    # leave room for it.
    use_optimizer: bool = True
    target_vol: float = 0.10
    risk_factors: int = 5
    risk_lookback: int = 252
    risk_refit_every: int = 21

    spread_bps: float = 4.0
    commission_bps: float = 1.0
    impact_eta: float = 0.10

    alpaca_api_key: str = ""
    alpaca_secret_key: str = ""
    alpaca_base_url: str = "https://paper-api.alpaca.markets"
    alpaca_data_url: str = "https://data.alpaca.markets"

    fred_api_key: str = ""
    finnhub_api_key: str = ""
    edgar_user_agent: str = "GIG Trading Algorithm (research; set EDGAR_USER_AGENT to your email)"

    # Local research desk (Ollama). Never used for order generation.
    ollama_host: str = "http://127.0.0.1:11434"
    ollama_model: str = "llama3.2"

    @model_validator(mode="after")
    def unprefixed_secrets(self) -> Settings:
        pairs = [
            ("alpaca_api_key", "ALPACA_API_KEY"),
            ("alpaca_secret_key", "ALPACA_SECRET_KEY"),
            ("alpaca_base_url", "ALPACA_BASE_URL"),
            ("alpaca_data_url", "ALPACA_DATA_URL"),
            ("fred_api_key", "FRED_API_KEY"),
            ("finnhub_api_key", "FINNHUB_API_KEY"),
            ("edgar_user_agent", "EDGAR_USER_AGENT"),
        ]
        for attr, env in pairs:
            current = getattr(self, attr)
            placeholder = isinstance(current, str) and (
                "set EDGAR_USER_AGENT" in current or current.startswith("YOUR_")
            )
            if not current or placeholder:
                env_val = os.environ.get(env, "")
                if env_val:
                    object.__setattr__(self, attr, env_val)
        if not Path(self.db_path).exists():
            legacy = Path("data/qalpha.duckdb")
            if legacy.exists():
                object.__setattr__(self, "db_path", legacy)
        return self

    def ensure_dirs(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        (self.data_dir / "cache").mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    def keys_status(self) -> dict[str, bool]:
        ua = self.edgar_user_agent
        edgar_ok = "@" in ua and "YOUR_EMAIL_HERE" not in ua and "set EDGAR_USER_AGENT" not in ua
        return {
            "fred": bool(self.fred_api_key),
            "finnhub": bool(self.finnhub_api_key),
            "alpaca": bool(self.alpaca_api_key) and bool(self.alpaca_secret_key),
            "edgar": edgar_ok,
        }


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
