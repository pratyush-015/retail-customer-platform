"""Central application configuration.

Every tunable value the platform needs is declared here exactly once and read
from the environment, so the ETL scripts, the ML training pipeline and the
FastAPI app all agree on things like the churn threshold and where model
artifacts live.

Design notes
------------
* ``pydantic-settings`` gives us typed, validated config with a single source of
  truth (``.env``) plus real environment variables taking precedence.
* Paths are resolved to absolute paths against the repository root, so scripts
  behave the same whether invoked from the repo root, from ``backend/``, or
  from inside a container with a different working directory.
* ``DATABASE_URL`` is deliberately dialect-agnostic. Postgres is the production
  target, but the whole pipeline also runs on SQLite so a reviewer without
  Docker can still execute everything end to end.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# backend/app/core/config.py -> parents[3] == repository root
REPO_ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    """Typed application settings loaded from the environment / ``.env``."""

    model_config = SettingsConfigDict(
        env_file=(REPO_ROOT / ".env", REPO_ROOT / ".env.local"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ---------------------------------------------------------------- app ---
    app_name: str = "Retail Insights Platform"
    environment: str = "development"
    log_level: str = "INFO"
    api_v1_prefix: str = "/api"

    # SQLite default keeps the project runnable with zero infrastructure.
    # docker-compose and the deployment configs override this with Postgres.
    database_url: str = "sqlite:///./data/retail_insights.db"

    cors_origins: str = (
        "http://localhost:5173,http://localhost:3000,http://localhost:4173"
    )

    # ----------------------------------------------------------- database ---
    db_connect_max_retries: int = 6
    db_connect_base_delay: float = 0.5
    db_connect_max_delay: float = 16.0
    db_pool_size: int = 5
    db_max_overflow: int = 10
    db_echo: bool = False

    # ------------------------------------------------------ analytics / ml ---
    churn_threshold_days: int = Field(default=90, ge=1, le=3650)
    model_test_size: float = Field(default=0.25, gt=0.0, lt=1.0)
    random_seed: int = 42

    # --------------------------------------------------------------- paths ---
    data_dir: str = "data"
    raw_data_dir: str = "data/raw"
    model_artifact_dir: str = "backend/app/ml/artifacts"

    # ----------------------------------------------------------- validators ---
    @field_validator("log_level")
    @classmethod
    def _normalise_log_level(cls, value: str) -> str:
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        upper = value.strip().upper()
        if upper not in allowed:
            raise ValueError(
                f"log_level must be one of {sorted(allowed)}, got {value!r}"
            )
        return upper

    @field_validator("environment")
    @classmethod
    def _normalise_environment(cls, value: str) -> str:
        allowed = {"development", "staging", "production", "test"}
        lower = value.strip().lower()
        if lower not in allowed:
            raise ValueError(
                f"environment must be one of {sorted(allowed)}, got {value!r}"
            )
        return lower

    # ------------------------------------------------------ derived helpers ---
    @property
    def cors_origin_list(self) -> list[str]:
        """CORS origins as a clean list.

        ``"*"`` is honoured as a wildcard so a reviewer can open the dashboard
        from an arbitrary host, but the default is an explicit allow-list.
        """
        origins = [o.strip() for o in self.cors_origins.split(",") if o.strip()]
        return origins or ["http://localhost:5173"]

    @property
    def is_sqlite(self) -> bool:
        return self.database_url.startswith("sqlite")

    @property
    def is_production(self) -> bool:
        return self.environment == "production"

    def _resolve(self, raw: str) -> Path:
        """Resolve a configured path against the repo root when relative."""
        path = Path(raw)
        return path if path.is_absolute() else (REPO_ROOT / path)

    @property
    def data_path(self) -> Path:
        return self._resolve(self.data_dir)

    @property
    def raw_data_path(self) -> Path:
        return self._resolve(self.raw_data_dir)

    @property
    def artifact_path(self) -> Path:
        return self._resolve(self.model_artifact_dir)

    @property
    def model_file(self) -> Path:
        return self.artifact_path / "churn_model.joblib"

    @property
    def model_metadata_file(self) -> Path:
        return self.artifact_path / "model_metadata.json"

    @property
    def data_quality_report_file(self) -> Path:
        return self.data_path / "data_quality_report.json"

    @property
    def effective_database_url(self) -> str:
        """DB URL with relative SQLite paths anchored to the repo root.

        Without this, ``sqlite:///./data/retail_insights.db`` would create a
        different database file for every working directory the scripts happen
        to be launched from - a subtle and very annoying class of bug.
        """
        url = self.database_url
        prefix = "sqlite:///"
        if url.startswith(prefix):
            raw = url[len(prefix) :]
            if raw and not raw.startswith("/") and ":" not in raw[:3]:
                resolved = self._resolve(raw.lstrip("./"))
                resolved.parent.mkdir(parents=True, exist_ok=True)
                return f"{prefix}{resolved.as_posix()}"
        return url

    def ensure_directories(self) -> None:
        """Create the directories the pipeline writes into."""
        for path in (self.data_path, self.raw_data_path, self.artifact_path):
            path.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    """Return the cached settings singleton (FastAPI-dependency friendly)."""
    return Settings()


settings = get_settings()
