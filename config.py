from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Always resolve to the backend/.env regardless of cwd
_DEFAULT_ENV_FILE: Path = Path(__file__).resolve().parent / ".env"

# Project root: backend/ directory (all data dirs are co-located here)
_PROJECT_ROOT: Path = Path(__file__).resolve().parent

# Module-level variable so tests can monkey-patch it
ENV_FILE: Path = _DEFAULT_ENV_FILE


class Settings(BaseSettings):
    openrouter_api_key: str = "sk-placeholder"
    model: str = "gemini-2.0-flash"
    log_level: str = "INFO"  # DEBUG/INFO/WARNING/ERROR, overridable via LOG_LEVEL env var
    cmd_mode: Literal["bypass", "permission"] = "permission"
    workspace_dir: str = str(_PROJECT_ROOT / "workspace")
    logs_dir: str = str(_PROJECT_ROOT / "logs")
    rag_docs_dir: str = str(_PROJECT_ROOT / "rag_docs")
    skills_dir: str = str(_PROJECT_ROOT / "skills")
    # PGVector, routed through pgbouncer (CNPG). Two DSNs so ingestion (write)
    # and retrieval (read) can be pointed at the primary vs. read replica(s).
    db_write_url: str = "postgresql+psycopg://postgres:postgres@localhost:6432/ragdb"
    db_read_url: str = "postgresql+psycopg://postgres:postgres@localhost:6432/ragdb"
    rag_collection_name: str = "default"
    rag_top_k: int = 5
    rag_score_threshold: float = 1.0
    openai_api_key: str = ""
    google_api_key: str = ""
    conversations_dir: str = str(_PROJECT_ROOT / "conversations")
    enabled_tools: Optional[list[str]] = None  # None = all tools enabled, [] = none enabled
    history_max_turns: int = 3  # total turns before compaction triggers
    history_recent_turns: int = 1  # how many recent turns to keep verbatim (must be < max)

    @field_validator("workspace_dir", "logs_dir", "rag_docs_dir", "skills_dir", "conversations_dir", mode="before")
    @classmethod
    def _resolve_path(cls, v: str) -> str:
        """Resolve relative paths against the project root, not the process cwd."""
        p = Path(v)
        if not p.is_absolute():
            p = _PROJECT_ROOT / p
        return str(p.resolve())

    model_config = SettingsConfigDict(
        env_file=str(_DEFAULT_ENV_FILE),
        env_file_encoding="utf-8",
        extra="ignore",
        # Suppress the pydantic v2 protected-namespace warning for 'model'
        protected_namespaces=(),
    )


_settings: Settings | None = None


def get_settings() -> Settings:
    """Return the cached Settings singleton; create it on first call."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def reset_settings() -> None:
    """Clear the cached singleton so the next call re-reads the environment."""
    global _settings
    _settings = None


def update_env_file(key: str, value: str) -> None:
    """Persist a single KEY=value pair to the .env file.

    - If the key already exists its line is updated in-place.
    - If it is absent it is appended.
    - All other lines are preserved unchanged.
    """
    env_path = ENV_FILE
    lines: list[str] = []
    if env_path.exists():
        lines = env_path.read_text(encoding="utf-8").splitlines()

    found = False
    new_lines: list[str] = []
    for line in lines:
        stripped = line.lstrip()
        if stripped.startswith(f"{key}=") or stripped.startswith(f"{key} ="):
            new_lines.append(f"{key}={value}")
            found = True
        else:
            new_lines.append(line)

    if not found:
        new_lines.append(f"{key}={value}")

    env_path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
