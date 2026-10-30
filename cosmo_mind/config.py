"""
cosmo_mind.config
=================

Central configuration for CosmoMind.

All runtime settings are loaded from a single ``.env`` file via
``pydantic-settings``.  Nothing here is ever hardcoded as a secret; secrets
live exclusively in ``.env`` (which is git-ignored).

Two operating modes are supported:

``LOCAL``  -> TF-IDF embeddings, in-memory vector store (Qdrant `:memory:`),
               NetworkX knowledge graph, deterministic LLM stub. No keys /
               external services required.
``PROD``   -> OpenAI embeddings+chat, remote Qdrant, Neo4j graph, and live
               astroquery calls (SIMBAD / VizieR / NASA Exoplanet Archive).
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class CosmoSettings(BaseSettings):
    """Strongly-typed application settings sourced from ``.env``."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        env_prefix="",
        case_sensitive=False,
    )

    # run mode
    cosmo_mode: str = Field(
        default="LOCAL",
        description="LOCAL = offline fallbacks; PROD = real cloud backends.",
    )

    # LLM backend (OpenAI chat completions with function-calling)
    openai_api_key: str | None = Field(default=None, validation_alias="OPENAI_API_KEY")
    openai_model: str = Field(default="gpt-4o", validation_alias="OPENAI_MODEL")
    openai_max_tokens: int = Field(default=4000, validation_alias="OPENAI_MAX_TOKENS")
    openai_temperature: float = Field(
        default=0.2, validation_alias="OPENAI_TEMPERATURE"
    )
    openai_embedding_model: str = Field(
        default="text-embedding-3-small",
        validation_alias="OPENAI_EMBEDDING_MODEL",
    )

    # Optional Anthropic backend (used if OPENAI_API_KEY is empty)
    anthropic_api_key: str | None = Field(
        default=None, validation_alias="ANTHROPIC_API_KEY"
    )
    anthropic_model: str = Field(
        default="claude-3-5-sonnet-20241022",
        validation_alias="ANTHROPIC_MODEL",
    )

    # Vector database (Qdrant)
    qdrant_url: str = Field(
        default="http://localhost:6333", validation_alias="QDRANT_URL"
    )
    qdrant_api_key: str | None = Field(
        default=None, validation_alias="QDRANT_API_KEY"
    )
    qdrant_prefer_grpc: bool = Field(
        default=False, validation_alias="QDRANT_PREFER_GRPC"
    )

    # Knowledge graph (Neo4j)
    neo4j_uri: str = Field(
        default="bolt://localhost:7687", validation_alias="NEO4J_URI"
    )
    neo4j_user: str = Field(default="neo4j", validation_alias="NEO4J_USER")
    neo4j_password: str = Field(
        default="neo4j", validation_alias="NEO4J_PASSWORD"
    )

    # Astronomy APIs
    nasa_api_key: str | None = None

    # Paths (data dirs are git-ignored)
    cosmo_data_dir: Path = Field(
        default=Path("./data"), validation_alias="COSMO_DATA_DIR"
    )
    cosmo_vector_cache: Path = Field(
        default=Path("./data/vector_cache"),
        validation_alias="COSMO_VECTOR_CACHE",
    )
    cosmo_kg_cache: Path = Field(
        default=Path("./data/kg_cache"),
        validation_alias="COSMO_KG_CACHE",
    )

    # Sandbox / code execution
    cosmo_enable_code_sandbox: bool = Field(
        default=False, validation_alias="COSMO_ENABLE_CODE_SANDBOX"
    )

    use_openai: bool = True
    use_anthropic: bool = True

    # validators / derived state
    @field_validator("cosmo_mode")
    @classmethod
    def _validate_mode(cls, v: str) -> str:
        v = v.upper()
        if v not in ("LOCAL", "PROD"):
            raise ValueError("COSMO_MODE must be 'LOCAL' or 'PROD'")
        return v

    @field_validator(
        "openai_api_key", "anthropic_api_key", "qdrant_api_key", mode="before"
    )
    @classmethod
    def _blank_to_none(cls, v: Any) -> str | None:
        if isinstance(v, str) and v.strip() == "":
            return None
        return v

    def _resolve_dependencies(self) -> None:
        """Compute derived boolean flags after env load."""
        self.use_openai = bool(self.openai_api_key)
        self.use_anthropic = bool(self.anthropic_api_key)
        if self.cosmo_mode == "LOCAL":
            # In LOCAL mode we deliberately bypass remote services unless a
            # key was genuinely supplied.
            self.use_openai = self.use_openai and self.openai_api_key is not None
        self.cosmo_data_dir.mkdir(parents=True, exist_ok=True)
        self.cosmo_vector_cache.mkdir(parents=True, exist_ok=True)
        self.cosmo_kg_cache.mkdir(parents=True, exist_ok=True)

    @property
    def is_local(self) -> bool:
        return self.cosmo_mode == "LOCAL"

    @property
    def is_prod(self) -> bool:
        return self.cosmo_mode == "PROD"

    def summary(self) -> dict[str, Any]:
        """Non-sensitive runtime summary (used by the ``--check`` CLI)."""
        return {
            "mode": self.cosmo_mode,
            "use_openai": self.use_openai,
            "use_anthropic": self.use_anthropic,
            "openai_model": self.openai_model,
            "openai_embedding_model": self.openai_embedding_model,
            "qdrant_url": self.qdrant_url,
            "neo4j_uri": self.neo4j_uri,
            "nasa_api_key_set": bool(self.nasa_api_key),
            "cosmo_data_dir": str(self.cosmo_data_dir),
            "enable_code_sandbox": self.cosmo_enable_code_sandbox,
        }


@lru_cache(maxsize=1)
def get_settings() -> CosmoSettings:
    """Return the singleton settings instance (auto-created on first call)."""
    settings = CosmoSettings()
    settings._resolve_dependencies()
    return settings


def reload_settings() -> CosmoSettings:
    """Force a re-read of ``.env`` (used by tests and hot-reloading)."""
    get_settings.cache_clear()
    return get_settings()


__all__ = ["CosmoSettings", "get_settings", "reload_settings"]
