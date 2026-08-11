from functools import lru_cache
import os
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


PROJECT_ROOT = Path(os.environ.get("SQMS_AI_PROJECT_ROOT", Path.cwd())).resolve()


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="SQMS_AI_",
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "SQMS AI Orchestrator"
    host: str = "0.0.0.0"
    port: int = 8200
    default_flow: str = "procurement"
    flows_directory: Path = PROJECT_ROOT / "flows"
    data_directory: Path = PROJECT_ROOT / "data"

    llm_base_url: str = "http://10.64.10.48/v1"
    llm_api_key: str = "EMPTY"
    llm_model: str = "auto"
    llm_context_tokens: int = 4096
    llm_timeout_seconds: float = 120.0

    semantic_enabled: bool = False
    embedding_model: str = "BAAI/bge-m3"
    rerank_enabled: bool = False
    rerank_model: str = "BAAI/bge-reranker-v2-m3"
    debug_responses: bool = False

    max_search_iterations: int = 2
    lexical_candidates: int = 15
    semantic_candidates: int = 15
    final_candidates: int = 8
    max_evidence_tokens: int = 2100
    max_history_messages: int = 8


@lru_cache
def get_settings() -> Settings:
    return Settings()
