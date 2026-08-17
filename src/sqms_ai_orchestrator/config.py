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
    llm_reasoning_effort: str = "auto"

    # O modelo do gateway raciocina antes de responder, e o raciocínio domina o
    # tempo: medido no gpt do gateway, _plan gastou 4.360 tokens de saída para
    # um JSON de 40 tokens (31,6s). Desligado, a mesma chamada leva 1,1s com o
    # mesmo JSON. Ligar de volta é só subir a env, sem deploy.
    llm_thinking_json: bool = False       # _plan / _select_evidence
    llm_thinking_answer: bool = False     # _answer
    llm_max_tokens_json: int = 512
    # The final answer must not be cut off by an artificial output cap. The
    # provider/context window remains the natural safety boundary.
    llm_max_tokens_answer: int | None = None

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
