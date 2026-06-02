"""
Attorney.AI — Centralized Configuration
Reads from environment variables / .env file.
"""
from functools import lru_cache
from typing import List

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── App ──────────────────────────────────────────────
    app_env: str = Field(default="development")
    app_host: str = Field(default="0.0.0.0")
    app_port: int = Field(default=8000)
    log_level: str = Field(default="INFO")
    cors_origins: List[str] = Field(default=["http://localhost:5173", "http://localhost:3000"])

    # ── OpenAI ───────────────────────────────────────────
    openai_api_key: str = Field(default="")
    openai_embedding_model: str = Field(default="text-embedding-3-large")
    openai_chat_model: str = Field(default="gpt-4o-mini")

    # ── Anthropic ─────────────────────────────────────────
    anthropic_api_key: str = Field(default="")

    # ── Qdrant ───────────────────────────────────────────
    qdrant_url: str = Field(default="http://localhost:6333")
    qdrant_api_key: str = Field(default="")
    qdrant_collection: str = Field(default="attorney_ai_legal")

    # ── Legal APIs ───────────────────────────────────────
    courtlistener_api_key: str = Field(default="")
    courtlistener_base_url: str = Field(default="https://www.courtlistener.com/api/rest/v4")
    govinfo_api_key: str = Field(default="")
    govinfo_base_url: str = Field(default="https://api.govinfo.gov")
    ecfr_base_url: str = Field(default="https://www.ecfr.gov/api/versioner/v1")
    federal_register_base_url: str = Field(default="https://www.federalregister.gov/api/v1")
    edgar_base_url: str = Field(default="https://efts.sec.gov/LATEST/search-index")

    # ── RAG Tuning ───────────────────────────────────────
    chunk_size: int = Field(default=512)
    chunk_overlap: int = Field(default=64)
    top_k_retrieve: int = Field(default=20)
    top_k_rerank: int = Field(default=5)
    reranker_model: str = Field(default="cross-encoder/ms-marco-MiniLM-L-12-v2")

    # ── Elasticsearch (BM25) ─────────────────────────────
    elasticsearch_url: str = Field(default="http://localhost:9200")
    elasticsearch_index: str = Field(default="attorney_ai_bm25")

    # ── Evaluation ───────────────────────────────────────
    legalbench_rag_path: str = Field(default="./data/legalbench_rag")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
