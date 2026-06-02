"""
Attorney.AI — Centralized Configuration
Reads from environment variables / .env file.

Default mode: 100% local & free
  - Ollama for LLM generation
  - BGE-large for embeddings (local sentence-transformers)
  - Qdrant local (Docker) for vector search
  - rank-bm25 for keyword search
  - HuggingFace models for NER, NLI, summarization
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

    # ── Ollama (local, FREE) ──────────────────────────────
    ollama_base_url: str = Field(default="http://localhost:11434")
    ollama_model: str = Field(default="llama3.1")   # Change to mistral/phi3/qwen2.5 etc.

    # ── OpenAI (OPTIONAL — only used if key is present) ───
    openai_api_key: str = Field(default="")          # Leave blank to use Ollama
    openai_embedding_model: str = Field(default="text-embedding-3-large")
    openai_chat_model: str = Field(default="gpt-4o-mini")

    # ── Embedding Backend ─────────────────────────────────
    # "local"  → BGE-large-en-v1.5 (free, ~1.2GB download once)
    # "openai" → OpenAI text-embedding-3-large (paid, needs key)
    embedding_backend: str = Field(default="local")
    local_embedding_model: str = Field(default="bge-large")  # see local_embedder.py registry

    # ── Supabase (FREE tier, pgvector + FTS) ────────────────
    # Get these from: https://supabase.com → Project Settings → API
    supabase_url: str = Field(default="")            # https://xxxx.supabase.co
    supabase_service_key: str = Field(default="")    # service_role key (not anon)
    supabase_table: str = Field(default="legal_chunks")
    # Embedding dimension — must match your chosen model:
    #   BGE-large-en-v1.5  → 1024
    #   legal-bert          → 768
    #   MiniLM-L6-v2        → 384
    #   OpenAI 3-large      → 3072
    embedding_dim: int = Field(default=1024)

    # ── Legal APIs (all FREE, no key needed except CourtListener) ────────
    courtlistener_api_key: str = Field(default="")   # Optional; unauthenticated is rate-limited
    courtlistener_base_url: str = Field(default="https://www.courtlistener.com/api/rest/v4")
    govinfo_api_key: str = Field(default="")         # Optional; most endpoints are open
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

    # ── HuggingFace Transformer Models (all FREE) ─────────
    ner_model: str = Field(default="dslim/bert-base-NER")
    nli_model: str = Field(default="cross-encoder/nli-deberta-v3-base")
    summarizer_model: str = Field(default="nsi319/legal-led-base-16384")
    classifier_model: str = Field(default="facebook/bart-large-mnli")

    # ── Elasticsearch (optional BM25 alternative) ─────────
    elasticsearch_url: str = Field(default="http://localhost:9200")
    elasticsearch_index: str = Field(default="attorney_ai_bm25")

    # ── Evaluation ───────────────────────────────────────
    legalbench_rag_path: str = Field(default="./data/legalbench_rag")

    @property
    def using_ollama(self) -> bool:
        """True if Ollama is the active LLM backend."""
        return not (self.openai_api_key and self.openai_api_key.startswith("sk-"))

    @property
    def using_local_embeddings(self) -> bool:
        """True if local embeddings are active."""
        return self.embedding_backend == "local" or not self.using_ollama is False


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
