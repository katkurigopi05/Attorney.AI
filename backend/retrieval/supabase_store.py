"""
Attorney.AI — Supabase Vector Store (pgvector + PostgreSQL FTS)
Replaces Qdrant. Uses Supabase free tier — no Docker, no local setup needed.

Setup:
  1. Create a project at https://supabase.com (free)
  2. Run supabase_setup.sql in the SQL Editor
  3. Set SUPABASE_URL and SUPABASE_SERVICE_KEY in .env
"""
import uuid
from typing import Dict, List, Optional, Tuple

from loguru import logger
from supabase import create_client, Client

from config import settings
from ingestion.metadata_schema import LegalChunkMetadata


def _get_client() -> Client:
    """Create and return Supabase client."""
    if not settings.supabase_url or not settings.supabase_service_key:
        raise RuntimeError(
            "Supabase credentials not set.\n"
            "Add SUPABASE_URL and SUPABASE_SERVICE_KEY to your .env file.\n"
            "Get them from: https://supabase.com → Project Settings → API"
        )
    return create_client(settings.supabase_url, settings.supabase_service_key)


class SupabaseVectorStore:
    """
    Supabase pgvector store for legal chunks.

    Uses two SQL functions defined in supabase_setup.sql:
      - match_legal_chunks()  → cosine similarity vector search
      - search_legal_chunks_fts() → PostgreSQL full-text search

    Free tier limits: 500MB database, 2GB bandwidth/month.
    Supports ~500K-1M legal chunks on free tier.
    """

    def __init__(self):
        self.client = _get_client()
        self.table = settings.supabase_table
        logger.info(f"Supabase vector store: {settings.supabase_url} → {self.table}")

    def upsert(
        self,
        chunks: List[LegalChunkMetadata],
        embeddings: List[List[float]],
        batch_size: int = 50,
    ) -> int:
        """
        Upsert chunks with their embeddings into Supabase.
        Uses ON CONFLICT (chunk_id) DO UPDATE to be idempotent.
        Returns number of rows upserted.
        """
        total = 0
        for i in range(0, len(chunks), batch_size):
            batch_chunks = chunks[i: i + batch_size]
            batch_embeds = embeddings[i: i + batch_size]

            rows = []
            for chunk, embedding in zip(batch_chunks, batch_embeds):
                row = chunk.to_qdrant_payload()  # reuse existing serializer
                row["embedding"] = embedding
                # Remove computed/extra fields not in DB schema
                row.pop("chunk_id", None)
                row["chunk_id"] = chunk.chunk_id
                rows.append(row)

            try:
                self.client.table(self.table).upsert(
                    rows, on_conflict="chunk_id"
                ).execute()
                total += len(rows)
                logger.debug(f"Upserted batch {i // batch_size + 1}: {len(rows)} rows")
            except Exception as e:
                logger.error(f"Supabase upsert error (batch {i // batch_size + 1}): {e}")

        return total

    def vector_search(
        self,
        query_vector: List[float],
        top_k: int = 20,
        filters: Optional[Dict] = None,
    ) -> List[Tuple[LegalChunkMetadata, float]]:
        """
        Cosine similarity search via the match_legal_chunks() SQL function.
        Returns list of (chunk, similarity_score).
        """
        filters = filters or {}
        params = {
            "query_embedding": query_vector,
            "match_count": top_k,
            "filter_jurisdiction": filters.get("jurisdiction"),
            "filter_source_type": filters.get("source_type"),
            "filter_court_level": filters.get("court_level"),
        }

        try:
            resp = self.client.rpc("match_legal_chunks", params).execute()
            return _parse_vector_results(resp.data or [])
        except Exception as e:
            logger.error(f"Supabase vector search error: {e}")
            return []

    def fts_search(
        self,
        query_text: str,
        top_k: int = 20,
        filters: Optional[Dict] = None,
    ) -> List[Tuple[LegalChunkMetadata, float]]:
        """
        PostgreSQL full-text search via search_legal_chunks_fts() SQL function.
        Returns list of (chunk, ts_rank score).
        """
        filters = filters or {}
        params = {
            "query_text": query_text,
            "match_count": top_k,
            "filter_jurisdiction": filters.get("jurisdiction"),
            "filter_source_type": filters.get("source_type"),
        }

        try:
            resp = self.client.rpc("search_legal_chunks_fts", params).execute()
            return _parse_fts_results(resp.data or [])
        except Exception as e:
            logger.error(f"Supabase FTS error: {e}")
            return []

    def get_stats(self) -> dict:
        """Return collection statistics."""
        try:
            resp = (
                self.client.table(self.table)
                .select("id", count="exact")
                .execute()
            )
            count = resp.count or 0
            return {
                "table": self.table,
                "total_chunks": count,
                "supabase_url": settings.supabase_url,
            }
        except Exception as e:
            return {"error": str(e)}


# ── Result parsers ────────────────────────────────────────────────────────────

def _parse_vector_results(
    rows: List[dict],
) -> List[Tuple[LegalChunkMetadata, float]]:
    """Convert Supabase RPC rows → (LegalChunkMetadata, score) tuples."""
    results = []
    for row in rows:
        try:
            score = float(row.pop("similarity", 0.0))
            row.pop("id", None)
            # Reconstruct LegalChunkMetadata
            chunk = LegalChunkMetadata(
                doc_id=row.get("doc_id", ""),
                chunk_id=row.get("chunk_id", ""),
                title=row.get("title", ""),
                citation=row.get("citation", ""),
                source_url=row.get("source_url", ""),
                jurisdiction=row.get("jurisdiction", "US-Federal"),
                court_or_agency=row.get("court_or_agency"),
                court_level=row.get("court_level"),
                date_str=row.get("date_str"),
                decision_date=_parse_date(row.get("decision_date")),
                source_type=row.get("source_type", "case"),
                parent_section=row.get("parent_section"),
                start_char=row.get("start_char", 0),
                end_char=row.get("end_char", 0),
                text=row.get("text", ""),
                docket_number=row.get("docket_number"),
                author_judge=row.get("author_judge"),
                practice_area=row.get("practice_area"),
            )
            results.append((chunk, score))
        except Exception as e:
            logger.warning(f"Could not parse vector result row: {e}")
    return results


def _parse_fts_results(
    rows: List[dict],
) -> List[Tuple[LegalChunkMetadata, float]]:
    """Convert FTS RPC rows → (LegalChunkMetadata, score) tuples."""
    results = []
    for row in rows:
        try:
            score = float(row.pop("rank", 0.0))
            chunk = LegalChunkMetadata(
                doc_id=row.get("doc_id", ""),
                chunk_id=row.get("chunk_id", ""),
                title=row.get("title", ""),
                citation=row.get("citation", ""),
                source_url=row.get("source_url", ""),
                jurisdiction=row.get("jurisdiction", "US-Federal"),
                court_or_agency=row.get("court_or_agency"),
                court_level=row.get("court_level"),
                date_str=row.get("date_str"),
                source_type=row.get("source_type", "case"),
                parent_section=row.get("parent_section"),
                start_char=row.get("start_char", 0),
                end_char=row.get("end_char", 0),
                text=row.get("text", ""),
                docket_number=row.get("docket_number"),
                author_judge=row.get("author_judge"),
                practice_area=row.get("practice_area"),
            )
            results.append((chunk, score))
        except Exception as e:
            logger.warning(f"Could not parse FTS result row: {e}")
    return results


def _parse_date(date_str):
    if not date_str:
        return None
    try:
        from datetime import date
        return date.fromisoformat(str(date_str))
    except Exception:
        return None
