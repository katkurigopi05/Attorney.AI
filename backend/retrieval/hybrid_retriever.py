"""
Attorney.AI — Hybrid Retriever (Supabase pgvector + PostgreSQL FTS)
Fuses vector similarity and full-text search using Reciprocal Rank Fusion (RRF).

Both search methods run against Supabase — no separate Qdrant or BM25 store needed.
  - Vector search: cosine similarity via pgvector (match_legal_chunks RPC)
  - FTS:           PostgreSQL tsvector + plainto_tsquery (search_legal_chunks_fts RPC)
  - Fusion:        Reciprocal Rank Fusion (RRF, k=60)
"""
import asyncio
from typing import Dict, List, Optional, Tuple

from loguru import logger

from ingestion.metadata_schema import LegalChunkMetadata
from retrieval.local_embedder import LocalEmbedder
from retrieval.supabase_store import SupabaseVectorStore

# RRF constant
_RRF_K = 60


class HybridRetriever:
    """
    Hybrid retrieval over Supabase (pgvector + PostgreSQL FTS).

    Both retrieval methods share the same Supabase table — zero extra infra.
    RRF fusion weights can be tuned (default: equal 50/50).
    """

    def __init__(
        self,
        embedder: Optional[LocalEmbedder] = None,
        store: Optional[SupabaseVectorStore] = None,
    ):
        self.embedder = embedder or _build_embedder()
        self.store = store or SupabaseVectorStore()

    async def retrieve(
        self,
        query: str,
        top_k: int = 20,
        filters: Optional[Dict] = None,
        vector_weight: float = 0.6,
        fts_weight: float = 0.4,
    ) -> List[Tuple[LegalChunkMetadata, float]]:
        """
        Hybrid retrieval: pgvector similarity + PostgreSQL full-text search, fused via RRF.

        Args:
            query:         Natural language legal query
            top_k:         Number of final results to return
            filters:       Dict with optional keys: jurisdiction, source_type, court_level
            vector_weight: RRF weight for vector results (default 0.6)
            fts_weight:    RRF weight for FTS results (default 0.4)

        Returns:
            List of (LegalChunkMetadata, rrf_score), sorted descending.
        """
        candidate_k = top_k * 3  # retrieve more candidates before fusion

        # Embed query for vector search (async, runs locally)
        query_vector = await self.embedder.embed_single(query)

        # Run vector search and FTS in parallel using asyncio
        vector_task = asyncio.get_event_loop().run_in_executor(
            None, self.store.vector_search, query_vector, candidate_k, filters
        )
        fts_task = asyncio.get_event_loop().run_in_executor(
            None, self.store.fts_search, query, candidate_k, filters
        )

        vector_results, fts_results = await asyncio.gather(vector_task, fts_task)

        logger.debug(
            f"Supabase hybrid: vector={len(vector_results)} hits, "
            f"fts={len(fts_results)} hits"
        )

        # Build RRF ranked lists (chunk_id, score)
        vector_ranked = [(c.chunk_id, s) for c, s in vector_results]
        fts_ranked    = [(c.chunk_id, s) for c, s in fts_results]

        fused_scores = _rrf_fusion(
            ranked_lists=[vector_ranked, fts_ranked],
            weights=[vector_weight, fts_weight],
            k=_RRF_K,
        )

        # Build chunk lookup
        chunk_lookup: Dict[str, LegalChunkMetadata] = {}
        for chunk, _ in vector_results + fts_results:
            chunk_lookup[chunk.chunk_id] = chunk

        results = []
        for chunk_id, rrf_score in fused_scores[:top_k]:
            if chunk_id in chunk_lookup:
                results.append((chunk_lookup[chunk_id], rrf_score))

        logger.debug(f"After RRF fusion: {len(results)} final candidates")
        return results


def _rrf_fusion(
    ranked_lists: List[List[Tuple[str, float]]],
    weights: List[float],
    k: int = 60,
) -> List[Tuple[str, float]]:
    """
    Reciprocal Rank Fusion.
    score(item) = Σ weight_i / (k + rank_i)
    """
    scores: Dict[str, float] = {}
    for ranked_list, weight in zip(ranked_lists, weights):
        for rank, (item_id, _) in enumerate(ranked_list, start=1):
            scores[item_id] = scores.get(item_id, 0.0) + weight / (k + rank)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def _build_embedder() -> LocalEmbedder:
    """Build the local embedder based on config."""
    from config import settings
    if settings.embedding_backend == "local":
        return LocalEmbedder(model_key=settings.local_embedding_model)
    # OpenAI embedder — wrap in a compatible async interface
    from retrieval.embedder import Embedder as OpenAIEmbedder

    class _AsyncWrapper:
        """Thin async wrapper around OpenAI embedder."""
        def __init__(self):
            self._inner = OpenAIEmbedder()
            self.dimension = self._inner.dimension

        async def embed_single(self, text: str):
            return await self._inner.embed_single(text)

        async def embed_batch(self, texts):
            return await self._inner.embed_batch(texts)

    return _AsyncWrapper()
