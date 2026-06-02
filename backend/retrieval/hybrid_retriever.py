"""
Attorney.AI — Hybrid Retriever
Fuses BM25 keyword search + Qdrant vector search using Reciprocal Rank Fusion (RRF).
"""
from typing import Dict, List, Optional, Tuple

from loguru import logger

from ingestion.metadata_schema import LegalChunkMetadata
from retrieval.bm25_store import BM25Store
from retrieval.embedder import Embedder
from retrieval.vector_store import VectorStore


# RRF constant (standard value, usually 60)
_RRF_K = 60


class HybridRetriever:
    """
    Hybrid retrieval combining:
    - BM25 keyword matching (good for exact legal terms, statute numbers)
    - Vector similarity (good for semantic / paraphrase matching)
    - Metadata filters (jurisdiction, court, source_type, date)
    
    Fusion via Reciprocal Rank Fusion (RRF).
    """

    def __init__(
        self,
        embedder: Optional[Embedder] = None,
        vector_store: Optional[VectorStore] = None,
        bm25_store: Optional[BM25Store] = None,
    ):
        self.embedder = embedder or Embedder()
        self.vector_store = vector_store or VectorStore()
        self.bm25_store = bm25_store or BM25Store()

    async def retrieve(
        self,
        query: str,
        top_k: int = 20,
        filters: Optional[Dict] = None,
        bm25_weight: float = 0.5,
        vector_weight: float = 0.5,
    ) -> List[Tuple[LegalChunkMetadata, float]]:
        """
        Retrieve top-k chunks using hybrid RRF fusion.
        
        Args:
            query: Natural language legal query
            top_k: Number of results to return
            filters: Metadata filters dict (jurisdiction, source_type, etc.)
            bm25_weight: Weight for BM25 score in fusion (default 0.5)
            vector_weight: Weight for vector score in fusion (default 0.5)
        
        Returns:
            List of (chunk, fused_score) sorted descending.
        """
        # Run BM25 and vector search in parallel
        bm25_results = self.bm25_store.search(query=query, top_k=top_k * 2, filters=filters)

        query_vector = await self.embedder.embed_single(query)
        vector_results = await self.vector_store.search(
            query_vector=query_vector, top_k=top_k * 2, filters=filters
        )

        logger.debug(
            f"Hybrid retrieve: BM25={len(bm25_results)} hits, "
            f"Vector={len(vector_results)} hits"
        )

        # Fuse with Reciprocal Rank Fusion
        fused = _rrf_fusion(
            ranked_lists=[
                [(c.chunk_id, s) for c, s in bm25_results],
                [(c.chunk_id, s) for c, s in vector_results],
            ],
            weights=[bm25_weight, vector_weight],
            k=_RRF_K,
        )

        # Build lookup for chunks by chunk_id
        chunk_lookup: Dict[str, LegalChunkMetadata] = {}
        for chunk, _ in bm25_results + vector_results:
            chunk_lookup[chunk.chunk_id] = chunk

        results = []
        for chunk_id, rrf_score in fused[:top_k]:
            if chunk_id in chunk_lookup:
                results.append((chunk_lookup[chunk_id], rrf_score))

        return results


def _rrf_fusion(
    ranked_lists: List[List[Tuple[str, float]]],
    weights: List[float],
    k: int = 60,
) -> List[Tuple[str, float]]:
    """
    Reciprocal Rank Fusion.
    
    For each item across all ranked lists, compute:
        RRF_score = sum_i( weight_i / (k + rank_i) )
    
    Items not appearing in a list get rank = infinity (no contribution).
    """
    scores: Dict[str, float] = {}

    for ranked_list, weight in zip(ranked_lists, weights):
        for rank, (item_id, _) in enumerate(ranked_list, start=1):
            scores[item_id] = scores.get(item_id, 0.0) + weight * (1.0 / (k + rank))

    sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_items
