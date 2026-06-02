"""
Attorney.AI — Cross-Encoder Reranker
Uses a lightweight cross-encoder to rerank retrieved chunks by relevance to the query.
Model: cross-encoder/ms-marco-MiniLM-L-12-v2 (free, fast, strong on legal text)
"""
from typing import List, Tuple

from loguru import logger
from sentence_transformers import CrossEncoder

from config import settings
from ingestion.metadata_schema import LegalChunkMetadata


class Reranker:
    """
    Cross-encoder reranker. Takes a query + list of candidate chunks,
    returns the top-k reranked by cross-encoder score.
    
    The cross-encoder jointly encodes (query, passage) — much better
    than cosine similarity for precise legal relevance judgments.
    """

    def __init__(self):
        logger.info(f"Loading reranker: {settings.reranker_model}")
        self.model = CrossEncoder(settings.reranker_model, max_length=512)

    def rerank(
        self,
        query: str,
        candidates: List[Tuple[LegalChunkMetadata, float]],
        top_k: int = 5,
    ) -> List[Tuple[LegalChunkMetadata, float]]:
        """
        Rerank candidates using cross-encoder scores.
        
        Args:
            query: The legal research question
            candidates: List of (chunk, retrieval_score) from hybrid retriever
            top_k: How many top results to return
        
        Returns:
            Top-k reranked list of (chunk, cross_encoder_score).
        """
        if not candidates:
            return []

        # Build (query, chunk_text) pairs for cross-encoder
        pairs = [(query, c.text[:500]) for c, _ in candidates]

        try:
            scores = self.model.predict(pairs, show_progress_bar=False)
        except Exception as e:
            logger.error(f"Reranker failed: {e} — returning original order")
            return candidates[:top_k]

        # Zip with original chunks and sort by cross-encoder score
        scored = list(zip([c for c, _ in candidates], scores.tolist()))
        scored.sort(key=lambda x: x[1], reverse=True)

        logger.debug(
            f"Reranked {len(candidates)} → top {top_k}: "
            f"scores {[f'{s:.3f}' for _, s in scored[:top_k]]}"
        )
        return scored[:top_k]
