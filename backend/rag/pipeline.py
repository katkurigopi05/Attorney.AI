"""
Attorney.AI — Full RAG Pipeline Orchestrator
Coordinates: classify → rewrite → retrieve → rerank → generate → verify
"""
import asyncio
from typing import Optional

from loguru import logger

from config import settings
from rag.classifier import QueryClassifier
from rag.generator import AnswerGenerator
from rag.query_rewriter import QueryRewriter
from rag.verifier import CitationVerifier
from retrieval.bm25_store import BM25Store
from retrieval.embedder import Embedder
from retrieval.hybrid_retriever import HybridRetriever
from retrieval.reranker import Reranker
from retrieval.vector_store import VectorStore


class RAGPipeline:
    """
    Full Attorney.AI RAG pipeline:
    
    user_query
       ↓
    QueryClassifier      → jurisdiction, task_type, court_level
       ↓
    QueryRewriter        → expanded queries
       ↓
    HybridRetriever      → BM25 + vector + metadata filters (RRF fusion)
       ↓
    Reranker             → cross-encoder reranking
       ↓
    AnswerGenerator      → citation-grounded answer
       ↓
    CitationVerifier     → PASS / FLAG / REJECT
       ↓
    Response             → answer + citations + verdict
    """

    def __init__(self):
        self.classifier = QueryClassifier()
        self.rewriter = QueryRewriter()
        self.embedder = Embedder()
        self.vector_store = VectorStore()
        self.bm25_store = BM25Store()
        self.retriever = HybridRetriever(
            embedder=self.embedder,
            vector_store=self.vector_store,
            bm25_store=self.bm25_store,
        )
        self.reranker = Reranker()
        self.generator = AnswerGenerator()
        self.verifier = CitationVerifier()

        # Try to load persisted BM25 index
        self.bm25_store.load()

    async def query(
        self,
        user_query: str,
        jurisdiction_override: Optional[str] = None,
        source_type_override: Optional[str] = None,
        top_k_retrieve: Optional[int] = None,
        top_k_rerank: Optional[int] = None,
        skip_verify: bool = False,
    ) -> dict:
        """
        Run the full RAG pipeline for a legal research query.
        
        Args:
            user_query: Natural language legal question
            jurisdiction_override: Force a specific jurisdiction filter
            source_type_override: Force a specific source_type filter
            top_k_retrieve: Override default retrieval count
            top_k_rerank: Override default rerank count
            skip_verify: Skip citation verification (faster, less safe)
        
        Returns:
            Full response dict with answer, citations, metadata, verdict.
        """
        top_k_retrieve = top_k_retrieve or settings.top_k_retrieve
        top_k_rerank = top_k_rerank or settings.top_k_rerank

        logger.info(f"RAG pipeline: query='{user_query[:80]}...'")

        # ── Step 1: Classify ──────────────────────────────────────────────
        classification = await self.classifier.classify(user_query)
        filters = self.classifier.classification_to_filters(classification)

        # Apply overrides
        if jurisdiction_override:
            filters["jurisdiction"] = jurisdiction_override
        if source_type_override:
            filters["source_type"] = source_type_override

        logger.debug(f"Classification: {classification}")
        logger.debug(f"Filters: {filters}")

        # ── Step 2: Rewrite query ─────────────────────────────────────────
        rewrite = await self.rewriter.rewrite(user_query, classification)
        primary_query = rewrite.get("primary_query", user_query)

        # ── Step 3: Hybrid retrieval ──────────────────────────────────────
        candidates = await self.retriever.retrieve(
            query=primary_query,
            top_k=top_k_retrieve,
            filters=filters if any(filters.values()) else None,
        )

        # If primary retrieval returns too few results, try without filters
        if len(candidates) < 3 and filters:
            logger.info("Few results with filters — retrying without filters")
            candidates = await self.retriever.retrieve(
                query=primary_query,
                top_k=top_k_retrieve,
                filters=None,
            )

        logger.debug(f"Retrieved {len(candidates)} candidates before reranking")

        # ── Step 4: Rerank ────────────────────────────────────────────────
        reranked = self.reranker.rerank(
            query=primary_query,
            candidates=candidates,
            top_k=top_k_rerank,
        )

        # ── Step 5: Generate answer ───────────────────────────────────────
        generation = await self.generator.generate(
            query=user_query,
            chunks=reranked,
        )

        # ── Step 6: Verify citations ──────────────────────────────────────
        if skip_verify or not generation["has_answer"]:
            verification = {
                "verdict": "FLAG" if not generation["has_answer"] else "PASS",
                "overall_faithfulness": "HIGH",
                "verified_answer": generation["answer"],
                "unsupported_count": 0,
            }
        else:
            verification = await self.verifier.verify(
                answer=generation["answer"],
                chunks=reranked,
            )

        # ── Compose final response ────────────────────────────────────────
        return {
            "query": user_query,
            "answer": verification.get("verified_answer", generation["answer"]),
            "citations": generation["citations"],
            "classification": classification,
            "rewritten_query": primary_query,
            "key_legal_terms": rewrite.get("key_legal_terms", []),
            "verdict": verification.get("verdict", "FLAG"),
            "faithfulness": verification.get("overall_faithfulness", "MEDIUM"),
            "uncertainty": generation.get("uncertainty", "medium"),
            "has_answer": generation["has_answer"],
            "chunks_retrieved": len(candidates),
            "chunks_used": len(reranked),
            "filters_applied": filters,
        }
