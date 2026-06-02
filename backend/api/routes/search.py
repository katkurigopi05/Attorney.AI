from fastapi import APIRouter, HTTPException
from loguru import logger

from api.models import SearchRequest, SearchResponse, SearchResultItem
from retrieval.bm25_store import BM25Store
from retrieval.embedder import Embedder
from retrieval.hybrid_retriever import HybridRetriever
from retrieval.vector_store import VectorStore

router = APIRouter()

_retriever = None


def _get_retriever() -> HybridRetriever:
    global _retriever
    if _retriever is None:
        embedder = Embedder()
        vs = VectorStore()
        bm25 = BM25Store()
        bm25.load()
        _retriever = HybridRetriever(embedder=embedder, vector_store=vs, bm25_store=bm25)
    return _retriever


@router.post("/search", response_model=SearchResponse)
async def search_legal_documents(request: SearchRequest):
    """
    Direct legal document search without answer generation.
    Returns ranked chunks matching the query with metadata.
    """
    try:
        retriever = _get_retriever()
        filters = {}
        if request.jurisdiction:
            filters["jurisdiction"] = request.jurisdiction
        if request.source_type:
            filters["source_type"] = request.source_type
        if request.court_level:
            filters["court_level"] = request.court_level

        results = await retriever.retrieve(
            query=request.query,
            top_k=request.top_k,
            filters=filters or None,
        )

        items = [
            SearchResultItem(
                chunk_id=chunk.chunk_id,
                title=chunk.title,
                citation=chunk.citation,
                text_excerpt=chunk.text[:400] + ("..." if len(chunk.text) > 400 else ""),
                jurisdiction=chunk.jurisdiction,
                court_or_agency=chunk.court_or_agency,
                source_type=chunk.source_type,
                date=chunk.date_str,
                url=chunk.source_url,
                relevance_score=round(score, 4),
            )
            for chunk, score in results
        ]

        return SearchResponse(
            query=request.query,
            results=items,
            total_results=len(items),
            filters_applied=filters,
        )
    except Exception as e:
        logger.error(f"Search endpoint error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
