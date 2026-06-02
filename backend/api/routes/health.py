from fastapi import APIRouter
from api.models import HealthResponse
from config import settings

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """API health check endpoint."""
    # Check Qdrant connectivity
    qdrant_ok = False
    try:
        from retrieval.vector_store import VectorStore
        vs = VectorStore()
        await vs.ensure_collection()
        qdrant_ok = True
    except Exception:
        pass

    # Check BM25 index
    bm25_count = 0
    try:
        from retrieval.bm25_store import BM25Store
        bm25 = BM25Store()
        bm25.load()
        bm25_count = len(bm25.chunks)
    except Exception:
        pass

    return HealthResponse(
        status="ok",
        version="0.1.0",
        qdrant_connected=qdrant_ok,
        bm25_chunks_loaded=bm25_count,
    )
