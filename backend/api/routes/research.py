from functools import lru_cache
from fastapi import APIRouter, HTTPException
from loguru import logger

from api.models import ResearchRequest, ResearchResponse
from rag.pipeline import RAGPipeline

router = APIRouter()


@lru_cache(maxsize=1)
def get_pipeline() -> RAGPipeline:
    return RAGPipeline()


@router.post("/research", response_model=ResearchResponse)
async def legal_research(request: ResearchRequest):
    """
    Main legal research endpoint.
    Runs the full RAG pipeline: classify → rewrite → retrieve → rerank → generate → verify.
    Every answer is grounded in authoritative legal citations.
    """
    try:
        pipeline = get_pipeline()
        result = await pipeline.query(
            user_query=request.query,
            jurisdiction_override=request.jurisdiction,
            source_type_override=request.source_type,
            top_k_rerank=request.top_k,
            skip_verify=request.skip_verify,
        )
        return ResearchResponse(**result)
    except Exception as e:
        logger.error(f"Research endpoint error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
