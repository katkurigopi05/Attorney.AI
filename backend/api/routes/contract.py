"""
Attorney.AI — Contract Review API Route
Accepts PDF or text upload, returns clause analysis.
"""
import io
from fastapi import APIRouter, File, HTTPException, UploadFile
from loguru import logger

from api.models import ContractAnalysisResponse
from contract.contract_pipeline import ContractPipeline

router = APIRouter()

_pipeline = None


def _get_pipeline() -> ContractPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = ContractPipeline()
    return _pipeline


@router.post("/contract/analyze", response_model=ContractAnalysisResponse)
async def analyze_contract(file: UploadFile = File(...)):
    """
    Contract review endpoint.
    Upload a PDF or text file; receive a structured clause analysis
    covering CUAD clause types, risk levels, and missing terms.
    """
    if file.content_type not in [
        "application/pdf",
        "text/plain",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ]:
        raise HTTPException(
            status_code=400,
            detail="Unsupported file type. Upload PDF, DOCX, or TXT.",
        )

    try:
        contents = await file.read()
        pipeline = _get_pipeline()
        result = await pipeline.analyze(
            file_bytes=contents,
            filename=file.filename,
            content_type=file.content_type,
        )
        return result
    except Exception as e:
        logger.error(f"Contract analysis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
