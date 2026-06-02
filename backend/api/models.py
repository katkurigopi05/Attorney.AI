"""
Attorney.AI — Pydantic Request/Response Models
"""
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ResearchRequest(BaseModel):
    query: str = Field(..., min_length=5, max_length=2000, description="Legal research question")
    jurisdiction: Optional[str] = Field(default=None, description="Override jurisdiction filter")
    source_type: Optional[str] = Field(default=None, description="Override source type filter")
    top_k: Optional[int] = Field(default=None, ge=1, le=20, description="Results to return")
    skip_verify: bool = Field(default=False, description="Skip citation verification (faster)")


class CitationModel(BaseModel):
    index: int
    title: str
    citation: str
    court_or_agency: Optional[str]
    jurisdiction: str
    date: Optional[str]
    url: str
    source_type: str
    relevance_score: float
    chunk_id: str
    text_excerpt: str


class ResearchResponse(BaseModel):
    query: str
    answer: str
    citations: List[CitationModel]
    classification: Dict[str, Any]
    rewritten_query: str
    key_legal_terms: List[str]
    verdict: str = Field(description="PASS | FLAG | REJECT")
    faithfulness: str = Field(description="HIGH | MEDIUM | LOW")
    uncertainty: str = Field(description="low | medium | high")
    has_answer: bool
    chunks_retrieved: int
    chunks_used: int
    filters_applied: Dict[str, Any]
    generated_at: datetime = Field(default_factory=datetime.utcnow)
    disclaimer: str = Field(
        default="⚠️ This is not legal advice. Consult a licensed attorney for advice on your specific situation."
    )


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=2, max_length=500)
    jurisdiction: Optional[str] = None
    source_type: Optional[str] = None
    court_level: Optional[str] = None
    top_k: int = Field(default=10, ge=1, le=50)


class SearchResultItem(BaseModel):
    chunk_id: str
    title: str
    citation: str
    text_excerpt: str
    jurisdiction: str
    court_or_agency: Optional[str]
    source_type: str
    date: Optional[str]
    url: str
    relevance_score: float


class SearchResponse(BaseModel):
    query: str
    results: List[SearchResultItem]
    total_results: int
    filters_applied: Dict[str, Any]


class ContractClause(BaseModel):
    clause_type: str = Field(description="CUAD clause category")
    present: bool
    text_excerpt: Optional[str]
    risk_level: str = Field(description="LOW | MEDIUM | HIGH | N/A")
    explanation: str


class ContractAnalysisResponse(BaseModel):
    filename: str
    document_type: str
    clauses: List[ContractClause]
    missing_clauses: List[str]
    risk_summary: str
    overall_risk: str = Field(description="LOW | MEDIUM | HIGH")
    recommendations: List[str]
    disclaimer: str = Field(
        default="⚠️ This is not legal advice. Contract review by an attorney is strongly recommended."
    )


class HealthResponse(BaseModel):
    status: str
    version: str
    qdrant_connected: bool
    bm25_chunks_loaded: int
    disclaimer: str = Field(default="Attorney.AI is not a licensed legal service.")
