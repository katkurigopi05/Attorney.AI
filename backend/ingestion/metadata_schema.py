"""
Attorney.AI — Legal Chunk Metadata Schema
Every chunk stored in the vector DB carries this metadata.
"""
from datetime import date
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class SourceType(str, Enum):
    CASE = "case"
    STATUTE = "statute"
    REGULATION = "regulation"
    CONTRACT = "contract"
    FILING = "filing"
    BILL = "bill"
    RULE = "rule"


class TaskType(str, Enum):
    CASE_LAW = "case_law"
    STATUTE = "statute"
    REGULATION = "regulation"
    CONTRACT_REVIEW = "contract_review"
    SUMMARIZATION = "summarization"
    GENERAL = "general"


class LegalChunkMetadata(BaseModel):
    """
    Canonical metadata schema for every legal chunk stored in the vector DB.
    Mirrors the spec from the implementation plan exactly.
    """
    # ── Identity ──────────────────────────────────────────
    doc_id: str = Field(description="Unique document identifier (case ID, statute ID, etc.)")
    chunk_id: str = Field(description="Unique chunk ID within document: doc_id + ':' + chunk_index")

    # ── Display ───────────────────────────────────────────
    title: str = Field(description="Case name / statute section / regulation title")
    citation: str = Field(description="Official legal citation string")
    source_url: str = Field(description="Canonical URL to the source document")

    # ── Jurisdiction / Court ──────────────────────────────
    jurisdiction: str = Field(description="e.g. US-Federal, California, EU, India")
    court_or_agency: Optional[str] = Field(
        default=None,
        description="e.g. Supreme Court, 9th Circuit, SEC, CFR Title 17"
    )

    # ── Date ──────────────────────────────────────────────
    decision_date: Optional[date] = Field(
        default=None,
        description="Decision date for cases, effective date for statutes/regs"
    )
    date_str: Optional[str] = Field(
        default=None,
        description="Human-readable date string for display"
    )

    # ── Source Classification ─────────────────────────────
    source_type: SourceType = Field(description="Document type classification")

    # ── Document Structure ────────────────────────────────
    parent_section: Optional[str] = Field(
        default=None,
        description="Section / clause / heading this chunk belongs to"
    )
    start_char: int = Field(default=0, description="Character offset start in source document")
    end_char: int = Field(default=0, description="Character offset end in source document")

    # ── Content ───────────────────────────────────────────
    text: str = Field(description="The chunk text content")

    # ── Extra ─────────────────────────────────────────────
    court_level: Optional[str] = Field(
        default=None,
        description="e.g. Supreme, Circuit, District, State-Appellate"
    )
    docket_number: Optional[str] = Field(default=None)
    author_judge: Optional[str] = Field(default=None, description="Authoring judge/justice")
    practice_area: Optional[str] = Field(
        default=None,
        description="e.g. Constitutional, Criminal, Contract, IP, Tax"
    )

    def to_qdrant_payload(self) -> dict:
        """Convert to Qdrant payload dict (JSON-serializable)."""
        data = self.model_dump()
        # Convert date to ISO string for Qdrant
        if data.get("decision_date"):
            data["decision_date"] = data["decision_date"].isoformat()
        # Keep text in payload for retrieval display
        return data

    @classmethod
    def from_qdrant_payload(cls, payload: dict) -> "LegalChunkMetadata":
        return cls(**payload)
