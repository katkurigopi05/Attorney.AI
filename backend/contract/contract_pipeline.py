"""
Attorney.AI — Contract Review Pipeline
CUAD clause extraction + risk classification using GPT-4o-mini.
"""
import io
import json
from typing import List

from loguru import logger
from openai import AsyncOpenAI

from config import settings
from api.models import ContractAnalysisResponse, ContractClause


# 41 CUAD clause types (simplified to 20 key types for MVP)
CUAD_CLAUSE_TYPES = [
    "Parties",
    "Agreement Date",
    "Effective Date",
    "Expiration Date",
    "Renewal Term",
    "Notice Period to Terminate Renewal",
    "Governing Law",
    "Most Favored Nation",
    "Non-Compete",
    "Exclusivity",
    "Liquidated Damages",
    "Warranty Duration",
    "IP Ownership Assignment",
    "License Grant",
    "Limitation of Liability",
    "Uncapped Liability",
    "Indemnification",
    "Insurance",
    "Audit Rights",
    "Anti-Assignment",
    "Change of Control",
    "Force Majeure",
    "Confidentiality",
    "Non-Disparagement",
    "Termination for Convenience",
]

_ANALYSIS_SYSTEM_PROMPT = """\
You are a contract review expert. Analyze the provided contract text for the presence and quality of specific clauses.

For each clause type, determine:
1. Whether it is present in the contract
2. The exact text excerpt (first 200 chars) if present
3. The risk level: LOW (favorable/standard), MEDIUM (needs negotiation), HIGH (unfavorable/missing key protection), N/A (not applicable)
4. A brief explanation

Respond ONLY with valid JSON matching this schema:
{
  "document_type": "Service Agreement | NDA | Employment | License | Other",
  "clauses": [
    {
      "clause_type": "...",
      "present": true/false,
      "text_excerpt": "...",
      "risk_level": "LOW|MEDIUM|HIGH|N/A",
      "explanation": "..."
    }
  ],
  "missing_clauses": ["clause types that are absent but should be present"],
  "risk_summary": "overall risk assessment paragraph",
  "overall_risk": "LOW|MEDIUM|HIGH",
  "recommendations": ["specific actionable recommendations"]
}
"""


class ContractPipeline:
    """
    Contract review pipeline:
    1. Extract text from PDF/DOCX/TXT
    2. Run CUAD clause analysis via GPT-4o-mini
    3. Return structured clause report with risk levels
    """

    def __init__(self):
        self.client = AsyncOpenAI(api_key=settings.openai_api_key)

    def _extract_text(self, file_bytes: bytes, content_type: str, filename: str) -> str:
        """Extract plain text from PDF, DOCX, or TXT."""
        if content_type == "application/pdf" or filename.endswith(".pdf"):
            import pypdf
            reader = pypdf.PdfReader(io.BytesIO(file_bytes))
            return "\n\n".join(page.extract_text() or "" for page in reader.pages)
        elif content_type in [
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        ] or filename.endswith(".docx"):
            import docx
            doc = docx.Document(io.BytesIO(file_bytes))
            return "\n\n".join(para.text for para in doc.paragraphs if para.text.strip())
        else:
            return file_bytes.decode("utf-8", errors="ignore")

    async def analyze(
        self,
        file_bytes: bytes,
        filename: str,
        content_type: str,
    ) -> ContractAnalysisResponse:
        """Analyze a contract document and return structured clause report."""
        text = self._extract_text(file_bytes, content_type, filename)
        if not text or len(text.strip()) < 100:
            raise ValueError("Could not extract readable text from the uploaded document.")

        # Truncate to 12,000 chars to fit in context window
        truncated = text[:12000]
        clause_list_str = "\n".join(f"- {c}" for c in CUAD_CLAUSE_TYPES)

        user_prompt = f"""Analyze this contract for the following clause types:

{clause_list_str}

CONTRACT TEXT:
{truncated}
"""

        resp = await self.client.chat.completions.create(
            model=settings.openai_chat_model,
            messages=[
                {"role": "system", "content": _ANALYSIS_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,
            response_format={"type": "json_object"},
            max_tokens=3000,
        )

        raw = json.loads(resp.choices[0].message.content)

        clauses = [ContractClause(**c) for c in raw.get("clauses", [])]

        return ContractAnalysisResponse(
            filename=filename,
            document_type=raw.get("document_type", "Unknown"),
            clauses=clauses,
            missing_clauses=raw.get("missing_clauses", []),
            risk_summary=raw.get("risk_summary", ""),
            overall_risk=raw.get("overall_risk", "MEDIUM"),
            recommendations=raw.get("recommendations", []),
        )
