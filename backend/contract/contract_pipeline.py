"""
Attorney.AI — Contract Review Pipeline (Ollama / free)
CUAD clause extraction using Ollama local LLM — no API key required.
"""
import io
import json
import re
from typing import List

from loguru import logger

from api.models import ContractAnalysisResponse, ContractClause
from llm_client import chat_complete


CUAD_CLAUSE_TYPES = [
    "Parties", "Agreement Date", "Effective Date", "Expiration Date",
    "Renewal Term", "Notice Period to Terminate Renewal", "Governing Law",
    "Most Favored Nation", "Non-Compete", "Exclusivity", "Liquidated Damages",
    "Warranty Duration", "IP Ownership Assignment", "License Grant",
    "Limitation of Liability", "Uncapped Liability", "Indemnification",
    "Insurance", "Audit Rights", "Anti-Assignment", "Change of Control",
    "Force Majeure", "Confidentiality", "Non-Disparagement",
    "Termination for Convenience",
]

_ANALYSIS_SYSTEM_PROMPT = """\
You are a contract review expert. Analyze the provided contract for specific clauses.

For each clause type listed, determine:
1. present: true/false
2. text_excerpt: first 200 chars of relevant text (or null)
3. risk_level: "LOW" (standard/favorable), "MEDIUM" (needs review), "HIGH" (unfavorable/missing), "N/A"
4. explanation: brief explanation

Respond ONLY with valid JSON:
{
  "document_type": "Service Agreement|NDA|Employment|License|Other",
  "clauses": [
    {"clause_type": "...", "present": true, "text_excerpt": "...", "risk_level": "LOW", "explanation": "..."}
  ],
  "missing_clauses": ["..."],
  "risk_summary": "...",
  "overall_risk": "LOW|MEDIUM|HIGH",
  "recommendations": ["..."]
}
"""


class ContractPipeline:
    """
    Contract review pipeline using Ollama (local, free).
    Extracts CUAD clause types with risk classification.
    """

    def _extract_text(self, file_bytes: bytes, content_type: str, filename: str) -> str:
        """Extract plain text from PDF, DOCX, or TXT."""
        if content_type == "application/pdf" or filename.endswith(".pdf"):
            import pypdf
            reader = pypdf.PdfReader(io.BytesIO(file_bytes))
            return "\n\n".join(page.extract_text() or "" for page in reader.pages)
        elif (
            "wordprocessingml" in content_type
            or filename.endswith(".docx")
        ):
            import docx
            doc = docx.Document(io.BytesIO(file_bytes))
            return "\n\n".join(p.text for p in doc.paragraphs if p.text.strip())
        else:
            return file_bytes.decode("utf-8", errors="ignore")

    async def analyze(
        self, file_bytes: bytes, filename: str, content_type: str
    ) -> ContractAnalysisResponse:
        """Analyze a contract document. Returns structured clause report."""
        text = self._extract_text(file_bytes, content_type, filename)
        if not text or len(text.strip()) < 100:
            raise ValueError("Could not extract readable text from the uploaded document.")

        # Keep first 8000 chars — balance context vs. Ollama context window
        truncated = text[:8000]
        clause_list_str = "\n".join(f"- {c}" for c in CUAD_CLAUSE_TYPES)

        user_prompt = f"""Analyze this contract for the following clause types:

{clause_list_str}

CONTRACT TEXT:
{truncated}

Respond with JSON only."""

        raw = await chat_complete(
            messages=[
                {"role": "system", "content": _ANALYSIS_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,
            max_tokens=3000,
            json_mode=True,
        )

        # Extract JSON robustly
        data = _extract_json(raw)

        clauses = [ContractClause(**c) for c in data.get("clauses", [])]

        return ContractAnalysisResponse(
            filename=filename,
            document_type=data.get("document_type", "Unknown"),
            clauses=clauses,
            missing_clauses=data.get("missing_clauses", []),
            risk_summary=data.get("risk_summary", ""),
            overall_risk=data.get("overall_risk", "MEDIUM"),
            recommendations=data.get("recommendations", []),
        )


def _extract_json(text: str) -> dict:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
        logger.error(f"Could not parse JSON from contract analysis: {text[:300]}")
        return {
            "document_type": "Unknown", "clauses": [], "missing_clauses": [],
            "risk_summary": "Analysis failed — could not parse model output.",
            "overall_risk": "MEDIUM", "recommendations": [],
        }
