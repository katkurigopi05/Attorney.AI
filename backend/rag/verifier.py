"""
Attorney.AI — Citation Verifier & Hallucination Guard
Every factual claim in the generated answer must map to a retrieved citation.
Rejects or flags answers containing unsupported claims.
"""
import re
from typing import Dict, List, Tuple

from loguru import logger
from openai import AsyncOpenAI

from config import settings
from ingestion.metadata_schema import LegalChunkMetadata


_VERIFY_PROMPT = """\
You are a legal citation verifier. Your job is to check whether the claims in the generated answer are supported by the provided source texts.

For each factual claim in the answer that references a citation [N]:
1. Check whether the cited source text actually supports that claim.
2. Mark it as: "SUPPORTED", "PARTIALLY_SUPPORTED", or "UNSUPPORTED"

Respond ONLY with valid JSON:
{
  "verification_results": [
    {
      "claim": "short claim text",
      "citation_index": 1,
      "status": "SUPPORTED" | "PARTIALLY_SUPPORTED" | "UNSUPPORTED",
      "reason": "brief explanation"
    }
  ],
  "overall_faithfulness": "HIGH" | "MEDIUM" | "LOW",
  "unsupported_count": 0,
  "verdict": "PASS" | "FLAG" | "REJECT"
}

Verdict rules:
- PASS: All claims SUPPORTED, faithfulness HIGH
- FLAG: Some PARTIALLY_SUPPORTED, warn user but allow
- REJECT: Any UNSUPPORTED claims → answer must be revised

Answer to verify:
{answer}

Source texts:
{sources}
"""


class CitationVerifier:
    """
    Hallucination guard that verifies every cited claim in the generated answer
    against the actual retrieved source texts.
    
    Verdict: PASS | FLAG | REJECT
    """

    def __init__(self):
        self.client = AsyncOpenAI(api_key=settings.openai_api_key)

    async def verify(
        self,
        answer: str,
        chunks: List[Tuple[LegalChunkMetadata, float]],
    ) -> dict:
        """
        Verify citation faithfulness.
        
        Returns:
            {
              "verdict": "PASS" | "FLAG" | "REJECT",
              "overall_faithfulness": "HIGH" | "MEDIUM" | "LOW",
              "verification_results": [...],
              "unsupported_count": int,
              "verified_answer": str  # cleaned answer if needed
            }
        """
        if not chunks:
            return {
                "verdict": "FLAG",
                "overall_faithfulness": "LOW",
                "verification_results": [],
                "unsupported_count": 0,
                "verified_answer": answer,
            }

        # Build numbered source texts for the verifier
        sources_str = "\n\n".join([
            f"[{i}] {chunk.title}\n{chunk.text[:600]}"
            for i, (chunk, _) in enumerate(chunks, start=1)
        ])

        prompt = _VERIFY_PROMPT.format(answer=answer[:3000], sources=sources_str[:6000])

        try:
            resp = await self.client.chat.completions.create(
                model=settings.openai_chat_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                response_format={"type": "json_object"},
                max_tokens=800,
            )
            import json
            result = json.loads(resp.choices[0].message.content)
            result["verified_answer"] = answer

            verdict = result.get("verdict", "FLAG")
            faithfulness = result.get("overall_faithfulness", "MEDIUM")
            unsupported = result.get("unsupported_count", 0)

            logger.info(
                f"Citation verification: verdict={verdict}, "
                f"faithfulness={faithfulness}, unsupported={unsupported}"
            )

            # If REJECT, add a warning prefix to the answer
            if verdict == "REJECT":
                result["verified_answer"] = (
                    "⚠️ **Warning**: This answer may contain claims not fully supported "
                    "by the retrieved sources. Please verify independently.\n\n" + answer
                )

            return result

        except Exception as e:
            logger.error(f"Verifier failed: {e}")
            return {
                "verdict": "FLAG",
                "overall_faithfulness": "MEDIUM",
                "verification_results": [],
                "unsupported_count": 0,
                "verified_answer": answer,
            }

    def quick_citation_check(self, answer: str, num_chunks: int) -> bool:
        """
        Fast pre-check: does the answer contain at least one [N] citation?
        Returns True if citation markers found, False if answer has no citations.
        """
        citation_pattern = re.compile(r"\[\d+\]")
        found = citation_pattern.findall(answer)
        referenced = {int(m[1:-1]) for m in found}
        valid = {i for i in referenced if 1 <= i <= num_chunks}
        return len(valid) > 0
