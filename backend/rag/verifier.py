"""
Attorney.AI — Citation Verifier & Hallucination Guard (Ollama / free)
Checks that every claim in the answer is supported by cited sources.
Uses Ollama (local, free) — no API key required.
"""
import json
import re
from typing import List, Tuple

from loguru import logger

from ingestion.metadata_schema import LegalChunkMetadata
from llm_client import chat_complete


_VERIFY_PROMPT = """\
You are a legal citation verifier. Check whether the answer's claims are supported by the provided sources.

For each factual claim that references a citation [N], determine:
- SUPPORTED: source clearly supports the claim
- PARTIALLY_SUPPORTED: source is related but not fully clear
- UNSUPPORTED: source does not support this claim

Respond ONLY with valid JSON:
{
  "verification_results": [
    {"claim": "...", "citation_index": 1, "status": "SUPPORTED", "reason": "..."}
  ],
  "overall_faithfulness": "HIGH",
  "unsupported_count": 0,
  "verdict": "PASS"
}

Verdict rules:
- PASS: all claims SUPPORTED
- FLAG: some PARTIALLY_SUPPORTED  
- REJECT: any UNSUPPORTED

Answer: {answer}

Sources:
{sources}
"""


class CitationVerifier:
    """
    Hallucination guard using Ollama (free, local).
    Verifies every cited claim against retrieved source texts.
    Verdict: PASS | FLAG | REJECT
    """

    async def verify(
        self,
        answer: str,
        chunks: List[Tuple[LegalChunkMetadata, float]],
    ) -> dict:
        """
        Verify citation faithfulness.
        Returns verdict, faithfulness, and verified_answer.
        """
        if not chunks:
            return _default_result(answer, "FLAG", "LOW")

        # Quick check: does the answer even have citation markers?
        if not self.quick_citation_check(answer, len(chunks)):
            logger.warning("Answer has no citation markers — flagging")
            flagged = (
                "⚠️ **Note**: This answer may lack explicit citations. "
                "Please verify claims independently.\n\n" + answer
            )
            return _default_result(flagged, "FLAG", "MEDIUM")

        sources_str = "\n\n".join([
            f"[{i}] {chunk.title}\n{chunk.text[:500]}"
            for i, (chunk, _) in enumerate(chunks, start=1)
        ])

        prompt = _VERIFY_PROMPT.format(
            answer=answer[:2000],
            sources=sources_str[:4000],
        )

        try:
            raw = await chat_complete(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=600,
                json_mode=True,
            )
            result = _extract_json(raw)
            result["verified_answer"] = answer

            verdict = result.get("verdict", "FLAG")
            if verdict == "REJECT":
                result["verified_answer"] = (
                    "⚠️ **Warning**: This answer may contain unsupported claims. "
                    "Please verify independently.\n\n" + answer
                )

            logger.info(
                f"Citation verification: verdict={verdict}, "
                f"faithfulness={result.get('overall_faithfulness')}"
            )
            return result

        except Exception as e:
            logger.error(f"Verifier failed: {e}")
            return _default_result(answer, "FLAG", "MEDIUM")

    def quick_citation_check(self, answer: str, num_chunks: int) -> bool:
        """Fast check: does the answer contain at least one [N] citation?"""
        found = re.findall(r'\[\d+\]', answer)
        referenced = {int(m[1:-1]) for m in found}
        valid = {i for i in referenced if 1 <= i <= num_chunks}
        return len(valid) > 0


def _default_result(answer: str, verdict: str, faithfulness: str) -> dict:
    return {
        "verdict": verdict,
        "overall_faithfulness": faithfulness,
        "verification_results": [],
        "unsupported_count": 0,
        "verified_answer": answer,
    }


def _extract_json(text: str) -> dict:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            return json.loads(match.group())
        raise ValueError("No JSON found in verifier response")
