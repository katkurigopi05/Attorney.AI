"""
Attorney.AI — Answer Generator (Ollama / OpenAI via unified client)
Citation-grounded answers using the local Ollama LLM (free) or OpenAI (optional).
"""
import json
from typing import List, Tuple

from loguru import logger

from ingestion.metadata_schema import LegalChunkMetadata
from llm_client import chat_complete


_SYSTEM_PROMPT = """\
You are Attorney.AI, a citation-first legal research assistant.

CRITICAL RULES — NEVER VIOLATE:
1. Answer ONLY from the provided legal sources. Do NOT use your training data for factual claims.
2. Every factual or legal claim MUST be supported by a citation using [1], [2], etc.
3. If the sources do not answer the question, say: "The provided sources do not contain sufficient information to answer this question."
4. NEVER present outputs as legal advice. Always include: "⚠️ This is not legal advice."
5. If you are uncertain, say so explicitly. Use hedging language: "According to [1]...", "As stated in [2]..."
6. Do NOT synthesize claims across sources unless the sources explicitly support the synthesis.

FORMAT YOUR RESPONSE:
- Answer in clear paragraphs with inline citation numbers [1], [2], etc.
- After your answer, list citations as:
  [1] Case Name / Statute Title — Citation — Court — Date — URL
- End with: "⚠️ This is not legal advice. Consult a licensed attorney for advice on your specific situation."
"""

_USER_PROMPT_TEMPLATE = """\
Question: {query}

Legal Sources:
{context}

Instructions: Answer the question using ONLY the above sources. Cite every claim with [N].
"""


class AnswerGenerator:
    """
    Generates citation-grounded legal answers.
    Uses Ollama (free, local) by default; OpenAI if key is configured.
    """

    async def generate(
        self,
        query: str,
        chunks: List[Tuple[LegalChunkMetadata, float]],
    ) -> dict:
        """
        Generate a citation-grounded answer.
        Returns: { answer, citations, has_answer, uncertainty }
        """
        if not chunks:
            return {
                "answer": (
                    "No relevant legal sources were found for this query. "
                    "Please try a more specific question or check that documents "
                    "have been indexed.\n\n"
                    "⚠️ This is not legal advice."
                ),
                "citations": [],
                "has_answer": False,
                "uncertainty": "high",
            }

        # Build numbered context
        context_parts = []
        citation_list = []

        for i, (chunk, score) in enumerate(chunks, start=1):
            source_label = (
                f"[{i}] {chunk.title} | {chunk.citation} | "
                f"{chunk.court_or_agency or chunk.jurisdiction} | "
                f"{chunk.date_str or 'n.d.'}"
            )
            context_parts.append(f"{source_label}\n{chunk.text}")
            citation_list.append({
                "index": i,
                "title": chunk.title,
                "citation": chunk.citation,
                "court_or_agency": chunk.court_or_agency,
                "jurisdiction": chunk.jurisdiction,
                "date": chunk.date_str,
                "url": chunk.source_url,
                "source_type": chunk.source_type,
                "relevance_score": round(score, 4),
                "chunk_id": chunk.chunk_id,
                "text_excerpt": chunk.text[:300] + ("..." if len(chunk.text) > 300 else ""),
            })

        context_str = "\n\n---\n\n".join(context_parts)
        user_prompt = _USER_PROMPT_TEMPLATE.format(query=query, context=context_str)

        try:
            answer_text = await chat_complete(
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.1,
                max_tokens=1500,
            )

            uncertainty = _estimate_uncertainty(answer_text, chunks)

            return {
                "answer": answer_text,
                "citations": citation_list,
                "has_answer": True,
                "uncertainty": uncertainty,
            }

        except Exception as e:
            logger.error(f"Answer generation failed: {e}")
            return {
                "answer": (
                    f"An error occurred while generating the answer. "
                    f"Make sure Ollama is running: `ollama serve`\n\nError: {e}\n\n"
                    "⚠️ This is not legal advice."
                ),
                "citations": citation_list,
                "has_answer": False,
                "uncertainty": "high",
            }


def _estimate_uncertainty(answer: str, chunks: List[Tuple[LegalChunkMetadata, float]]) -> str:
    lower = answer.lower()
    if any(phrase in lower for phrase in [
        "not contain", "insufficient", "cannot answer", "no relevant", "unclear"
    ]):
        return "high"
    if chunks:
        avg_score = sum(s for _, s in chunks) / len(chunks)
        if avg_score < 0.3:
            return "medium"
    if any(phrase in lower for phrase in ["may", "might", "possibly", "unclear", "uncertain"]):
        return "medium"
    return "low"
