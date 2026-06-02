"""
Attorney.AI — Legal Query Rewriter (Ollama / free)
Expands natural language questions into optimized legal search queries.
Uses Ollama (local, free) — no API key required.
"""
import json
import re
from typing import List

from loguru import logger

from llm_client import chat_complete


_REWRITE_PROMPT = """\
You are a legal research expert. Rewrite this legal question into better search queries.

Given the question and context, provide:
1. A rewritten primary query (clear, precise legal phrasing)
2. 2 alternative queries with different phrasings
3. Key legal terms to search for (statute numbers, doctrine names, etc.)

Respond ONLY with valid JSON in this exact format:
{
  "primary_query": "...",
  "alternative_queries": ["...", "..."],
  "key_legal_terms": ["...", "..."],
  "suggested_citation_format": "..."
}

Question: {query}
Task type: {task_type}
Jurisdiction: {jurisdiction}
"""


class QueryRewriter:
    """
    Rewrites user queries into legal-optimized search queries.
    Uses Ollama (local, free) — no paid API needed.
    """

    async def rewrite(self, query: str, classification: dict) -> dict:
        """Expand and refine the query for legal retrieval."""
        prompt = _REWRITE_PROMPT.format(
            query=query,
            task_type=classification.get("task_type", "general"),
            jurisdiction=classification.get("jurisdiction", "US-Federal"),
        )

        try:
            raw = await chat_complete(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=400,
                json_mode=True,
            )
            # Extract JSON even if model adds extra text
            result = _extract_json(raw)
            logger.debug(f"Query rewritten: primary='{result.get('primary_query')}'")
            return result
        except Exception as e:
            logger.warning(f"Query rewriter failed ({e}), using original query")
            return {
                "primary_query": query,
                "alternative_queries": [query],
                "key_legal_terms": [],
                "suggested_citation_format": "",
            }

    def get_all_queries(self, rewrite_result: dict) -> List[str]:
        """Return all queries to use for retrieval (primary + alternatives)."""
        queries = [rewrite_result.get("primary_query", "")]
        queries += rewrite_result.get("alternative_queries", [])
        return [q for q in queries if q]


def _extract_json(text: str) -> dict:
    """Extract JSON object from text, even if surrounded by prose."""
    # Try direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Try to find JSON block
    match = re.search(r'\{[^{}]*\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    raise ValueError(f"Could not parse JSON from: {text[:200]}")
