"""
Attorney.AI — Legal Query Rewriter
Expands natural language questions into structured legal search queries.
"""
import json
from typing import List

from loguru import logger
from openai import AsyncOpenAI

from config import settings


_REWRITE_PROMPT = """\
You are a legal research expert helping rewrite a user's question into optimized search queries.

Given the user's question and its classification, produce:
1. A rewritten primary query optimized for legal document retrieval (clear, precise)
2. 2-3 alternative search queries covering different phrasings or related legal concepts
3. Key legal terms to emphasize (statute numbers, legal doctrine names, key terms)

Respond ONLY with valid JSON:
{
  "primary_query": "...",
  "alternative_queries": ["...", "..."],
  "key_legal_terms": ["...", "..."],
  "suggested_citation_format": "e.g. 42 U.S.C. § 1983 or Miranda v. Arizona"
}

User question: {query}
Task type: {task_type}
Jurisdiction: {jurisdiction}
Practice area: {practice_area}
"""


class QueryRewriter:
    """
    Rewrites user queries into legal-optimized search queries.
    Produces a primary query + alternatives to maximize recall.
    """

    def __init__(self):
        self.client = AsyncOpenAI(api_key=settings.openai_api_key)

    async def rewrite(
        self,
        query: str,
        classification: dict,
    ) -> dict:
        """
        Expand and refine the query for legal retrieval.
        Returns dict with primary_query, alternative_queries, key_legal_terms.
        """
        prompt = _REWRITE_PROMPT.format(
            query=query,
            task_type=classification.get("task_type", "general"),
            jurisdiction=classification.get("jurisdiction", "US-Federal"),
            practice_area=classification.get("practice_area", "Unknown"),
        )

        try:
            resp = await self.client.chat.completions.create(
                model=settings.openai_chat_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                response_format={"type": "json_object"},
                max_tokens=400,
            )
            result = json.loads(resp.choices[0].message.content)
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
