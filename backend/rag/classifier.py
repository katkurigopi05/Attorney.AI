"""
Attorney.AI — Jurisdiction + Task Classifier
Zero-shot classification to route queries to the right retrieval pipeline.
"""
import json
import re
from typing import Tuple

from loguru import logger
from openai import AsyncOpenAI

from config import settings
from ingestion.metadata_schema import TaskType


_CLASSIFICATION_PROMPT = """\
You are a legal query classifier for a U.S. legal research assistant.

Given the user's question, classify it into:

1. task_type: one of
   - "case_law"       → looking for court opinions, precedents, holdings
   - "statute"        → looking for laws passed by Congress (U.S. Code)
   - "regulation"     → looking for federal regulations (CFR, Federal Register)
   - "contract_review" → reviewing a contract or agreement clause
   - "summarization"  → wants a summary of a case, statute, or document
   - "general"        → general legal question not fitting above

2. jurisdiction: one of
   - "US-Federal", "Alabama", "Alaska", "Arizona", "Arkansas", "California",
     "Colorado", "Connecticut", "Delaware", "Florida", "Georgia", "Hawaii",
     "Idaho", "Illinois", "Indiana", "Iowa", "Kansas", "Kentucky", "Louisiana",
     "Maine", "Maryland", "Massachusetts", "Michigan", "Minnesota", "Mississippi",
     "Missouri", "Montana", "Nebraska", "Nevada", "New Hampshire", "New Jersey",
     "New Mexico", "New York", "North Carolina", "North Dakota", "Ohio", "Oklahoma",
     "Oregon", "Pennsylvania", "Rhode Island", "South Carolina", "South Dakota",
     "Tennessee", "Texas", "Utah", "Vermont", "Virginia", "Washington",
     "West Virginia", "Wisconsin", "Wyoming", "EU", "India", "Unknown"

3. court_level (if case_law): one of "Supreme", "Circuit", "District", "State-Appellate", "Unknown"

4. practice_area: one of
   - "Constitutional", "Criminal", "Civil", "Contract", "Tort", "IP",
     "Tax", "Immigration", "Labor", "Environmental", "Corporate", "Securities",
     "Family", "Real Estate", "Bankruptcy", "Administrative", "Unknown"

Respond ONLY with valid JSON matching this schema:
{
  "task_type": "...",
  "jurisdiction": "...",
  "court_level": "...",
  "practice_area": "...",
  "needs_date_filter": false,
  "confidence": 0.0
}

User question: {query}
"""


class QueryClassifier:
    """
    Classifies a legal research question into task type, jurisdiction,
    court level, and practice area. Uses GPT-4o-mini for zero-shot classification.
    """

    def __init__(self):
        self.client = AsyncOpenAI(api_key=settings.openai_api_key)

    async def classify(self, query: str) -> dict:
        """
        Classify a legal query. Returns a dict with task_type, jurisdiction,
        court_level, practice_area, needs_date_filter, confidence.
        """
        prompt = _CLASSIFICATION_PROMPT.format(query=query)
        try:
            resp = await self.client.chat.completions.create(
                model=settings.openai_chat_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                response_format={"type": "json_object"},
                max_tokens=200,
            )
            raw = resp.choices[0].message.content
            result = json.loads(raw)
            logger.debug(f"Query classified: {result}")
            return result
        except Exception as e:
            logger.warning(f"Classifier failed ({e}), using defaults")
            return {
                "task_type": "case_law",
                "jurisdiction": "US-Federal",
                "court_level": "Unknown",
                "practice_area": "Unknown",
                "needs_date_filter": False,
                "confidence": 0.5,
            }

    def classification_to_filters(self, classification: dict) -> dict:
        """Convert classification results to retrieval metadata filters."""
        filters = {}
        task = classification.get("task_type", "general")
        if task == "case_law":
            filters["source_type"] = "case"
        elif task == "statute":
            filters["source_type"] = "statute"
        elif task == "regulation":
            filters["source_type"] = "regulation"

        jurisdiction = classification.get("jurisdiction", "")
        if jurisdiction and jurisdiction != "Unknown":
            filters["jurisdiction"] = jurisdiction

        court_level = classification.get("court_level", "")
        if court_level and court_level not in ("Unknown", ""):
            filters["court_level"] = court_level

        return filters
