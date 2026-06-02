"""
Attorney.AI — Jurisdiction + Task Classifier
Uses HuggingFace zero-shot classification (facebook/bart-large-mnli, FREE)
as primary method. Falls back to Ollama for detailed classification.
"""
import json
import re
from functools import lru_cache
from typing import Optional

from loguru import logger

from config import settings
from ingestion.metadata_schema import TaskType
from llm_client import chat_complete


# ── HuggingFace zero-shot classifier (completely free, no API) ───────────────
@lru_cache(maxsize=1)
def _get_zero_shot_pipeline():
    """Lazy-load the zero-shot classification pipeline."""
    try:
        from transformers import pipeline
        logger.info(f"Loading zero-shot classifier: {settings.classifier_model}")
        pipe = pipeline(
            "zero-shot-classification",
            model=settings.classifier_model,
            device=-1,  # CPU
        )
        logger.info("Zero-shot classifier loaded")
        return pipe
    except Exception as e:
        logger.warning(f"Zero-shot classifier failed to load: {e}")
        return None


_TASK_LABELS = [
    "case law and court opinions",
    "statutes and legislation",
    "federal regulations and compliance",
    "contract review and clause analysis",
    "legal document summarization",
    "general legal question",
]

_TASK_LABEL_MAP = {
    "case law and court opinions": "case_law",
    "statutes and legislation": "statute",
    "federal regulations and compliance": "regulation",
    "contract review and clause analysis": "contract_review",
    "legal document summarization": "summarization",
    "general legal question": "general",
}

_JURISDICTION_LABELS = [
    "US Federal law",
    "California state law",
    "New York state law",
    "Texas state law",
    "European Union law",
    "Indian law",
    "Other US state law",
]

_JURISDICTION_MAP = {
    "US Federal law": "US-Federal",
    "California state law": "California",
    "New York state law": "New York",
    "Texas state law": "Texas",
    "European Union law": "EU",
    "Indian law": "India",
    "Other US state law": "US-Federal",
}


class QueryClassifier:
    """
    Classifies a legal research question using:
    1. HuggingFace zero-shot (bart-large-mnli) — primary, FREE, no API
    2. Ollama LLM — for detailed structured output, FREE
    """

    async def classify(self, query: str) -> dict:
        """Classify a legal query. Returns task_type, jurisdiction, etc."""
        # Try HuggingFace zero-shot first (fast, no API needed)
        pipe = _get_zero_shot_pipeline()

        task_type = "case_law"
        jurisdiction = "US-Federal"
        confidence = 0.5

        if pipe:
            try:
                # Classify task type
                task_result = pipe(query, candidate_labels=_TASK_LABELS, multi_label=False)
                best_task_label = task_result["labels"][0]
                task_type = _TASK_LABEL_MAP.get(best_task_label, "general")
                confidence = task_result["scores"][0]

                # Classify jurisdiction
                jur_result = pipe(query, candidate_labels=_JURISDICTION_LABELS, multi_label=False)
                best_jur_label = jur_result["labels"][0]
                jurisdiction = _JURISDICTION_MAP.get(best_jur_label, "US-Federal")

                logger.debug(
                    f"Zero-shot classify: task={task_type} ({confidence:.2f}), "
                    f"jurisdiction={jurisdiction}"
                )
            except Exception as e:
                logger.warning(f"Zero-shot classify failed: {e}")

        # Detect court level from query text (simple heuristic)
        court_level = _detect_court_level(query)
        practice_area = _detect_practice_area(query)

        result = {
            "task_type": task_type,
            "jurisdiction": jurisdiction,
            "court_level": court_level,
            "practice_area": practice_area,
            "needs_date_filter": any(
                kw in query.lower() for kw in ["recent", "latest", "2024", "2023", "current"]
            ),
            "confidence": round(confidence, 3),
        }
        logger.debug(f"Classification result: {result}")
        return result

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


def _detect_court_level(query: str) -> str:
    q = query.lower()
    if any(kw in q for kw in ["supreme court", "scotus", "u.s. supreme"]):
        return "Supreme"
    if any(kw in q for kw in ["circuit", "court of appeals", "appellate"]):
        return "Circuit"
    if any(kw in q for kw in ["district court", "trial court"]):
        return "District"
    return "Unknown"


def _detect_practice_area(query: str) -> str:
    q = query.lower()
    area_map = {
        "Constitutional": ["constitutional", "first amendment", "due process", "equal protection"],
        "Criminal": ["criminal", "murder", "felony", "misdemeanor", "prosecution"],
        "Contract": ["contract", "breach", "consideration", "agreement", "clause"],
        "IP": ["patent", "copyright", "trademark", "trade secret", "intellectual property"],
        "Tax": ["tax", "irs", "income tax", "deduction", "revenue"],
        "Immigration": ["immigration", "visa", "deportation", "asylum", "citizenship"],
        "Labor": ["employment", "discrimination", "harassment", "labor", "workers"],
        "Corporate": ["corporate", "securities", "merger", "acquisition", "shareholder"],
    }
    for area, keywords in area_map.items():
        if any(kw in q for kw in keywords):
            return area
    return "Unknown"
