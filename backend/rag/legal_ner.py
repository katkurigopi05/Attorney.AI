"""
Attorney.AI — Legal Named Entity Recognition (NER)
Extracts structured legal entities from queries and documents using Transformers.

Extracts:
  - CASE_NAME    → "Miranda v. Arizona", "Roe v. Wade"
  - STATUTE      → "42 U.S.C. § 1983", "Title VII"
  - CITATION     → "410 U.S. 113", "9th Cir. 2019"
  - JURISDICTION → "California", "US-Federal", "9th Circuit"
  - DATE         → "1973", "January 22, 1973"
  - PARTY        → plaintiff/defendant names
  - COURT        → "Supreme Court", "District Court"

Uses:
  - Primary:  dslim/bert-base-NER (fast, good general NER)
  - Legal:    Regex patterns for statute/citation/case-name patterns
  - Optional: law-ai/InLegalBERT for Indian law entities
"""
import re
from functools import lru_cache
from typing import Dict, List, Optional

from loguru import logger


# ── Legal-specific regex patterns ────────────────────────────────────────────

_STATUTE_PATTERN = re.compile(
    r"""
    (?:
        \d+\s+U\.S\.C\.?\s*§+\s*[\d.-]+   |  # 42 U.S.C. § 1983
        \d+\s+C\.F\.R\.?\s*(?:§+\s*)?[\d.-]+  |  # 29 C.F.R. § 825
        Title\s+[IVXLCDM\d]+\s+of\s+\w+    |  # Title VII of...
        Pub(?:lic)?\s*L(?:aw)?\.?\s*[\d-]+    # Public Law 104-191
    )
    """,
    re.VERBOSE | re.IGNORECASE,
)

_CITATION_PATTERN = re.compile(
    r"""
    (?:
        \d+\s+U\.S\.\s+\d+               |  # 410 U.S. 113
        \d+\s+F\.\d+d?\s+\d+             |  # 997 F.3d 184
        \d+\s+S\.\s*Ct\.\s+\d+           |  # 143 S. Ct. 2355
        \d+\s+L\.\s*Ed\.\s*(?:2d\s+)?\d+ |  # 93 L. Ed. 2d 649
        \d+\s+F\.\s*Supp\.\s*(?:3d\s+)?\d+  # 512 F. Supp. 3d 100
    )
    """,
    re.VERBOSE,
)

_CASE_NAME_PATTERN = re.compile(
    r"([A-Z][A-Za-z,.'&\s]{2,40})\s+v(?:s)?\.?\s+([A-Z][A-Za-z,.'&\s]{2,40}?)(?=\s*,|\s*\(|\s+\d|\.|$)"
)

_COURT_PATTERN = re.compile(
    r"""
    (?:
        Supreme\s+Court(?:\s+of\s+the\s+United\s+States)?  |
        (?:\d+(?:st|nd|rd|th)\s+)?Circuit\s+(?:Court\s+of\s+Appeals)?  |
        (?:U\.S\.\s+)?District\s+Court(?:\s+for\s+the\s+\w+\s+District)?  |
        Court\s+of\s+Appeals
    )
    """,
    re.VERBOSE | re.IGNORECASE,
)

_JURISDICTION_KEYWORDS = {
    "US-Federal": ["federal", "u.s.", "united states", "congress", "constitution"],
    "Supreme": ["supreme court", "scotus"],
    "California": ["california", "cal.", "9th circuit"],
    "New York": ["new york", "ny", "2nd circuit"],
    "Texas": ["texas", "tx", "5th circuit"],
    "EU": ["european union", "eu", "cjeu", "directive"],
    "India": ["india", "supreme court of india", "high court", "ipc"],
}


# ── Transformer-based NER (lazy loaded) ──────────────────────────────────────

_ner_pipeline = None


@lru_cache(maxsize=1)
def _get_ner_pipeline():
    """Lazy-load the HuggingFace NER pipeline."""
    global _ner_pipeline
    if _ner_pipeline is None:
        try:
            from transformers import pipeline
            logger.info("Loading NER model: dslim/bert-base-NER")
            _ner_pipeline = pipeline(
                "ner",
                model="dslim/bert-base-NER",
                aggregation_strategy="simple",
                device=-1,  # CPU; set to 0 for GPU
            )
            logger.info("NER model loaded")
        except Exception as e:
            logger.warning(f"NER model load failed: {e} — using regex-only mode")
            _ner_pipeline = None
    return _ner_pipeline


class LegalNER:
    """
    Legal Named Entity Recognizer.
    Combines transformer NER with legal-specific regex patterns.

    Usage:
        ner = LegalNER()
        entities = ner.extract("What is the holding in Miranda v. Arizona, 384 U.S. 436?")
    """

    def extract(self, text: str) -> Dict[str, List[str]]:
        """
        Extract legal entities from text.
        Returns a dict mapping entity type → list of extracted strings.
        """
        entities: Dict[str, List[str]] = {
            "STATUTE": [],
            "CITATION": [],
            "CASE_NAME": [],
            "COURT": [],
            "JURISDICTION": [],
            "PERSON": [],
            "ORG": [],
        }

        # ── Regex extraction ──────────────────────────────────────────────
        for match in _STATUTE_PATTERN.finditer(text):
            val = match.group().strip()
            if val not in entities["STATUTE"]:
                entities["STATUTE"].append(val)

        for match in _CITATION_PATTERN.finditer(text):
            val = match.group().strip()
            if val not in entities["CITATION"]:
                entities["CITATION"].append(val)

        for match in _CASE_NAME_PATTERN.finditer(text):
            val = match.group().strip()
            if val not in entities["CASE_NAME"]:
                entities["CASE_NAME"].append(val)

        for match in _COURT_PATTERN.finditer(text):
            val = match.group().strip()
            if val not in entities["COURT"]:
                entities["COURT"].append(val)

        # Jurisdiction detection
        text_lower = text.lower()
        for jur, keywords in _JURISDICTION_KEYWORDS.items():
            if any(kw in text_lower for kw in keywords):
                if jur not in entities["JURISDICTION"]:
                    entities["JURISDICTION"].append(jur)

        # ── Transformer NER (persons + organizations) ─────────────────────
        ner = _get_ner_pipeline()
        if ner and len(text) > 10:
            try:
                results = ner(text[:512])  # NER model max 512 tokens
                for ent in results:
                    label = ent.get("entity_group", "")
                    word = ent.get("word", "").strip()
                    score = ent.get("score", 0)
                    if score < 0.7 or not word:
                        continue
                    if label == "PER" and word not in entities["PERSON"]:
                        entities["PERSON"].append(word)
                    elif label == "ORG" and word not in entities["ORG"]:
                        entities["ORG"].append(word)
            except Exception as e:
                logger.debug(f"NER inference error: {e}")

        # Remove empties
        return {k: v for k, v in entities.items() if v}

    def extract_query_hints(self, text: str) -> dict:
        """
        Extract structured hints from a legal query to improve retrieval.
        Returns jurisdiction, source_type hints, and key legal terms.
        """
        entities = self.extract(text)
        hints = {}

        # Jurisdiction hint
        jurisdictions = entities.get("JURISDICTION", [])
        if jurisdictions:
            hints["jurisdiction"] = jurisdictions[0]

        # Source type hint based on what was found
        if entities.get("STATUTE"):
            hints["likely_source_type"] = "statute"
        elif entities.get("CITATION") or entities.get("CASE_NAME"):
            hints["likely_source_type"] = "case"

        # Key legal terms for BM25 boost
        key_terms = (
            entities.get("STATUTE", [])
            + entities.get("CITATION", [])
            + entities.get("CASE_NAME", [])
        )
        if key_terms:
            hints["key_legal_terms"] = key_terms

        return hints
