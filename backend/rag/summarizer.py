"""
Attorney.AI — Legal Summarizer (HuggingFace Transformers)

Generates structured legal summaries (case briefs, statute summaries)
using long-context transformer models.

Models:
  - nsi319/legal-led-base-16384   — Legal LED, 16K token context, legal-domain
  - facebook/bart-large-cnn       — Strong general summarization (CNN/DM)
  - sshleifer/distilbart-cnn-12-6 — Fast/lightweight fallback

Legal-LED (Longformer Encoder-Decoder) was specifically trained for
legal document summarization and handles long case opinions well.

Output formats:
  - BRIEF    → structured case brief (Facts/Issue/Holding/Reasoning/Disposition)
  - SUMMARY  → plain paragraph summary
  - BULLET   → bullet-point key holdings
"""
from enum import Enum
from functools import lru_cache
from typing import Optional

from loguru import logger


class SummaryFormat(str, Enum):
    BRIEF = "brief"       # Structured case brief
    SUMMARY = "summary"   # Plain paragraph
    BULLET = "bullet"     # Key bullet points


@lru_cache(maxsize=1)
def _get_summarizer(model_name: str = "nsi319/legal-led-base-16384"):
    """Lazy-load the summarization pipeline."""
    try:
        from transformers import pipeline
        logger.info(f"Loading summarizer: {model_name}")
        summarizer = pipeline(
            "summarization",
            model=model_name,
            device=-1,  # CPU; set to 0 for GPU
            max_length=512,
            min_length=64,
            truncation=True,
        )
        logger.info(f"Summarizer loaded: {model_name}")
        return summarizer
    except Exception as e:
        logger.warning(f"Primary summarizer failed ({e}), trying BART-CNN")
        try:
            from transformers import pipeline
            return pipeline(
                "summarization",
                model="sshleifer/distilbart-cnn-12-6",
                device=-1,
                max_length=256,
                min_length=40,
                truncation=True,
            )
        except Exception as e2:
            logger.error(f"All summarizers failed: {e2}")
            return None


class LegalSummarizer:
    """
    Summarizes legal documents using transformer models.

    Usage:
        summarizer = LegalSummarizer()
        brief = summarizer.summarize(case_text, format=SummaryFormat.BRIEF)
    """

    def __init__(self, model_name: str = "nsi319/legal-led-base-16384"):
        self.model_name = model_name

    def summarize(
        self,
        text: str,
        format: SummaryFormat = SummaryFormat.SUMMARY,
        max_length: int = 512,
        min_length: int = 64,
    ) -> dict:
        """
        Summarize a legal document.

        Args:
            text: The legal text to summarize
            format: Output format (BRIEF, SUMMARY, BULLET)
            max_length: Maximum summary token length
            min_length: Minimum summary token length

        Returns:
            {
              "summary": str,
              "format": str,
              "model": str,
              "input_length": int,
              "truncated": bool
            }
        """
        if not text or len(text.strip()) < 100:
            return {
                "summary": "Insufficient text for summarization.",
                "format": format,
                "model": self.model_name,
                "input_length": len(text),
                "truncated": False,
            }

        # Legal-LED handles 16K tokens; BART handles ~1024 tokens
        max_input_chars = 12000 if "legal-led" in self.model_name else 3000
        truncated = len(text) > max_input_chars
        input_text = text[:max_input_chars]

        if format == SummaryFormat.BRIEF:
            input_text = self._add_brief_prefix(input_text)

        summarizer = _get_summarizer(self.model_name)
        if not summarizer:
            # Fallback: extractive truncation
            sentences = text.split(". ")[:10]
            return {
                "summary": ". ".join(sentences) + ".",
                "format": format,
                "model": "extractive-fallback",
                "input_length": len(text),
                "truncated": truncated,
            }

        try:
            result = summarizer(
                input_text,
                max_length=max_length,
                min_length=min_length,
                do_sample=False,
            )
            raw_summary = result[0]["summary_text"].strip()

            # Post-process by format
            if format == SummaryFormat.BRIEF:
                summary = self._format_as_brief(raw_summary)
            elif format == SummaryFormat.BULLET:
                summary = self._format_as_bullets(raw_summary)
            else:
                summary = raw_summary

            return {
                "summary": summary,
                "format": format,
                "model": self.model_name,
                "input_length": len(text),
                "truncated": truncated,
            }

        except Exception as e:
            logger.error(f"Summarization error: {e}")
            return {
                "summary": f"Summarization failed: {e}",
                "format": format,
                "model": self.model_name,
                "input_length": len(text),
                "truncated": truncated,
            }

    def _add_brief_prefix(self, text: str) -> str:
        """Prefix text to guide model toward case brief structure."""
        return (
            "Summarize this legal case as a structured brief covering: "
            "Facts, Issue, Holding, Reasoning, Disposition.\n\n" + text
        )

    def _format_as_brief(self, raw: str) -> str:
        """Post-process raw summary into structured brief sections."""
        sections = ["Facts:", "Issue:", "Holding:", "Reasoning:", "Disposition:"]
        for section in sections:
            if section.lower() not in raw.lower():
                # Insert section header before first relevant sentence
                continue
        # If model didn't produce structured output, wrap it
        if not any(s.lower() in raw.lower() for s in sections):
            return (
                f"**Summary**\n{raw}\n\n"
                "_Note: Full structured brief requires longer context._"
            )
        return raw

    def _format_as_bullets(self, raw: str) -> str:
        """Convert summary sentences into bullet points."""
        sentences = [s.strip() for s in raw.split(".") if len(s.strip()) > 20]
        return "\n".join(f"• {s}." for s in sentences[:8])

    def summarize_case(self, case_text: str) -> dict:
        """Convenience method for case opinion summarization."""
        return self.summarize(case_text, format=SummaryFormat.BRIEF, max_length=600)

    def summarize_statute(self, statute_text: str) -> dict:
        """Convenience method for statute/regulation summarization."""
        return self.summarize(statute_text, format=SummaryFormat.BULLET, max_length=300)
