"""
Attorney.AI — eCFR API Fetcher
Fetches current U.S. federal regulations from the Electronic Code of Federal Regulations.
https://www.ecfr.gov/api/versioner/v1/

Note: eCFR is continuously updated but is NOT the official legal edition.
Production systems should verify against official CFR PDFs on GovInfo.
"""
import asyncio
from typing import AsyncGenerator, List, Optional

import httpx
from loguru import logger

from config import settings
from ingestion.chunker import chunk_legal_document
from ingestion.metadata_schema import LegalChunkMetadata, SourceType


class ECFRFetcher:
    """
    Async fetcher for current federal regulations from eCFR.
    Streams regulations by CFR Title and optional Part filter.
    """

    def __init__(self):
        self.client = httpx.AsyncClient(
            base_url=settings.ecfr_base_url,
            timeout=60.0,
            headers={"Accept": "application/json"},
        )

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        await self.client.aclose()

    async def list_titles(self) -> List[dict]:
        """List all available CFR titles."""
        resp = await self.client.get("/titles")
        resp.raise_for_status()
        return resp.json().get("titles", [])

    async def list_parts(self, title: int) -> List[dict]:
        """List all parts within a CFR title."""
        resp = await self.client.get(f"/structure/{title}/chapter")
        resp.raise_for_status()
        return resp.json().get("children", [])

    async def fetch_part_text(self, title: int, part: int) -> str:
        """Fetch the full XML text of a CFR part and convert to plain text."""
        try:
            resp = await self.client.get(
                f"/full/{title}/part-{part}",
                params={"format": "xml"},
            )
            resp.raise_for_status()
            # Parse XML to extract text content
            from lxml import etree
            root = etree.fromstring(resp.content)
            texts = root.itertext()
            return " ".join(t.strip() for t in texts if t.strip())
        except Exception as e:
            logger.debug(f"XML fetch failed for Title {title} Part {part}: {e}")
            # Try JSON format
            try:
                resp = await self.client.get(
                    f"/full/{title}/part-{part}",
                    params={"format": "json"},
                )
                resp.raise_for_status()
                data = resp.json()
                return _extract_ecfr_text(data)
            except Exception as e2:
                logger.warning(f"eCFR fetch failed Title {title} Part {part}: {e2}")
                return ""

    async def stream_chunks(
        self,
        titles: Optional[List[int]] = None,
        parts: Optional[List[int]] = None,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
    ) -> AsyncGenerator[LegalChunkMetadata, None]:
        """
        Stream regulation chunks. If titles is None, fetches from titles 1-50.
        If parts is None, fetches all parts within each title.
        """
        if titles is None:
            titles = list(range(1, 51))  # CFR titles 1–50

        for title_num in titles:
            logger.info(f"eCFR: Fetching CFR Title {title_num}")
            try:
                if parts:
                    part_numbers = parts
                else:
                    # Limit to first 5 parts per title for MVP
                    part_list = await self.list_parts(title_num)
                    part_numbers = [
                        int(p.get("identifier", 0))
                        for p in part_list[:5]
                        if p.get("identifier", "").isdigit()
                    ]
            except Exception as e:
                logger.warning(f"Could not list parts for Title {title_num}: {e}")
                continue

            for part_num in part_numbers:
                text = await self.fetch_part_text(title_num, part_num)
                if not text or len(text) < 100:
                    continue

                metadata_base = {
                    "doc_id": f"ecfr_t{title_num}_p{part_num}",
                    "title": f"CFR Title {title_num}, Part {part_num}",
                    "citation": f"{title_num} C.F.R. Part {part_num}",
                    "source_url": (
                        f"https://www.ecfr.gov/current/title-{title_num}/part-{part_num}"
                    ),
                    "jurisdiction": "US-Federal",
                    "court_or_agency": f"CFR Title {title_num}",
                    "source_type": SourceType.REGULATION,
                    "parent_section": f"Title {title_num}",
                }

                chunks = chunk_legal_document(
                    text=text,
                    metadata_base=metadata_base,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                )
                for chunk in chunks:
                    yield chunk

                await asyncio.sleep(0.5)  # Respectful rate limiting


def _extract_ecfr_text(data: dict, depth: int = 0) -> str:
    """Recursively extract text from eCFR JSON structure."""
    parts = []
    if "label" in data:
        parts.append(data["label"])
    if "subject_group" in data:
        parts.append(data["subject_group"])
    if "children" in data:
        for child in data["children"]:
            parts.append(_extract_ecfr_text(child, depth + 1))
    if "paragraphs" in data:
        for para in data["paragraphs"]:
            if isinstance(para, dict):
                parts.append(para.get("text", ""))
            elif isinstance(para, str):
                parts.append(para)
    if "content" in data and isinstance(data["content"], str):
        parts.append(data["content"])
    return "\n".join(p for p in parts if p)
