"""
Attorney.AI — CourtListener API Fetcher
Fetches U.S. case opinions from CourtListener's REST API v4.
https://www.courtlistener.com/api/rest/v4/
"""
import asyncio
from datetime import date
from typing import AsyncGenerator, Dict, List, Optional

import httpx
from loguru import logger

from config import settings
from ingestion.chunker import chunk_legal_document
from ingestion.metadata_schema import LegalChunkMetadata, SourceType


COURT_LEVEL_MAP = {
    "scotus": "Supreme",
    "ca1": "Circuit", "ca2": "Circuit", "ca3": "Circuit",
    "ca4": "Circuit", "ca5": "Circuit", "ca6": "Circuit",
    "ca7": "Circuit", "ca8": "Circuit", "ca9": "Circuit",
    "ca10": "Circuit", "ca11": "Circuit", "cadc": "Circuit",
    "cafc": "Circuit",
}


class CourtListenerFetcher:
    """
    Asynchronous fetcher for CourtListener opinions.

    Usage:
        fetcher = CourtListenerFetcher()
        async for chunk in fetcher.stream_chunks(jurisdiction="scotus", page_limit=5):
            # index chunk into vector DB
    """

    def __init__(self):
        headers = {
            "Accept": "application/json",
        }
        if settings.courtlistener_api_key:
            headers["Authorization"] = f"Token {settings.courtlistener_api_key}"

        self.client = httpx.AsyncClient(
            base_url=settings.courtlistener_base_url,
            headers=headers,
            timeout=30.0,
        )

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        await self.client.aclose()

    async def fetch_opinions_page(
        self,
        court: Optional[str] = None,
        after_date: Optional[str] = None,
        before_date: Optional[str] = None,
        page_size: int = 20,
        cursor: Optional[str] = None,
    ) -> Dict:
        """Fetch one page of opinions from CourtListener."""
        params: Dict = {
            "format": "json",
            "page_size": page_size,
            "order_by": "-date_filed",
        }
        if court:
            params["court"] = court
        if after_date:
            params["date_filed__gte"] = after_date
        if before_date:
            params["date_filed__lte"] = before_date
        if cursor:
            params["cursor"] = cursor

        resp = await self.client.get("/opinions/", params=params)
        resp.raise_for_status()
        return resp.json()

    async def fetch_opinion_text(self, opinion_id: int) -> str:
        """Fetch the full plain text of a single opinion."""
        resp = await self.client.get(f"/opinions/{opinion_id}/")
        resp.raise_for_status()
        data = resp.json()
        # CourtListener returns text in multiple formats; prefer plain_text
        text = (
            data.get("plain_text")
            or data.get("html_with_citations")
            or data.get("html")
            or ""
        )
        # Strip HTML tags if needed
        if text.startswith("<"):
            import html2text
            h2t = html2text.HTML2Text()
            h2t.ignore_links = False
            text = h2t.handle(text)
        return text.strip()

    async def fetch_cluster(self, cluster_id: int) -> Dict:
        """Fetch the OpinionCluster (case-level metadata)."""
        resp = await self.client.get(f"/clusters/{cluster_id}/")
        resp.raise_for_status()
        return resp.json()

    async def stream_chunks(
        self,
        court: Optional[str] = None,
        after_date: Optional[str] = None,
        before_date: Optional[str] = None,
        page_limit: int = 10,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
    ) -> AsyncGenerator[LegalChunkMetadata, None]:
        """
        Stream LegalChunkMetadata objects for all opinions matching filters.
        Automatically paginates up to page_limit pages.
        """
        cursor = None
        pages_fetched = 0

        while pages_fetched < page_limit:
            try:
                page = await self.fetch_opinions_page(
                    court=court,
                    after_date=after_date,
                    before_date=before_date,
                    cursor=cursor,
                )
            except httpx.HTTPStatusError as e:
                logger.error(f"CourtListener API error: {e.response.status_code} — {e}")
                break

            results = page.get("results", [])
            logger.info(
                f"CourtListener page {pages_fetched + 1}: "
                f"{len(results)} opinions (court={court})"
            )

            for opinion in results:
                opinion_id = opinion.get("id")
                cluster_url = opinion.get("cluster", "")

                try:
                    # Extract cluster ID from URL
                    cluster_id = int(cluster_url.rstrip("/").split("/")[-1])
                    cluster = await self.fetch_cluster(cluster_id)
                    text = await self.fetch_opinion_text(opinion_id)
                except Exception as e:
                    logger.warning(f"Skipping opinion {opinion_id}: {e}")
                    continue

                if not text or len(text) < 100:
                    logger.debug(f"Skipping empty opinion {opinion_id}")
                    continue

                # Build citation string
                citations = cluster.get("citations", [])
                citation_str = (
                    citations[0].get("cite", "") if citations else f"Opinion #{opinion_id}"
                )

                # Parse date
                date_filed = cluster.get("date_filed", "")
                try:
                    decision_date = date.fromisoformat(date_filed) if date_filed else None
                except ValueError:
                    decision_date = None

                court_id = opinion.get("court_id", court or "")
                court_level = COURT_LEVEL_MAP.get(court_id, "Unknown")

                metadata_base = {
                    "doc_id": f"cl_opinion_{opinion_id}",
                    "title": cluster.get("case_name", f"Opinion {opinion_id}"),
                    "citation": citation_str,
                    "source_url": f"https://www.courtlistener.com{cluster.get('absolute_url', '')}",
                    "jurisdiction": "US-Federal",
                    "court_or_agency": cluster.get("docket", {}).get("court_id", court_id).upper(),
                    "decision_date": decision_date,
                    "date_str": date_filed,
                    "source_type": SourceType.CASE,
                    "court_level": court_level,
                    "docket_number": cluster.get("docket_number"),
                    "author_judge": opinion.get("author_str"),
                }

                chunks = chunk_legal_document(
                    text=text,
                    metadata_base=metadata_base,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                )
                for chunk in chunks:
                    yield chunk

                # Be polite to the API
                await asyncio.sleep(0.2)

            # Paginate
            next_cursor = page.get("next")
            if not next_cursor:
                break
            # Extract cursor from next URL
            if "cursor=" in next_cursor:
                cursor = next_cursor.split("cursor=")[-1].split("&")[0]
            else:
                break

            pages_fetched += 1
