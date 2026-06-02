"""
Attorney.AI — Ingestion Pipeline (Supabase backend)
Fetches from CourtListener / GovInfo / eCFR, embeds locally, upserts to Supabase.
"""
import asyncio
from typing import List, Optional

from loguru import logger

from config import settings
from ingestion.courtlistener import CourtListenerFetcher
from ingestion.ecfr import ECFRFetcher
from ingestion.govinfo import GovInfoFetcher
from ingestion.metadata_schema import LegalChunkMetadata
from retrieval.local_embedder import LocalEmbedder
from retrieval.supabase_store import SupabaseVectorStore


class IngestionPipeline:
    """
    Orchestrates the full data ingestion pipeline:
    1. Fetch + chunk legal documents from APIs
    2. Embed chunks locally (BGE-large, no API cost)
    3. Upsert into Supabase (pgvector + FTS auto-indexed)
    """

    def __init__(self):
        self.embedder = LocalEmbedder(model_key=settings.local_embedding_model)
        self.store = SupabaseVectorStore()

    async def index_chunks(
        self,
        chunks: List[LegalChunkMetadata],
        batch_size: int = 50,
    ) -> int:
        """Embed and upsert a batch of chunks. Returns count indexed."""
        total = 0
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i: i + batch_size]
            texts = [c.text for c in batch]

            # Local embedding — runs on CPU, no API needed
            embeddings = await self.embedder.embed_batch(texts, batch_size=32)

            # Upsert to Supabase (idempotent ON CONFLICT chunk_id)
            indexed = self.store.upsert(chunks=batch, embeddings=embeddings)
            total += indexed
            logger.info(f"Indexed {total} chunks so far...")

        return total

    async def run_courtlistener(
        self,
        courts: Optional[List[str]] = None,
        after_date: Optional[str] = None,
        page_limit: int = 5,
    ):
        """Ingest case law from CourtListener."""
        if courts is None:
            courts = ["scotus", "ca9", "ca2", "ca5"]

        logger.info(f"Starting CourtListener ingestion: courts={courts}")
        total = 0
        buffer: List[LegalChunkMetadata] = []

        async with CourtListenerFetcher() as fetcher:
            for court in courts:
                async for chunk in fetcher.stream_chunks(
                    court=court,
                    after_date=after_date,
                    page_limit=page_limit,
                    chunk_size=settings.chunk_size,
                    chunk_overlap=settings.chunk_overlap,
                ):
                    buffer.append(chunk)
                    if len(buffer) >= 100:
                        total += await self.index_chunks(buffer)
                        buffer = []

        if buffer:
            total += await self.index_chunks(buffer)

        logger.info(f"CourtListener ingestion complete: {total} chunks")
        return total

    async def run_ecfr(
        self,
        titles: Optional[List[int]] = None,
        parts: Optional[List[int]] = None,
    ):
        """Ingest current federal regulations from eCFR."""
        logger.info("Starting eCFR ingestion")
        total = 0
        buffer: List[LegalChunkMetadata] = []

        async with ECFRFetcher() as fetcher:
            async for chunk in fetcher.stream_chunks(
                titles=titles,
                parts=parts,
                chunk_size=settings.chunk_size,
                chunk_overlap=settings.chunk_overlap,
            ):
                buffer.append(chunk)
                if len(buffer) >= 100:
                    total += await self.index_chunks(buffer)
                    buffer = []

        if buffer:
            total += await self.index_chunks(buffer)

        logger.info(f"eCFR ingestion complete: {total} chunks")
        return total

    async def run_govinfo(
        self,
        uscode_titles: Optional[List[int]] = None,
        query: str = "",
    ):
        """Ingest U.S. Code from GovInfo."""
        logger.info("Starting GovInfo ingestion")
        total = 0
        buffer: List[LegalChunkMetadata] = []

        async with GovInfoFetcher() as fetcher:
            async for chunk in fetcher.stream_uscode_chunks(
                title=uscode_titles[0] if uscode_titles else None,
                query=query,
                chunk_size=settings.chunk_size,
                chunk_overlap=settings.chunk_overlap,
            ):
                buffer.append(chunk)
                if len(buffer) >= 100:
                    total += await self.index_chunks(buffer)
                    buffer = []

        if buffer:
            total += await self.index_chunks(buffer)

        logger.info(f"GovInfo ingestion complete: {total} chunks")
        return total

    def get_stats(self) -> dict:
        """Return current index statistics."""
        return self.store.get_stats()
