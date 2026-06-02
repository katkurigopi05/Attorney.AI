"""
Attorney.AI — Ingestion Pipeline Orchestrator
Coordinates fetching from all legal data sources and indexing into Qdrant + BM25.
"""
import asyncio
from typing import List, Optional

from loguru import logger
from tqdm.asyncio import tqdm

from config import settings
from ingestion.courtlistener import CourtListenerFetcher
from ingestion.ecfr import ECFRFetcher
from ingestion.govinfo import GovInfoFetcher
from ingestion.metadata_schema import LegalChunkMetadata
from retrieval.embedder import Embedder
from retrieval.vector_store import VectorStore
from retrieval.bm25_store import BM25Store


class IngestionPipeline:
    """
    Orchestrates the full data ingestion pipeline:
    1. Fetch from data source
    2. Chunk with legal-aware chunker
    3. Embed with OpenAI embeddings
    4. Upsert into Qdrant (vector search)
    5. Index into BM25 store (keyword search)
    """

    def __init__(self):
        self.embedder = Embedder()
        self.vector_store = VectorStore()
        self.bm25_store = BM25Store()

    async def index_chunks(
        self,
        chunks: List[LegalChunkMetadata],
        batch_size: int = 50,
    ) -> int:
        """Embed and index a batch of chunks. Returns count indexed."""
        total = 0
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            texts = [c.text for c in batch]
            embeddings = await self.embedder.embed_batch(texts)

            await self.vector_store.upsert(chunks=batch, embeddings=embeddings)
            self.bm25_store.add_chunks(batch)

            total += len(batch)
            logger.debug(f"Indexed batch {i // batch_size + 1}: {len(batch)} chunks")

        return total

    async def run_courtlistener(
        self,
        courts: Optional[List[str]] = None,
        after_date: Optional[str] = None,
        page_limit: int = 5,
    ):
        """Ingest case law from CourtListener."""
        if courts is None:
            courts = ["scotus", "ca9", "ca2", "ca5"]  # Default: key federal courts

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
                        indexed = await self.index_chunks(buffer)
                        total += indexed
                        buffer = []
                        logger.info(f"CourtListener: {total} chunks indexed so far")

        if buffer:
            total += await self.index_chunks(buffer)

        logger.info(f"CourtListener ingestion complete: {total} chunks total")
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

        logger.info(f"eCFR ingestion complete: {total} chunks total")
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

        logger.info(f"GovInfo ingestion complete: {total} chunks total")
        return total

    async def run_all(self, page_limit: int = 3):
        """Run full ingestion pipeline for all sources."""
        results = await asyncio.gather(
            self.run_courtlistener(page_limit=page_limit),
            self.run_ecfr(titles=[1, 2, 3, 4, 5]),
            self.run_govinfo(query="constitution"),
            return_exceptions=True,
        )
        for source, result in zip(["CourtListener", "eCFR", "GovInfo"], results):
            if isinstance(result, Exception):
                logger.error(f"{source} ingestion failed: {result}")
            else:
                logger.info(f"{source}: {result} chunks indexed")
