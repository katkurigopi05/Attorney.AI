"""
Attorney.AI — GovInfo API Fetcher
Fetches U.S. Code, CFR, bills, and other official government documents.
https://api.govinfo.gov/docs
"""
import asyncio
from typing import AsyncGenerator, List, Optional

import httpx
from loguru import logger

from config import settings
from ingestion.chunker import chunk_legal_document
from ingestion.metadata_schema import LegalChunkMetadata, SourceType


# GovInfo collection codes
COLLECTIONS = {
    "USCODE": "United States Code",
    "CFR": "Code of Federal Regulations",
    "BILLS": "Congressional Bills",
    "FR": "Federal Register",
    "STATUTE": "U.S. Statutes at Large",
    "PLAW": "Public and Private Laws",
}


class GovInfoFetcher:
    """
    Async fetcher for GovInfo.gov official government documents.
    Supports U.S. Code, CFR, and Congressional Bills.
    """

    def __init__(self):
        self.client = httpx.AsyncClient(
            base_url=settings.govinfo_base_url,
            params={"api_key": settings.govinfo_api_key} if settings.govinfo_api_key else {},
            timeout=30.0,
            headers={"Accept": "application/json"},
        )

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        await self.client.aclose()

    async def search_collection(
        self,
        collection: str,
        query: str = "",
        page_size: int = 10,
        offset_mark: str = "*",
    ) -> dict:
        """Search a GovInfo collection."""
        params = {
            "query": query or f"collection:{collection}",
            "pageSize": page_size,
            "offsetMark": offset_mark,
            "resultLevel": "default",
        }
        resp = await self.client.get("/search", params=params)
        resp.raise_for_status()
        return resp.json()

    async def fetch_package_summary(self, package_id: str) -> dict:
        """Fetch summary metadata for a specific package."""
        resp = await self.client.get(f"/packages/{package_id}/summary")
        resp.raise_for_status()
        return resp.json()

    async def fetch_package_content(self, package_id: str) -> str:
        """Fetch the plain text content of a package."""
        # Try to get HTM then TXT granules
        for fmt in ["htm", "txt"]:
            try:
                resp = await self.client.get(
                    f"/packages/{package_id}/content",
                    params={"contentType": fmt},
                )
                if resp.status_code == 200:
                    content = resp.text
                    if fmt == "htm":
                        import html2text
                        h = html2text.HTML2Text()
                        h.ignore_links = False
                        content = h.handle(content)
                    return content.strip()
            except Exception:
                continue
        return ""

    async def stream_uscode_chunks(
        self,
        title: Optional[int] = None,
        query: str = "",
        page_limit: int = 5,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
    ) -> AsyncGenerator[LegalChunkMetadata, None]:
        """Stream U.S. Code chunks from GovInfo."""
        search_query = f"collection:USCODE"
        if title:
            search_query += f" title:{title}"
        if query:
            search_query += f" {query}"

        offset_mark = "*"
        pages = 0

        while pages < page_limit:
            try:
                result = await self.search_collection(
                    "USCODE", query=search_query, offset_mark=offset_mark
                )
            except Exception as e:
                logger.error(f"GovInfo search error: {e}")
                break

            packages = result.get("results", [])
            logger.info(f"GovInfo USCODE page {pages + 1}: {len(packages)} packages")

            for pkg in packages:
                pkg_id = pkg.get("packageId", "")
                if not pkg_id:
                    continue

                try:
                    summary = await self.fetch_package_summary(pkg_id)
                    text = await self.fetch_package_content(pkg_id)
                except Exception as e:
                    logger.warning(f"Skipping GovInfo package {pkg_id}: {e}")
                    continue

                if not text or len(text) < 50:
                    continue

                section_title = summary.get("title", pkg_id)
                date_issued = summary.get("dateIssued", "")

                metadata_base = {
                    "doc_id": f"govinfo_{pkg_id}",
                    "title": section_title,
                    "citation": summary.get("citation", section_title),
                    "source_url": f"https://www.govinfo.gov/content/pkg/{pkg_id}/html/{pkg_id}.htm",
                    "jurisdiction": "US-Federal",
                    "court_or_agency": "U.S. Congress",
                    "date_str": date_issued,
                    "source_type": SourceType.STATUTE,
                    "parent_section": summary.get("collection", "USCODE"),
                }

                chunks = chunk_legal_document(
                    text=text,
                    metadata_base=metadata_base,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                )
                for chunk in chunks:
                    yield chunk

                await asyncio.sleep(0.3)

            next_offset = result.get("nextPage")
            if not next_offset:
                break
            offset_mark = next_offset
            pages += 1
