"""
Attorney.AI — Qdrant Vector Store Wrapper
Manages collection creation, upsert, and filtered vector search.
"""
import uuid
from typing import Dict, List, Optional, Tuple

from loguru import logger
from qdrant_client import AsyncQdrantClient, models
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    Range,
    VectorParams,
)

from config import settings
from ingestion.metadata_schema import LegalChunkMetadata


class VectorStore:
    """
    Qdrant vector store for legal chunks.
    Supports upsert, similarity search with metadata filters.
    """

    def __init__(self):
        kwargs = {"url": settings.qdrant_url}
        if settings.qdrant_api_key:
            kwargs["api_key"] = settings.qdrant_api_key
        self.client = AsyncQdrantClient(**kwargs)
        self.collection = settings.qdrant_collection
        # Infer dimension from model name
        self.dimension = 3072 if "3-large" in settings.openai_embedding_model else 1536

    async def ensure_collection(self):
        """Create the Qdrant collection if it doesn't exist."""
        existing = await self.client.get_collections()
        names = [c.name for c in existing.collections]
        if self.collection not in names:
            await self.client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(
                    size=self.dimension,
                    distance=Distance.COSINE,
                ),
            )
            # Create payload indexes for fast metadata filtering
            for field in ["jurisdiction", "source_type", "court_level", "court_or_agency"]:
                await self.client.create_payload_index(
                    collection_name=self.collection,
                    field_name=field,
                    field_schema=models.PayloadSchemaType.KEYWORD,
                )
            # Date range index
            await self.client.create_payload_index(
                collection_name=self.collection,
                field_name="decision_date",
                field_schema=models.PayloadSchemaType.KEYWORD,
            )
            logger.info(f"Created Qdrant collection: {self.collection} (dim={self.dimension})")
        else:
            logger.debug(f"Qdrant collection '{self.collection}' already exists")

    async def upsert(
        self,
        chunks: List[LegalChunkMetadata],
        embeddings: List[List[float]],
    ):
        """Upsert chunks with their embeddings into Qdrant."""
        await self.ensure_collection()
        points = []
        for chunk, embedding in zip(chunks, embeddings):
            # Use chunk_id as a deterministic UUID
            point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk.chunk_id))
            points.append(
                PointStruct(
                    id=point_id,
                    vector=embedding,
                    payload=chunk.to_qdrant_payload(),
                )
            )
        if points:
            await self.client.upsert(collection_name=self.collection, points=points)
            logger.debug(f"Upserted {len(points)} points into Qdrant")

    async def search(
        self,
        query_vector: List[float],
        top_k: int = 20,
        filters: Optional[Dict] = None,
    ) -> List[Tuple[LegalChunkMetadata, float]]:
        """
        Perform filtered vector similarity search.
        Returns list of (chunk, score) sorted by descending score.
        """
        qdrant_filter = _build_qdrant_filter(filters) if filters else None

        results = await self.client.search(
            collection_name=self.collection,
            query_vector=query_vector,
            query_filter=qdrant_filter,
            limit=top_k,
            with_payload=True,
        )

        output = []
        for hit in results:
            try:
                chunk = LegalChunkMetadata.from_qdrant_payload(hit.payload)
                output.append((chunk, hit.score))
            except Exception as e:
                logger.warning(f"Failed to deserialize Qdrant hit: {e}")
        return output

    async def get_collection_stats(self) -> dict:
        """Return collection info."""
        info = await self.client.get_collection(self.collection)
        return {
            "name": self.collection,
            "points_count": info.points_count,
            "dimension": self.dimension,
            "status": str(info.status),
        }


def _build_qdrant_filter(filters: Dict) -> Filter:
    """
    Convert a filter dict to a Qdrant Filter object.
    Supported keys: jurisdiction, source_type, court_level, court_or_agency
    """
    must_conditions = []

    for key in ["jurisdiction", "source_type", "court_level", "court_or_agency"]:
        val = filters.get(key)
        if val:
            must_conditions.append(
                FieldCondition(key=key, match=MatchValue(value=val))
            )

    if must_conditions:
        return Filter(must=must_conditions)
    return None
