"""
Attorney.AI — Embedding Model Wrapper
Supports OpenAI text-embedding-3-large with async batching.
"""
import asyncio
from typing import List

import numpy as np
from loguru import logger
from openai import AsyncOpenAI

from config import settings


class Embedder:
    """
    Wraps OpenAI embedding API with async batch processing.
    Produces 3072-dim vectors for text-embedding-3-large.
    """

    def __init__(self):
        self.client = AsyncOpenAI(api_key=settings.openai_api_key)
        self.model = settings.openai_embedding_model
        self.dimension = 3072 if "3-large" in self.model else 1536

    async def embed_single(self, text: str) -> List[float]:
        """Embed a single text string."""
        text = text.replace("\n", " ").strip()
        if not text:
            return [0.0] * self.dimension
        resp = await self.client.embeddings.create(
            model=self.model,
            input=text,
        )
        return resp.data[0].embedding

    async def embed_batch(
        self, texts: List[str], batch_size: int = 100
    ) -> List[List[float]]:
        """
        Embed a list of texts efficiently.
        OpenAI allows up to 2048 inputs per request; we batch by token safety.
        """
        all_embeddings: List[List[float]] = []
        cleaned = [t.replace("\n", " ").strip() or " " for t in texts]

        for i in range(0, len(cleaned), batch_size):
            batch = cleaned[i : i + batch_size]
            try:
                resp = await self.client.embeddings.create(
                    model=self.model,
                    input=batch,
                )
                embeddings = [item.embedding for item in sorted(resp.data, key=lambda x: x.index)]
                all_embeddings.extend(embeddings)
            except Exception as e:
                logger.error(f"Embedding batch {i // batch_size} failed: {e}")
                # Fall back to zeros for failed batch
                all_embeddings.extend([[0.0] * self.dimension] * len(batch))

        return all_embeddings

    def embed_batch_sync(self, texts: List[str]) -> List[List[float]]:
        """Synchronous wrapper for embed_batch."""
        return asyncio.run(self.embed_batch(texts))
