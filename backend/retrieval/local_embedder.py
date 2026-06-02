"""
Attorney.AI — Local Embedding Model (HuggingFace Transformers)
Supports domain-specific legal embeddings without OpenAI API calls.

Recommended models (in priority order):
  1. BAAI/bge-large-en-v1.5       — best general quality, 1024-dim
  2. BAAI/bge-small-en-v1.5       — fast, lightweight, 384-dim
  3. nlpaueb/legal-bert-base-uncased — legal-domain BERT, 768-dim
  4. sentence-transformers/all-MiniLM-L6-v2 — smallest/fastest, 384-dim

Usage: Set EMBEDDING_BACKEND=local in .env to use this instead of OpenAI.
"""
import asyncio
from typing import List

import numpy as np
from loguru import logger

# Lazy import — only load model when needed to avoid startup delay
_model = None
_model_name = None


def _get_model(model_name: str):
    """Lazy-load the sentence transformer model."""
    global _model, _model_name
    if _model is None or _model_name != model_name:
        from sentence_transformers import SentenceTransformer
        logger.info(f"Loading local embedding model: {model_name}")
        _model = SentenceTransformer(model_name)
        _model_name = model_name
        dim = _model.get_sentence_embedding_dimension()
        logger.info(f"Model loaded: {model_name} (dim={dim})")
    return _model


class LocalEmbedder:
    """
    Local CPU/GPU embedding model using sentence-transformers.
    Drop-in replacement for the OpenAI Embedder.

    Advantages over OpenAI:
    - No API cost or rate limits
    - Legal-domain models (legal-bert) available
    - Runs fully offline / air-gapped
    - Lower latency for large batch ingestion

    Disadvantages:
    - Lower quality than text-embedding-3-large for general text
    - Requires local compute (CPU works, GPU much faster)
    """

    # Model registry with their embedding dimensions
    MODEL_REGISTRY = {
        "bge-large": ("BAAI/bge-large-en-v1.5", 1024),
        "bge-small": ("BAAI/bge-small-en-v1.5", 384),
        "legal-bert": ("nlpaueb/legal-bert-base-uncased", 768),
        "minilm": ("sentence-transformers/all-MiniLM-L6-v2", 384),
        "inlegal-bert": ("law-ai/InLegalBERT", 768),  # India-focused
    }

    def __init__(self, model_key: str = "bge-large"):
        if model_key in self.MODEL_REGISTRY:
            self.model_name, self.dimension = self.MODEL_REGISTRY[model_key]
        else:
            # Treat as a full HuggingFace model ID
            self.model_name = model_key
            self.dimension = 768  # Default, will be overridden after load
        self._model = None

    def _load(self):
        if self._model is None:
            self._model = _get_model(self.model_name)
            self.dimension = self._model.get_sentence_embedding_dimension()
        return self._model

    def embed_single_sync(self, text: str) -> List[float]:
        """Synchronous single embed."""
        model = self._load()
        text = text.replace("\n", " ").strip() or " "
        vec = model.encode(text, normalize_embeddings=True, show_progress_bar=False)
        return vec.tolist()

    def embed_batch_sync(
        self,
        texts: List[str],
        batch_size: int = 32,
        show_progress: bool = True,
    ) -> List[List[float]]:
        """
        Synchronous batch embed. More efficient than calling embed_single in a loop.
        Uses sentence-transformers' native batching with optional GPU acceleration.
        """
        model = self._load()
        cleaned = [t.replace("\n", " ").strip() or " " for t in texts]
        vecs = model.encode(
            cleaned,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
        )
        return vecs.tolist()

    async def embed_single(self, text: str) -> List[float]:
        """Async wrapper for embed_single_sync (runs in thread pool)."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.embed_single_sync, text)

    async def embed_batch(
        self, texts: List[str], batch_size: int = 32
    ) -> List[List[float]]:
        """Async wrapper for embed_batch_sync."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self.embed_batch_sync, texts, batch_size, False
        )

    def get_model_info(self) -> dict:
        """Return info about the loaded model."""
        return {
            "model_name": self.model_name,
            "dimension": self.dimension,
            "backend": "local (sentence-transformers)",
        }
