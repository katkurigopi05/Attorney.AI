"""
Attorney.AI — BM25 Keyword Search Store
In-memory BM25 index for fast keyword retrieval.
Can be swapped for Elasticsearch for production scale.
"""
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from loguru import logger
from rank_bm25 import BM25Okapi

from ingestion.metadata_schema import LegalChunkMetadata


_INDEX_PATH = Path("./data/bm25_index.pkl")


class BM25Store:
    """
    BM25 keyword search over legal chunks.
    Stores all chunk texts in memory; persists to disk via pickle.
    For production: replace with Elasticsearch for scale.
    """

    def __init__(self):
        self.chunks: List[LegalChunkMetadata] = []
        self.corpus: List[List[str]] = []  # tokenized texts
        self._bm25: Optional[BM25Okapi] = None

    def _tokenize(self, text: str) -> List[str]:
        """Simple whitespace tokenizer; lowercase, strip punctuation."""
        import re
        text = text.lower()
        text = re.sub(r"[^\w\s]", " ", text)
        return text.split()

    def add_chunks(self, chunks: List[LegalChunkMetadata]):
        """Add chunks to the BM25 index."""
        for chunk in chunks:
            self.chunks.append(chunk)
            self.corpus.append(self._tokenize(chunk.text))
        self._bm25 = None  # Invalidate index; rebuild on next search

    def _ensure_index(self):
        """Build/rebuild the BM25 index if needed."""
        if self._bm25 is None and self.corpus:
            self._bm25 = BM25Okapi(self.corpus)

    def search(
        self,
        query: str,
        top_k: int = 20,
        filters: Optional[Dict] = None,
    ) -> List[Tuple[LegalChunkMetadata, float]]:
        """
        BM25 keyword search with optional metadata filters.
        Returns list of (chunk, bm25_score).
        """
        if not self.corpus:
            logger.warning("BM25 index is empty — no documents indexed yet")
            return []

        self._ensure_index()
        tokens = self._tokenize(query)
        scores = self._bm25.get_scores(tokens)

        # Combine scores with chunk indices
        indexed = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)

        results = []
        for idx, score in indexed:
            if len(results) >= top_k:
                break
            if score <= 0:
                break
            chunk = self.chunks[idx]
            # Apply metadata filters
            if filters and not _matches_filters(chunk, filters):
                continue
            results.append((chunk, float(score)))

        return results

    def save(self, path: Optional[Path] = None):
        """Persist the BM25 index to disk."""
        save_path = path or _INDEX_PATH
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "wb") as f:
            pickle.dump({"chunks": self.chunks, "corpus": self.corpus}, f)
        logger.info(f"BM25 index saved: {len(self.chunks)} chunks → {save_path}")

    def load(self, path: Optional[Path] = None) -> bool:
        """Load a persisted BM25 index from disk."""
        load_path = path or _INDEX_PATH
        if not load_path.exists():
            logger.warning(f"BM25 index not found at {load_path}")
            return False
        with open(load_path, "rb") as f:
            data = pickle.load(f)
        self.chunks = data["chunks"]
        self.corpus = data["corpus"]
        self._bm25 = None
        self._ensure_index()
        logger.info(f"BM25 index loaded: {len(self.chunks)} chunks from {load_path}")
        return True


def _matches_filters(chunk: LegalChunkMetadata, filters: Dict) -> bool:
    """Check if a chunk matches the given metadata filters."""
    for key, value in filters.items():
        if not value:
            continue
        chunk_val = getattr(chunk, key, None)
        if chunk_val and chunk_val.lower() != value.lower():
            return False
    return True
