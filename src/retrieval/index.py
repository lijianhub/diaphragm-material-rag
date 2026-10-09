from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional

from chunking import RecursiveChunker
from embedding.embedder import SimpleEmbedder
from ingestion.loader import Document

from .hybrid import hybrid_scores


@dataclass
class SearchResult:
    text: str
    source: str
    start: int
    end: int
    score: float


class SearchIndex:
    """In-memory index that wires chunking, embedding and hybrid retrieval together.

    The chunker only needs a ``chunk(text, source)`` method returning objects with
    ``text, source, start, end``; the embedder only needs ``encode`` and
    ``encode_many``. Either can be swapped without touching this class.
    """

    def __init__(self, embedder=None, chunker=None, alpha: float = 0.5):
        self.embedder = embedder or SimpleEmbedder()
        self.chunker = chunker or RecursiveChunker(chunk_size=300, overlap=50)
        self.alpha = alpha
        self._chunks: list = []
        self._vectors: List[List[float]] = []

    def __len__(self) -> int:
        return len(self._chunks)

    def add_documents(self, documents: Iterable[Document]) -> None:
        new_chunks = [chunk for document in documents for chunk in self.chunker.chunk(document.content, source=document.source)]
        self._chunks.extend(new_chunks)
        self._vectors.extend(self.embedder.encode_many([chunk.text for chunk in new_chunks]))

    def search(self, query: str, top_k: int = 3, alpha: Optional[float] = None) -> List[SearchResult]:
        if not self._chunks:
            return []
        weight = self.alpha if alpha is None else alpha
        texts = [chunk.text for chunk in self._chunks]
        scores = hybrid_scores(query, self.embedder.encode(query), texts, self._vectors, weight)
        # Rank by index so duplicate chunk texts keep their own source and offsets.
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [
            SearchResult(
                text=self._chunks[i].text,
                source=self._chunks[i].source,
                start=self._chunks[i].start,
                end=self._chunks[i].end,
                score=scores[i],
            )
            for i in ranked
        ]
