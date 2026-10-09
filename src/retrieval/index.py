from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional

from chunking import RecursiveChunker
from embedding.embedder import SimpleEmbedder
from ingestion.loader import Document

from tokenization import tokenize

from .bm25 import BM25
from .dense import cosine_similarity
from .hybrid import combine_scores, lexical_coverage, max_normalize

LEXICAL_SCORERS = ("coverage", "bm25")


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

    def __init__(
        self,
        embedder=None,
        chunker=None,
        alpha: float = 0.5,
        lexical: str = "bm25",
        stem_tokens: bool = True,
    ):
        if lexical not in LEXICAL_SCORERS:
            raise ValueError(f"lexical must be one of {LEXICAL_SCORERS}")
        self.embedder = embedder or SimpleEmbedder()
        self.chunker = chunker or RecursiveChunker(chunk_size=300, overlap=50)
        self.alpha = alpha
        self.lexical = lexical
        self.stem_tokens = stem_tokens
        self._chunks: list = []
        self._vectors: List[List[float]] = []
        self._bm25: Optional[BM25] = None

    def __len__(self) -> int:
        return len(self._chunks)

    def add_documents(self, documents: Iterable[Document]) -> None:
        new_chunks = [chunk for document in documents for chunk in self.chunker.chunk(document.content, source=document.source)]
        self._chunks.extend(new_chunks)
        self._vectors.extend(self.embedder.encode_many([chunk.text for chunk in new_chunks]))
        self._bm25 = None  # IDF depends on the whole corpus, so rebuild lazily on next search.

    def _lexical_scores(self, query: str, texts: List[str]) -> List[float]:
        if self.lexical == "bm25":
            if self._bm25 is None:
                self._bm25 = BM25(texts, stem_tokens=self.stem_tokens)
            # BM25 is unbounded; scale to [0, 1] so it can be blended with cosine.
            return max_normalize(self._bm25.scores(query))
        query_terms = tokenize(query, stem_tokens=self.stem_tokens)
        return [lexical_coverage(query_terms, text, self.stem_tokens) for text in texts]

    def search(self, query: str, top_k: int = 3, alpha: Optional[float] = None) -> List[SearchResult]:
        if not self._chunks:
            return []
        weight = self.alpha if alpha is None else alpha
        texts = [chunk.text for chunk in self._chunks]
        query_vector = self.embedder.encode(query)
        dense = [cosine_similarity(query_vector, vector) for vector in self._vectors]
        scores = combine_scores(dense, self._lexical_scores(query, texts), weight)
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
