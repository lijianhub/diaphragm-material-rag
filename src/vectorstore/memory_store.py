from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from .base import ChunkRecord, VectorHit, VectorStore, l2_normalize

VECTORS_FILE = "vectors.json"


class InMemoryVectorStore(VectorStore):
    """Exact brute-force search in pure Python.

    Every query is compared with every vector: O(N x dim) per search. That is
    exact and fine for thousands of chunks, and it is the reference that
    approximate stores are checked against.
    """

    kind = "memory"

    def __init__(self, dim: Optional[int] = None):
        super().__init__(dim)
        self._vectors: Dict[str, List[float]] = {}

    def add(self, records: Sequence[ChunkRecord], vectors: Sequence[Sequence[float]]) -> None:
        self._validate_new(records, vectors)
        for record, vector in zip(records, vectors):
            self._records[record.id] = record
            self._vectors[record.id] = l2_normalize(vector)

    def delete_document(self, doc_id: str) -> int:
        doomed = [chunk_id for chunk_id, record in self._records.items() if record.doc_id == doc_id]
        for chunk_id in doomed:
            del self._records[chunk_id]
            del self._vectors[chunk_id]
        return len(doomed)

    def search(self, query_vector: Sequence[float], top_k: int) -> List[VectorHit]:
        if not self._records:
            return []
        self._check_dim(query_vector)
        query = l2_normalize(query_vector)
        hits = [
            VectorHit(id=chunk_id, score=sum(q * v for q, v in zip(query, vector)))
            for chunk_id, vector in self._vectors.items()
        ]
        # Stable sort: equal scores keep insertion order, matching the FAISS store.
        hits.sort(key=lambda hit: hit.score, reverse=True)
        return hits[:top_k]

    def _save_vectors(self, directory: Path) -> None:
        (directory / VECTORS_FILE).write_text(json.dumps(self._vectors), encoding="utf-8")

    @classmethod
    def _load(cls, directory: Path, dim: Optional[int], records: List[ChunkRecord]) -> "InMemoryVectorStore":
        store = cls(dim)
        vectors = json.loads((directory / VECTORS_FILE).read_text(encoding="utf-8"))
        for record in records:
            store._records[record.id] = record
            store._vectors[record.id] = vectors[record.id]
        return store
