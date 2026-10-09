from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from .base import ChunkRecord, VectorHit, VectorStore, l2_normalize

INDEX_FILE = "index.faiss"
ID_MAP_FILE = "faiss_ids.json"


def _require_faiss():
    try:
        import faiss
        import numpy as np
    except ImportError as exc:  # pragma: no cover - exercised only without the optional extra
        raise ImportError("FaissVectorStore needs the optional extra: pip install faiss-cpu numpy") from exc
    return faiss, np


class FaissVectorStore(VectorStore):
    """FAISS-backed store using an exact inner-product index over unit vectors.

    ``IndexFlatIP`` on L2-normalised vectors is exact cosine search, so results
    match ``InMemoryVectorStore``; the gain is speed (SIMD, batched BLAS) and a
    compact binary file. ``IndexIDMap2`` maps FAISS's int64 ids to our string ids
    and supports deletes. For millions of vectors, swap the inner index for an
    approximate one (HNSW or IVF) and re-check recall on the benchmark.
    """

    kind = "faiss"

    def __init__(self, dim: Optional[int] = None):
        super().__init__(dim)
        self._faiss, self._np = _require_faiss()
        self._index = None
        self._int_ids: Dict[str, int] = {}
        self._str_ids: Dict[int, str] = {}
        self._next_id = 0

    def _ensure_index(self) -> None:
        if self._index is None and self.dim is not None:
            self._index = self._faiss.IndexIDMap2(self._faiss.IndexFlatIP(self.dim))

    def _matrix(self, vectors: Sequence[Sequence[float]]):
        return self._np.asarray([l2_normalize(v) for v in vectors], dtype="float32")

    def add(self, records: Sequence[ChunkRecord], vectors: Sequence[Sequence[float]]) -> None:
        self._validate_new(records, vectors)
        if not records:
            return
        self._ensure_index()
        ids = []
        for record in records:
            self._records[record.id] = record
            self._int_ids[record.id] = self._next_id
            self._str_ids[self._next_id] = record.id
            ids.append(self._next_id)
            self._next_id += 1
        self._index.add_with_ids(self._matrix(vectors), self._np.asarray(ids, dtype="int64"))

    def delete_document(self, doc_id: str) -> int:
        doomed = [chunk_id for chunk_id, record in self._records.items() if record.doc_id == doc_id]
        if doomed:
            int_ids = [self._int_ids.pop(chunk_id) for chunk_id in doomed]
            self._index.remove_ids(self._np.asarray(int_ids, dtype="int64"))
            for chunk_id, int_id in zip(doomed, int_ids):
                del self._records[chunk_id]
                del self._str_ids[int_id]
        return len(doomed)

    def search(self, query_vector: Sequence[float], top_k: int) -> List[VectorHit]:
        if not self._records:
            return []
        self._check_dim(query_vector)
        k = min(top_k, len(self._records))
        scores, ids = self._index.search(self._matrix([query_vector]), k)
        hits = [VectorHit(id=self._str_ids[int(i)], score=float(s)) for s, i in zip(scores[0], ids[0]) if i != -1]
        # FAISS breaks ties arbitrarily; re-sort stably by (score, insertion id) so ties
        # resolve like the in-memory store. Round to float32 precision first.
        hits.sort(key=lambda hit: (-round(hit.score, 6), self._int_ids[hit.id]))
        return hits

    def _save_vectors(self, directory: Path) -> None:
        if self._index is not None:
            self._faiss.write_index(self._index, str(directory / INDEX_FILE))
        (directory / ID_MAP_FILE).write_text(
            json.dumps({"next_id": self._next_id, "ids": self._int_ids}), encoding="utf-8"
        )

    @classmethod
    def _load(cls, directory: Path, dim: Optional[int], records: List[ChunkRecord]) -> "FaissVectorStore":
        store = cls(dim)
        id_map = json.loads((directory / ID_MAP_FILE).read_text(encoding="utf-8"))
        store._next_id = id_map["next_id"]
        store._int_ids = {chunk_id: int(i) for chunk_id, i in id_map["ids"].items()}
        store._str_ids = {i: chunk_id for chunk_id, i in store._int_ids.items()}
        for record in records:
            store._records[record.id] = record
        if (directory / INDEX_FILE).exists():
            store._index = store._faiss.read_index(str(directory / INDEX_FILE))
        return store
