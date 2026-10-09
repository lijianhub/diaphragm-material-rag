from __future__ import annotations

import json
import math
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

PathLike = Union[str, Path]
STORE_META_FILE = "store.json"
RECORDS_FILE = "records.jsonl"


@dataclass
class ChunkRecord:
    """One stored chunk. ``id`` is unique per chunk; ``doc_id`` groups a document's chunks."""

    id: str
    doc_id: str
    text: str
    source: str
    start: int
    end: int
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class VectorHit:
    id: str
    score: float


def l2_normalize(vector: Sequence[float]) -> List[float]:
    norm = math.sqrt(sum(float(x) * float(x) for x in vector))
    return [float(x) / norm for x in vector] if norm else [float(x) for x in vector]


class VectorStore(ABC):
    """Stores chunk records with their embeddings and finds nearest neighbours by cosine.

    Implementations normalise vectors on the way in, so similarity is the cosine of
    the angle even when the embedder does not return unit vectors. The dimension
    is fixed by the first ``add`` and enforced afterwards.
    """

    kind: str = "abstract"

    def __init__(self, dim: Optional[int] = None):
        self.dim = dim
        self._records: Dict[str, ChunkRecord] = {}  # insertion-ordered

    def __len__(self) -> int:
        return len(self._records)

    def records(self) -> List[ChunkRecord]:
        return list(self._records.values())

    def get(self, chunk_id: str) -> ChunkRecord:
        return self._records[chunk_id]

    def document_ids(self) -> List[str]:
        return list(dict.fromkeys(record.doc_id for record in self._records.values()))

    def _check_dim(self, vector: Sequence[float]) -> None:
        if self.dim is not None and len(vector) != self.dim:
            raise ValueError(f"vector has dimension {len(vector)}, store expects {self.dim}")

    def _validate_new(self, records: Sequence[ChunkRecord], vectors: Sequence[Sequence[float]]) -> None:
        if len(records) != len(vectors):
            raise ValueError("records and vectors differ in length")
        seen = set()
        for record in records:
            if record.id in self._records or record.id in seen:
                raise ValueError(f"duplicate chunk id {record.id!r}")
            seen.add(record.id)
        if vectors and self.dim is None:
            self.dim = len(vectors[0])
        for vector in vectors:
            self._check_dim(vector)

    @abstractmethod
    def add(self, records: Sequence[ChunkRecord], vectors: Sequence[Sequence[float]]) -> None: ...

    @abstractmethod
    def delete_document(self, doc_id: str) -> int:
        """Remove every chunk of ``doc_id``; return how many were removed."""

    @abstractmethod
    def search(self, query_vector: Sequence[float], top_k: int) -> List[VectorHit]: ...

    @abstractmethod
    def _save_vectors(self, directory: Path) -> None: ...

    @classmethod
    @abstractmethod
    def _load(cls, directory: Path, dim: Optional[int], records: List[ChunkRecord]) -> "VectorStore": ...

    def save(self, directory: PathLike) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / STORE_META_FILE).write_text(
            json.dumps({"kind": self.kind, "dim": self.dim, "count": len(self)}, indent=2), encoding="utf-8"
        )
        with (directory / RECORDS_FILE).open("w", encoding="utf-8") as handle:
            for record in self._records.values():
                handle.write(json.dumps(asdict(record)) + "\n")
        self._save_vectors(directory)


def read_records(directory: Path) -> List[ChunkRecord]:
    with (directory / RECORDS_FILE).open(encoding="utf-8") as handle:
        return [ChunkRecord(**json.loads(line)) for line in handle if line.strip()]
