"""Vector stores: where chunk embeddings live between indexing and querying.

``create_store("memory" | "faiss")`` builds an empty store; ``load_store(path)``
reopens a saved one, whichever kind it is.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional, Type

from .base import STORE_META_FILE, ChunkRecord, PathLike, VectorHit, VectorStore, read_records
from .memory_store import InMemoryVectorStore

STORE_KINDS = ("memory", "faiss")


def _store_class(kind: str) -> Type[VectorStore]:
    if kind == "memory":
        return InMemoryVectorStore
    if kind == "faiss":
        from .faiss_store import FaissVectorStore  # optional dependency, imported lazily

        return FaissVectorStore
    raise ValueError(f"unknown vector store {kind!r}; expected one of {STORE_KINDS}")


def create_store(kind: str = "memory", dim: Optional[int] = None) -> VectorStore:
    return _store_class(kind)(dim)


def load_store(directory: PathLike) -> VectorStore:
    directory = Path(directory)
    meta: Dict = json.loads((directory / STORE_META_FILE).read_text(encoding="utf-8"))
    return _store_class(meta["kind"])._load(directory, meta["dim"], read_records(directory))


__all__ = [
    "STORE_KINDS",
    "ChunkRecord",
    "InMemoryVectorStore",
    "VectorHit",
    "VectorStore",
    "create_store",
    "load_store",
]
