from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Union

from chunking import RecursiveChunker
from embedding.embedder import SimpleEmbedder
from ingestion.loader import Document
from tokenization import tokenize
from vectorstore import ChunkRecord, VectorStore, create_store, load_store

from .bm25 import BM25
from .fusion import rank_by_score, reciprocal_rank_fusion
from .hybrid import combine_scores, lexical_coverage, max_normalize

LEXICAL_SCORERS = ("coverage", "bm25")
FUSION_METHODS = ("weighted", "rrf")
MANIFEST_FILE = "manifest.json"
STORE_DIR = "store"
FORMAT_VERSION = 1


@dataclass
class SearchResult:
    text: str
    source: str
    start: int
    end: int
    score: float
    metadata: Dict = field(default_factory=dict)


@dataclass
class SyncStats:
    added: int = 0
    updated: int = 0
    removed: int = 0
    unchanged: int = 0
    chunks: int = 0  # chunks written in this sync

    def __str__(self) -> str:
        return (
            f"added={self.added} updated={self.updated} removed={self.removed} "
            f"unchanged={self.unchanged} chunks_written={self.chunks}"
        )


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _embedder_name(embedder) -> str:
    return getattr(embedder, "name", type(embedder).__name__)


class SearchIndex:
    """Hybrid search over a persistent vector store plus an in-memory BM25.

    Indexing: documents are chunked, embedded and written to a ``VectorStore``
    (in-memory or FAISS). ``sync`` re-indexes only documents whose content hash
    changed and deletes documents that disappeared. ``save``/``load`` persist the
    store together with a manifest recording the embedder, chunker and per-document
    hashes, so an index is never queried with a different embedding model.

    Querying: the store returns the ``dense_candidates`` nearest chunks; BM25
    scores every chunk; the two are fused (weighted sum or RRF). Chunks outside
    the dense candidate set get a dense score of 0, as with a real ANN index.
    """

    def __init__(
        self,
        embedder=None,
        chunker=None,
        store: Union[str, VectorStore] = "memory",
        alpha: float = 0.5,
        lexical: str = "bm25",
        stem_tokens: bool = True,
        fusion: str = "weighted",
        rrf_k: int = 60,
        dense_candidates: int = 100,
    ):
        if lexical not in LEXICAL_SCORERS:
            raise ValueError(f"lexical must be one of {LEXICAL_SCORERS}")
        if fusion not in FUSION_METHODS:
            raise ValueError(f"fusion must be one of {FUSION_METHODS}")
        self.embedder = embedder or SimpleEmbedder()
        self.chunker = chunker or RecursiveChunker(chunk_size=300, overlap=50)
        self.store = create_store(store) if isinstance(store, str) else store
        self.alpha = alpha
        self.lexical = lexical
        self.stem_tokens = stem_tokens
        self.fusion = fusion
        self.rrf_k = rrf_k
        self.dense_candidates = dense_candidates
        self._doc_hashes: Dict[str, str] = {}
        self._bm25: Optional[BM25] = None
        self._records: Optional[List[ChunkRecord]] = None

    def __len__(self) -> int:
        return len(self.store)

    def document_ids(self) -> List[str]:
        return list(self._doc_hashes)

    # ---- indexing -------------------------------------------------------

    def _invalidate(self) -> None:
        # IDF and chunk positions depend on the whole corpus; rebuild lazily on next search.
        self._bm25 = None
        self._records = None

    def _write_document(self, document: Document) -> int:
        chunks = self.chunker.chunk(document.content, source=document.source)
        doc_type = Path(document.source).suffix.lstrip(".").lower() or "text"
        records = [
            ChunkRecord(
                id=f"{document.source}#{n}",
                doc_id=document.source,
                text=chunk.text,
                source=document.source,
                start=chunk.start,
                end=chunk.end,
                metadata={"doc_type": doc_type},
            )
            for n, chunk in enumerate(chunks)
        ]
        self.store.add(records, self.embedder.encode_many([record.text for record in records]))
        return len(records)

    def upsert_documents(self, documents: Iterable[Document]) -> SyncStats:
        """Add new documents and re-index changed ones; unchanged ones are skipped."""
        stats = SyncStats()
        for document in documents:
            digest = content_hash(document.content)
            previous = self._doc_hashes.get(document.source)
            if previous == digest:
                stats.unchanged += 1
                continue
            if previous is not None:
                self.store.delete_document(document.source)
                stats.updated += 1
            else:
                stats.added += 1
            stats.chunks += self._write_document(document)
            self._doc_hashes[document.source] = digest
        if stats.added or stats.updated:
            self._invalidate()
        return stats

    def remove_documents(self, doc_ids: Iterable[str]) -> int:
        removed = 0
        for doc_id in doc_ids:
            if doc_id in self._doc_hashes:
                self.store.delete_document(doc_id)
                del self._doc_hashes[doc_id]
                removed += 1
        if removed:
            self._invalidate()
        return removed

    def sync(self, documents: Iterable[Document]) -> SyncStats:
        """Make the index match ``documents`` exactly: upsert them, delete everything else."""
        documents = list(documents)
        stats = self.upsert_documents(documents)
        present = {document.source for document in documents}
        stats.removed = self.remove_documents([doc_id for doc_id in self.document_ids() if doc_id not in present])
        return stats

    def add_documents(self, documents: Iterable[Document]) -> SyncStats:
        return self.upsert_documents(documents)

    # ---- persistence ----------------------------------------------------

    def _manifest(self) -> Dict:
        return {
            "format_version": FORMAT_VERSION,
            "embedder": {"name": _embedder_name(self.embedder), "dim": getattr(self.embedder, "dim", None)},
            "chunker": {
                "type": type(self.chunker).__name__,
                "chunk_size": getattr(self.chunker, "chunk_size", None),
                "overlap": getattr(self.chunker, "overlap", None),
            },
            "store": self.store.kind,
            "search": {
                "alpha": self.alpha,
                "lexical": self.lexical,
                "stem_tokens": self.stem_tokens,
                "fusion": self.fusion,
                "rrf_k": self.rrf_k,
                "dense_candidates": self.dense_candidates,
            },
            "documents": self._doc_hashes,
        }

    def save(self, directory: Union[str, Path]) -> None:
        directory = Path(directory)
        self.store.save(directory / STORE_DIR)
        (directory / MANIFEST_FILE).write_text(json.dumps(self._manifest(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, directory: Union[str, Path], embedder=None, chunker=None, **search_overrides) -> "SearchIndex":
        directory = Path(directory)
        manifest_path = directory / MANIFEST_FILE
        if not manifest_path.exists():
            raise FileNotFoundError(f"no index at {directory} (missing {MANIFEST_FILE}); build one with scripts/index.py")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("format_version") != FORMAT_VERSION:
            raise ValueError(f"index format {manifest.get('format_version')} is not supported; rebuild the index")

        built_with = manifest["embedder"]["name"]
        if embedder is None:
            if not built_with.startswith("simple-hash-"):
                raise ValueError(f"index was built with embedder {built_with!r}; pass that embedder to load()")
            embedder = SimpleEmbedder(dim=manifest["embedder"]["dim"])
        if _embedder_name(embedder) != built_with:
            raise ValueError(
                f"index was built with embedder {built_with!r} but {_embedder_name(embedder)!r} was given; "
                "vectors from different models are not comparable, so rebuild the index"
            )
        if chunker is None:
            spec = manifest["chunker"]
            chunker = RecursiveChunker(chunk_size=spec["chunk_size"], overlap=spec["overlap"])

        options = {**manifest["search"], **search_overrides}
        index = cls(embedder=embedder, chunker=chunker, store=load_store(directory / STORE_DIR), **options)
        index._doc_hashes = dict(manifest["documents"])
        return index

    # ---- querying -------------------------------------------------------

    def _corpus(self) -> List[ChunkRecord]:
        if self._records is None:
            self._records = self.store.records()
        return self._records

    def _lexical_scores(self, query: str, texts: List[str]) -> List[float]:
        """Raw lexical scores: unbounded for BM25, in [0, 1] for coverage."""
        if self.lexical == "bm25":
            if self._bm25 is None:
                self._bm25 = BM25(texts, stem_tokens=self.stem_tokens)
            return self._bm25.scores(query)
        query_terms = tokenize(query, stem_tokens=self.stem_tokens)
        return [lexical_coverage(query_terms, text, self.stem_tokens) for text in texts]

    def search(self, query: str, top_k: int = 3, alpha: Optional[float] = None) -> List[SearchResult]:
        records = self._corpus()
        if not records:
            return []
        weight = self.alpha if alpha is None else alpha
        texts = [record.text for record in records]
        position = {record.id: i for i, record in enumerate(records)}

        dense = [0.0] * len(records)
        for hit in self.store.search(self.embedder.encode(query), top_k=self.dense_candidates):
            dense[position[hit.id]] = hit.score
        lexical = self._lexical_scores(query, texts)

        if self.fusion == "rrf":
            # Ranks only, so BM25 needs no normalization. alpha weights the two
            # rankings; 0.5 gives both a weight of 1, as in plain RRF.
            fused = reciprocal_rank_fusion(
                [rank_by_score(dense), rank_by_score(lexical)],
                k=self.rrf_k,
                weights=[2.0 * weight, 2.0 * (1.0 - weight)],
            )
            scores = [fused.get(i, 0.0) for i in range(len(records))]
        else:
            # BM25 is unbounded; scale it to [0, 1] so it can be blended with cosine.
            if self.lexical == "bm25":
                lexical = max_normalize(lexical)
            scores = combine_scores(dense, lexical, weight)

        # Rank by position so duplicate chunk texts keep their own source and offsets.
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [
            SearchResult(
                text=records[i].text,
                source=records[i].source,
                start=records[i].start,
                end=records[i].end,
                score=scores[i],
                metadata=dict(records[i].metadata),
            )
            for i in ranked
        ]
