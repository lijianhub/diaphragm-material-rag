from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Union

from chunking import RecursiveChunker
from embedding import SimpleEmbedder, create_embedder, query_vector
from ingestion.loader import Document
from ingestion.parser import Section
from reranking import create_reranker
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
    Optionally, ``dedupe_overlaps`` drops results that overlap a better-ranked
    chunk of the same section, and a ``reranker`` reorders the top
    ``rerank_depth`` candidates by scoring each (query, passage) pair jointly.
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
        reranker=None,
        rerank_depth: int = 20,
        dedupe_overlaps: bool = False,
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
        self.reranker = reranker
        self.rerank_depth = rerank_depth
        self.dedupe_overlaps = dedupe_overlaps
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
        """Chunk each section separately, so a chunk never spans two pages or sheets and
        carries its section's citation metadata (page, sheet, heading). Offsets are
        relative to the section."""
        doc_meta = {"doc_type": document.metadata.get("doc_type") or Path(document.source).suffix.lstrip(".").lower() or "text"}
        if document.metadata.get("title"):
            doc_meta["title"] = document.metadata["title"]
        sections = document.sections or [Section(document.content)]

        records: List[ChunkRecord] = []
        for section_index, section in enumerate(sections):
            for chunk in self.chunker.chunk(section.text, source=document.source):
                records.append(
                    ChunkRecord(
                        id=f"{document.source}#{len(records)}",
                        doc_id=document.source,
                        text=chunk.text,
                        source=document.source,
                        start=chunk.start,
                        end=chunk.end,
                        metadata={**doc_meta, **section.metadata, "section": section_index},
                    )
                )
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

    def sync(self, documents: Iterable[Document], keep: Iterable[str] = ()) -> SyncStats:
        """Make the index match ``documents``: upsert them and delete every other document,
        except those listed in ``keep`` (e.g. files that failed to parse this run)."""
        documents = list(documents)
        stats = self.upsert_documents(documents)
        present = {document.source for document in documents} | set(keep)
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
                "rerank_depth": self.rerank_depth,
                "dedupe_overlaps": self.dedupe_overlaps,
            },
            "reranker": getattr(self.reranker, "name", None),
            "documents": self._doc_hashes,
        }

    def save(self, directory: Union[str, Path]) -> None:
        directory = Path(directory)
        self.store.save(directory / STORE_DIR)
        (directory / MANIFEST_FILE).write_text(json.dumps(self._manifest(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, directory: Union[str, Path], embedder=None, chunker=None, reranker=None, **search_overrides) -> "SearchIndex":
        directory = Path(directory)
        manifest_path = directory / MANIFEST_FILE
        if not manifest_path.exists():
            raise FileNotFoundError(f"no index at {directory} (missing {MANIFEST_FILE}); build one with scripts/index.py")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("format_version") != FORMAT_VERSION:
            raise ValueError(f"index format {manifest.get('format_version')} is not supported; rebuild the index")

        built_with = manifest["embedder"]["name"]
        if embedder is None:
            # The manifest name is a spec, so the same model is rebuilt (local or Bedrock).
            embedder = create_embedder(built_with)
        if _embedder_name(embedder) != built_with:
            raise ValueError(
                f"index was built with embedder {built_with!r} but {_embedder_name(embedder)!r} was given; "
                "vectors from different models are not comparable, so rebuild the index"
            )
        if chunker is None:
            spec = manifest["chunker"]
            chunker = RecursiveChunker(chunk_size=spec["chunk_size"], overlap=spec["overlap"])

        if reranker is None and manifest.get("reranker"):
            reranker = create_reranker(manifest["reranker"])  # query-time model, rebuilt by name
        options = {**manifest["search"], **search_overrides}
        index = cls(embedder=embedder, chunker=chunker, store=load_store(directory / STORE_DIR), reranker=reranker, **options)
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

    def term_idf(self, term: str) -> float:
        """Corpus IDF of a term (BM25's, with this index's stemming), e.g. for query expansion."""
        if self._bm25 is None:
            self._bm25 = BM25([record.text for record in self._corpus()], stem_tokens=self.stem_tokens)
        return self._bm25.idf(term)

    def search(self, query: str, top_k: int = 3, alpha: Optional[float] = None, rerank: bool = True) -> List[SearchResult]:
        """Hybrid retrieval, then (if configured and ``rerank``) cross-encoder reranking.

        ``rerank=False`` returns the first-stage ranking, which multi-query retrieval
        fuses before reranking once against the original question."""
        records = self._corpus()
        if not records:
            return []
        weight = self.alpha if alpha is None else alpha
        texts = [record.text for record in records]
        position = {record.id: i for i, record in enumerate(records)}

        dense = [0.0] * len(records)
        for hit in self.store.search(query_vector(self.embedder, query), top_k=self.dense_candidates):
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
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        use_reranker = self.reranker is not None and rerank
        needed = self.rerank_depth if use_reranker else top_k
        candidates = self._dedupe(order, records, needed) if self.dedupe_overlaps else order[:needed]

        if not use_reranker:
            return [self._result(records[i], scores[i]) for i in candidates[:top_k]]

        rerank_scores = self.reranker.score(query, [records[i].text for i in candidates])
        reranked = sorted(zip(candidates, rerank_scores), key=lambda pair: pair[1], reverse=True)[:top_k]
        return [self._result(records[i], score, retrieval_score=scores[i]) for i, score in reranked]

    @staticmethod
    def _result(record: ChunkRecord, score: float, retrieval_score: Optional[float] = None) -> SearchResult:
        metadata = dict(record.metadata)
        if retrieval_score is not None:
            metadata["retrieval_score"] = retrieval_score  # first-stage score, kept for debugging
        return SearchResult(text=record.text, source=record.source, start=record.start, end=record.end, score=score, metadata=metadata)

    @staticmethod
    def _is_duplicate(a: ChunkRecord, b: ChunkRecord) -> bool:
        """Two chunks of the same section are duplicates when their overlap covers at least
        half of the shorter one. Neighbouring chunks share only the configured overlap
        (a small border) and stay distinct; an overlap fragment lies (almost) entirely
        inside a full chunk and is a duplicate."""
        if a.doc_id != b.doc_id or a.metadata.get("section") != b.metadata.get("section"):
            return False
        shared = min(a.end, b.end) - max(a.start, b.start)
        shorter = min(a.end - a.start, b.end - b.start)
        return shared > 0 and shared * 2 >= shorter

    @classmethod
    def _dedupe(cls, order: List[int], records: List[ChunkRecord], limit: int) -> List[int]:
        """Collapse duplicate chunks (see ``_is_duplicate``) into one result.

        The short overlap fragment often outranks the full chunk that contains it
        (shorter text scores higher on BM25 and cosine), so each group keeps the best
        rank position but the longest span. The whole ranking is scanned because the
        longer chunk can rank below the fragment.
        """
        kept: List[int] = []
        for i in order:
            for slot, j in enumerate(kept):
                if cls._is_duplicate(records[i], records[j]):
                    if records[i].end - records[i].start > records[j].end - records[j].start:
                        kept[slot] = i  # same rank position, more context
                    break
            else:
                kept.append(i)
        # A longer chunk swapped in above may duplicate a second kept fragment; keep the first.
        result: List[int] = []
        for i in kept:
            if not any(cls._is_duplicate(records[i], records[j]) for j in result):
                result.append(i)
        return result[:limit]
