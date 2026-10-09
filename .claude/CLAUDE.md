# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

An open-source (MIT), educational RAG project for materials-engineering knowledge retrieval. The initial domain is diaphragm forming with ELGILOY® / HAVAR® Co-Cr-Ni-Mo alloys. Requires Python >= 3.10.

## Commands

```bash
pip install -r requirements.txt               # pytest + pypdf; the only runtime deps today
pytest -q                                     # full suite
pytest tests/test_chunking.py -q              # one file
pytest tests/test_chunking.py::test_name -q   # one test
python scripts/index.py [--store faiss] [--rebuild]   # build/sync the persistent index data/raw -> data/index
python scripts/query.py "How is springback compensated?"   # retrieve + answer from data/index (in-memory fallback)
python scripts/evaluate.py --k 3 --alpha 0.5  # retrieval metrics; writes evaluation/results.jsonl
python experiments/01_naive_rag/run.py        # naive demo: whole documents, no chunking or retrieval
```

Run `scripts/evaluate.py` before and after any change to chunking, embedding or retrieval, and report the metric change. It indexes only the tracked `data/raw/*.txt` samples, so scores are reproducible on every clone. Look at the per-category lines as well as the overall score. Never tune defaults (alpha, k1, rrf_k) to maximize the score on this set, because it is also the test set; justify defaults from first principles and use sweeps only as analysis.

There is no linter or formatter configured, and `Makefile`, `.env.example`, `docker/` and `config/*.yaml` are currently empty.

## Import layout

`src/` is the import root, not a package prefix. `pyproject.toml` sets `pythonpath = ["src"]` for pytest and `package-dir = {"" = "src"}` for setuptools, so modules are imported as top-level packages:

```python
from ingestion.loader import load_documents   # correct
from src.ingestion.loader import ...          # wrong
```

Code run outside pytest must add `src/` to the import path itself. Scripts and experiments insert `REPO_ROOT / "src"` into `sys.path` (see `experiments/01_naive_rag/run.py`); for ad-hoc scripts, use `PYTHONPATH=src` or `pip install -e .`.

## Architecture

The pipeline is ingestion → chunking → embedding → retrieval → generation. Each stage is a separate package under `src/`, and the stages are deliberately decoupled: they exchange plain strings and lists rather than shared types. `retrieval/index.py` is the only place that composes them.

- `ingestion/loader.py`: `load_documents()` accepts a file, a directory (recursive; `.txt .md .json .csv .pdf`) or an iterable of paths, and returns `Document(source, content)`. PDF support goes through an optional `pypdf` import.
- `tokenization.py`: the shared `tokenize()` (lowercase, hyphenated terms kept whole, stopwords removed, optional `stem_tokens`). Embedding and lexical scoring must both use it so they agree on what a term is. `stem()` is a light Porter-style suffix stripper; stems are index keys (`verify` → `verifi`), not words. Query and corpus must use the same stemming setting.
- `chunking/`: `FixedSizeChunker` (character windows with overlap) and `RecursiveChunker` (collapses spaces but keeps line and paragraph breaks, then splits after the last `\n\n`, `\n`, `. ` or ` ` separator). Each chunker returns its own dataclass (`Chunk` / `RecursiveChunk`) with `text, source, start, end`. In `RecursiveChunker`, offsets refer to the normalized text, consecutive chunks never leave gaps, and the overlap always starts on a word boundary.
- `embedding/embedder.py`: `SimpleEmbedder` uses feature hashing into a fixed `dim` (default 512) of L2-normalized vectors. It is deterministic and offline, and captures word overlap, not meaning.
- `retrieval/dense.py`: `cosine_similarity` and `dense_search`.
- `retrieval/hybrid.py`: `hybrid_search(query, query_vector, texts, vectors, top_k, alpha)` scores `alpha * cosine + (1 - alpha) * lexical coverage`, both in [0, 1]. The caller embeds the query with the same embedder as the corpus.
- `retrieval/bm25.py`: `BM25(texts, k1, b, stem_tokens)` with Lucene IDF, which is never negative. `.scores(query)` is unbounded.
- `retrieval/fusion.py`: `reciprocal_rank_fusion(rankings, k=60, weights)` and `rank_by_score`, which drops zero-score items so that "not retrieved" never gets a rank.
- `vectorstore/`: the `VectorStore` interface (`add`, `delete_document`, `search`, `save`, and `load_store`) with `InMemoryVectorStore` (pure Python, exact, the reference) and `FaissVectorStore` (`IndexIDMap2(IndexFlatIP)`, exact; optional `faiss-cpu` dependency, imported lazily). Stores L2-normalize vectors, so scores are cosines. A new backend (OpenSearch, pgvector) must pass the contract tests in `tests/test_vectorstore.py`, which run against every store kind.
- `retrieval/index.py`: `SearchIndex(store="memory"|"faiss")`. `sync(documents)` upserts by SHA-256 content hash and deletes missing sources; `save`/`load` write `manifest.json` plus `store/`. `load` refuses an embedder whose `name` differs from the one that built the index. Search asks the store for `dense_candidates` (default 100) neighbours; other chunks get dense score 0. `.search()` returns `SearchResult(text, source, start, end, score, metadata)`. It ranks by position, so duplicate chunk texts keep their own sources. The lexical side is `lexical="bm25"` (default) or `"coverage"`, with `stem_tokens=True` by default. `fusion="weighted"` (default) max-normalizes BM25 and blends it with cosine by `alpha`; `fusion="rrf"` fuses the two rankings, with `alpha` weighting them (0.5 means equal). BM25 is rebuilt lazily after any change, because IDF depends on the whole corpus.
- `evaluation/`: relevance is judged by **evidence phrases** in `evaluation/ground_truth.jsonl` that must appear verbatim in a retrieved chunk, not by chunk ids, so the ground truth survives chunking changes. The metrics are hit@k, recall@k (over evidence phrases) and MRR, overall and per question `category` (`keyword`, `paraphrase`, `distractor`, `multi`). `tests/test_evaluation.py` checks that every evidence phrase exists in the corpus; when you edit the samples or questions, keep them in sync. When the benchmark saturates, expand it and freeze it **before** implementing the method it will judge, and record results in `docs/11-rag-evaluation.md`.
- `generation/rag_chain.py`: `RAGPipeline.answer(question, documents)` takes raw strings, does keyword filtering and concatenation (no LLM call yet), and returns `{"answer", "context", "sources"}`. Its `sources` are placeholder `doc_N` labels; real sources come from `SearchResult`.

### Scaffolded but empty

Many files exist only as zero-byte placeholders for the planned learning path. Check a file's contents before assuming it implements anything:

- `src/`: `agent/`, `api/`, `query/`, `vectorstore/pgvector_store.py`, `generation/claude.py`, `generation/prompt.py`, `ingestion/parser.py`, `config.py`
- `scripts/` other than `index.py`, `query.py` and `evaluate.py`; `experiments/02_*`–`11_*`; `notebooks/`; `docs/*.md` other than `06-vector-database.md` and `11-rag-evaluation.md`

Each `docs/NN-*.md` topic is meant to pair with an `experiments/NN_*` directory and a `src/` module. When you implement a stage, keep that numbering consistent.

## Conventions

- Optional heavy dependencies (`faiss-cpu`, and later model SDKs) are imported lazily inside the backend that needs them, declared under `[project.optional-dependencies]`, and their tests are skipped when missing. The core install and test suite must work without them.

- Every module starts with `from __future__ import annotations` and uses `@dataclass` for records with `typing` hints.
- Tests and the default demo must run offline, with no network or API keys. This is why the embedder is deterministic. Real model or API backends should be added alongside the offline implementation, not replace it. Test-specific rules are in `.claude/rules/testing.md`.
- `data/raw/Elgiloy_Havar_Material_Requirements_for_Diaphragm.pdf` is git-ignored and must not be committed. Only redistributable sample text belongs in `data/raw/`.
