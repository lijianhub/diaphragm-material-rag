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
python scripts/query.py "How is springback compensated?"   # retrieve + answer over data/raw
python scripts/evaluate.py --k 3 --alpha 0.5  # retrieval metrics; writes evaluation/results.jsonl
python experiments/01_naive_rag/run.py        # naive demo: whole documents, no chunking or retrieval
```

Run `scripts/evaluate.py` before and after any change to chunking, embedding or retrieval, and report the metric change. It indexes only the tracked `data/raw/*.txt` samples, so scores are reproducible on every clone.

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
- `tokenization.py`: the shared `tokenize()` (lowercase, hyphenated terms kept whole, stopwords removed). Embedding and lexical scoring must both use it so they agree on what a term is.
- `chunking/`: `FixedSizeChunker` (character windows with overlap) and `RecursiveChunker` (collapses spaces but keeps line and paragraph breaks, then splits after the last `\n\n`, `\n`, `. ` or ` ` separator). Each chunker returns its own dataclass (`Chunk` / `RecursiveChunk`) with `text, source, start, end`. In `RecursiveChunker`, offsets refer to the normalized text, consecutive chunks never leave gaps, and the overlap always starts on a word boundary.
- `embedding/embedder.py`: `SimpleEmbedder` uses feature hashing into a fixed `dim` (default 512) of L2-normalized vectors. It is deterministic and offline, and captures word overlap, not meaning.
- `retrieval/dense.py`: `cosine_similarity` and `dense_search`.
- `retrieval/hybrid.py`: `hybrid_search(query, query_vector, texts, vectors, top_k, alpha)` scores `alpha * cosine + (1 - alpha) * lexical coverage`, both in [0, 1]. The caller embeds the query with the same embedder as the corpus.
- `retrieval/index.py`: `SearchIndex.add_documents()` chunks and embeds; `.search()` returns `SearchResult(text, source, start, end, score)`. It ranks by position, so duplicate chunk texts keep their own sources.
- `evaluation/`: relevance is judged by **evidence phrases** in `evaluation/ground_truth.jsonl` that must appear verbatim in a retrieved chunk, not by chunk ids, so the ground truth survives chunking changes. The metrics are hit@k, recall@k (over evidence phrases) and MRR. `tests/test_evaluation.py` checks that every evidence phrase exists in the corpus; when you edit the samples or questions, keep them in sync.
- `generation/rag_chain.py`: `RAGPipeline.answer(question, documents)` takes raw strings, does keyword filtering and concatenation (no LLM call yet), and returns `{"answer", "context", "sources"}`. Its `sources` are placeholder `doc_N` labels; real sources come from `SearchResult`.

### Scaffolded but empty

Many files exist only as zero-byte placeholders for the planned learning path. Check a file's contents before assuming it implements anything:

- `src/`: `agent/`, `api/`, `query/`, `vectorstore/` (FAISS / pgvector), `retrieval/bm25.py`, `generation/claude.py`, `generation/prompt.py`, `ingestion/parser.py`, `config.py`
- `scripts/` other than `query.py` and `evaluate.py`, `experiments/02_*`–`11_*`, `notebooks/`, `docs/*.md` other than `11-rag-evaluation.md`

Each `docs/NN-*.md` topic is meant to pair with an `experiments/NN_*` directory and a `src/` module. When you implement a stage, keep that numbering consistent.

## Conventions

- Every module starts with `from __future__ import annotations` and uses `@dataclass` for records with `typing` hints.
- Tests and the default demo must run offline, with no network or API keys. This is why the embedder is deterministic. Real model or API backends should be added alongside the offline implementation, not replace it. Test-specific rules are in `.claude/rules/testing.md`.
- `data/raw/Elgiloy_Havar_Material_Requirements_for_Diaphragm.pdf` is git-ignored and must not be committed. Only redistributable sample text belongs in `data/raw/`.
