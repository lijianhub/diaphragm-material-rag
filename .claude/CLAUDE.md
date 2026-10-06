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
python experiments/01_naive_rag/run.py        # end-to-end demo on data/raw/diaphragm_material_notes.txt
```

There is no linter or formatter configured, and `Makefile`, `.env.example`, `docker/` and `config/*.yaml` are currently empty.

## Import layout

`src/` is the import root, not a package prefix. `pyproject.toml` sets `pythonpath = ["src"]` for pytest and `package-dir = {"" = "src"}` for setuptools, so modules are imported as top-level packages:

```python
from ingestion.loader import load_documents   # correct
from src.ingestion.loader import ...          # wrong
```

Code run outside pytest must add `src/` to the import path itself. Experiments insert `REPO_ROOT / "src"` into `sys.path` (see `experiments/01_naive_rag/run.py`); for ad-hoc scripts, use `PYTHONPATH=src` or `pip install -e .`.

## Architecture

The pipeline is ingestion → chunking → embedding → retrieval → generation. Each stage is a separate package under `src/`, and the stages are deliberately decoupled: they exchange plain strings and lists rather than shared types.

- `ingestion/loader.py`: `load_documents()` accepts a file, a directory (recursive; `.txt .md .json .csv .pdf`) or an iterable of paths, and returns `Document(source, content)`. PDF support goes through an optional `pypdf` import.
- `chunking/`: `FixedSizeChunker` (character windows with overlap) and `RecursiveChunker` (collapses whitespace, then backs off to the last `\n\n`, `\n`, `. ` or ` ` separator). Each chunker returns its own dataclass (`Chunk` / `RecursiveChunk`) with `text, source, start, end`. Offsets in `RecursiveChunker` refer to the whitespace-normalized text, not the original.
- `embedding/embedder.py`: `SimpleEmbedder` is a deterministic, offline placeholder (a token-hash vector whose length equals the token count), not a semantic embedding. Vectors differ in length, so code must not assume a fixed dimension or use cosine similarity on them.
- `retrieval/hybrid.py`: `hybrid_search(query, texts, vectors, top_k)` scores `2 * lexical term overlap + mean(vector) / 100`.
- `generation/rag_chain.py`: `RAGPipeline.answer(question, documents)` takes raw strings, does keyword filtering and concatenation (no LLM call yet), and returns `{"answer", "context", "sources"}`.

The pipeline does not yet wire chunking or retrieval in; the demo passes whole documents straight to `RAGPipeline`.

### Scaffolded but empty

Many files exist only as zero-byte placeholders for the planned learning path. Check a file's contents before assuming it implements anything:

- `src/`: `agent/`, `api/`, `evaluation/`, `query/`, `vectorstore/` (FAISS / pgvector), `retrieval/bm25.py`, `retrieval/dense.py`, `generation/claude.py`, `generation/prompt.py`, `ingestion/parser.py`, `config.py`
- `scripts/*.py`, `experiments/02_*`–`11_*`, `notebooks/`, `docs/*.md`, `evaluation/*.jsonl`

Each `docs/NN-*.md` topic is meant to pair with an `experiments/NN_*` directory and a `src/` module. When you implement a stage, keep that numbering consistent.

## Conventions

- Every module starts with `from __future__ import annotations` and uses `@dataclass` for records with `typing` hints.
- Tests and the default demo must run offline, with no network or API keys. This is why the embedder is deterministic. Real model or API backends should be added alongside the offline implementation, not replace it. Test-specific rules are in `.claude/rules/testing.md`.
- `data/raw/Elgiloy_Havar_Material_Requirements_for_Diaphragm.pdf` is git-ignored and must not be committed. Only redistributable sample text belongs in `data/raw/`.
