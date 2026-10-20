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
python scripts/evaluate.py --embedder st:sentence-transformers/all-MiniLM-L6-v2   # real embedding model
python experiments/01_naive_rag/run.py        # naive demo: whole documents, no chunking or retrieval
```

Run `scripts/evaluate.py` before and after any change to chunking, embedding or retrieval, and report the metric change. It indexes only the tracked samples returned by `evaluation.benchmark_corpus()` (txt, md, html and csv in `data/raw/`; never the private PDF), so scores are reproducible on every clone. Look at the per-category lines as well as the overall score. Never tune defaults (alpha, k1, rrf_k) to maximize the score on this set, because it is also the test set; justify defaults from first principles and use sweeps only as analysis.

There is no linter or formatter configured, and `Makefile`, `.env.example`, `docker/` and `config/*.yaml` are currently empty. `README.md` has a capability status table (implemented and tested / tested with fakes only / planned); keep it accurate when a capability changes status, and never mark something implemented that is only designed.

## Import layout

`src/` is the import root, not a package prefix. `pyproject.toml` sets `pythonpath = ["src"]` for pytest and `package-dir = {"" = "src"}` for setuptools, so modules are imported as top-level packages:

```python
from ingestion.loader import load_documents   # correct
from src.ingestion.loader import ...          # wrong
```

Code run outside pytest must add `src/` to the import path itself. Scripts and experiments insert `REPO_ROOT / "src"` into `sys.path` (see `experiments/01_naive_rag/run.py`); for ad-hoc scripts, use `PYTHONPATH=src` or `pip install -e .`.

## Architecture

The pipeline is ingestion → chunking → embedding → retrieval → generation. Each stage is a separate package under `src/`, and the stages are deliberately decoupled: they exchange plain strings and lists rather than shared types. `retrieval/index.py` is the only place that composes them.

- `ingestion/`: `parser.py` holds one parser per format (txt/json, md, html, pdf, docx, xlsx, csv) behind `parse_file()` → `Parsed(sections, metadata)`; DOCX/XLSX are read with `zipfile` + `xml.etree` (no dependency). A `Section` is a citation unit (PDF page, sheet, heading section) with metadata such as `page`, `sheet` or `heading`. `loader.ingest(paths)` returns `IngestResult(documents, failures, skipped, duplicates)` and never stops at a bad file; `load_documents()` is the strict variant that raises. `Document(source, content, metadata, sections)`; empty `sections` means one section equal to `content`. New formats register via `register_parser`. Tests build PDF/DOCX/XLSX files in code with `tests/file_builders.py`; never commit binary fixtures.
- `tokenization.py`: the shared `tokenize()` (lowercase, hyphenated terms kept whole, stopwords removed, optional `stem_tokens`). Embedding and lexical scoring must both use it so they agree on what a term is. `stem()` is a light Porter-style suffix stripper; stems are index keys (`verify` → `verifi`), not words. Query and corpus must use the same stemming setting.
- `chunking/`: `FixedSizeChunker` (character windows with overlap) and `RecursiveChunker` (collapses spaces but keeps line and paragraph breaks, then splits after the last `\n\n`, `\n`, `. ` or ` ` separator). Each chunker returns its own dataclass (`Chunk` / `RecursiveChunk`) with `text, source, start, end`. In `RecursiveChunker`, offsets refer to the normalized text, consecutive chunks never leave gaps, and the overlap always starts on a word boundary.
- `embedding/`: every embedder has `name`, `dim`, `encode`, `encode_many` and optionally `encode_query`. `create_embedder(name)` builds one from a spec: `simple-hash[-dim]` (`SimpleEmbedder`: offline feature hashing, the default and what tests use; word overlap, not meaning), `st:<hf-model>` (`SentenceTransformerEmbedder`, optional `[local-embeddings]` extra), and `bedrock:<model>[@dims]` (`BedrockEmbedder`: Titan Text Embeddings V2, optional `[aws]` extra; one InvokeModel per text with backoff on throttling). An embedder's `name` is itself a valid spec, so `SearchIndex.load` rebuilds the embedder from the manifest. Queries always go through `embedding.query_vector()`, which uses `encode_query`, so asymmetric models get their query prefix. Tests use the fakes in `tests/fakes.py` and never download models or call AWS; the one real-model test needs `RUN_MODEL_TESTS=1`.
- `retrieval/dense.py`: `cosine_similarity` and `dense_search`.
- `retrieval/hybrid.py`: `hybrid_search(query, query_vector, texts, vectors, top_k, alpha)` scores `alpha * cosine + (1 - alpha) * lexical coverage`, both in [0, 1]. The caller embeds the query with the same embedder as the corpus.
- `retrieval/bm25.py`: `BM25(texts, k1, b, stem_tokens)` with Lucene IDF, which is never negative. `.scores(query)` is unbounded.
- `reranking/`: second-stage rerankers with `name` and `score(query, passages) -> List[float]`. `create_reranker(name)` builds `ce:<model>` (`CrossEncoderReranker`, sentence-transformers `CrossEncoder`, `[local-embeddings]` extra) or `bedrock-rerank:<model>` (`BedrockReranker`, bedrock-agent-runtime `rerank` with inline documents, `[aws]` extra). Tests use `FakeCrossEncoder` and `FakeBedrockAgentClient` from `tests/fakes.py`.
- `retrieval/fusion.py`: `reciprocal_rank_fusion(rankings, k=60, weights)` and `rank_by_score`, which drops zero-score items so that "not retrieved" never gets a rank.
- `vectorstore/`: the `VectorStore` interface (`add`, `delete_document`, `search`, `save`, and `load_store`) with `InMemoryVectorStore` (pure Python, exact, the reference) and `FaissVectorStore` (`IndexIDMap2(IndexFlatIP)`, exact; optional `faiss-cpu` dependency, imported lazily). Stores L2-normalize vectors, so scores are cosines. A new backend (OpenSearch, pgvector) must pass the contract tests in `tests/test_vectorstore.py`, which run against every store kind.
- `retrieval/index.py`: `SearchIndex(store="memory"|"faiss")`. Chunking runs per section, and chunks inherit `doc_type`, `title` and the section metadata (offsets are relative to the section). `sync(documents, keep=...)` upserts by SHA-256 content hash and deletes missing sources except those in `keep` (scripts pass files that failed to parse, so they retain their last good version); `save`/`load` write `manifest.json` plus `store/`. `load` refuses an embedder whose `name` differs from the one that built the index. Search asks the store for `dense_candidates` (default 100) neighbours; other chunks get dense score 0. `.search()` returns `SearchResult(text, source, start, end, score, metadata)`. After fusion, `dedupe_overlaps=True` collapses duplicate chunks (same section, overlap ≥ half of the shorter span) into the longest one at the best rank. Neighbouring chunks that share only the overlap border are not duplicates, and treating them as such dropped hit@1 from 0.889 to 0.422. With a `reranker`, the top `rerank_depth` (default 20) candidates are rescored and `score` becomes the reranker score; the first-stage score goes to `metadata["retrieval_score"]`. The manifest stores `rerank_depth`, `dedupe_overlaps` and the reranker name, and `load` rebuilds the reranker unless one is passed in. It ranks by position, so duplicate chunk texts keep their own sources. The lexical side is `lexical="bm25"` (default) or `"coverage"`, with `stem_tokens=True` by default. `fusion="weighted"` (default) max-normalizes BM25 and blends it with cosine by `alpha`; `fusion="rrf"` fuses the two rankings, with `alpha` weighting them (0.5 means equal). BM25 is rebuilt lazily after any change, because IDF depends on the whole corpus.
- `query/`: query transformation on top of `SearchIndex`. `decompose(question)` splits compound questions ("…, and what …") by rule; `expansion_terms(...)` is pseudo-relevance feedback (frequent in the top passages, rare in the corpus, returned as surface words); `LLMQueryRewriter(complete, mode="multi-query"|"hyde")` takes any text-in, text-out function and falls back to the original question if it fails. `retrieve(index, question, transforms=..., prf_terms=...)` runs every query first-stage only (`search(..., rerank=False)`), fuses the rankings (`fusion="rrf"` for paraphrases of one question, which rewards consensus; `"interleave"` for decomposed sub-questions, which needs coverage, because RRF there let a generic chunk found by every sub-query crowd out each part's best match), and reranks against the **original** question. PRF causes query drift on this corpus (hit@1 0.889 → 0.600), so it stays opt-in. Never add hand-written synonyms that target known benchmark failures; that is tuning on the test set.
- `evaluation/`: relevance is judged by **evidence phrases** in `evaluation/ground_truth.jsonl` that must appear verbatim in a retrieved chunk, not by chunk ids, so the ground truth survives chunking changes. The metrics are hit@k, recall@k (over evidence phrases) and MRR, overall and per question `category` (`keyword`, `paraphrase`, `distractor`, `multi`). `tests/test_evaluation.py` checks that every evidence phrase exists in the corpus; when you edit the samples or questions, keep them in sync. When the benchmark saturates, expand it and freeze it **before** implementing the method it will judge, and record results in `docs/11-rag-evaluation.md`.
- `generation/`: `prompt.py` builds grounded prompts: `build_prompt(question, results, max_context_tokens)` → `Prompt(system, user, included)` with numbered `<source id="n" label="…">` blocks in rank order, `<`/`>` escaped against prompt injection, and a ~4-chars-per-token budget (the first source is truncated rather than dropped). It also provides `citation_label`, `parse_citations` and `check_citations`. `rag_chain.py`: `RAGPipeline(index, complete=None, top_k, retrieve_fn=None).answer(question)` → `Answer(text, citations, sources, used_llm, invalid_citations, fallback_reason, prompt)`. `complete` is any text-in, text-out LLM function; without one, or when it raises, `ExtractiveAnswerer` quotes matching sentences with citations. It cannot judge answerability ("price of titanium" quotes "titanium nitride"), which is a known limitation for the LLM step and generation evaluation to address. With nothing retrieved, no LLM call is made. `answer(question, documents=[...])` accepts plain passages for quick experiments.

### Scaffolded but empty

Many files exist only as zero-byte placeholders for the planned learning path. Check a file's contents before assuming it implements anything:

- `src/`: `agent/`, `api/`, `vectorstore/pgvector_store.py`, `generation/claude.py`, `config.py`
- `scripts/` other than `index.py`, `query.py` and `evaluate.py`; `experiments/02_*`–`11_*`; `notebooks/`; `docs/*.md` other than `01`–`06`, `08`–`11`, `architecture.md` and `claude-code-workflows.md`

Each `docs/NN-*.md` topic is meant to pair with an `experiments/NN_*` directory and a `src/` module. When you implement a stage, keep that numbering consistent.

## Conventions

- Optional heavy dependencies (`faiss-cpu`, `sentence-transformers`, `boto3`) are imported lazily inside the backend that needs them, declared under `[project.optional-dependencies]`, and their tests are skipped when missing. The core install and test suite must work without them.

- Every module starts with `from __future__ import annotations` and uses `@dataclass` for records with `typing` hints.
- Tests and the default demo must run offline, with no network or API keys. This is why the embedder is deterministic. Real model or API backends should be added alongside the offline implementation, not replace it. Test-specific rules are in `.claude/rules/testing.md`.
- `data/raw/Elgiloy_Havar_Material_Requirements_for_Diaphragm.pdf` is git-ignored and must not be committed. Only redistributable sample text belongs in `data/raw/`.
