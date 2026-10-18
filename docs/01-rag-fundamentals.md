# RAG Fundamentals

Retrieval-augmented generation (RAG) answers a question in two stages: **retrieve** the passages from your own documents that are most likely to contain the answer, then **generate** an answer from those passages. The language model is given the evidence instead of being asked to remember it.

## Why RAG

| Problem with a model alone | How retrieval helps |
|---|---|
| It doesn't know your internal documents | Relevant passages are fetched at question time |
| Knowledge is frozen at training time | Update the index, not the model |
| It can state wrong facts confidently (hallucination) | Answers are grounded in retrieved text |
| Answers can't be checked | Every passage has a source, so answers can cite it |
| Fine-tuning is slow and costly to repeat | Indexing a new document takes seconds |

RAG is about **knowledge**. Fine-tuning is better at changing **behaviour** (style, format, task skills). The two can be combined.

## The pipeline in this repository

```
 files ──► ingestion ──► chunking ──► embedding ──► vector store ─┐
 (pdf, docx, xlsx,  text +      passages      vectors      persisted   │
  html, md, csv, txt) sections                              index      │
                                                                       ▼
 question ─────────────────────────────► hybrid retrieval (dense + BM25) ──► generation ──► answer + sources
```

| Stage | What it does | Code | Doc |
|---|---|---|---|
| Ingestion | Read files of many formats into text, split into citation sections (page, sheet, heading) | `src/ingestion/` | [02](02-document-ingestion.md), [03](03-document-parsing.md) |
| Chunking | Cut sections into retrievable passages that respect paragraph and sentence boundaries | `src/chunking/` | [04](04-chunking.md) |
| Embedding | Turn each passage into a vector that encodes its meaning | `src/embedding/` | [05](05-embeddings.md) |
| Vector store | Persist vectors and chunk records; nearest-neighbour search; incremental sync | `src/vectorstore/`, `src/retrieval/index.py` | [06](06-vector-database.md) |
| Retrieval | Combine dense (vector) and lexical (BM25) scores; return the top-k chunks with metadata | `src/retrieval/` | [11](11-rag-evaluation.md) |
| Generation | Produce the answer from the retrieved chunks | `src/generation/` | — (an LLM backend is not implemented yet) |
| Evaluation | Score retrieval on a labelled question set | `src/evaluation/`, `scripts/evaluate.py` | [11](11-rag-evaluation.md) |

Two phases run at different times:

- **Indexing (offline):** `python scripts/index.py` ingests, chunks, embeds and stores. Only new or changed files are processed on later runs.
- **Querying (online):** `python scripts/query.py "question"` embeds the question, retrieves, and answers with citations.

## Core ideas

### Retrieval quality caps answer quality
If the passage that contains the answer is not retrieved, no prompt or model can produce a grounded answer. This project therefore measures retrieval first (hit@k, recall@k, MRR) and changes one component at a time against a frozen benchmark.

### Dense and lexical retrieval are complementary
- **Dense retrieval** compares embedding vectors. It matches meaning across different wording ("rust in the sea" ↔ "oxide film … seawater").
- **Lexical retrieval (BM25)** matches terms, weighted by rarity. It is precise for exact identifiers (alloy names, part numbers, "molybdenum").

On this project's benchmark each one misses questions the other answers, so the default is a **hybrid** of both.

### Chunks are the unit of retrieval
Documents are too long to embed or to pass to a model whole. Chunk size trades precision (small chunks) against context (large chunks); see [04-chunking.md](04-chunking.md).

### Citations make answers checkable
Every chunk keeps its source file and section metadata, so results are reported as `manual.pdf, page 2` or `lot_results.xlsx, sheet Q3 lots`.

## Common failure modes

| Failure | Symptom | Where to look |
|---|---|---|
| Text lost or garbled at ingestion | The answer exists in the file but is never retrieved | Ingestion report; parser for that format |
| Bad chunk boundaries | Evidence split across chunks; fragments rank high | Chunker settings; chunk coverage |
| Vocabulary mismatch | Paraphrased questions miss | Embedding model; stemming; query rewriting |
| Distractors | A passage that shares the question's words wins | BM25 weighting; reranking |
| Mixed embedding models | Scores look normal but rankings are random | The index manifest records the embedder and refuses mismatches |
| Generation ignores context | The answer contradicts the sources | Prompt design; faithfulness evaluation |

## Quick start

```bash
pip install -r requirements.txt
pytest -q
python scripts/index.py                                       # build data/index from data/raw
python scripts/query.py "How is springback compensated?"     # answer with sources
python scripts/evaluate.py --k 3                              # retrieval metrics on the benchmark
```

For semantic retrieval, install a real embedding model and rebuild the index:

```bash
pip install -e .[local-embeddings]
python scripts/index.py --rebuild --embedder st:sentence-transformers/all-MiniLM-L6-v2
```

## Glossary

| Term | Meaning |
|---|---|
| Chunk | A passage of a document, the unit that is embedded and retrieved |
| Embedding | A vector representing the meaning of a text |
| Vector store | Storage that finds the vectors nearest to a query vector |
| BM25 | A term-weighting ranking function used by search engines |
| Hybrid search | Combining dense and lexical scores or rankings |
| top-k | The number of chunks returned for a question |
| hit@k / recall@k / MRR | Retrieval metrics; see [11-rag-evaluation.md](11-rag-evaluation.md) |
| Grounding | Basing an answer only on retrieved evidence |
