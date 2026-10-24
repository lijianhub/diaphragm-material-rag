# Architecture

## Problem

Engineers need answers from a growing set of internal documents (material specifications, forming and heat-treatment procedures, inspection records, supplier information) in many formats. Keyword search misses paraphrased questions, and a chatbot without sources can't be trusted for engineering decisions.

| | |
|---|---|
| **Input** | A natural-language question; a folder of documents (PDF, DOCX, XLSX, HTML, Markdown, CSV, TXT) |
| **Output** | A concise answer in which every statement cites a source (file plus page, sheet or section), or an explicit statement that the sources don't contain the answer |
| **Users** | Engineers and quality staff asking questions; administrators adding documents and watching quality |
| **Out of scope** | Decisions without human review (the system informs, people decide); scanned documents without OCR; questions requiring calculation or data outside the indexed documents; legacy binary Office formats |

## Components

| Layer | Package | Responsibility | Interface (swappable) |
|---|---|---|---|
| Ingestion | `ingestion/` | Parse each format into text sections with citation metadata; report failures without stopping | `register_parser(suffixes, fn)` |
| Chunking | `chunking/` | Split sections into paragraph-aware chunks with overlap | `chunk(text, source)` |
| Embedding | `embedding/` | Text → vectors | `name`, `dim`, `encode_many`, `encode_query`; `create_embedder(name)` |
| Vector store | `vectorstore/` | Store chunk records and vectors; nearest-neighbour search; persistence | `VectorStore` (memory, FAISS; later OpenSearch or pgvector) |
| Retrieval | `retrieval/` | BM25 + dense fusion, dedupe, optional reranking; incremental `sync`; `save`/`load` | `SearchIndex` |
| Reranking | `reranking/` | Rescore top candidates with a cross-encoder | `score(query, passages)`; `create_reranker(name)` |
| Query transformation | `query/` | Decompose, expand, rewrite; fuse multi-query rankings | `retrieve(index, question, transforms=…)` |
| Generation | `generation/` | Grounded prompt, LLM call or extractive fallback, citation checking | `complete(prompt) -> str` |
| Evaluation | `evaluation/` | Benchmark corpus, metrics, per-category reports | `evaluate_retrieval(examples, search_fn, k)` |

## Flows

### Indexing (offline, incremental)
```
ingest(folder) ─► IngestResult(documents, failures, skipped, duplicates)
   └─► SearchIndex.sync(documents, keep=failed)
         per document: SHA-256 unchanged → skip | new or changed → chunk per section → embed → store
         documents no longer present → delete (except those that failed to parse this run)
   └─► save(): manifest.json (embedder, chunker, search settings, document hashes) + store/
```

### Querying
```
question ─► [transforms] ─► first-stage retrieval per query (BM25 + dense)
        ─► fuse (RRF for paraphrases, interleave for sub-questions) ─► [dedupe] ─► [rerank vs original question]
        ─► build_prompt (numbered, labelled, escaped sources within a token budget)
        ─► LLM complete() │ extractive fallback ─► check citations ─► Answer
```

## Failure Handling

| Failure | Current handling | Planned |
|---|---|---|
| **Invalid input** (unsupported, corrupted or scanned file; empty or oversized question) | Files: reported per file in `IngestResult.failures`; the batch continues; a file that fails keeps its last indexed version. Questions: Pydantic validation in the API → 422 with a request ID | — |
| **No retrieval results** | No LLM call; answer says the information isn't in the sources; counted as `refused` in `rag_answers_total` | Surfaced on the ops dashboard |
| **Model or tool timeout / outage** | Bedrock calls: exponential backoff on throttling, bounded retries; LLM calls in the service: 30 s timeout, 1 retry; LLM failure → extractive answer with `fallback_reason`, counted as `fallback` | Alerts on error and fallback rates |
| **Malformed model output** | Citation numbers are parsed and validated; numbers that point to no source go to `invalid_citations` | Structured-output validation; regression tests on fixed samples |
| **Insufficient evidence** | Prompt rule: say when the answer isn't in the sources, with no citation on that sentence (enforced in code). Measured: 12/12 clean refusals with `gpt-5.4-mini@low`. ⚠️ The offline extractive answerer cannot detect this (0/12) | — |
| **Embedding model changed** | `load()` refuses an index built with a different embedder; a rebuild is required | — |
| **Prompt injection in documents or questions** | Angle brackets escaped in sources and in the question; sources declared as data; 8 malicious cases in the generation benchmark | Upload permissions, audit logging |
| **Unexpected server error** | API returns `500 internal_error` with the request ID and no internals; the exception type is logged | Alerts |

## Deterministic Code vs LLM

| Step | Who does it | Why |
|---|---|---|
| Parsing, chunking, hashing, sync | Deterministic code | Must be reproducible and testable |
| Retrieval scoring and fusion | Deterministic code + embedding models | Measurable on the benchmark |
| Reranking | Cross-encoder model | Pairwise relevance judgement |
| Query rewriting (multi-query, HyDE) | LLM, optional | Bridging vocabulary gaps; falls back to the original question |
| Writing the answer | LLM; extractive fallback | Synthesis across sources |
| Citation checking, budget, escaping | Deterministic code | Safety checks must not depend on the model |
| Deciding to act (agent, planned) | LLM proposes, code enforces limits, a human approves side effects | Bounded autonomy |

## Versioning and Traceability

- **Index manifest:** format version, embedder name and dimension, chunker settings, search settings (alpha, fusion, rerank depth, dedupe), reranker name and a content hash per document.
- **Answers:** citations map to source file, section and character offsets; the prompt used is attached to the `Answer`.
- **Answers** also record the served model, the prompt version (label + SHA-256 fingerprint), token usage and stage timings.
- **Evaluation runs** are tracked in MLflow or JSONL with their configuration, git commit, dirty flag and data hashes ([15-experiment-tracking.md](15-experiment-tracking.md)).
- **Requests** carry an `X-Request-ID` that appears in the response, the error body and the log line ([14-production-rag.md](14-production-rag.md)). Planned: the same ID through the backend-for-frontend.

## Service Layers

```
Browser
 ├─ React user app        ask, read cited answers, open sources, give feedback
 └─ React ops dashboard   latency, errors, cost, evaluation trends, ingestion failures, approval queue
         │ HTTP + Server-Sent Events
 Node.js backend-for-frontend (TypeScript, Fastify)
         request IDs, rate limiting, authentication, streaming, dashboard aggregation
         │ HTTP JSON
 Python FastAPI service (implemented: src/api)
         /ask /health /ready /metrics · request IDs · JSON logs · versioned answers
```

The React apps and the backend-for-frontend are planned. The backend-for-frontend is a common pattern that keeps browser concerns (sessions, streaming, aggregation) out of the Python service. A minimal deployment could have React call FastAPI directly.

## Deployment Target

The project will run as an enterprise trial on AWS: documents in S3, vectors in a managed store (OpenSearch or pgvector) behind the `VectorStore` interface, embeddings and reranking through Bedrock, services in containers. Configuration comes from environment variables and secrets from a secrets manager, never from code or images.
