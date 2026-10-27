# Technology Stack

Every tool, model and data store the project uses, with the version installed during development, its role, and how far it has been exercised. Status:

- **Used**: runs in the default setup or was used for the reported measurements.
- **Measured**: compared on a benchmark; the numbers are in the linked documents.
- **Fakes only**: implemented and tested with test doubles, never against the real service.
- **Planned**: designed, not implemented.

## At a glance

| Layer | Choice |
|---|---|
| Language | Python 3.10+ (developed on 3.14, image on 3.12); TypeScript for the BFF and the web app |
| Retrieval | hybrid: own BM25 (Lucene IDF) + dense vectors, weighted fusion or RRF; optional cross-encoder reranking; metadata filter inside retrieval |
| Embeddings | feature hashing (512 dimensions, offline default), `all-MiniLM-L6-v2` (384), Amazon Titan Text Embeddings V2 (fakes only) |
| Vector store | in-memory exact search (default), FAISS `IndexFlatIP`; PostgreSQL + pgvector planned |
| LLM | OpenAI `gpt-5.4-mini` at low reasoning effort (all real measurements); Anthropic Claude Opus 5.5 supported (fakes only) |
| Orchestration | plain Python: no LangChain or LlamaIndex |
| Service | FastAPI + Uvicorn (Python), Fastify (Node.js BFF), React + Vite (web), nginx |
| State | JSONL files, SQLite (agent runs, MLflow), the index as JSON files |
| Evaluation and tracking | own rule-based benchmarks (retrieval, generation, agent), MLflow or JSONL run log |
| Delivery | Docker Compose, GitHub Actions; AWS designed, not deployed |

## Document ingestion

| Tool | Version | Role | Status |
|---|---|---|---|
| Own parsers (`ingestion/parser.py`) | — | one parser per format behind `parse_file()`; sections carry page, sheet or heading for citations | Used |
| `pypdf` | 6.16 | PDF text per page | Used (the private manual) |
| Python `zipfile` + `xml.etree` | standard library | DOCX (headings, tables) and XLSX (sheets) without a dependency | Used |
| Python `html.parser`, `csv` | standard library | HTML and CSV | Used |
| OCR for scanned PDFs | — | — | Not included (known limitation) |

## Chunking and tokenization

| Component | Setting | Status |
|---|---|---|
| `RecursiveChunker` (own) | 300 characters, 50 overlap; splits after `\n\n`, `\n`, `. `, ` `; no gaps; overlap starts on a word boundary | Used, measured ([04](04-chunking.md)) |
| `FixedSizeChunker` (own) | character windows | Baseline only |
| Tokenizer and light Porter-style stemmer (own, `tokenization.py`) | shared by BM25 and the hashing embedder, so both agree on what a term is | Used |

## Embedding models

| Model | Dimensions | Where it runs | Status | Result (hit@3 on the 45-question benchmark) |
|---|---|---|---|---|
| Feature hashing (`simple-hash-512`, own) | 512 | in process, offline, deterministic | Used: the default, all tests, CI, Docker image | 0.956 (hybrid with BM25) |
| `sentence-transformers/all-MiniLM-L6-v2` (sentence-transformers 6.1) | 384 | local CPU | Measured | 0.978; hit@1 0.933 ([05](05-embeddings.md)) |
| Amazon Titan Text Embeddings V2 (`amazon.titan-embed-text-v2:0`, boto3) | 256 / 512 / 1024 | AWS Bedrock | Fakes only | — |

## Vector stores and retrieval

| Component | Detail | Status |
|---|---|---|
| `InMemoryVectorStore` (own) | exact cosine, pure Python; the reference implementation | Used (default) |
| FAISS (`faiss-cpu` 1.15) | `IndexIDMap2(IndexFlatIP)` on L2-normalized vectors = exact cosine; deletes; `IDSelectorBatch` for filters | Used, identical ranking to the reference ([06](06-vector-database.md)) |
| BM25 (own, `retrieval/bm25.py`) | k1 = 1.5, b = 0.75, Lucene IDF (never negative), stemming | Used |
| Hybrid fusion (own) | weighted: alpha 0.5 × cosine + 0.5 × max-normalized BM25 (default); or RRF with k = 60 | Used, measured ([11](11-rag-evaluation.md)) |
| Dense candidates | 100 nearest neighbours per query | Used |
| Metadata filter | `doc_type`, `source`, `title`, `heading`, `page`, `sheet`; applied inside retrieval | Used |
| Query transformation (own) | rule-based decomposition, pseudo-relevance feedback, LLM multi-query and HyDE | Measured; opt-in ([09](09-query-transformation.md)) |
| PostgreSQL + pgvector | for the cloud deployment | Planned ([20](20-aws-deployment.md)) |
| OpenSearch Serverless | — | Rejected for the trial on cost ([20](20-aws-deployment.md)) |

## Reranking

| Model | Status | Result |
|---|---|---|
| `cross-encoder/ms-marco-MiniLM-L-6-v2` (sentence-transformers `CrossEncoder`) | Measured, opt-in | lifts a weak first stage (hit@1 0.689 → 0.911 with hashing dense only), but did not improve the good hybrid; +70–100 ms per query ([08](08-reranking.md)) |
| Amazon Bedrock Rerank (`amazon.rerank-v1:0`) | Fakes only | — |

## Language models

| Model | Through | Used for | Status |
|---|---|---|---|
| OpenAI `gpt-5.4-mini` (served as `gpt-5.4-mini-2026-03-17`), reasoning effort `low` | `openai` SDK 3.17, Responses API | answers, streaming, query rewriting, agent tool calling | **Used for every real measurement**: generation gates 6/6, agent gates 9/9 |
| Anthropic Claude Opus 5.5 (`claude-opus-5-5`, the code's default for `anthropic:`), explicit `effort` | `anthropic` SDK 1.15, Messages API (with server-side refusal fallback) | the same features | Fakes only (no Anthropic key used yet) |
| Extractive answerer (own) | — | offline answers and the fallback when the LLM fails | Used |

LLM features built on top: grounded prompt `grounded-v2` with numbered sources and escaping, citation validation, refusal citations removed in code, `CachedLLM` (on-disk cache, free reruns), native tool calling with a provider-neutral transcript, token and model recorded on every answer.

## Agent

| Component | Detail | Status |
|---|---|---|
| State machine (own, `agent/`) | running → awaiting approval → completed / stopped / failed | Used, benchmarked ([13](13-agentic-rag.md)) |
| Tools | `search_documents`, `lookup_alloy`, `raise_nonconformance_report` (side effect: human approval) | Used |
| Run store | SQLite (standard library) with compare-and-set | Used |
| Frameworks (LangGraph, LangChain agents) | — | Not used, by design |

## Service and web

| Tool | Version | Role |
|---|---|---|
| FastAPI | 0.162 | HTTP API: `/ask`, `/ask/stream` (Server-Sent Events), feedback, agent, ops views |
| Uvicorn | 0.54 | ASGI server |
| Pydantic | 2.13 | request validation |
| Own Prometheus-format metrics | — | counters and histograms at `/metrics` |
| Node.js | 22 (image), 24 (development) | BFF runtime |
| Fastify | 5.12 | BFF: auth stub, admin keys, rate limits (`@fastify/rate-limit`), CORS (`@fastify/cors`), SSE pass-through |
| TypeScript | 7.0 (BFF), 6.0 (web) | types |
| React | 19.3 | user app and ops dashboard |
| Vite | 8.3 | build and dev server with proxy |
| Tailwind CSS | 4.3 | styling |
| nginx (`nginxinc/nginx-unprivileged` 1.27) | — | static files, `/api` proxy without buffering |

## Data and state

| Data | Stored in | Status |
|---|---|---|
| Source documents | `data/raw/` (tracked samples; the private PDF is git-ignored) | Used; S3 planned |
| Index | `data/index/`: `manifest.json` + store files (JSON, or FAISS binary) + `ingest_report.json` | Used |
| Feedback | `logs/feedback.jsonl` | Used; PostgreSQL planned |
| Agent runs | `logs/agent.db` (SQLite) | Used; PostgreSQL planned |
| Nonconformance reports | `logs/ncr.jsonl` (idempotent by key) | Used (a stand-in for a quality system) |
| LLM cache | `.cache/llm.jsonl` | Used |
| Experiment runs | `mlflow.db` (SQLite) + `mlartifacts/`, or `runs/runs.jsonl` | Used |

## Evaluation, tracking and testing

| Tool | Version | Role |
|---|---|---|
| Own benchmarks | — | retrieval (45 questions: hit@k, recall@k, MRR), generation (71 cases, 6 gates), agent (28 cases, 9 gates) |
| MLflow | 3.17 | experiment tracking (optional; JSONL tracker without dependencies) |
| pytest | 9.1 | Python tests (311 locally) |
| Vitest | 5.0 | BFF tests (46) and web tests (43) |
| React Testing Library + jsdom | 16 / 29 | web component tests |
| oxlint | 1.88 | web lint |
| httpx, httpx2 | — | HTTP clients for the API tests |

## Delivery and operations

| Tool | Version | Role | Status |
|---|---|---|---|
| Docker Desktop / Engine, Docker Compose | 29.9 / 5.6 | three images (api, bff, web), health-checked start-up | Used |
| GitHub Actions | — | CI: Python 3.10 and 3.12, BFF, web, retrieval gate, Compose smoke test; LLM gates on manual runs | Used, green |
| AWS (CloudFront, S3, ALB, ECS Fargate, RDS PostgreSQL + pgvector, Secrets Manager, CloudWatch, ECR, Budgets) | — | trial deployment | Planned ([20](20-aws-deployment.md)) |
| Terraform, GitHub OIDC | — | infrastructure as code, keyless deploys | Planned |
| Amazon Cognito | — | sign-in and roles | Planned ([21](21-enterprise-features.md)) |

## Deliberately not used

| Tool | Why not |
|---|---|
| LangChain, LlamaIndex | the project is about understanding each stage; every component is small, tested and swappable behind an interface, and benchmark results can be traced to code that is fully visible |
| A hosted vector database (Pinecone, Weaviate, OpenSearch) | the corpus is small; exact search is fast and is the reference; pgvector covers the cloud case without another service |
| An LLM-as-judge for evaluation | rule-based scoring is free, deterministic and re-scorable; the checks (facts, citations, tools, lots) are crisp |
