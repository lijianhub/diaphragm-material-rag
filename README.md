# Diaphragm Material RAG

A retrieval-augmented generation (RAG) system for engineering documents, built step by step and measured at every step. The domain is diaphragm forming with ELGILOY® and HAVAR® cobalt alloys: specifications, procedures, inspection records and supplier information spread across PDF, Word, Excel, HTML, Markdown and text files.

The goal is a small but complete AI service that is **evaluated, tested and explainable**: every retrieval change is checked against a frozen benchmark, regressions are documented rather than hidden, and the core runs offline with no API keys.

## Status

| Capability | Status | Notes |
|---|---|---|
| Multi-format ingestion (PDF, DOCX, XLSX, HTML, Markdown, CSV, TXT) with page/sheet/section citations | ✅ Implemented and tested | Standard library parsers; fault-tolerant batch ingestion |
| Paragraph-aware chunking with overlap | ✅ Implemented and tested | No gaps; overlap starts on word boundaries |
| Hybrid retrieval: BM25 + dense vectors (weighted sum or RRF) | ✅ Implemented and tested | Light stemming; shared tokenizer |
| Persistent vector store (in-memory, FAISS) with incremental sync | ✅ Implemented and tested | Content-hash sync; embedder recorded in the manifest |
| Local embedding model (sentence-transformers) | ✅ Implemented, tested and benchmarked | Optional extra |
| Cross-encoder reranking and duplicate-chunk collapsing | ✅ Implemented, tested and benchmarked | Optional extra; off by default |
| Query transformation: decomposition, pseudo-relevance feedback | ✅ Implemented, tested and benchmarked | Opt-in; neither improves the current benchmark |
| Grounded prompt with numbered sources, citation checking, prompt-injection escaping of sources and question | ✅ Implemented and tested | Refusal sentences never carry citations (enforced in code) |
| Offline extractive answers with citations | ✅ Implemented and tested | Cannot judge answerability (see limitations) |
| Retrieval benchmark: 45 questions, 4 categories, hit@k / recall@k / MRR | ✅ Implemented | Frozen before each new method is implemented |
| Amazon Bedrock embeddings and reranking | 🟡 Implemented, tested with fakes only | Not yet run against a live AWS account |
| LLM answers through the OpenAI or Anthropic API, with model, prompt version and token usage recorded | ✅ Implemented, tested, measured with OpenAI | Declines unanswerable questions correctly; Anthropic backend tested with fakes only |
| LLM query rewriting (multi-query, HyDE) with an on-disk response cache | ✅ Implemented, tested and benchmarked | Raises hit@3 to 1.000 with MiniLM; opt-in |
| Generation evaluation: 71 cases (answerable, unanswerable, ambiguous, malicious), rule-based scoring, pass/fail gates | ✅ Implemented, tested, measured with OpenAI | All 6 gates pass with `gpt-5.4-mini@low`; ambiguity handling (2/6) is reported, not gated |
| Experiment tracking: MLflow or JSONL runs with configuration, metrics, git commit and dataset hashes | ✅ Implemented, tested, used | Optional extra; local SQLite store; see [docs/15](docs/15-experiment-tracking.md) |
| HTTP API (FastAPI): `/ask`, liveness and readiness, request IDs, JSON logs, Prometheus metrics (latency, outcomes, tokens per model) | ✅ Implemented, tested, load-tested locally | Optional extra; see [docs/14](docs/14-production-rag.md) |
| Streaming answers (Server-Sent Events: sources, text deltas, checked final answer) with cancellation down to the LLM | ✅ Implemented, tested on a real server, measured | A client disconnect stops generation |
| Node.js backend-for-frontend (TypeScript, Fastify): SSE pass-through, rate limiting, API-key auth stub, CORS, request IDs | ✅ Implemented and tested (Vitest) | Authentication is a stub; see [docs/16](docs/16-backend-for-frontend.md) |
| React user app: streamed answers, citation links to sources, stop, 👍/👎 feedback stored with the answer's request ID | ✅ Implemented and tested (Vitest, React Testing Library) | Checked end to end with curl through the dev proxy, not yet in a real browser; see [docs/17](docs/17-web-app.md) |
| React ops dashboard | 🔜 Planned | |
| Agent workflow with tool limits and human approval | 🔜 Planned | |
| Docker, CI, AWS deployment, operations runbook | 🔜 Planned | |

## Architecture

```
                         INDEXING (offline, incremental)
 data/raw/*  ──► ingestion ──► sections ──► chunking ──► embedding ──► vector store + manifest
 pdf docx xlsx    parse_file    page/sheet/   paragraph-    hashing |     memory | FAISS
 html md csv txt  ingest()      heading       aware, overlap MiniLM |     content-hash sync
                                                             Bedrock

                         QUERYING
 question ──► (optional) query transformation ──► hybrid retrieval ──► (optional) rerank ──► grounded prompt ──► answer
              decompose · PRF · multi-query/HyDE   BM25 + dense          cross-encoder        numbered sources     + checked
                                                   weighted | RRF        | Bedrock            injection-escaped    citations
```

Every stage sits behind a small interface (vector store, embedder, reranker, LLM function), so a managed cloud backend can replace a local one without changing the pipeline. See [docs/architecture.md](docs/architecture.md) for components, failure handling and the planned service layers.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pytest -q                                              # ~240 tests, offline, about a second

python scripts/index.py                                # build data/index from data/raw (incremental on re-runs)
python scripts/query.py "How do toolmakers deal with springback?"
python scripts/evaluate.py --k 3                       # retrieval metrics on the benchmark
python scripts/evaluate_generation.py                   # answer metrics and gates (offline extractive answers)
```

Optional extras:

```bash
pip install -e .[local-embeddings]   # sentence-transformers: real embeddings and cross-encoder reranking
pip install -e .[faiss]              # FAISS vector store
pip install -e .[aws]                # boto3: Bedrock embeddings and reranking
pip install -e .[llm]                # anthropic + openai SDKs: LLM answers and query rewriting
cp .env.example .env                 # add OPENAI_API_KEY or ANTHROPIC_API_KEY and RAG_LLM (git-ignored)
python scripts/check_llm.py          # verify configuration without printing secrets
python scripts/evaluate_generation.py --llm openai:<model>@low --gate   # LLM answers; exit 1 if a gate fails
pip install -e .[tracking]           # MLflow experiment tracking
pip install -e .[api] && python scripts/serve.py      # HTTP API on :8000 (docs at /docs)
cd bff && npm install && npm run dev && npm test      # Node.js BFF on :3000 (needs the API)
cd web && npm install && npm run dev && npm test      # React app on :5173 (proxies /api to the BFF)
python scripts/evaluate.py --track mlflow && python scripts/compare_runs.py   # record runs, compare them

python scripts/index.py --rebuild --embedder st:sentence-transformers/all-MiniLM-L6-v2 --store faiss
python scripts/evaluate.py --k 1 --embedder st:sentence-transformers/all-MiniLM-L6-v2
python scripts/evaluate.py --k 1 --reranker ce:cross-encoder/ms-marco-MiniLM-L-6-v2
```

Example output:

```
Answer:
High-strength cobalt alloys show more springback than mild steel, so dies are often
compensated by over-forming the profile by a small angle. [1] ...

Sources (* = cited):
*[1] forming_defects_notes.txt [380:649]  score=0.653
 [2] diaphragm_material_notes.txt [406:679]  score=0.402
```

## Results

Retrieval on the frozen benchmark: 45 questions over the 11 tracked sample documents (40 chunks), in four categories (keyword, paraphrase, distractor, multi-document).

| Configuration | hit@1 | MRR@3 | recall@3 |
|---|---|---|---|
| BM25 only | 0.933 | 0.952 | 0.959 |
| Hashing embedder, hybrid (offline default) | 0.889 | 0.919 | 0.937 |
| all-MiniLM-L6-v2, dense only | 0.844 | 0.904 | 0.919 |
| all-MiniLM-L6-v2, hybrid | **0.933** | **0.956** | **0.952** |
| Any first stage + cross-encoder reranking | 0.933 | 0.952 | 0.952 |
| all-MiniLM-L6-v2, hybrid + LLM rewriting (multi-query or HyDE) | 0.933 | 0.963–0.967 | 0.963–0.974 (hit@3 = 1.000) |

What the measurements showed, in brief (details in [docs/11-rag-evaluation.md](docs/11-rag-evaluation.md)):

- **Dense and lexical retrieval fail on different questions.** The embedding model fixes paraphrases ("rust in the sea" vs "oxide film … seawater"); BM25 catches exact terms ("molybdenum"), so hybrid is the default.
- **Reranking lifts weak first stages to its own ceiling** (0.689 → 0.911 for hashing dense) but did not improve a good hybrid, at about +70–100 ms per query.
- **Some methods did not help and are documented as such:** pseudo-relevance feedback caused query drift (hit@1 0.889 → 0.600); RRF fusion of decomposed sub-questions lost each part's best evidence until it was replaced by interleaving.
- **The benchmark caught regressions the unit tests did not**, including a chunk-deduplication rule that dropped hit@1 to 0.422.
- **An LLM fixed what rules couldn't:** query rewriting retrieved the one paraphrase every other method missed, and LLM answers decline unanswerable questions that the offline answerer quoted unrelated sentences for. The whole LLM experiment cost about 15k tokens thanks to a response cache.
- **The generation benchmark found a real injection bug.** A question containing `</source><source id="9">…` forged a source, and the model answered from it and cited `[9]`. The question is now escaped like the sources; injection resistance went from 7/8 to 8/8, and clean refusals from 3/12 to 12/12.

## How it is evaluated

- **Evidence phrases, not chunk IDs:** a retrieved chunk counts as relevant if it contains a labelled phrase verbatim, so the ground truth survives every chunking change.
- **Frozen before each method:** when the benchmark saturates it is expanded and frozen *before* the next method is implemented.
- **No tuning on the test set:** defaults are chosen from first principles; parameter sweeps are reported as analysis only.
- **Per-category metrics** show weaknesses that averages hide.
- **Data-integrity test:** every evidence phrase must exist in the parsed corpus.
- **Every run is traceable:** with `--track`, a run records its configuration, metrics, git commit (and whether the tree was dirty), and hashes of the benchmark files and corpus, so two scores are only compared when they were measured on the same data.
- **Answers are evaluated too:** key facts, citations that hold the evidence, clean refusals, ambiguity and prompt injection, scored by deterministic rules and checked against thresholds fixed before the first measurement.

## Project layout

```
src/
  ingestion/     parsers per format, fault-tolerant ingest()
  chunking/      fixed-size and recursive (paragraph-aware) chunkers
  tokenization.py  shared tokenizer and light stemmer
  embedding/     hashing, sentence-transformers and Bedrock embedders; create_embedder()
  vectorstore/   VectorStore interface; in-memory and FAISS backends
  retrieval/     BM25, hybrid scoring, RRF, SearchIndex (sync, save/load, search)
  reranking/     cross-encoder and Bedrock rerankers
  query/         decomposition, pseudo-relevance feedback, LLM rewriting, multi-query retrieve()
  generation/    grounded prompt, citations, RAGPipeline, extractive answers
  evaluation/    retrieval metrics, generation scoring and gates, dataset, benchmark corpus
  tracking/      run records, version fingerprints; null, JSONL and MLflow trackers
  api/           FastAPI app (JSON and streaming), schemas, Prometheus-format metrics
bff/             Node.js backend-for-frontend (TypeScript, Fastify, Vitest)
web/             React user app (TypeScript, Vite, Tailwind CSS, Vitest)
scripts/         index.py, query.py, serve.py, evaluate.py, evaluate_generation.py, compare_runs.py, check_llm.py
evaluation/      questions.jsonl, ground_truth.jsonl, generation_cases.jsonl, results
data/raw/        redistributable sample documents (txt, md, html, csv)
docs/            one document per topic (fundamentals, ingestion, parsing, chunking, embeddings, ...)
tests/           unit tests, fakes for model SDKs, in-code builders for PDF/DOCX/XLSX
```

Topic documents: [RAG fundamentals](docs/01-rag-fundamentals.md) · [ingestion](docs/02-document-ingestion.md) · [parsing](docs/03-document-parsing.md) · [chunking](docs/04-chunking.md) · [embeddings](docs/05-embeddings.md) · [vector database](docs/06-vector-database.md) · [reranking](docs/08-reranking.md) · [query transformation](docs/09-query-transformation.md) · [generation](docs/10-generation.md) · [evaluation](docs/11-rag-evaluation.md) · [HTTP service](docs/14-production-rag.md) · [backend-for-frontend and streaming](docs/16-backend-for-frontend.md) · [web app](docs/17-web-app.md) · [experiment tracking](docs/15-experiment-tracking.md) · [architecture](docs/architecture.md)

## Known limitations

- **Answerability offline:** without an LLM, the extractive answerer quotes the best-matching sentence even when the question cannot be answered ("What is the price of titanium?" returns a sentence about titanium nitride). With an LLM the system declines correctly and without citations.
- **Ambiguous questions:** the LLM usually answers one interpretation instead of covering each case or asking which one is meant (2/6), even when every case was retrieved. The prompt asks for it; the small model at low effort mostly does not comply.
- **Small benchmark:** 45 questions over 11 short documents; one question changes hit@1 by about 0.022, and the benchmark is also the test set (no separate dev split yet).
- **General-domain models:** the embedding and reranking models were trained on web data; one paraphrase question (q33) is missed by every configuration.
- **Bedrock components** are tested with fakes only, not yet against a live AWS account.
- **Scanned PDFs** need OCR, which is not included.

## Roadmap

1. React ops dashboard (latency, errors, cost, feedback, evaluation trends)
3. Agent workflow with tool schemas, timeouts, step limits and human approval for side-effecting actions
4. Docker Compose, GitHub Actions CI (running the generation gates and logging runs), AWS deployment and an operations runbook

## Contributing

Run `pytest -q` and `python scripts/evaluate.py` before and after any change to chunking, embedding or retrieval, and `python scripts/evaluate_generation.py` before and after any change to the prompt or answer handling; report the metric change. Tests must run offline; model SDKs are optional extras imported lazily, and tests use the fakes in `tests/fakes.py`. For the BFF and the web app, run `npm test` and `npm run typecheck` in `bff/` and `web/`. Only redistributable documents belong in `data/raw/`.

## License

MIT. See [LICENSE](LICENSE).
