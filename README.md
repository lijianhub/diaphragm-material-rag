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
| Grounded prompt with numbered sources, citation checking, prompt-injection escaping | ✅ Implemented and tested | |
| Offline extractive answers with citations | ✅ Implemented and tested | Cannot judge answerability (see limitations) |
| Retrieval benchmark: 45 questions, 4 categories, hit@k / recall@k / MRR | ✅ Implemented | Frozen before each new method is implemented |
| Amazon Bedrock embeddings and reranking | 🟡 Implemented, tested with fakes only | Not yet run against a live AWS account |
| LLM answers through the Anthropic or OpenAI API, with model, prompt version and token usage recorded | 🟡 Implemented, tested with fakes only | Optional `[llm]` extra; keys from the environment or a git-ignored `.env`; not yet run against a live API |
| LLM query rewriting (multi-query, HyDE) | 🟡 Implemented, tested with fakes only | Not yet measured on the benchmark |
| Generation evaluation (faithfulness, citations, refusals on unanswerable questions) | 🔜 Planned | |
| Experiment tracking (MLflow) | 🔜 Planned | |
| FastAPI service with structured logs and metrics | 🔜 Planned | |
| Node.js backend-for-frontend, React user app and ops dashboard | 🔜 Planned | |
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
pytest -q                                              # ~160 tests, offline, about a second

python scripts/index.py                                # build data/index from data/raw (incremental on re-runs)
python scripts/query.py "How do toolmakers deal with springback?"
python scripts/evaluate.py --k 3                       # retrieval metrics on the benchmark
```

Optional extras:

```bash
pip install -e .[local-embeddings]   # sentence-transformers: real embeddings and cross-encoder reranking
pip install -e .[faiss]              # FAISS vector store
pip install -e .[aws]                # boto3: Bedrock embeddings and reranking
pip install -e .[llm]                # anthropic + openai SDKs: LLM answers and query rewriting
cp .env.example .env                 # add OPENAI_API_KEY or ANTHROPIC_API_KEY and RAG_LLM (git-ignored)
python scripts/check_llm.py          # verify configuration without printing secrets

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

What the measurements showed, in brief (details in [docs/11-rag-evaluation.md](docs/11-rag-evaluation.md)):

- **Dense and lexical retrieval fail on different questions.** The embedding model fixes paraphrases ("rust in the sea" vs "oxide film … seawater"); BM25 catches exact terms ("molybdenum"), so hybrid is the default.
- **Reranking lifts weak first stages to its own ceiling** (0.689 → 0.911 for hashing dense) but did not improve a good hybrid, at about +70–100 ms per query.
- **Some methods did not help and are documented as such:** pseudo-relevance feedback caused query drift (hit@1 0.889 → 0.600); RRF fusion of decomposed sub-questions lost each part's best evidence until it was replaced by interleaving.
- **The benchmark caught regressions the unit tests did not**, including a chunk-deduplication rule that dropped hit@1 to 0.422.

## How it is evaluated

- **Evidence phrases, not chunk IDs:** a retrieved chunk counts as relevant if it contains a labelled phrase verbatim, so the ground truth survives every chunking change.
- **Frozen before each method:** when the benchmark saturates it is expanded and frozen *before* the next method is implemented.
- **No tuning on the test set:** defaults are chosen from first principles; parameter sweeps are reported as analysis only.
- **Per-category metrics** show weaknesses that averages hide.
- **Data-integrity test:** every evidence phrase must exist in the parsed corpus.

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
  evaluation/    metrics, dataset, benchmark corpus
scripts/         index.py, query.py, evaluate.py
evaluation/      questions.jsonl, ground_truth.jsonl, results.jsonl
data/raw/        redistributable sample documents (txt, md, html, csv)
docs/            one document per topic (fundamentals, ingestion, parsing, chunking, embeddings, ...)
tests/           unit tests, fakes for model SDKs, in-code builders for PDF/DOCX/XLSX
```

Topic documents: [RAG fundamentals](docs/01-rag-fundamentals.md) · [ingestion](docs/02-document-ingestion.md) · [parsing](docs/03-document-parsing.md) · [chunking](docs/04-chunking.md) · [embeddings](docs/05-embeddings.md) · [vector database](docs/06-vector-database.md) · [reranking](docs/08-reranking.md) · [query transformation](docs/09-query-transformation.md) · [generation](docs/10-generation.md) · [evaluation](docs/11-rag-evaluation.md) · [architecture](docs/architecture.md)

## Known limitations

- **Answerability:** the offline extractive answerer quotes the best-matching sentence even when the question cannot be answered ("What is the price of titanium?" returns a sentence about titanium nitride). Deciding whether the sources answer a question needs the planned LLM step and generation evaluation.
- **Small benchmark:** 45 questions over 11 short documents; one question changes hit@1 by about 0.022, and the benchmark is also the test set (no separate dev split yet).
- **General-domain models:** the embedding and reranking models were trained on web data; one paraphrase question (q33) is missed by every configuration.
- **Bedrock components** are tested with fakes only, not yet against a live AWS account.
- **Scanned PDFs** need OCR, which is not included.

## Roadmap

1. Generation evaluation: faithfulness, citation accuracy, refusals on unanswerable, ambiguous and malicious questions
2. Experiment tracking with MLflow
3. FastAPI service with request IDs, structured logs and metrics
4. Node.js backend-for-frontend (TypeScript), React user app and React ops dashboard
5. Agent workflow with tool schemas, timeouts, step limits and human approval for side-effecting actions
6. Docker Compose, GitHub Actions CI, AWS deployment and an operations runbook

## Contributing

Run `pytest -q` and `python scripts/evaluate.py` before and after any change to chunking, embedding or retrieval, and report the metric change. Tests must run offline; model SDKs are optional extras imported lazily, and tests use the fakes in `tests/fakes.py`. Only redistributable documents belong in `data/raw/`.

## License

MIT. See [LICENSE](LICENSE).
