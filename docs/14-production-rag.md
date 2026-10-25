# Production RAG: the HTTP Service

The pipeline becomes a service: an HTTP API with validated requests, request IDs, one structured log line per request and Prometheus metrics. It is the layer that the Node.js backend-for-frontend and the React apps (planned) will call.

## Run it

```bash
pip install -e .[api]                 # FastAPI + Uvicorn (optional extra)
python scripts/index.py               # build data/index first
python scripts/serve.py               # http://127.0.0.1:8000, interactive docs at /docs

curl -s localhost:8000/ask -H "Content-Type: application/json" \
     -d '{"question": "How is springback compensated?", "top_k": 3}'
curl -s localhost:8000/metrics
```

| Variable | Default | Meaning |
|---|---|---|
| `RAG_INDEX_DIR` | `data/index` | index loaded at startup |
| `RAG_LLM` | unset | e.g. `openai:<model>@low`; unset → offline extractive answers |
| `RAG_LLM_TIMEOUT` | `30` | seconds per LLM call (the SDK default of 120 s is too long for a user request) |
| `RAG_LLM_MAX_RETRIES` | `1` | SDK retries per LLM call |

## Endpoints

| Endpoint | Purpose | Status codes |
|---|---|---|
| `POST /ask` | answer with citations, sources, refused flag, model, prompt version, tokens, stage timings | 200; 422 invalid request; 503 no index; 500 unexpected error |
| `POST /ask/stream` | the same answer as Server-Sent Events: `sources`, `delta`…, `final` (or `error`); see [16-backend-for-frontend.md](16-backend-for-frontend.md) | 422 and 503 before the stream starts; afterwards an in-band `error` event |
| `POST /feedback` | records a 👍/👎 rating (`request_id` of the answer, `rating` `up`/`down`, optional `comment` up to 500 characters) as one line in `logs/feedback.jsonl` (`RAG_FEEDBACK_PATH`); see [17-web-app.md](17-web-app.md) | 201; 422 invalid request |
| `GET /health` | **liveness**: the process is up | 200 |
| `GET /ready` | **readiness**: the index is loaded; also reports chunks, embedder and answerer | 200, or 503 with a reason |
| `GET /metrics` | Prometheus text format | 200 |
| `POST /agent/runs`, `GET /agent/runs/{id}` | start an agent run (it completes, stops at a limit, or pauses for approval); read one with its trace, answer and citations (no transcript); see [13-agentic-rag.md](13-agentic-rag.md) | 200; 404; 422; 503 no index |
| `GET /ops/approvals`, `POST /ops/approvals/{id}` | the approval queue; `{decision: approve\|reject, reviewer, reason}` resumes the run | 200; 404; 409 already decided; 422 |
| `GET /ops/summary`, `/ops/feedback`, `/ops/evaluations`, `/ops/ingestion`, `/ops/agent-runs` | read-only JSON for the ops dashboard (metrics with estimated percentiles, recent feedback, tracked runs, last ingestion report); internal, behind the BFF's admin key; see [18-ops-dashboard.md](18-ops-dashboard.md) | 200; 422 invalid `limit` |

Request body: `question` (1–2,000 characters, not blank), `top_k` (1–10, default 3) and an optional `filter` on chunk metadata, e.g. `{"doc_type": ["md", "pdf"]}` (fields `source`, `doc_type`, `title`, `heading`, `page`, `sheet`; 1–20 values of at most 200 characters each; an unknown field is a 422), validated by Pydantic. The filter acts **inside** retrieval (see `SearchIndex.search`), so the `top_k` slots are filled with matching chunks; only the filter's field names are logged, never its values. Errors have one shape: `{"request_id", "error", "detail"}`.

Liveness and readiness are separate on purpose: an orchestrator restarts a process that fails liveness, but only stops sending traffic to one that is not ready. A missing index is a readiness problem, not a reason to restart.

## Request IDs

Every response carries `X-Request-ID`. If the caller sends one (the backend-for-frontend will), it is reused, so one ID follows a request through every layer; otherwise a new UUID is generated. A caller's ID is accepted only if it matches `[A-Za-z0-9._:-]{1,64}`, so it cannot inject fake lines into the logs. Error bodies include the ID too, so a user can quote it to support.

## Structured logs

One JSON object per request on stdout:

```json
{"ts": "2026-10-24T09:20:42.188+00:00", "event": "request", "request_id": "load-299", "method": "POST",
 "route": "/ask", "status": 200, "latency_ms": 2.28, "question_chars": 61, "question_sha256": "296e245f307a",
 "top_k": 3, "model": "extractive", "prompt_version": "grounded-v2+5647c36b", "used_llm": false,
 "refused": false, "fallback_reason": null, "sources": 3, "citations": 2, "invalid_citations": 0,
 "tokens_in": 0, "tokens_out": 0, "retrieval_ms": 1.12, "generation_ms": 0.12}
```

- **The question text is not logged**, only its length and a short SHA-256. Questions and documents may be confidential, and logs are copied, retained and read by more people than the documents. The hash still lets you find repeated questions.
- **Unexpected errors** log the exception type; the response is a generic `internal_error` with no stack trace or message, which could leak internals.
- `/metrics` scrapes are not logged; they arrive every few seconds and would drown the log.

## Metrics

| Metric | Type | Labels | Answers |
|---|---|---|---|
| `http_requests_total` | counter | method, route, status | traffic and error rate |
| `http_request_duration_seconds` | histogram | route | latency percentiles |
| `rag_stage_duration_seconds` | histogram | stage (retrieval, first_token, generation) | where the time goes; `first_token` only for streamed answers |
| `rag_answers_total` | counter | outcome (answered, refused, fallback) | refusal and fallback rates |
| `rag_llm_calls_total` | counter | model | call volume per model |
| `rag_llm_tokens_total` | counter | model, direction (input, output) | **cost**: tokens × the model's price |
| `rag_feedback_total` | counter | rating (up, down) | user satisfaction over time |
| `rag_agent_runs_total` | counter | status | agent outcomes, including pauses for approval |
| `rag_agent_tool_calls_total` | counter | tool, outcome | failing or misused tools (labels are registered tool names, never model text) |
| `rag_agent_approvals_total` | counter | decision | how often reviewers reject what the agent proposes |

- **Route templates, never raw paths:** `/ask`, and `unmatched` for 404s. A label taken from user input would create one time series per distinct URL (unbounded cardinality).
- **Cost is tracked as tokens per model**, not dollars: prices change and differ by contract, while tokens are exact. Dollars are a dashboard calculation.
- The registry is a small in-house implementation of the Prometheus text format (counters and histograms only), to avoid a dependency; Prometheus, Grafana or the ops dashboard can read it. Histogram buckets start at 1 ms: percentiles are estimated inside a bucket, and with 5 ms as the first bound every 2 ms request showed p95 = 4.8 ms.

## Measured on a laptop

Persistent index (52 chunks, hashing embedder), benchmark questions sent over HTTP from the same machine.

| Mode | Concurrency | Client latency p50 / p95 | Throughput | Server: retrieval p50 / generation p50 |
|---|---|---|---|---|
| extractive | 1 | 2.4 / 11.3 ms | ~262 req/s | 0.8 ms / 0.1 ms |
| extractive | 8 | 35 / 104 ms | ~175 req/s | 1.5 ms / 0.1 ms |
| `gpt-5.4-mini@low` | 1 | 1,908 / 2,489 ms | ~0.5 req/s | 9.7 ms / 1,885 ms |

- **The LLM is over 99% of the latency** when enabled. Retrieval optimisation will not change user-perceived speed; streaming the answer (planned in the backend-for-frontend) and smaller contexts will.
- **More threads lowered throughput.** The endpoint is synchronous and BM25 scoring is pure Python, so threads in one process compete for the interpreter lock. The service scales with **worker processes** (`uvicorn --workers N`) or replicas, not threads.
- **Measure the client too.** The first load test reported p50 ≈ 690 ms while the server logged ≈ 2 ms. The client created a new HTTP client per request, and constructing one costs about 200 ms. With one pooled client the true latency appeared. The server-side timings in the log made the discrepancy obvious.
- Six real LLM requests used 2,809 input and 354 output tokens; the refusal question returned `refused: true` with no citations, as in the generation benchmark.

## Design decisions

| Decision | Alternatives | Why |
|---|---|---|
| `create_app(settings, index=, llm=)` factory | A module-level `app` | Tests inject an index and a fake LLM; nothing loads at import time |
| Synchronous `/ask` (run in FastAPI's thread pool) | `async` endpoint | The pipeline and SDK calls are blocking; an `async` handler would block the event loop |
| A configured but unbuildable LLM fails at startup | Fail on the first request | Misconfiguration should stop a deployment, not the first user |
| Missing index → not ready (503) | Crash on startup | The index can be built after the container starts; readiness expresses that |
| No response cache in the service | Reuse `CachedLLM` | Caching users' questions and answers needs a retention and access policy first |
| In-house metrics registry | `prometheus_client` | Two metric types; no dependency; fully unit-tested |
| Rate limiting and authentication in the backend-for-frontend | In this service | Browser-facing concerns belong in the edge layer; this service stays an internal API |

## Not yet included

- **Ingestion and evaluation endpoints**: indexing and evaluation run as scripts.
- **Authentication, rate limiting** (backend-for-frontend), **TLS** (deployment).
- **Distributed tracing** (OpenTelemetry spans); the request ID is the current correlation key.
- **Alerting** on the metrics (deployment and operations steps).
