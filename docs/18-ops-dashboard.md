# The Ops Dashboard

`/ops` in the web app ([17-web-app.md](17-web-app.md)) is a page for the people who run the service. It answers five questions:

| Question | Panel | Source |
|---|---|---|
| Is it up, fast and error-free? | Service: status, requests, server-error rate, `/ask` p50/p95, refusal and fallback rates, requests per minute, stage latencies | `GET /ops/summary` (the metrics registry) |
| What does it cost? | LLM usage: calls, input and output tokens per model, tokens per call | `GET /ops/summary` |
| Do users find the answers helpful? | Feedback: share of 👍, recent ratings with comments and the answer's request ID | `GET /ops/feedback` (`logs/feedback.jsonl`) |
| Did a change make the system better or worse? | Evaluation runs: trend of a chosen metric, runs table with commit, dirty flag and data version | `GET /ops/evaluations` (`runs/runs.jsonl`) |
| Did the last indexing run fail on any files? | Ingestion: loaded, failed (with reason), duplicates, skipped | `GET /ops/ingestion` (`data/index/ingest_report.json`) |

```
Browser /ops ──/api/ops/*──► BFF (admin key) ──/ops/*──► API ── metrics registry, feedback file, run log, ingestion report
```

## Run it

```bash
python scripts/index.py                              # writes data/index/ingest_report.json
python scripts/evaluate.py --track jsonl             # each run appends to runs/runs.jsonl (commit first)
python scripts/serve.py                              # API on :8000
cd bff && BFF_API_KEYS=<user key> BFF_ADMIN_KEYS=<admin key> npm run dev   # or no keys at all: everything open (development)
cd web && npm run dev                                # http://localhost:5173/ops
```

The page refreshes every 10 seconds (Pause stops it). It is loaded as a separate JavaScript chunk (5.7 kB gzipped), so users of the Q&A page never download it.

## Access

The ops views show user comments and service internals, so the BFF puts `/api/ops/*` behind **admin keys** (`BFF_ADMIN_KEYS`):

| Keys configured | No key | User key | Admin key |
|---|---|---|---|
| none (development) | open | open | open |
| user keys only | 401 | **403** | — |
| user and admin keys | 401 | **403** | 200 |

A user key is valid but not enough (403, not 401). With only user keys configured, the ops routes are closed to everyone, rather than open by accident. The dashboard asks for the key on 401/403 and keeps it in the tab's memory only, never in browser storage. Real identity (Cognito) replaces the keys in step 24. The API's `/ops/*` routes have no authentication of their own: the API is internal, reached only through the BFF.

## How the numbers are made

**Traffic excludes health checks and the dashboard itself.** The totals count only `/ask`, `/ask/stream` and `/feedback`. Otherwise watching the dashboard would inflate the request count it shows, and so would orchestrator health checks. Every route is still listed in `routes`.

**Requests per minute come from counter differences.** The API's counters only go up, from process start. The page takes a snapshot every 10 s and plots (requests now − requests before) / minutes between them, over the time the page has been open (up to 15 minutes). A restart resets the counters, so a lower count or a new start time starts a new series instead of a negative rate. Longer history is a job for Prometheus (below).

**Percentiles are estimates from histogram buckets**, computed like Prometheus' `histogram_quantile`: find the bucket that holds the p-th observation and interpolate linearly inside it. An estimate is only as fine as its bucket. The first measurement showed **p95 = 4.8 ms for every route**: all requests took about 2 ms and fell into the first bucket (0–5 ms), and 0.95 × 5 = 4.75. Adding 1 ms and 2.5 ms buckets fixed the resolution. With 20 offline requests:

| `/ask` latency | Estimated from buckets | Exact, from the request log |
|---|---|---|
| p50 | 1.9 ms | 2.18 ms |
| p95 | 4.4 ms | 3.61 ms |

The error stays within the bucket (2.5–5 ms). Above the last bucket (30 s) the estimate is the bound itself, a lower limit.

**`/ask/stream` latency is time to the first byte**, not to the end of the stream: the HTTP middleware measures until the response starts. The stream's real timings are the `first_token` and `generation` stages.

**Requests rejected by the BFF never reach the API**, so the API's 4xx count only covers requests that passed BFF validation. The BFF logs its own rejections.

**Feedback totals come from the file, not the counter**, so they survive restarts. The counter (`rag_feedback_total`) is still exported for Prometheus.

**Evaluation trends join only comparable runs.** Each run carries a **data version**: a hash of the SHA-256 tags of its evaluation files and corpus (step 15). The line joins only runs with the same data version as the newest run that has the chosen metric. Runs on other data are listed in amber and counted ("1 run on other data not joined"), because a score change across data versions may come from the data, not the system. Runs measured with uncommitted changes are marked **dirty**.

**Damaged lines are skipped and counted**, never fatal: a dashboard that breaks on one bad line in a log is useless exactly when something has gone wrong.

## Why a custom dashboard, and when to use Grafana instead

The standard tooling for service metrics is Prometheus (scraping `/metrics`, which already exists) plus Grafana. It keeps history across restarts, supports alerting, and aggregates across replicas, none of which this page does. This dashboard is worth having for the **domain views** that generic tools do poorly: feedback with comments linked to request IDs, evaluation runs with comparability rules, and ingestion failures with reasons. It is also a project goal for learning React. In a deployment (step 23), Prometheus and Grafana would take over the service panels, with alerts on the same numbers, and this page would keep the domain panels.

## Design decisions

| Decision | Alternatives | Why |
|---|---|---|
| Read-only `/ops/*` JSON endpoints in the API | The BFF parsing `/metrics`; the browser reading files | One place computes the numbers, tested in Python; the BFF stays a thin proxy |
| Polling every 10 s | WebSockets, SSE | Numbers change slowly; polling is simple and cache-friendly |
| Rates computed in the browser from snapshots | Server-side time series | No storage in the API; long history belongs to Prometheus |
| Separate admin keys | One key for everything | Users must not read other users' comments |
| Feedback and runs in JSONL files | A database | Enough for one process; a database comes with deployment |
| Single-series SVG charts, written by hand | A chart library | Two small charts; each has a table with the same numbers |
| `ingest_report.json` next to the index | Parsing script output | Console output is gone after the run; the report stays until the next one |

## Tests

- **Python (`tests/test_api.py`):** percentile interpolation, including the empty case and values above the last bucket; traffic excluding health checks and dashboard polls; 422s counted as client errors; feedback newest first with damaged lines skipped; runs newest first, with data versions that differ when a question file changes; the ingestion report with repository-relative paths.
- **BFF (`bff/test/app.test.ts`):** each view forwarded as a GET with the request ID; `limit` validated, other query parameters never forwarded; the 401/403/200 key matrix; ops closed when only user keys exist; 502 when the API is down.
- **Web (`web/src/ops/*.test.ts(x)`, `Root.test.tsx`):** rates from snapshots, including a restart; trend joins only comparable runs; every panel rendered from fixtures; the key prompt on 401 and the key sent afterwards; one failing view leaves the others intact; polling stops when paused; the chart tooltip; `/ops` routing.

A local end-to-end run (API, BFF with user and admin keys, Vite) returned the 401/403/200 matrix through the Vite proxy and real numbers in every view: the traffic count excluded the dashboard's own polls, and five tracked runs pointed at one clean commit. The page itself has not yet been checked in a real browser.

## Not yet included

- **History and alerts** (Prometheus and Grafana, at deployment)
- **Aggregation across replicas:** each API process has its own counters
- **The approval queue**, which arrives with the agent (step 20): actions with side effects wait there for a person
- **MLflow runs:** the page reads the JSONL run log; use `mlflow ui` for MLflow runs
