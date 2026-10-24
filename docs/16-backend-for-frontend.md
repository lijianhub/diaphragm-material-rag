# Backend-for-Frontend (Node.js) and Streaming

Browsers do not call the Python API directly. A small Node.js service in `bff/` (TypeScript, Fastify) sits in between and owns the browser-facing concerns: request IDs, authentication, rate limiting, CORS, validation and streaming. The Python service stays an internal API.

```text
Browser ──HTTP / SSE──▶ BFF (Node.js, :3000) ──HTTP / SSE──▶ RAG API (Python, :8000) ──▶ LLM
         X-Request-ID ─────────────────────────────▶ same ID in all three logs
         disconnect ──────▶ abort upstream ─────────▶ close generator ──▶ close LLM stream
```

## Run it

```bash
python scripts/serve.py                          # the RAG API on :8000
cd bff && npm install && npm run dev             # the BFF on :3000 (tsx watch)
curl -N localhost:3000/api/ask/stream -H "Content-Type: application/json" -d '{"question": "What is galling?"}'
npm test                                         # Vitest, offline (a fake upstream stands in for the API)
npm run typecheck && npm run build && npm start  # production build in bff/dist
```

| Variable | Default | Meaning |
|---|---|---|
| `RAG_API_URL` | `http://127.0.0.1:8000` | the Python API |
| `BFF_HOST`, `BFF_PORT` | `127.0.0.1`, `3000` | where the BFF listens |
| `BFF_API_KEYS` | empty | comma-separated Bearer keys; **empty turns authentication off** (local development only; a warning is logged) |
| `BFF_AGENT_TIMEOUT_MS` | `180000` | for agent runs and approvals, which take several model turns |
| `BFF_ADMIN_KEYS` | empty | Bearer keys for `/api/ops/*`; a user key gets 403. With user keys but no admin keys, the ops routes are closed |
| `BFF_RATE_LIMIT_MAX`, `BFF_RATE_LIMIT_WINDOW` | `30`, `1 minute` | per client, on the `/api/ask` routes |
| `BFF_CORS_ORIGINS` | `http://localhost:5173` | allowed browser origins (the React dev server) |
| `BFF_UPSTREAM_TIMEOUT_MS` | `60000` | until the API answers (JSON) or starts streaming |

## Routes

| Route | Does | Errors |
|---|---|---|
| `POST /api/ask` | validates, authenticates, rate-limits, forwards to `/ask`, passes status and body through | 400 invalid, 401, 429 (with `Retry-After`), 502 API unreachable, 504 API timeout, API errors passed through |
| `POST /api/ask/stream` | the same, then pipes the API's Server-Sent Events to the browser without buffering | as above before the stream starts; an `error` event if the upstream stream breaks |
| `POST /api/feedback` | validates (`request_id` pattern, `rating` `up`/`down`, `comment` up to 500 characters), authenticates, rate-limits, forwards to `/feedback` | as for `/api/ask` |
| `POST /api/agent/runs`, `GET /api/agent/runs/:id` | user key, rate limit; question 1–2,000 characters; run ID must be 32 hex characters | 400, 401, 429, 502, 504 |
| `POST /api/ops/approvals/:id` | admin key; `decision` and `reason` only: **the BFF sets `reviewer` from the matched admin key** (`admin-1`, …; `local-dev` without keys), and a `reviewer` in the body is stripped | 400, 401, 403, 409 passed through |
| `GET /api/ops/summary`, `feedback`, `evaluations`, `ingestion`, `approvals`, `agent-runs` | admin key, rate limit, valid `limit` only; forwards to `/ops/*` as GET | 401, 403 (user key), 400, 502, 504 |
| `GET /api/health` | liveness of the BFF itself (no upstream call, no authentication) | — |
| `GET /api/ready` | the API's readiness, passed through | 503 when the API is down or not ready |

Every response carries `X-Request-ID`. A caller's ID is kept only if it matches `[A-Za-z0-9._:-]{1,64}`; otherwise a new one is generated. The ID is forwarded to the API, so one request has the same ID in the browser, the BFF log (Fastify's JSON logger, field `request_id`) and the API log.

## The stream protocol

`POST /ask/stream` (API) and `POST /api/ask/stream` (BFF) send:

| Event | When | Data |
|---|---|---|
| `sources` | right after retrieval (~5–20 ms) | `{request_id, sources: [{number, label, source, text, score}]}` |
| `delta` | for each piece of generated text | `{text}` |
| `final` | at the end | the same body as `POST /ask`: checked answer, citations, refused flag, model, prompt version, tokens, timings (including `first_token_ms`) |
| `error` | on an unexpected failure after the stream started | `{request_id, error: "internal_error" \| "upstream_error"}` |

**The `final` event is authoritative.** Refusal citations are removed and citation numbers are validated on the complete text (the generation-evaluation guards), so a client shows the deltas while they arrive and then **replaces** them with `final.answer`. If the LLM fails mid-stream, `final` carries the extractive fallback answer with `fallback_reason`.

## Cancellation

When the user closes the page, generation should stop: the model is billed per generated token.

1. The browser disconnects → the BFF's response emits `close` → it aborts its upstream `fetch`.
2. The API's next write fails → `ClosingStreamingResponse` closes the body iterator → the pipeline generator's `finally` closes the LLM stream → the provider stops generating.
3. The API logs the stream with `"disconnected": true`; the BFF logs `client_disconnected: true`.

**Step 2 was broken until a real-server test showed it.** With an ASGI 2.4 server (Uvicorn), Starlette does not watch for disconnects while streaming. The next write after a disconnect raises, and the exception leaves the `async for` loop without `aclose()`, so the generator was abandoned mid-stream: its `finally` never ran, and the LLM kept generating for a client that was gone. Unit tests had passed, because they closed the generator directly. A test now starts a real Uvicorn server, disconnects after the first delta, and checks that the fake LLM's stream was closed. It fails without the fix.

The API notices a disconnect at its next write, so cancellation takes effect within one text piece. During a model's silent reasoning phase there are no writes, so cancellation waits for the first token.

## Measured (laptop, `gpt-5.4-mini@low`, through the BFF)

| Question | Sources shown | First text | Final answer | Output tokens |
|---|---|---|---|---|
| springback compensation | 20 ms | 2,400 ms | 2,704 ms | 60 |
| aging atmosphere | 4 ms | 1,272 ms | 1,811 ms | 134 |
| fibrous fracture surface | 4 ms | 902 ms | 1,132 ms | 30 |

- **Sources appear almost at once**, and they are what users start reading.
- **Streaming the text saves 0.2–0.5 s here, not the full answer time.** A reasoning model spends most of the time before its first token, and grounded answers are short. Longer answers or non-reasoning models gain more.
- A disconnect after the first delta stopped the stream after 4 deltas (1.2 s); the API logged `disconnected: true`.

## Design decisions

| Decision | Alternatives | Why |
|---|---|---|
| A BFF in Node.js | React calls FastAPI directly | Keeps sessions, rate limits, CORS and streaming out of the Python service; a common pattern. A minimal deployment could skip it |
| Fastify + official plugins (`@fastify/rate-limit`, `@fastify/cors`) | Express, custom middleware | Schema validation, a JSON logger and request IDs built in; maintained plugins |
| SSE | WebSockets | One-way server-to-browser text over plain HTTP; proxies and load balancers handle it; the LLM APIs use it too |
| Pass the upstream stream through byte for byte | Parse and re-emit events | Nothing to keep in sync with the API's event format; no added latency |
| Node's built-in `fetch` | `axios`, a new client per request | Connection pooling by default; a client per request cost ~200 ms in the step-16 load test |
| Authentication as a Bearer API-key stub | Real identity now | Proves the hook and the 401 path; real identity (Cognito) is a later step. Keys are compared in constant time |
| Rate limit per API key, else per IP | Per IP only | Several users behind one office IP should not share one limit |
| Unknown request fields stripped (Fastify's default `removeAdditional`) | Reject them | Matches the API, which ignores extra fields; a test pins the behaviour so it is a choice, not an accident |

## Lessons from the tests

- **A framework option silently bypassed the request-ID check.** With `requestIdHeader: "x-request-id"`, Fastify uses the header value as-is and never calls `genReqId`, so a 100-character or quoted ID went straight into logs and upstream. `requestIdHeader: false` makes `genReqId` (with the whitelist) run for every request. Found by a test that sent an over-long ID.
- **`additionalProperties: false` does not reject in Fastify by default;** it removes the field. Found by a test that expected 400.
- **A header containing a newline cannot reach a handler:** Node's HTTP parser rejects it first, so it is not a useful log-injection test case; over-long and quoted IDs are.
