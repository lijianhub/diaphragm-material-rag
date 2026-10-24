# The User Web App (React)

`web/` is the browser app for asking questions: it streams the answer, shows the sources it was built from, links each citation to its source, and collects 👍/👎 feedback. It talks only to the backend-for-frontend ([16-backend-for-frontend.md](16-backend-for-frontend.md)), which talks to the Python API ([14-production-rag.md](14-production-rag.md)).

```
Browser (React, :5173) ──/api──► BFF (Node.js, :3000) ──► API (FastAPI, :8000)
```

Stack: React 19, TypeScript, Vite, Tailwind CSS 4, Vitest with React Testing Library and jsdom.

## Run it

```bash
pip install -e .[api] && python scripts/serve.py      # API on :8000 (RAG_LLM unset → offline extractive answers)
cd bff && npm install && npm run dev                  # BFF on :3000
cd web && npm install && npm run dev                  # app on http://localhost:5173
cd web && npm test && npm run typecheck && npm run build
```

In development Vite proxies `/api` to the BFF (`BFF_URL`, default `http://127.0.0.1:3000`). The browser therefore sees one origin: no CORS and no API key in the browser. In production the same layout comes from serving `web/dist` and the BFF behind one host.

## What the user sees

| Element | Behaviour |
|---|---|
| Question box | 1–2,000 characters; Enter asks, Shift+Enter adds a line; Ask is disabled while the question is blank |
| "Searching the documents…" | until the `sources` event arrives (tens of milliseconds) |
| Sources | shown as soon as retrieval finishes, before any answer text; each card has its label, a two-line preview and Show text / Hide text |
| Answer | text appears as it is generated, with a cursor; when `final` arrives, the checked answer **replaces** the streamed text |
| Citations | `[1]` becomes a button; clicking it highlights, expands and scrolls to source 1; sources the answer cites are marked "cited" |
| Refusal | amber panel titled "Not found in the documents" |
| Fallback | a note when the LLM was unavailable and the answer is quoted sentences |
| Invalid citations | a warning listing citation numbers that match no source |
| Stop | aborts the request; the BFF and API then stop the model, and the partial text stays, marked as incomplete |
| Errors | a plain-language message (rate limit with the `Retry-After` wait, not ready, unavailable) and the request ID to quote to support |
| Feedback | 👍 sends at once; 👎 opens an optional comment (up to 500 characters) and sends on Send |
| Details | model, prompt version, tokens in/out, retrieval / first-text / generation times, request ID |

## Why the final answer replaces the streamed text

The streamed deltas are the model's raw output. The `final` event carries the output **after** the pipeline's checks: citations validated against the sources, citations stripped from refusals, the refusal flag set. Only the final text is rendered with citation links, and only numbers that exist in `citations` become links; any other `[n]` stays plain text. A link to a source that does not exist would make an unsupported claim look supported.

## Reading the stream

`EventSource`, the browser's built-in SSE client, supports only GET, and asking is a POST with a JSON body. The app therefore reads the response with `fetch` and parses it with a small incremental parser (`src/lib/sse.ts`):

- **Network chunks ignore event boundaries.** One chunk can hold half an event or several events, so text after the last blank line is kept for the next chunk.
- **Network chunks ignore character boundaries.** A multi-byte UTF-8 character ("–", "µ") can be split between two chunks. `TextDecoder` in stream mode keeps the incomplete bytes until the rest arrives; decoding each chunk separately would produce replacement characters.
- CRLF line endings, comment lines (keep-alives) and multi-line `data:` are handled as the SSE specification says.
- A stream that ends without `final`, or with an `error` event, is reported as an error with the request ID.

## Feedback

`POST /api/feedback` → BFF (validation, authentication, rate limit) → `POST /feedback` on the API:

```json
{"request_id": "fecc9541d2954ffb863f5171ffc7eae7", "rating": "down", "comment": "missing the die angle"}
```

The API appends one JSON line per rating to `logs/feedback.jsonl` (`RAG_FEEDBACK_PATH`, git-ignored) and counts `rag_feedback_total{rating}` in `/metrics`. The rating is tied to the answer's **request ID**, which is also in the API's log line for that answer (model, prompt version, retrieval and generation times). A 👎 can therefore be traced to the exact model, prompt and sources that produced it, which is what makes feedback useful for evaluation: down-rated questions are candidates for new benchmark cases. The comment text is stored in the feedback file but never written to the request log, for the same reason the question text is not logged.

## Design decisions

| Decision | Alternatives | Why |
|---|---|---|
| Vite + React, no framework | Next.js | The app is one client-side page behind a BFF; server rendering adds nothing here |
| `fetch` + own SSE parser | `EventSource`, a library | `EventSource` cannot POST; the parser is ~40 lines and fully tested |
| Plain React state | Redux, TanStack Query | One screen, one request at a time; a state library would be more code than the state |
| Tailwind CSS | CSS modules, a component library | Small bundle (16 kB CSS, 4 kB gzipped), no design system to maintain |
| Vite dev proxy to the BFF | CORS from the browser | Same origin in development and production; the API key stays on the server |
| Tests with React Testing Library | Snapshot tests, end-to-end browser tests | Tests what the user sees and does (roles, labels, clicks); runs in seconds without a browser |

## Tests

`npm test` (24 tests, offline, about 5 s). `fetch` is replaced with a stub that returns a `ReadableStream`, so the app's real stream-reading code runs:

- **SSE parser:** split events, several events per chunk, CRLF split between chunks, comments, multi-line data
- **Citations:** valid, grouped, and invalid markers
- **API client:** protocol order, a UTF-8 character split between chunks, `error` events, a stream with no `final`, 429 with `Retry-After`, feedback bodies
- **App:** sources before the final answer; the final answer replaces the draft; citation click highlights and expands; refusal; rate-limit error with request ID; Stop aborts the request and keeps the partial text; 👍 and 👎-with-comment request bodies; blank question disables Ask

A local end-to-end check (API, BFF and Vite dev server, offline extractive answers) streamed `sources`, `delta` and `final` through the Vite proxy, and a feedback POST returned 201 and was written to the feedback file. The tests run in jsdom, which has no layout engine. The page has not yet been checked in a real browser, and browser end-to-end tests (Playwright) are a later step.

## Not yet included

- Conversation history and follow-up questions; each question is independent
- Sign-in (the BFF's API key is a stub; real identity is a later step)
- The ops dashboard (latency, errors, cost, feedback, evaluation trends), which is the next step
