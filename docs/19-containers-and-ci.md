# Containers and Continuous Integration

The whole system runs with one command, and every push is checked by GitHub Actions.

```
                 published :8080                     Compose network only
Browser ──► web (nginx: React files, /api proxy) ──► bff (Node.js) ──► api (Python, index built in)
                                                                         └─ volume api-logs: feedback, agent runs, reports
```

## Run it

```bash
docker compose up -d --build --wait     # http://localhost:8080, ops dashboard: /ops
python scripts/smoke.py            # 14 end-to-end checks through nginx
docker compose logs -f api              # one JSON line per request
docker compose down                     # keeps the api-logs volume; add -v to delete it
```

Configuration comes from the shell or the git-ignored `.env`, which Compose reads for the `${...}` values in `compose.yaml`:

| Variable | Default | Meaning |
|---|---|---|
| `RAG_LLM` | empty | model for answers and the agent; empty means offline answers and the rule-based planner (0 tokens) |
| `OPENAI_API_KEY`, `ANTHROPIC_API_KEY` | empty | passed to the API container at run time, never built into an image |
| `BFF_API_KEYS`, `BFF_ADMIN_KEYS` | empty | as in [16-backend-for-frontend.md](16-backend-for-frontend.md); empty means authentication off |
| `BFF_RATE_LIMIT_MAX` | 30 | requests per client per minute |
| `WEB_PORT` | 8080 | the only published port |

`docker compose config` prints the resolved values, keys included: don't paste its output anywhere.

## The images

| Image | Base | Size | Notes |
|---|---|---|---|
| `api` | `python:3.12-slim` | 297 MB | `pip install .[api,llm]`; the index is built at image build time from the tracked samples; runs as uid 10001; health check on `/ready` |
| `bff` | build on `node:22-bookworm-slim`, run on `node:22-alpine` | 261 MB | `tsc` in the build stage, production dependencies only at run time (16 MB); the Node binary alone is 124 MB, so Debian slim (348 MB) was mostly runtime; user `node` |
| `web` | build on Node, serve on `nginxinc/nginx-unprivileged:1.27-alpine` | 74 MB | Vite build; nginx serves the files and proxies `/api`; non-root, port 8080 |

**What never goes into an image** (`.dockerignore`): `.env`, the private PDF in `data/raw/`, the local index, logs, runs, the MLflow store, the LLM cache, `node_modules`. A check on the built API image found only the 11 tracked sample documents and no `.env`.

**Start-up order** comes from health checks: `bff` waits until `api` is ready (index loaded), and `web` waits for `bff`. `docker compose up --wait` returns only when all three are healthy.

**Only `web` is published.** The API and the BFF are reachable inside the Compose network only, so every browser request passes nginx, then the BFF's validation, keys and rate limits.

## nginx

- **`proxy_buffering off`** for `/api/`, or Server-Sent Events would arrive all at once at the end. Measured through nginx with `gpt-5.4-mini@low`: sources at once, then 23 text pieces between 1.9 and 2.0 s, final at 2.2 s.
- **SPA fallback:** `/ops` and any other app route serve `index.html`, which is never cached; hashed `/assets/` files are cached for a year.
- **`X-Forwarded-For` is overwritten, not appended** (see below).

## The proxy and rate limits: a hole found by testing the running stack

Behind nginx, the BFF sees every request coming from nginx's address, so a per-IP rate limit becomes one shared limit for all users. The obvious fix, `trustProxy: true` in Fastify plus nginx's default `$proxy_add_x_forwarded_for`, made it worse:

| With a limit of 2 per minute | 4 normal requests | 4 requests with a fake `X-Forwarded-For` |
|---|---|---|
| `trustProxy: true`, nginx appends | 200 200 429 429 | **200 200 200 200**: limit bypassed |
| hop count 1, nginx overwrites | 200 200 429 429 | 200 200 429 429 |

nginx **appends** the real address to whatever the client sent, and Fastify with `trustProxy: true` takes the **leftmost** address, which the client wrote. Every request could claim a new address and a fresh limit. The fix has two layers:

1. **nginx is the edge** and sets `X-Forwarded-For $remote_addr`, replacing any value from the client.
2. **The BFF trusts a fixed number of hops** (`BFF_TRUST_PROXY_HOPS=1`, a trust function `hop < 1`), so the client address is the entry the trusted proxy wrote, never the first one. Without a proxy (the default, 0), `X-Forwarded-For` is ignored.

A BFF test pins each case, including a spoofed first entry. A number for `trustProxy` did not type-check with Fastify 5 and was not applied at run time; the explicit function is both clearer and correct.

## A test runner surprise

The smoke test was first named `scripts/smoke_test.py`. It matches pytest's default `*_test.py` pattern, so pytest collected it and, in its default import mode, put `scripts/` on `sys.path`: `import agent` and `import query` then resolved to `scripts/agent.py` and `scripts/query.py`, and two test files failed to import. `testpaths = ["tests"]` in `pyproject.toml` keeps pytest out of `scripts/`, and the script is now `scripts/smoke.py`. It was caught by running the full suite before committing, which is also what CI does on every push.

## Continuous integration

`.github/workflows/ci.yml` runs on every push and pull request:

| Job | What runs | Cost |
|---|---|---|
| `python` (3.10 and 3.12) | `pytest`; on 3.12, the **retrieval regression gate** `evaluate.py --k 3 --min-hit 0.95`, and the generation and agent benchmarks offline as a harness check | free |
| `bff` | `npm ci`, typecheck, Vitest | free |
| `web` | `npm ci`, typecheck, lint, Vitest, production build | free |
| `compose` | after the three above: build all images, `up --wait`, `smoke.py` through nginx, logs on failure | free (offline answers) |
| `llm-gates` | **only when started by hand** with a model: one ping, then the generation and agent benchmarks with `--gate` | tokens: about 60k for the agent benchmark alone, more for generation |

**Why the gates are split.** The retrieval benchmark is deterministic and offline, so it gates every push. Its threshold is the recorded baseline: hit@3 is 0.956 (43 of 45), and one lost question (0.933) fails. The generation and agent gates need a model: offline, the extractive answerer never declines and the rule planner never acts, so their gates fail by design. On every push they run only as a harness check (the scripts must not break); the gated runs cost tokens and are started by hand, with the key as a repository secret (`OPENAI_API_KEY`).

**Injection-safe manual input.** The model name from the manual run is passed through an environment variable and used as `"$RAG_LLM"`, never pasted into a shell command with `${{ }}`, so a crafted input cannot run commands.

## Not yet included

- **Pushing images to a registry** and deploying them (step 23, AWS)
- **Pinning base images and actions by digest**; tags are used today for readability
- **Image vulnerability scanning** (for example Trivy) in CI
- **A cache of LLM turns in CI**, so the manual gate runs would be cheaper on reruns
- **Multiple API replicas:** counters, the run store and the report file are per container; shared stores come with the cloud deployment
