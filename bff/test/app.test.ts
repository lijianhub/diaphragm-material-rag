import { createServer, type IncomingMessage, type Server } from "node:http";
import type { AddressInfo } from "node:net";
import type { FastifyInstance } from "fastify";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { buildApp } from "../src/app.js";
import { type Config, loadConfig } from "../src/config.js";

/** A stand-in for the Python RAG API: records requests, streams SSE, can fail or hang. */
class FakeUpstream {
  server!: Server;
  url = "";
  requests: { path: string; headers: IncomingMessage["headers"]; body: string }[] = [];
  streamClosedByPeer = false;
  mode: "ok" | "error" | "slow" = "ok";

  async start() {
    this.server = createServer((req, res) => {
      let body = "";
      req.on("data", (chunk) => (body += chunk));
      req.on("end", () => {
        this.requests.push({ path: req.url ?? "", headers: req.headers, body });
        const id = String(req.headers["x-request-id"] ?? "");
        if (this.mode === "slow") return; // never answers: the BFF must time out
        if (req.url === "/ready") {
          res.writeHead(200, { "content-type": "application/json" }).end(JSON.stringify({ ready: true }));
        } else if (req.url === "/ask") {
          if (this.mode === "error") {
            res.writeHead(503, { "content-type": "application/json" }).end(JSON.stringify({ request_id: id, error: "not_ready" }));
          } else {
            res.writeHead(200, { "content-type": "application/json" }).end(JSON.stringify({ request_id: id, answer: "Over-formed [1]." }));
          }
        } else if (req.url === "/feedback") {
          res.writeHead(201, { "content-type": "application/json" }).end(JSON.stringify({ status: "recorded", request_id: id }));
        } else if (req.url?.startsWith("/ops/") || req.url?.startsWith("/agent/")) {
          res.writeHead(200, { "content-type": "application/json" }).end(JSON.stringify({ path: req.url, method: req.method, body: body ? JSON.parse(body) : null }));
        } else if (req.url === "/ask/stream") {
          res.writeHead(200, { "content-type": "text/event-stream" });
          res.on("close", () => {
            if (!res.writableFinished) this.streamClosedByPeer = true;
          });
          const events = [
            `event: sources\ndata: ${JSON.stringify({ request_id: id, sources: [] })}\n\n`,
            `event: delta\ndata: ${JSON.stringify({ text: "Over-formed " })}\n\n`,
            `event: delta\ndata: ${JSON.stringify({ text: "[1]." })}\n\n`,
            `event: final\ndata: ${JSON.stringify({ request_id: id, answer: "Over-formed [1]." })}\n\n`,
          ];
          let i = 0;
          const timer = setInterval(() => {
            if (res.destroyed) return clearInterval(timer);
            res.write(events[i++]);
            if (i === events.length) {
              clearInterval(timer);
              res.end();
            }
          }, 40);
        }
      });
    });
    await new Promise<void>((resolve) => this.server.listen(0, "127.0.0.1", resolve));
    this.url = `http://127.0.0.1:${(this.server.address() as AddressInfo).port}`;
  }

  async stop() {
    this.server.closeAllConnections();
    await new Promise((resolve) => this.server.close(resolve));
  }
}

let upstream: FakeUpstream;
let app: FastifyInstance;

const config = (overrides: Partial<Config> = {}): Config => ({
  ...loadConfig({}),
  upstreamUrl: upstream.url,
  logLevel: "silent",
  ...overrides,
});

const ask = (body: unknown, headers: Record<string, string> = {}, path = "/api/ask") =>
  app.inject({ method: "POST", url: path, payload: body as object, headers });

beforeEach(async () => {
  upstream = new FakeUpstream();
  await upstream.start();
  app = await buildApp(config());
});

afterEach(async () => {
  await app.close();
  await upstream.stop();
});

describe("request IDs", () => {
  it("generates one, or keeps a safe one from the caller, and forwards it upstream", async () => {
    const generated = await app.inject({ url: "/api/health" });
    const kept = await ask({ question: "What is galling?" }, { "x-request-id": "browser-42" });
    // Values that can arrive over HTTP but are not plain IDs (a newline never gets past Node's parser).
    const tooLong = await app.inject({ url: "/api/health", headers: { "x-request-id": "x".repeat(100) } });
    const quoted = await app.inject({ url: "/api/health", headers: { "x-request-id": 'id "with" spaces' } });

    expect(generated.headers["x-request-id"]).toMatch(/^[0-9a-f]{32}$/);
    expect(kept.headers["x-request-id"]).toBe("browser-42");
    expect(upstream.requests.at(-1)?.headers["x-request-id"]).toBe("browser-42");
    expect(tooLong.headers["x-request-id"]).toMatch(/^[0-9a-f]{32}$/);
    expect(quoted.headers["x-request-id"]).toMatch(/^[0-9a-f]{32}$/);
  });
});

describe("POST /api/ask", () => {
  it("proxies the question and passes the API's status and body through", async () => {
    const response = await ask({ question: "How is springback compensated?", top_k: 2 });

    expect(response.statusCode).toBe(200);
    expect(response.json().answer).toBe("Over-formed [1].");
    expect(JSON.parse(upstream.requests.at(-1)!.body)).toEqual({ question: "How is springback compensated?", top_k: 2 });
  });

  it("passes upstream errors through unchanged", async () => {
    upstream.mode = "error";

    const response = await ask({ question: "anything?" });

    expect(response.statusCode).toBe(503);
    expect(response.json().error).toBe("not_ready");
  });

  it.each([
    [{ question: "" }],
    [{ question: "   " }],
    [{ question: "x".repeat(2001) }],
    [{ question: "ok", top_k: 0 }],
    [{}],
  ])("rejects %j with 400 before calling the API", async (body) => {
    const response = await ask(body);

    expect(response.statusCode).toBe(400);
    expect(response.json()).toMatchObject({ error: "invalid_request", request_id: response.headers["x-request-id"] });
    expect(upstream.requests).toHaveLength(0);
  });

  it("strips unknown fields instead of forwarding them (Fastify's default removeAdditional)", async () => {
    const response = await ask({ question: "What is galling?", debug: true, top_k: 2 });

    expect(response.statusCode).toBe(200);
    expect(JSON.parse(upstream.requests.at(-1)!.body)).toEqual({ question: "What is galling?", top_k: 2 });
  });

  it("returns 502 when the API is unreachable", async () => {
    await app.close();
    app = await buildApp(config({ upstreamUrl: "http://127.0.0.1:1" }));

    const response = await ask({ question: "anything?" });

    expect(response.statusCode).toBe(502);
    expect(response.json().error).toBe("upstream_unavailable");
  });

  it("returns 504 when the API does not answer in time", async () => {
    upstream.mode = "slow";
    await app.close();
    app = await buildApp(config({ upstreamTimeoutMs: 150 }));

    const response = await ask({ question: "anything?" });

    expect(response.statusCode).toBe(504);
    expect(response.json().error).toBe("upstream_timeout");
  });
});

describe("authentication stub", () => {
  beforeEach(async () => {
    await app.close();
    app = await buildApp(config({ apiKeys: ["key-one", "key-two"] }));
  });

  it("requires a valid Bearer key on /api/ask but not on /api/health", async () => {
    expect((await ask({ question: "q?" })).statusCode).toBe(401);
    expect((await ask({ question: "q?" }, { authorization: "Bearer wrong" })).statusCode).toBe(401);
    expect((await ask({ question: "q?" }, { authorization: "Bearer key-two" })).statusCode).toBe(200);
    expect((await app.inject({ url: "/api/health" })).statusCode).toBe(200);
    expect(upstream.requests).toHaveLength(1);
  });
});

describe("rate limiting", () => {
  it("allows the configured number of requests per window, then answers 429", async () => {
    await app.close();
    app = await buildApp(config({ rateLimitMax: 2 }));

    const codes = [];
    for (let i = 0; i < 3; i++) codes.push((await ask({ question: "q?" })).statusCode);
    const limited = await ask({ question: "q?" });

    expect(codes).toEqual([200, 200, 429]);
    expect(limited.json()).toMatchObject({ error: "rate_limited", request_id: limited.headers["x-request-id"] });
    expect(limited.headers["retry-after"]).toBeDefined();
    expect(upstream.requests).toHaveLength(2);
  });
});

describe("CORS", () => {
  it("allows the configured browser origin only", async () => {
    const allowed = await app.inject({ url: "/api/health", headers: { origin: "http://localhost:5173" } });
    const other = await app.inject({ url: "/api/health", headers: { origin: "https://evil.example" } });

    expect(allowed.headers["access-control-allow-origin"]).toBe("http://localhost:5173");
    expect(other.headers["access-control-allow-origin"]).toBeUndefined();
  });
});

describe("POST /api/ask/stream", () => {
  it("passes the Server-Sent Events through with the request ID", async () => {
    const response = await ask({ question: "How is springback compensated?" }, { "x-request-id": "stream-1" }, "/api/ask/stream");

    expect(response.statusCode).toBe(200);
    expect(response.headers["content-type"]).toContain("text/event-stream");
    expect(response.headers["x-request-id"]).toBe("stream-1");
    expect(response.body.match(/^event: (\w+)$/gm)).toEqual(["event: sources", "event: delta", "event: delta", "event: final"]);
    expect(upstream.requests.at(-1)?.headers.accept).toBe("text/event-stream");
  });

  it("aborts the upstream request when the browser disconnects", async () => {
    // A real socket is needed to disconnect, so this test listens on a port.
    await app.listen({ port: 0, host: "127.0.0.1" });
    const port = (app.server.address() as AddressInfo).port;
    const browser = new AbortController();

    const response = await fetch(`http://127.0.0.1:${port}/api/ask/stream`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ question: "How is springback compensated?" }),
      signal: browser.signal,
    });
    const reader = response.body!.getReader();
    await reader.read(); // first event arrived
    browser.abort(); // the user closed the page
    await new Promise((resolve) => setTimeout(resolve, 300));

    expect(upstream.streamClosedByPeer).toBe(true);
  });
});

describe("POST /api/feedback", () => {
  it("forwards a rating with the request ID", async () => {
    const response = await ask({ request_id: "ans-1", rating: "down", comment: "wrong alloy" }, { "x-request-id": "fb-1" }, "/api/feedback");

    expect(response.statusCode).toBe(201);
    expect(response.json()).toEqual({ status: "recorded", request_id: "fb-1" });
    expect(upstream.requests.at(-1)?.path).toBe("/feedback");
    expect(JSON.parse(upstream.requests.at(-1)!.body)).toEqual({ request_id: "ans-1", rating: "down", comment: "wrong alloy" });
  });

  it.each([
    [{ request_id: "ans-1", rating: "meh" }],
    [{ request_id: "bad id", rating: "up" }],
    [{ request_id: "ans-1", rating: "up", comment: "x".repeat(501) }],
  ])("rejects %j with 400", async (body) => {
    const response = await ask(body, {}, "/api/feedback");

    expect(response.statusCode).toBe(400);
    expect(upstream.requests).toHaveLength(0);
  });

  it("requires the API key when authentication is on", async () => {
    await app.close();
    app = await buildApp(config({ apiKeys: ["k"] }));

    expect((await ask({ request_id: "a", rating: "up" }, {}, "/api/feedback")).statusCode).toBe(401);
    expect((await ask({ request_id: "a", rating: "up" }, { authorization: "Bearer k" }, "/api/feedback")).statusCode).toBe(201);
  });
});

describe("GET /api/ready", () => {
  it("reports the API's readiness, and 503 when the API is down", async () => {
    const ready = await app.inject({ url: "/api/ready" });
    await upstream.stop();
    const down = await app.inject({ url: "/api/ready" });
    await upstream.start(); // so afterEach can stop it again

    expect(ready.statusCode).toBe(200);
    expect(ready.json()).toEqual({ ready: true, upstream: { ready: true } });
    expect(down.statusCode).toBe(503);
  });
});

describe("GET /api/ops/*", () => {
  it.each(["summary", "feedback", "evaluations", "ingestion", "approvals", "agent-runs"])("forwards %s to the API as a GET", async (view) => {
    const response = await app.inject({ url: `/api/ops/${view}`, headers: { "x-request-id": "ops-1" } });

    expect(response.statusCode).toBe(200);
    expect(response.json()).toEqual({ path: `/ops/${view}`, method: "GET", body: null });
    expect(upstream.requests[0]!.headers["x-request-id"]).toBe("ops-1");
  });

  it("forwards a valid limit, rejects an invalid one, and never forwards other parameters", async () => {
    const ok = await app.inject({ url: "/api/ops/feedback?limit=20" });
    const bad = await Promise.all(["limit=0", "limit=501", "limit=abc"].map((q) => app.inject({ url: `/api/ops/feedback?${q}` })));
    const extra = await app.inject({ url: "/api/ops/feedback?path=../../etc" }); // stripped, like unknown body fields

    expect(ok.json().path).toBe("/ops/feedback?limit=20");
    expect(bad.map((r) => r.statusCode)).toEqual([400, 400, 400]);
    expect(extra.json().path).toBe("/ops/feedback");
    expect(upstream.requests).toHaveLength(2);
  });

  it("needs an admin key when authentication is on: a user key is forbidden", async () => {
    await app.close();
    app = await buildApp(config({ apiKeys: ["user-key"], adminKeys: ["admin-key"] }));
    const as = (key?: string) => app.inject({ url: "/api/ops/summary", headers: key ? { authorization: `Bearer ${key}` } : {} });

    expect((await as()).statusCode).toBe(401);
    expect((await as("wrong")).statusCode).toBe(401);
    expect((await as("user-key")).statusCode).toBe(403);
    expect((await as("admin-key")).statusCode).toBe(200);
    expect(upstream.requests).toHaveLength(1);
  });

  it("stays closed when only user keys are configured", async () => {
    await app.close();
    app = await buildApp(config({ apiKeys: ["user-key"] }));

    expect((await app.inject({ url: "/api/ops/summary", headers: { authorization: "Bearer user-key" } })).statusCode).toBe(403);
    expect(upstream.requests).toHaveLength(0);
  });

  it("returns 502 when the API is unreachable", async () => {
    await app.close();
    app = await buildApp(config({ upstreamUrl: "http://127.0.0.1:1" }));

    expect((await app.inject({ url: "/api/ops/summary" })).statusCode).toBe(502);
  });
});

describe("agent routes", () => {
  const runId = "0123456789abcdef0123456789abcdef";

  it("starts a run and reads one back for a user", async () => {
    const started = await ask({ question: "Raise an NCR for lot L-302." }, {}, "/api/agent/runs");
    const read = await app.inject({ url: `/api/agent/runs/${runId}` });

    expect(started.json()).toEqual({ path: "/agent/runs", method: "POST", body: { question: "Raise an NCR for lot L-302." } });
    expect(read.json()).toEqual({ path: `/agent/runs/${runId}`, method: "GET", body: null });
  });

  it("rejects a blank question and a malformed run ID before calling the API", async () => {
    const blank = await ask({ question: "   " }, {}, "/api/agent/runs");
    const traversal = await app.inject({ url: "/api/agent/runs/..%2F..%2Fops%2Fsummary" });

    expect([blank.statusCode, traversal.statusCode]).toEqual([400, 400]);
    expect(upstream.requests).toHaveLength(0);
  });

  it("sets the reviewer from the admin key and ignores a reviewer claimed in the body", async () => {
    await app.close();
    app = await buildApp(config({ apiKeys: ["user-key"], adminKeys: ["first-admin", "second-admin"] }));

    const response = await app.inject({
      method: "POST",
      url: `/api/ops/approvals/${runId}`,
      headers: { authorization: "Bearer second-admin" },
      payload: { decision: "approve", reviewer: "the-ceo", reason: "checked" },
    });

    expect(response.json()).toEqual({ path: `/ops/approvals/${runId}`, method: "POST", body: { decision: "approve", reason: "checked", reviewer: "admin-2" } });
  });

  it("lets only admins decide: users get 403, anonymous 401", async () => {
    await app.close();
    app = await buildApp(config({ apiKeys: ["user-key"], adminKeys: ["admin-key"] }));
    const decide = (headers: Record<string, string>) =>
      app.inject({ method: "POST", url: `/api/ops/approvals/${runId}`, headers, payload: { decision: "reject" } });

    expect((await decide({})).statusCode).toBe(401);
    expect((await decide({ authorization: "Bearer user-key" })).statusCode).toBe(403);
    expect((await decide({ authorization: "Bearer admin-key" })).statusCode).toBe(200);
    expect((await ask({ question: "q" }, {}, "/api/agent/runs")).statusCode).toBe(401); // users need their key to start runs
    expect(upstream.requests).toHaveLength(1);
  });

  it("names the reviewer local-dev when authentication is off, and validates the decision", async () => {
    const ok = await app.inject({ method: "POST", url: `/api/ops/approvals/${runId}`, payload: { decision: "reject", reason: "wrong lot" } });
    const bad = await app.inject({ method: "POST", url: `/api/ops/approvals/${runId}`, payload: { decision: "maybe" } });

    expect(ok.json().body).toEqual({ decision: "reject", reason: "wrong lot", reviewer: "local-dev" });
    expect(bad.statusCode).toBe(400);
  });
});

describe("behind a reverse proxy", () => {
  const fromClient = (ip: string) =>
    app.inject({ method: "POST", url: "/api/ask", payload: { question: "q" }, headers: { "x-forwarded-for": ip } });

  it("rate-limits each client separately when the proxy is trusted", async () => {
    await app.close();
    app = await buildApp(config({ trustProxyHops: 1, rateLimitMax: 1 }));

    const statuses = [(await fromClient("10.0.0.1")).statusCode, (await fromClient("10.0.0.2")).statusCode, (await fromClient("10.0.0.1")).statusCode];

    expect(statuses).toEqual([200, 200, 429]);
  });

  it("ignores X-Forwarded-For when no proxy is trusted, so clients cannot dodge the limit", async () => {
    await app.close();
    app = await buildApp(config({ rateLimitMax: 1 }));

    const statuses = [(await fromClient("10.0.0.1")).statusCode, (await fromClient("10.0.0.2")).statusCode];

    expect(statuses).toEqual([200, 429]);
  });

  it("uses the address the proxy added, so a client-written first entry cannot dodge the limit", async () => {
    await app.close();
    app = await buildApp(config({ trustProxyHops: 1, rateLimitMax: 1 }));
    // A client sends its own X-Forwarded-For; a proxy that appends would produce "fake, real".
    const spoofed = (fake: string) => fromClient(`${fake}, 10.0.0.7`);

    const statuses = [(await spoofed("203.0.113.1")).statusCode, (await spoofed("203.0.113.2")).statusCode];

    expect(statuses).toEqual([200, 429]); // both counted as 10.0.0.7
  });
});
