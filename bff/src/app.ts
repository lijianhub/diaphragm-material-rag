/**
 * Backend-for-frontend (BFF) between browsers and the Python RAG API.
 *
 * Browser-facing concerns live here so the Python service stays an internal API:
 * request IDs (generated or accepted, forwarded upstream), API-key authentication (a stub
 * until real identity arrives), per-client rate limiting, CORS, request validation, and
 * Server-Sent Events passed through without buffering. When a browser disconnects from a
 * stream, the upstream request is aborted, so the model stops generating.
 *
 * Upstream calls use Node's built-in fetch, which keeps connections alive and pools them:
 * creating a new HTTP client per request is what made a 2 ms service look like 690 ms.
 */
import { randomUUID, timingSafeEqual } from "node:crypto";
import cors from "@fastify/cors";
import rateLimit from "@fastify/rate-limit";
import Fastify, { type FastifyInstance, type FastifyReply, type FastifyRequest } from "fastify";
import type { Config } from "./config.js";

const REQUEST_ID = /^[A-Za-z0-9._:-]{1,64}$/;

const askSchema = {
  type: "object",
  required: ["question"],
  additionalProperties: false,
  properties: {
    question: { type: "string", minLength: 1, maxLength: 2000, pattern: "\\S" },
    top_k: { type: "integer", minimum: 1, maximum: 10 },
  },
} as const;

const feedbackSchema = {
  type: "object",
  required: ["request_id", "rating"],
  additionalProperties: false,
  properties: {
    request_id: { type: "string", pattern: "^[A-Za-z0-9._:-]{1,64}$" },
    rating: { type: "string", enum: ["up", "down"] },
    comment: { type: "string", maxLength: 500 },
  },
} as const;

interface FeedbackBody {
  request_id: string;
  rating: "up" | "down";
  comment?: string;
}

interface AskBody {
  question: string;
  top_k?: number;
}

function sameSecret(given: string, expected: string): boolean {
  const a = Buffer.from(given);
  const b = Buffer.from(expected);
  return a.length === b.length && timingSafeEqual(a, b); // constant time: no key guessing by timing
}

function bearer(request: FastifyRequest): string | undefined {
  const header = request.headers.authorization;
  return header?.startsWith("Bearer ") ? header.slice("Bearer ".length).trim() : undefined;
}

function errorBody(request: FastifyRequest, error: string, detail?: unknown) {
  return { request_id: request.id, error, ...(detail === undefined ? {} : { detail }) };
}

export async function buildApp(config: Config): Promise<FastifyInstance> {
  const app = Fastify({
    logger: { level: config.logLevel },
    // false, so genReqId always runs: with requestIdHeader set, Fastify would take the
    // caller's header as-is and skip genReqId, bypassing the check below (a test caught it).
    requestIdHeader: false,
    requestIdLogLabel: "request_id",
    // Accept the caller's ID only if it is short and plain (no log injection); else make one.
    genReqId: (req) => {
      const given = req.headers["x-request-id"];
      return typeof given === "string" && REQUEST_ID.test(given) ? given : randomUUID().replaceAll("-", "");
    },
  });

  app.addHook("onRequest", async (request, reply) => {
    reply.header("x-request-id", request.id);
  });

  await app.register(cors, { origin: config.corsOrigins, exposedHeaders: ["x-request-id", "retry-after"] });
  await app.register(rateLimit, {
    global: false,
    // Per API key when authenticated, otherwise per IP address.
    keyGenerator: (request) => bearer(request) ?? request.ip,
    errorResponseBuilder: (request, context) => ({
      statusCode: 429,
      request_id: request.id,
      error: "rate_limited",
      detail: `limit ${context.max} per ${context.after}`,
    }),
  });

  app.setErrorHandler((error: { validation?: unknown; statusCode?: number; message: string }, request, reply) => {
    if (error.validation) {
      return reply.code(400).send(errorBody(request, "invalid_request", error.message));
    }
    if (error.statusCode === 429) {
      return reply.code(429).send(error);
    }
    request.log.error({ err: error }, "unhandled error");
    return reply.code(500).send(errorBody(request, "internal_error"));
  });

  const authenticate = async (request: FastifyRequest, reply: FastifyReply) => {
    if (config.apiKeys.length === 0) return; // authentication off (local development)
    const token = bearer(request);
    if (!token || !config.apiKeys.some((key) => sameSecret(token, key))) {
      return reply.code(401).send(errorBody(request, "unauthorized"));
    }
  };

  const upstreamHeaders = (request: FastifyRequest) => ({
    "content-type": "application/json",
    "x-request-id": request.id, // one ID across browser, BFF and API logs
  });

  const askRoute = {
    schema: { body: askSchema },
    preHandler: authenticate,
    config: { rateLimit: { max: config.rateLimitMax, timeWindow: config.rateLimitWindow } },
  };

  app.get("/api/health", async () => ({ status: "ok" }));

  app.get("/api/ready", async (request, reply) => {
    try {
      const upstream = await fetch(`${config.upstreamUrl}/ready`, {
        headers: upstreamHeaders(request),
        signal: AbortSignal.timeout(5_000),
      });
      const body = await upstream.json();
      return reply.code(upstream.ok ? 200 : 503).send({ ready: upstream.ok, upstream: body });
    } catch {
      return reply.code(503).send({ ready: false, upstream: null, reason: "upstream_unavailable" });
    }
  });

  /** Forward a JSON body to the API and pass its status and body back (its errors carry the same request ID). */
  const proxyJson = async (request: FastifyRequest, reply: FastifyReply, path: string) => {
    let upstream: Response;
    try {
      upstream = await fetch(`${config.upstreamUrl}${path}`, {
        method: "POST",
        headers: upstreamHeaders(request),
        body: JSON.stringify(request.body),
        signal: AbortSignal.timeout(config.upstreamTimeoutMs),
      });
    } catch (error) {
      const timedOut = (error as Error).name === "TimeoutError";
      request.log.warn({ err: error }, "upstream request failed");
      return reply.code(timedOut ? 504 : 502).send(errorBody(request, timedOut ? "upstream_timeout" : "upstream_unavailable"));
    }
    return reply.code(upstream.status).type("application/json").send(await upstream.text());
  };

  app.post<{ Body: AskBody }>("/api/ask", askRoute, (request, reply) => proxyJson(request, reply, "/ask"));

  app.post<{ Body: FeedbackBody }>(
    "/api/feedback",
    {
      schema: { body: feedbackSchema },
      preHandler: authenticate,
      config: { rateLimit: { max: config.rateLimitMax, timeWindow: config.rateLimitWindow } },
    },
    (request, reply) => proxyJson(request, reply, "/feedback"),
  );

  app.post<{ Body: AskBody }>("/api/ask/stream", askRoute, async (request, reply) => {
    const controller = new AbortController();
    const headerTimer = setTimeout(() => controller.abort(new DOMException("upstream timeout", "TimeoutError")), config.upstreamTimeoutMs);
    let clientGone = false;
    // 'close' on the response fires when the browser disconnects (or after we finish).
    reply.raw.on("close", () => {
      if (!reply.raw.writableFinished) {
        clientGone = true;
        controller.abort(); // stop the upstream request, and with it the model's generation
      }
    });

    let upstream: Response;
    try {
      upstream = await fetch(`${config.upstreamUrl}/ask/stream`, {
        method: "POST",
        headers: { ...upstreamHeaders(request), accept: "text/event-stream" },
        body: JSON.stringify(request.body),
        signal: controller.signal,
      });
    } catch (error) {
      clearTimeout(headerTimer);
      const timedOut = (error as Error).name === "TimeoutError";
      return reply.code(timedOut ? 504 : 502).send(errorBody(request, timedOut ? "upstream_timeout" : "upstream_unavailable"));
    }
    clearTimeout(headerTimer);
    if (!upstream.ok || !upstream.body) {
      return reply.code(upstream.status).type("application/json").send(await upstream.text());
    }

    // From here we write the raw response ourselves, so Fastify must not touch it.
    reply.hijack();
    reply.raw.writeHead(200, {
      ...(reply.getHeaders() as Record<string, string>), // request ID and CORS headers set earlier
      "content-type": "text/event-stream; charset=utf-8",
      "cache-control": "no-cache",
      connection: "keep-alive",
      "x-accel-buffering": "no",
    });
    const started = performance.now();
    let bytes = 0;
    try {
      for await (const chunk of upstream.body) {
        bytes += chunk.byteLength;
        reply.raw.write(chunk); // pass through as it arrives: no buffering
      }
    } catch (error) {
      if (!clientGone) {
        request.log.warn({ err: error }, "upstream stream broke");
        reply.raw.write(`event: error\ndata: ${JSON.stringify(errorBody(request, "upstream_error"))}\n\n`);
      }
    } finally {
      request.log.info({ event: "stream_proxy", bytes, stream_ms: Math.round(performance.now() - started), client_disconnected: clientGone }, "stream finished");
      reply.raw.end();
    }
  });

  return app;
}
