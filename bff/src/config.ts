/** Configuration from environment variables, with defaults for local development. */
export interface Config {
  /** Base URL of the Python RAG API. */
  upstreamUrl: string;
  host: string;
  port: number;
  /** Accepted API keys (Bearer tokens). Empty means authentication is off (local development only). */
  apiKeys: string[];
  /** Keys for the ops dashboard routes (/api/ops/*). A user key is not enough. */
  adminKeys: string[];
  /** Requests per client per window on the /api/ask routes. */
  rateLimitMax: number;
  rateLimitWindow: string;
  /** Browser origins allowed by CORS. */
  corsOrigins: string[];
  /** Time allowed until the upstream API answers (JSON) or starts the stream (headers). */
  upstreamTimeoutMs: number;
  logLevel: string;
}

const list = (value: string | undefined): string[] =>
  (value ?? "")
    .split(",")
    .map((item) => item.trim())
    .filter(Boolean);

export function loadConfig(env: NodeJS.ProcessEnv = process.env): Config {
  return {
    upstreamUrl: (env.RAG_API_URL ?? "http://127.0.0.1:8000").replace(/\/+$/, ""),
    host: env.BFF_HOST ?? "127.0.0.1",
    port: Number(env.BFF_PORT ?? 3000),
    apiKeys: list(env.BFF_API_KEYS),
    adminKeys: list(env.BFF_ADMIN_KEYS),
    rateLimitMax: Number(env.BFF_RATE_LIMIT_MAX ?? 30),
    rateLimitWindow: env.BFF_RATE_LIMIT_WINDOW ?? "1 minute",
    corsOrigins: env.BFF_CORS_ORIGINS === undefined ? ["http://localhost:5173"] : list(env.BFF_CORS_ORIGINS),
    upstreamTimeoutMs: Number(env.BFF_UPSTREAM_TIMEOUT_MS ?? 60_000),
    logLevel: env.BFF_LOG_LEVEL ?? "info",
  };
}
