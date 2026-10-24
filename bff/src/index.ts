import { buildApp } from "./app.js";
import { loadConfig } from "./config.js";

const config = loadConfig();
const app = await buildApp(config);

if (config.apiKeys.length === 0) {
  app.log.warn("BFF_API_KEYS is empty: authentication is OFF (local development only)");
}

// Finish in-flight requests on shutdown (containers send SIGTERM before killing).
for (const signal of ["SIGINT", "SIGTERM"] as const) {
  process.on(signal, () => {
    app.log.info({ signal }, "shutting down");
    void app.close().then(() => process.exit(0));
  });
}

await app.listen({ host: config.host, port: config.port });
app.log.info({ upstream: config.upstreamUrl }, "BFF ready");
