import type { Answer, Source } from "../lib/api";

export const sources: Source[] = [
  { number: 1, label: "springback.md › Compensation", source: "springback.md", text: "Toolmakers overbend the die to compensate for springback.", score: 0.91 },
  { number: 2, label: "heat_treatment.txt", source: "heat_treatment.txt", text: "Age hardening raises strength after forming.", score: 0.55 },
];

export function answer(overrides: Partial<Answer> = {}): Answer {
  return {
    request_id: "req-123",
    answer: "Toolmakers overbend the die [1].",
    refused: false,
    citations: [{ number: 1, label: sources[0]!.label, source: sources[0]!.source }],
    sources,
    invalid_citations: [],
    model: "fake-model",
    prompt_version: "grounded-v2@abc123",
    used_llm: true,
    fallback_reason: null,
    usage: { input_tokens: 812, output_tokens: 24 },
    timings_ms: { retrieval_ms: 4, first_token_ms: 900, generation_ms: 1400 },
    ...overrides,
  };
}

export const sse = (event: string, data: unknown) => `event: ${event}\ndata: ${JSON.stringify(data)}\n\n`;

/** A streaming response whose body yields `chunks` one by one (strings or raw bytes). */
export function streamResponse(chunks: (string | Uint8Array)[], init: ResponseInit = {}): Response {
  const encoder = new TextEncoder();
  const body = new ReadableStream<Uint8Array>({
    start(controller) {
      for (const chunk of chunks) controller.enqueue(typeof chunk === "string" ? encoder.encode(chunk) : chunk);
      controller.close();
    },
  });
  return new Response(body, { status: 200, headers: { "content-type": "text/event-stream" }, ...init });
}

/** A response that sends `head` and then stays open until the request is aborted. */
export function hangingResponse(head: string, signal: AbortSignal): Response {
  const body = new ReadableStream<Uint8Array>({
    start(controller) {
      controller.enqueue(new TextEncoder().encode(head));
      signal.addEventListener("abort", () => controller.error(new DOMException("Aborted", "AbortError")));
    },
  });
  return new Response(body, { status: 200, headers: { "content-type": "text/event-stream" } });
}

export const jsonResponse = (body: unknown, status: number, headers: Record<string, string> = {}) =>
  new Response(JSON.stringify(body), { status, headers: { "content-type": "application/json", ...headers } });
