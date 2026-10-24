/** Types and calls for the BFF (`/api/...`), mirroring the RAG API's schemas. */
import { SSEParser } from "./sse";

export interface Source {
  number: number;
  label: string;
  source: string;
  text: string;
  score: number;
}

export interface Citation {
  number: number;
  label: string;
  source: string;
}

export interface Answer {
  request_id: string;
  answer: string;
  refused: boolean;
  citations: Citation[];
  sources: Source[];
  invalid_citations: number[];
  model: string;
  prompt_version: string;
  used_llm: boolean;
  fallback_reason: string | null;
  usage: { input_tokens: number; output_tokens: number } | null;
  timings_ms: Record<string, number>;
}

export class ApiError extends Error {
  readonly status: number;
  readonly requestId?: string;

  constructor(message: string, status: number, requestId?: string) {
    super(message);
    this.status = status;
    this.requestId = requestId;
  }
}

export interface StreamHandlers {
  onSources: (sources: Source[], requestId: string) => void;
  onDelta: (text: string) => void;
  onFinal: (answer: Answer) => void;
}

export async function errorFrom(response: Response): Promise<ApiError> {
  const requestId = response.headers.get("x-request-id") ?? undefined;
  let body: { error?: string; request_id?: string } = {};
  try {
    body = await response.json();
  } catch {
    // not JSON (e.g. a proxy error page)
  }
  const id = body.request_id ?? requestId;
  if (response.status === 429) {
    const retry = response.headers.get("retry-after");
    return new ApiError(`Too many requests. Try again${retry ? ` in ${retry} s` : " later"}.`, 429, id);
  }
  if (response.status === 401) return new ApiError("Not authorised: the API key is missing or wrong.", 401, id);
  if (response.status === 403) return new ApiError("This key is not allowed here: the ops dashboard needs an admin key.", 403, id);
  if (response.status === 400 || response.status === 422) return new ApiError("The question was not accepted (empty or too long).", response.status, id);
  if (response.status === 503) return new ApiError("The service is not ready yet (no document index).", 503, id);
  if (response.status === 502 || response.status === 504) return new ApiError("The answering service is unavailable. Please try again.", response.status, id);
  return new ApiError(`Something went wrong (${body.error ?? response.status}).`, response.status, id);
}

/**
 * Ask a question and receive the answer as a stream: sources first, then text pieces,
 * then the checked final answer (which replaces the pieces). Aborting `signal` stops the
 * stream, and the BFF and API stop the model's generation.
 */
export async function streamAnswer(question: string, handlers: StreamHandlers, signal: AbortSignal): Promise<Answer> {
  const response = await fetch("/api/ask/stream", {
    method: "POST",
    headers: { "content-type": "application/json", accept: "text/event-stream" },
    body: JSON.stringify({ question }),
    signal,
  });
  if (!response.ok || !response.body) throw await errorFrom(response);

  const parser = new SSEParser();
  const decoder = new TextDecoder(); // stream mode: keeps a split multi-byte character for the next chunk
  const reader = response.body.getReader();
  let final: Answer | undefined;
  for (;;) {
    const { value, done } = await reader.read();
    const text = done ? decoder.decode() : decoder.decode(value, { stream: true });
    for (const event of parser.push(text)) {
      const data = JSON.parse(event.data);
      if (event.event === "sources") handlers.onSources(data.sources, data.request_id);
      else if (event.event === "delta") handlers.onDelta(data.text);
      else if (event.event === "final") {
        final = data as Answer;
        handlers.onFinal(final);
      } else if (event.event === "error") {
        throw new ApiError("The answer could not be completed.", 500, data.request_id);
      }
    }
    if (done) break;
  }
  if (!final) throw new ApiError("The answer stream ended unexpectedly.", 502, response.headers.get("x-request-id") ?? undefined);
  return final;
}

export async function sendFeedback(requestId: string, rating: "up" | "down", comment?: string): Promise<void> {
  const response = await fetch("/api/feedback", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ request_id: requestId, rating, ...(comment ? { comment } : {}) }),
  });
  if (!response.ok) throw await errorFrom(response);
}
