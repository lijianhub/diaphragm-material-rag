import { describe, expect, it, vi } from "vitest";
import { answer, jsonResponse, sources, sse, streamResponse } from "../test/fixtures";
import { ApiError, sendFeedback, streamAnswer } from "./api";

const handlers = () => ({ onSources: vi.fn(), onDelta: vi.fn(), onFinal: vi.fn() });

describe("streamAnswer", () => {
  it("calls the handlers in protocol order and returns the final answer", async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      streamResponse([sse("sources", { request_id: "req-123", sources }), sse("delta", { text: "Tool" }), sse("delta", { text: "makers" }), sse("final", answer())]),
    );
    vi.stubGlobal("fetch", fetchMock);
    const h = handlers();

    const result = await streamAnswer("How?", h, new AbortController().signal);

    expect(result.request_id).toBe("req-123");
    expect(h.onSources).toHaveBeenCalledWith(sources, "req-123");
    expect(h.onDelta.mock.calls).toEqual([["Tool"], ["makers"]]);
    expect(h.onFinal).toHaveBeenCalledOnce();
    const [url, init] = fetchMock.mock.calls[0]!;
    expect(url).toBe("/api/ask/stream");
    expect(JSON.parse(init.body)).toEqual({ question: "How?" });
  });

  it("decodes a multi-byte character split between network chunks", async () => {
    const bytes = new TextEncoder().encode(sse("delta", { text: "Ni–Mo µm" }));
    const cut = bytes.indexOf(0xe2) + 1; // inside the three bytes of the en dash
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(streamResponse([bytes.slice(0, cut), bytes.slice(cut), sse("final", answer())])));
    const h = handlers();

    await streamAnswer("q", h, new AbortController().signal);

    expect(h.onDelta).toHaveBeenCalledWith("Ni–Mo µm");
  });

  it("throws an ApiError with the request ID on an error event", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(streamResponse([sse("sources", { request_id: "r1", sources }), sse("error", { request_id: "r1", error: "generation_failed" })])));

    await expect(streamAnswer("q", handlers(), new AbortController().signal)).rejects.toMatchObject({ requestId: "r1" });
  });

  it("throws when the stream ends without a final event", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(streamResponse([sse("delta", { text: "half" })])));

    await expect(streamAnswer("q", handlers(), new AbortController().signal)).rejects.toThrow("ended unexpectedly");
  });

  it("maps 429 to a message with Retry-After", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(jsonResponse({ error: "rate_limited", request_id: "r9" }, 429, { "retry-after": "12" })));

    const error = await streamAnswer("q", handlers(), new AbortController().signal).catch((e) => e);

    expect(error).toBeInstanceOf(ApiError);
    expect(error.message).toBe("Too many requests. Try again in 12 s.");
    expect(error.requestId).toBe("r9");
  });
});

describe("sendFeedback", () => {
  it("omits an empty comment", async () => {
    const fetchMock = vi.fn().mockResolvedValue(jsonResponse({ status: "recorded" }, 201));
    vi.stubGlobal("fetch", fetchMock);

    await sendFeedback("req-1", "up");

    expect(JSON.parse(fetchMock.mock.calls[0]![1].body)).toEqual({ request_id: "req-1", rating: "up" });
  });

  it("throws on a rejected request", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(jsonResponse({ error: "bad_gateway" }, 502)));

    await expect(sendFeedback("req-1", "down", "wrong")).rejects.toThrow("unavailable");
  });
});
