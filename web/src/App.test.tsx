import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import App from "./App";
import { answer, hangingResponse, jsonResponse, sources, sse, streamResponse } from "./test/fixtures";

async function ask(question = "How is springback compensated?") {
  const user = userEvent.setup();
  render(<App />);
  await user.type(screen.getByLabelText(/ask about/i), question);
  await user.click(screen.getByRole("button", { name: "Ask" }));
  return user;
}

describe("App", () => {
  it("disables Ask until a question is typed", async () => {
    const user = userEvent.setup();
    render(<App />);
    const button = screen.getByRole("button", { name: "Ask" });
    expect(button).toBeDisabled();
    await user.type(screen.getByLabelText(/ask about/i), "   ");
    expect(button).toBeDisabled();
    await user.type(screen.getByLabelText(/ask about/i), "x");
    expect(button).toBeEnabled();
  });

  it("shows the sources, then replaces the streamed text with the checked final answer", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        streamResponse([sse("sources", { request_id: "req-123", sources }), sse("delta", { text: "draft text [9]" }), sse("final", answer())]),
      ),
    );
    await ask();

    const text = await screen.findByTestId("answer-text");
    await waitFor(() => expect(text).toHaveTextContent("Toolmakers overbend the die 1."));
    expect(text).not.toHaveTextContent("draft");
    expect(screen.getByRole("heading", { name: "Sources (2)" })).toBeInTheDocument();
    expect(screen.getByText("springback.md › Compensation").parentElement).toHaveTextContent("cited");
    expect(screen.getByText("heat_treatment.txt").parentElement).not.toHaveTextContent("cited");
    expect(screen.getByText("req-123")).toBeInTheDocument();
    expect(screen.getByText("812 in / 24 out")).toBeInTheDocument();
  });

  it("highlights and expands a source when its citation is clicked", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(streamResponse([sse("sources", { request_id: "req-123", sources }), sse("final", answer())])));
    const user = await ask();

    await user.click(await screen.findByRole("button", { name: "Show source 1" }));

    const card = document.getElementById("source-1")!;
    expect(card).toHaveAttribute("data-active", "true");
    expect(within(card).getByRole("button", { name: "Hide text" })).toHaveAttribute("aria-expanded", "true");
    expect(document.getElementById("source-2")).not.toHaveAttribute("data-active");
  });

  it("marks a refusal", async () => {
    const refusal = answer({ answer: "The documents do not contain the price of titanium.", refused: true, citations: [] });
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(streamResponse([sse("sources", { request_id: "req-123", sources }), sse("final", refusal)])));
    await ask("What is the price of titanium?");

    expect(await screen.findByRole("heading", { name: "Not found in the documents" })).toBeInTheDocument();
  });

  it("shows the error message and request ID when rate limited", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(jsonResponse({ error: "rate_limited", request_id: "req-429" }, 429, { "retry-after": "30" })));
    await ask();

    const alert = await screen.findByRole("alert");
    expect(alert).toHaveTextContent("Too many requests. Try again in 30 s.");
    expect(alert).toHaveTextContent("req-429");
    expect(screen.getByRole("button", { name: "Ask" })).toBeEnabled();
  });

  it("stops the stream and keeps the partial text", async () => {
    let signal: AbortSignal | undefined;
    vi.stubGlobal(
      "fetch",
      vi.fn((_url: string, init: RequestInit) => {
        signal = init.signal!;
        return Promise.resolve(hangingResponse(sse("sources", { request_id: "req-123", sources }) + sse("delta", { text: "Partial" }), signal));
      }),
    );
    const user = await ask();
    await waitFor(() => expect(screen.getByTestId("answer-text")).toHaveTextContent("Partial"));

    await user.click(screen.getByRole("button", { name: "Stop" }));

    expect(signal!.aborted).toBe(true);
    expect(await screen.findByText(/Stopped/)).toBeInTheDocument();
    expect(screen.getByTestId("answer-text")).toHaveTextContent("Partial");
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
  });

  it("sends thumbs-up at once and thumbs-down with an optional comment", async () => {
    const fetchMock = vi.fn().mockResolvedValueOnce(streamResponse([sse("sources", { request_id: "req-123", sources }), sse("final", answer())]));
    fetchMock.mockResolvedValue(jsonResponse({ status: "recorded", request_id: "fb" }, 201));
    vi.stubGlobal("fetch", fetchMock);
    const user = await ask();

    await user.click(await screen.findByRole("button", { name: "Not helpful" }));
    expect(fetchMock).toHaveBeenCalledTimes(1); // nothing sent before the comment step
    await user.type(screen.getByLabelText(/what was wrong/i), "  missing the die angle ");
    await user.click(screen.getByRole("button", { name: "Send" }));

    expect(await screen.findByText(/Thanks for the feedback/)).toBeInTheDocument();
    const [url, init] = fetchMock.mock.calls[1]!;
    expect(url).toBe("/api/feedback");
    expect(JSON.parse(init.body)).toEqual({ request_id: "req-123", rating: "down", comment: "missing the die angle" });
  });

  it("sends thumbs-up without a comment", async () => {
    const fetchMock = vi.fn().mockResolvedValueOnce(streamResponse([sse("sources", { request_id: "req-123", sources }), sse("final", answer())]));
    fetchMock.mockResolvedValue(jsonResponse({ status: "recorded", request_id: "fb" }, 201));
    vi.stubGlobal("fetch", fetchMock);
    const user = await ask();

    await user.click(await screen.findByRole("button", { name: "Helpful" }));

    await screen.findByText(/Thanks for the feedback/);
    expect(JSON.parse(fetchMock.mock.calls[1]![1].body)).toEqual({ request_id: "req-123", rating: "up" });
  });
});
