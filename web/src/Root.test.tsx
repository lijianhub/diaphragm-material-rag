import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { Root } from "./Root";
import { jsonResponse } from "./test/fixtures";

describe("Root", () => {
  it("serves the Q&A page by default and the ops dashboard on /ops", async () => {
    vi.stubGlobal("fetch", vi.fn(() => Promise.resolve(jsonResponse({ error: "unauthorized" }, 401))));

    const { unmount } = render(<Root path="/" />);
    expect(screen.getByRole("heading", { name: "Diaphragm Material Assistant" })).toBeInTheDocument();
    unmount();

    render(<Root path="/ops/" />);
    expect(await screen.findByRole("heading", { name: "Operations" })).toBeInTheDocument();
  });
});
