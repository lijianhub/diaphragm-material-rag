import { fireEvent, render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { jsonResponse } from "../test/fixtures";
import Dashboard from "./Dashboard";
import { evaluations, feedback, ingestion, summary } from "./fixtures";
import { LineChart } from "./LineChart";

type Views = Record<string, unknown>;

/** fetch stub that answers each /api/ops/<view> from `views`, or with `status` for all. */
function stubOps(views: Views = { summary: summary(), feedback, evaluations, ingestion }, status = 200) {
  const mock = vi.fn((url: string, init?: RequestInit) => {
    const view = url.replace("/api/ops/", "");
    void init;
    return Promise.resolve(status === 200 ? jsonResponse(views[view], 200) : jsonResponse({ error: "unauthorized", request_id: "ops-1" }, status));
  });
  vi.stubGlobal("fetch", mock);
  return mock;
}

const section = (name: string) => screen.getByRole("region", { name });

describe("Dashboard", () => {
  it("shows service health, usage, feedback, evaluation runs and ingestion", async () => {
    stubOps();
    render(<Dashboard />);

    const service = await screen.findByRole("region", { name: "Service" });
    expect(within(service).getByText("Ready")).toBeInTheDocument();
    expect(within(service).getByText("1.7%")).toBeInTheDocument(); // 2 server errors / 120 requests
    expect(within(service).getByText("2.5 s")).toBeInTheDocument(); // /ask p95
    expect(within(service).getByText("15.0%")).toBeInTheDocument(); // refused 15 of 100
    expect(within(service).getByText(/started 1 h 2 min ago/)).toBeInTheDocument();

    expect(within(section("LLM usage")).getByText("76,000")).toBeInTheDocument();
    expect(within(section("LLM usage")).getByText("860")).toBeInTheDocument(); // tokens per call

    const fb = section("Feedback");
    expect(within(fb).getByText("75.0%")).toBeInTheDocument();
    expect(within(fb).getByText("missing the die angle")).toBeInTheDocument();
    expect(within(fb).getByText("ans-42")).toBeInTheDocument();

    const ing = section("Ingestion");
    expect(within(ing).getByText("data/raw/scan.pdf")).toBeInTheDocument();
    expect(within(ing).getByText(/no extractable text/)).toBeInTheDocument();
  });

  it("plots only comparable runs, flags dirty runs, and switches kind and metric", async () => {
    stubOps();
    const user = userEvent.setup();
    render(<Dashboard />);

    const evals = await screen.findByRole("region", { name: "Evaluation runs" });
    await within(evals).findByText("hit@1 (data d1)");
    expect(within(evals).getByRole("img", { name: /2 points, latest 0.911/ })).toBeInTheDocument();
    expect(within(evals).getByText("1 run(s) on other data not joined")).toBeInTheDocument();
    expect(within(evals).getByText("dirty")).toBeInTheDocument();

    await user.selectOptions(within(evals).getByLabelText("Kind"), "generation");
    await user.selectOptions(within(evals).getByLabelText("Metric"), "gates_passed");

    expect(within(evals).getByText("gates_passed (data d1)")).toBeInTheDocument();
    expect(within(evals).getByText("pass")).toBeInTheDocument();
  });

  it("asks for an admin key on 401 and sends it on the next requests", async () => {
    const mock = stubOps(undefined, 401);
    const user = userEvent.setup();
    render(<Dashboard />);

    await user.type(await screen.findByLabelText("Admin key"), "secret-admin");
    stubOps(); // the key works
    const next = vi.mocked(fetch);
    await user.click(screen.getByRole("button", { name: "Use key" }));

    await screen.findByText("Ready");
    expect(mock.mock.calls[0]![1]).toEqual({ headers: {}, signal: undefined });
    expect(next.mock.calls.at(-1)![1]).toEqual({ headers: { authorization: "Bearer secret-admin" }, signal: undefined });
    expect(screen.queryByLabelText("Admin key")).not.toBeInTheDocument();
  });

  it("keeps the other sections when one view fails", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn((url: string) =>
        Promise.resolve(url.endsWith("evaluations") ? jsonResponse({ error: "bad_gateway", request_id: "r-502" }, 502) : jsonResponse({ summary: summary(), feedback, ingestion }[url.replace("/api/ops/", "")], 200)),
      ),
    );
    render(<Dashboard />);

    expect(await within(await screen.findByRole("region", { name: "Evaluation runs" })).findByText(/unavailable/)).toBeInTheDocument();
    expect(within(section("Evaluation runs")).getByText("(r-502)")).toBeInTheDocument();
    expect(within(section("Service")).getByText("Ready")).toBeInTheDocument();
  });

  it("refreshes every interval until paused, and draws the traffic rate from the snapshots", async () => {
    let requests = 100;
    const mock = vi.fn((url: string) => {
      const view = url.replace("/api/ops/", "");
      if (view === "summary") requests += 10;
      const body = { summary: summary({ traffic: { requests, client_errors: 0, server_errors: 0 } }), feedback, evaluations, ingestion }[view];
      return Promise.resolve(jsonResponse(body, 200));
    });
    vi.stubGlobal("fetch", mock);
    const summaryCalls = () => mock.mock.calls.filter(([url]) => url.endsWith("summary")).length;
    render(<Dashboard refreshMs={40} />);

    // three snapshots give two rates, enough for a line
    expect(await screen.findByRole("img", { name: /Requests per minute/ }, { timeout: 2000 })).toBeInTheDocument();
    expect(summaryCalls()).toBeGreaterThanOrEqual(3);

    fireEvent.click(screen.getByRole("button", { name: /Pause/ }));
    const paused = summaryCalls();
    await new Promise((resolve) => setTimeout(resolve, 200));
    expect(summaryCalls()).toBe(paused);
  });

  it("explains an empty run log and a missing ingestion report", async () => {
    stubOps({ summary: summary(), feedback, evaluations: { total: 0, malformed: 0, runs: [] }, ingestion: { report: null, reason: "no ingestion report: run scripts/index.py" } });
    render(<Dashboard />);

    expect(await screen.findByText(/No tracked runs yet/)).toBeInTheDocument();
    expect(await screen.findByText(/no ingestion report/)).toBeInTheDocument();
  });
});

describe("LineChart", () => {
  it("shows a tooltip for the point under the pointer", () => {
    render(
      <LineChart
        title="hit@1"
        points={[{ key: "a", label: "21 Oct", value: 0.7 }, { key: "b", label: "22 Oct", value: 0.9, detail: "run b" }]}
        format={(v) => v.toFixed(2)}
      />,
    );
    const area = screen.getByTestId("chart-hit-area");
    area.getBoundingClientRect = () => ({ left: 0, width: 100, top: 0, height: 50, right: 100, bottom: 50, x: 0, y: 0, toJSON: () => ({}) });

    fireEvent.mouseMove(area, { clientX: 90 });
    expect(screen.getByRole("tooltip")).toHaveTextContent("0.90");
    expect(screen.getByRole("tooltip")).toHaveTextContent("run b");

    fireEvent.mouseLeave(area);
    expect(screen.queryByRole("tooltip")).not.toBeInTheDocument();
  });
});
