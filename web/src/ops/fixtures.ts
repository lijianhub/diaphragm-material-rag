import type { EvaluationsView, FeedbackView, IngestionView, Run, Summary } from "./api";

export function summary(overrides: Partial<Summary> = {}): Summary {
  return {
    started_at: 1_000,
    uptime_s: 3_725,
    ready: true,
    not_ready_reason: null,
    answerer: "openai:gpt-5.4-mini@low",
    index_chunks: 52,
    traffic: { requests: 120, client_errors: 3, server_errors: 2 },
    routes: [
      { route: "/ask", count: 80, client_errors: 2, server_errors: 2, p50_ms: 1_800, p95_ms: 2_450 },
      { route: "/health", count: 400, client_errors: 0, server_errors: 0, p50_ms: 1, p95_ms: 2 },
    ],
    stages: [
      { stage: "retrieval", count: 100, p50_ms: 3, p95_ms: 8 },
      { stage: "first_token", count: 20, p50_ms: 1_100, p95_ms: 2_300 },
    ],
    answers: { answered: 80, refused: 15, fallback: 5 },
    models: [{ model: "gpt-5.4-mini-2026-03-17", calls: 95, input_tokens: 76_000, output_tokens: 5_700 }],
    feedback: { up: 6, down: 2 },
    ...overrides,
  };
}

export const feedback: FeedbackView = {
  total: 9,
  malformed: 1,
  counts: { up: 6, down: 2 },
  recent: [
    { ts: "2026-10-24T10:15:30+00:00", rating: "down", comment: "missing the die angle", answer_request_id: "ans-42" },
    { ts: "2026-10-24T10:10:00+00:00", rating: "up", comment: null, answer_request_id: "ans-41" },
  ],
};

const run = (id: string, time: string, value: number, data = "d1", extra: Partial<Run> = {}): Run => ({
  run_id: id,
  time,
  name: `retrieval ${id}`,
  kind: "retrieval",
  params: { kind: "retrieval" },
  metrics: { "hit@1": value, "mrr@1": value },
  git_commit: "abc1234",
  git_dirty: false,
  data_version: data,
  gates: null,
  ...extra,
});

export const evaluations: EvaluationsView = {
  total: 4,
  malformed: 0,
  runs: [
    run("r4", "2026-10-23T10:00:00+00:00", 0.911, "d1", { git_dirty: true }),
    run("r3", "2026-10-22T10:00:00+00:00", 0.889),
    run("r2", "2026-10-21T10:00:00+00:00", 0.7, "d0"),
    {
      ...run("g1", "2026-10-22T11:00:00+00:00", 0),
      name: "generation llm",
      kind: "generation",
      params: { kind: "generation" },
      metrics: { gates_passed: 6, "answerable.correct": 0.93 },
      gates: "pass",
    },
  ],
};

export const ingestion: IngestionView = {
  report: {
    time: "2026-10-24T21:55:03+00:00",
    loaded: 11,
    failed: [{ source: "data/raw/scan.pdf", reason: "no extractable text (scanned PDF?)" }],
    skipped: ["data/raw/.gitkeep"],
    duplicates: [{ source: "data/raw/copy.txt", same_as: "data/raw/forming.txt" }],
    sync: { added: 1, updated: 0, removed: 0, unchanged: 10, chunks: 4 },
    seconds: 0.07,
  },
  reason: null,
};
