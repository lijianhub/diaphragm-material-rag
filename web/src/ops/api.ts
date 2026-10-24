/** Types and calls for the ops dashboard's read-only views (`/api/ops/...`). */
import { errorFrom } from "../lib/api";

export interface RouteStats {
  route: string;
  count: number;
  client_errors: number;
  server_errors: number;
  p50_ms: number | null;
  p95_ms: number | null;
}

export interface Summary {
  started_at: number;
  uptime_s: number;
  ready: boolean;
  not_ready_reason: string | null;
  answerer: string;
  index_chunks: number;
  traffic: { requests: number; client_errors: number; server_errors: number };
  routes: RouteStats[];
  stages: { stage: string; count: number; p50_ms: number | null; p95_ms: number | null }[];
  answers: { answered: number; refused: number; fallback: number };
  models: { model: string; calls: number; input_tokens: number; output_tokens: number }[];
  feedback: { up: number; down: number };
}

export interface FeedbackView {
  total: number;
  malformed: number;
  counts: { up: number; down: number };
  recent: { ts: string; rating: "up" | "down"; comment: string | null; answer_request_id: string }[];
}

export interface Run {
  run_id: string;
  time: string;
  name: string;
  kind: string;
  params: Record<string, string>;
  metrics: Record<string, number>;
  git_commit: string;
  git_dirty: boolean;
  data_version: string;
  gates: string | null;
}

export interface EvaluationsView {
  total: number;
  malformed: number;
  runs: Run[];
}

export interface IngestionView {
  report: {
    time: string;
    loaded: number;
    failed: { source: string; reason: string }[];
    skipped: string[];
    duplicates: { source: string; same_as: string }[];
    sync: Record<string, number>;
    seconds: number | null;
  } | null;
  reason: string | null;
}

export interface Views {
  summary: Summary;
  feedback: FeedbackView;
  evaluations: EvaluationsView;
  ingestion: IngestionView;
}

/** The admin key is held in memory only (never in storage), and sent only to the BFF. */
export async function fetchView<K extends keyof Views>(view: K, adminKey: string, signal?: AbortSignal): Promise<Views[K]> {
  const response = await fetch(`/api/ops/${view}`, {
    headers: adminKey ? { authorization: `Bearer ${adminKey}` } : {},
    signal,
  });
  if (!response.ok) throw await errorFrom(response);
  return response.json();
}
