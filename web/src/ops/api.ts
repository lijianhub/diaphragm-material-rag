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
  agent?: { runs: Record<string, number>; approvals: { approve: number; reject: number } };
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

export interface TraceEntry {
  step: number;
  type: "tool" | "approval" | "answer" | "limit";
  tool: string | null;
  arguments: Record<string, unknown> | null;
  status: string;
  summary: string;
  ms: number;
  reviewer?: string;
}

export interface Approval {
  call_id: string;
  tool: string;
  arguments: Record<string, unknown>;
  decision: "approve" | "reject";
  reviewer: string;
  reason: string;
  ts: string;
}

export interface AgentRun {
  id: string;
  question: string;
  status: "running" | "awaiting_approval" | "deciding" | "completed" | "stopped" | "failed";
  stop_reason: string | null;
  created: string;
  updated: string;
  planner: string;
  steps: number;
  tool_calls: number;
  usage: { input_tokens: number; output_tokens: number };
  pending: { call_id: string; tool: string; arguments: Record<string, unknown>; requested: string } | null;
  approvals: Approval[];
  trace?: TraceEntry[];
  sources?: { number: number; label: string; source: string; text: string }[];
  answer?: string | null;
}

export interface Views {
  summary: Summary;
  feedback: FeedbackView;
  evaluations: EvaluationsView;
  ingestion: IngestionView;
  approvals: { pending: AgentRun[] };
  "agent-runs": { runs: AgentRun[] };
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

/** Approve or reject a run's pending action. The BFF records who decided, from the key. */
export async function decide(runId: string, decision: "approve" | "reject", reason: string, adminKey: string): Promise<AgentRun> {
  const response = await fetch(`/api/ops/approvals/${runId}`, {
    method: "POST",
    headers: { "content-type": "application/json", ...(adminKey ? { authorization: `Bearer ${adminKey}` } : {}) },
    body: JSON.stringify(reason ? { decision, reason } : { decision }),
  });
  if (!response.ok) throw await errorFrom(response);
  return response.json();
}
