import { useState } from "react";
import { ApiError } from "../lib/api";
import { count, dateTime } from "../lib/format";
import { type AgentRun, decide } from "./api";

const label = (tool: string) => tool.replaceAll("_", " ");

/** The approval queue: each agent action with a side effect waits here for a person. */
export function ApprovalQueue({ pending, adminKey, onDecided }: { pending: AgentRun[]; adminKey: string; onDecided: () => void }) {
  if (pending.length === 0) return <p className="text-sm text-slate-500">Nothing waiting. Actions with side effects appear here before they run.</p>;
  return (
    <ul className="space-y-3" aria-label="Waiting for approval">
      {pending.map((run) => (
        <PendingAction key={run.id} run={run} adminKey={adminKey} onDecided={onDecided} />
      ))}
    </ul>
  );
}

function PendingAction({ run, adminKey, onDecided }: { run: AgentRun; adminKey: string; onDecided: () => void }) {
  const [reason, setReason] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<ApiError>();
  const [result, setResult] = useState<AgentRun>();
  const action = run.pending!;

  const submit = async (decision: "approve" | "reject") => {
    setBusy(true);
    setError(undefined);
    try {
      setResult(await decide(run.id, decision, reason.trim(), adminKey));
    } catch (e) {
      setError(e instanceof ApiError ? e : new ApiError("Could not reach the service.", 0));
    } finally {
      setBusy(false);
    }
  };

  if (result) {
    return (
      <li className="rounded-lg border border-emerald-200 bg-emerald-50 p-3 text-sm">
        <p className="font-medium text-emerald-900">
          {result.approvals.at(-1)?.decision === "approve" ? "Approved" : "Rejected"}. The run is now <strong>{result.status.replace("_", " ")}</strong>.
        </p>
        {result.answer && <p className="mt-1 text-slate-700">{result.answer}</p>}
        <button type="button" onClick={onDecided} className="mt-2 text-xs text-sky-700 hover:underline">Refresh the queue</button>
      </li>
    );
  }

  return (
    <li className="rounded-lg border border-amber-300 bg-amber-50 p-3 text-sm" aria-label={`Pending ${label(action.tool)}`}>
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <p className="font-semibold text-amber-900">The agent wants to {label(action.tool)}</p>
        <span className="text-xs text-slate-500">requested {dateTime(action.requested)} · run <span className="font-mono">{run.id.slice(0, 8)}</span></span>
      </div>
      <p className="mt-1 text-slate-700">
        <span className="text-slate-500">User asked:</span> {run.question}
      </p>
      <dl className="mt-2 grid grid-cols-[auto_1fr] gap-x-3 gap-y-1 rounded bg-white p-2">
        {Object.entries(action.arguments).map(([key, value]) => (
          <div key={key} className="contents">
            <dt className="text-xs text-slate-500">{key.replaceAll("_", " ")}</dt>
            <dd className="text-slate-900">{String(value)}</dd>
          </div>
        ))}
      </dl>
      {run.trace && run.trace.length > 0 && (
        <details className="mt-2 text-xs text-slate-600">
          <summary className="cursor-pointer">What the agent did first ({run.trace.length} steps)</summary>
          <ol className="mt-1 space-y-1">
            {run.trace.map((t, i) => (
              <li key={i}>
                <span className="font-mono">{t.tool ?? t.type}</span> · {t.status}
                {t.summary && <span className="block truncate text-slate-500">{t.summary}</span>}
              </li>
            ))}
          </ol>
        </details>
      )}
      {run.sources && run.sources.length > 0 && (
        <p className="mt-2 text-xs text-slate-500">Evidence found: {run.sources.map((s) => `[${s.number}] ${s.label}`).join(" · ")}</p>
      )}
      <div className="mt-3 flex flex-wrap items-center gap-2">
        <input
          aria-label="Reason (required to reject)"
          placeholder="Reason (required to reject)"
          maxLength={500}
          value={reason}
          onChange={(e) => setReason(e.target.value)}
          className="min-w-48 flex-1 rounded-lg border border-slate-300 bg-white px-2 py-1"
        />
        <button type="button" disabled={busy} onClick={() => void submit("approve")} className="rounded-lg bg-emerald-700 px-3 py-1 text-white hover:bg-emerald-800 disabled:opacity-50">
          Approve
        </button>
        <button
          type="button"
          disabled={busy || !reason.trim()}
          onClick={() => void submit("reject")}
          className="rounded-lg bg-red-700 px-3 py-1 text-white hover:bg-red-800 disabled:cursor-not-allowed disabled:opacity-50"
        >
          Reject
        </button>
      </div>
      {error && (
        <p className="mt-2 text-red-700" role="alert">
          {error.message}
          {error.status === 409 && (
            <button type="button" onClick={onDecided} className="ml-2 text-xs text-sky-700 hover:underline">Refresh</button>
          )}
        </p>
      )}
    </li>
  );
}

const STATUS_TONE: Record<string, string> = {
  completed: "bg-emerald-100 text-emerald-800",
  awaiting_approval: "bg-amber-100 text-amber-800",
  stopped: "bg-slate-200 text-slate-700",
  failed: "bg-red-100 text-red-800",
};

/** Recent agent runs: outcome, effort and who decided. */
export function AgentRunsTable({ runs }: { runs: AgentRun[] }) {
  if (runs.length === 0) return <p className="text-sm text-slate-500">No agent runs yet.</p>;
  return (
    <div className="max-h-72 overflow-auto">
      <table className="w-full text-sm">
        <thead className="sticky top-0 bg-white text-left text-xs text-slate-500">
          <tr><th className="py-1">Updated</th><th>Question</th><th>Status</th><th className="text-right">Steps</th><th className="text-right">Tools</th><th className="text-right">Tokens</th><th>Decision</th></tr>
        </thead>
        <tbody>
          {runs.map((r) => {
            const last = r.approvals.at(-1);
            return (
              <tr key={r.id} className="border-t border-slate-100">
                <td className="py-1 whitespace-nowrap">{dateTime(r.updated)}</td>
                <td className="max-w-xs truncate" title={r.question}>{r.question}</td>
                <td>
                  <span className={`rounded px-1.5 text-xs ${STATUS_TONE[r.status] ?? "bg-slate-100"}`}>{r.status.replace("_", " ")}</span>
                  {r.stop_reason && <span className="ml-1 text-xs text-slate-500">{r.stop_reason}</span>}
                </td>
                <td className="text-right">{r.steps}</td>
                <td className="text-right">{r.tool_calls}</td>
                <td className="text-right">{count(r.usage.input_tokens + r.usage.output_tokens)}</td>
                <td className="text-xs">{last ? `${last.decision} by ${last.reviewer}` : "–"}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
