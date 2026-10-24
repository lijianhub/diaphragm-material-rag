import { type FormEvent, type ReactNode, useCallback, useEffect, useState } from "react";
import { ApiError } from "../lib/api";
import { count, dateTime, duration, ms, pct } from "../lib/format";
import { type EvaluationsView, type FeedbackView, fetchView, type IngestionView, type Run, type Summary } from "./api";
import { LineChart } from "./LineChart";
import { metricNames, type Sample, trafficRates, trend } from "./series";

export const REFRESH_MS = 10_000;
const MAX_SAMPLES = 90; // 15 minutes at one sample per 10 s

type Loaded<T> = { data?: T; error?: ApiError };

/** The ops dashboard: service health, cost, feedback, evaluation runs and ingestion. */
export default function Dashboard({ refreshMs = REFRESH_MS }: { refreshMs?: number }) {
  const [adminKey, setAdminKey] = useState("");
  const [summary, setSummary] = useState<Loaded<Summary>>({});
  const [feedback, setFeedback] = useState<Loaded<FeedbackView>>({});
  const [evaluations, setEvaluations] = useState<Loaded<EvaluationsView>>({});
  const [ingestion, setIngestion] = useState<Loaded<IngestionView>>({});
  const [samples, setSamples] = useState<Sample[]>([]);
  const [paused, setPaused] = useState(false);
  const [updatedAt, setUpdatedAt] = useState<number>();
  const refresh = useCallback(async () => {
    const key = adminKey;
    const settle = <T,>(set: (value: Loaded<T>) => void) => [
      (data: T) => set({ data }),
      (error: unknown) => set({ error: error instanceof ApiError ? error : new ApiError("Could not reach the service.", 0) }),
    ] as const;
    // Each view loads on its own: one broken source must not blank the whole page.
    await Promise.all([
      fetchView("summary", key).then((data) => {
        setSummary({ data });
        setSamples((previous) => [...previous, { at: Date.now(), summary: data }].slice(-MAX_SAMPLES));
      }, settle(setSummary)[1]),
      fetchView("feedback", key).then(...settle(setFeedback)),
      fetchView("evaluations", key).then(...settle(setEvaluations)),
      fetchView("ingestion", key).then(...settle(setIngestion)),
    ]);
    setUpdatedAt(Date.now());
  }, [adminKey]);

  // Load on mount and whenever the key changes. Fetching from the API is synchronising with
  // an external system, and every state update in refresh() happens after an await, so this
  // is not the cascading render the lint rule targets.
  useEffect(() => {
    // oxlint-disable-next-line react/set-state-in-effect
    void refresh();
  }, [refresh]);

  // Then poll, unless paused. Pausing or resuming does not reload by itself.
  useEffect(() => {
    if (paused) return;
    const timer = setInterval(() => void refresh(), refreshMs);
    return () => clearInterval(timer);
  }, [paused, refresh, refreshMs]);

  const authError = [summary, feedback, evaluations, ingestion].map((v) => v.error).find((e) => e && (e.status === 401 || e.status === 403));

  return (
    <div className="min-h-screen bg-slate-50">
      <header className="border-b border-slate-200 bg-white">
        <div className="mx-auto flex max-w-6xl flex-wrap items-center justify-between gap-3 px-4 py-4">
          <div>
            <h1 className="text-lg font-semibold text-slate-900">Operations</h1>
            <p className="text-sm text-slate-500">Diaphragm Material Assistant · service, cost, feedback, evaluation, ingestion</p>
          </div>
          <div className="flex items-center gap-3 text-sm text-slate-600">
            <span>{updatedAt ? `Updated ${new Date(updatedAt).toLocaleTimeString("en-GB")}` : "Loading…"}</span>
            <button type="button" onClick={() => setPaused((p) => !p)} className="rounded-lg border border-slate-300 bg-white px-3 py-1 hover:bg-slate-50">
              {paused ? "Resume" : "Pause"} auto-refresh
            </button>
            <button type="button" onClick={() => void refresh()} className="rounded-lg bg-sky-600 px-3 py-1 text-white hover:bg-sky-700">
              Refresh
            </button>
          </div>
        </div>
      </header>
      <main className="mx-auto max-w-6xl space-y-6 px-4 py-6">
        {authError && <KeyForm error={authError} onKey={setAdminKey} />}
        <ServiceSection loaded={summary} samples={samples} refreshMs={refreshMs} />
        <div className="grid gap-6 lg:grid-cols-2">
          <CostSection loaded={summary} />
          <FeedbackSection loaded={feedback} />
        </div>
        <EvaluationSection loaded={evaluations} />
        <IngestionSection loaded={ingestion} />
      </main>
    </div>
  );
}

function KeyForm({ error, onKey }: { error: ApiError; onKey: (key: string) => void }) {
  const [value, setValue] = useState("");
  const submit = (event: FormEvent) => {
    event.preventDefault();
    if (value.trim()) onKey(value.trim());
  };
  return (
    <form onSubmit={submit} role="alert" className="flex flex-wrap items-center gap-2 rounded-lg border border-amber-300 bg-amber-50 p-3 text-sm text-amber-900">
      <span>{error.message}</span>
      <input
        type="password"
        aria-label="Admin key"
        placeholder="Admin key"
        value={value}
        onChange={(e) => setValue(e.target.value)}
        className="rounded-lg border border-amber-300 bg-white px-2 py-1"
      />
      <button type="submit" className="rounded-lg bg-amber-700 px-3 py-1 text-white hover:bg-amber-800">Use key</button>
      <span className="text-xs text-amber-700">Kept in this tab's memory only.</span>
    </form>
  );
}

function Section({ title, children, note }: { title: string; children: ReactNode; note?: ReactNode }) {
  return (
    <section aria-label={title} className="rounded-lg border border-slate-200 bg-white p-4 shadow-sm">
      <div className="mb-3 flex flex-wrap items-baseline justify-between gap-2">
        <h2 className="text-base font-semibold text-slate-900">{title}</h2>
        {note && <p className="text-xs text-slate-500">{note}</p>}
      </div>
      {children}
    </section>
  );
}

function Failed({ error }: { error: ApiError }) {
  return (
    <p className="text-sm text-red-700">
      {error.message}
      {error.requestId && <span className="ml-2 font-mono text-xs">({error.requestId})</span>}
    </p>
  );
}

function Tile({ label, value, sub, tone = "normal" }: { label: string; value: ReactNode; sub?: ReactNode; tone?: "normal" | "bad" | "good" }) {
  const ring = tone === "bad" ? "border-red-300 bg-red-50" : tone === "good" ? "border-emerald-200 bg-emerald-50" : "border-slate-200 bg-slate-50";
  return (
    <div className={`rounded-lg border p-3 ${ring}`}>
      <div className="text-xs text-slate-500">{label}</div>
      <div className="mt-1 text-xl font-semibold text-slate-900">{value}</div>
      {sub && <div className="mt-0.5 text-xs text-slate-500">{sub}</div>}
    </div>
  );
}

function ServiceSection({ loaded, samples, refreshMs }: { loaded: Loaded<Summary>; samples: Sample[]; refreshMs: number }) {
  const s = loaded.data;
  if (!s) return <Section title="Service">{loaded.error ? <Failed error={loaded.error} /> : <p className="text-sm text-slate-500">Loading…</p>}</Section>;
  const answers = s.answers.answered + s.answers.refused + s.answers.fallback;
  const ask = s.routes.find((r) => r.route === "/ask");
  const rates = trafficRates(samples);
  return (
    <Section title="Service" note={`Totals since the API started ${duration(s.uptime_s)} ago; counters reset on restart. Latency percentiles are estimated from histogram buckets.`}>
      {loaded.error && <Failed error={loaded.error} />}
      <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-6">
        <Tile label="Status" value={s.ready ? "Ready" : "Not ready"} sub={s.ready ? `${count(s.index_chunks)} chunks · ${s.answerer}` : s.not_ready_reason} tone={s.ready ? "good" : "bad"} />
        <Tile label="Requests" value={count(s.traffic.requests)} sub="ask, stream, feedback" />
        <Tile
          label="Server errors"
          value={pct(s.traffic.server_errors, s.traffic.requests)}
          sub={`${count(s.traffic.server_errors)} · ${count(s.traffic.client_errors)} rejected (4xx)`}
          tone={s.traffic.server_errors > 0 ? "bad" : "normal"}
        />
        <Tile label="/ask p95" value={ms(ask?.p95_ms)} sub={`p50 ${ms(ask?.p50_ms)}`} />
        <Tile label="Refused" value={pct(s.answers.refused, answers)} sub={`${count(s.answers.refused)} of ${count(answers)} answers`} />
        <Tile label="LLM fallback" value={pct(s.answers.fallback, answers)} sub="quoted sentences instead" tone={s.answers.fallback > 0 ? "bad" : "normal"} />
      </div>
      <div className="mt-4 grid gap-6 lg:grid-cols-2">
        <div>
          {rates.length >= 2 ? (
            <LineChart
              title="Requests per minute (since this page opened)"
              points={rates.map((r) => ({
                key: String(r.at),
                label: new Date(r.at).toLocaleTimeString("en-GB"),
                value: r.perMinute,
                detail: r.serverErrors ? `${r.serverErrors} server errors` : undefined,
              }))}
              format={(v) => v.toFixed(1)}
            />
          ) : (
            <p className="text-sm text-slate-500">Requests per minute appear after a few refreshes (every {refreshMs / 1000} s).</p>
          )}
        </div>
        <table className="w-full text-sm">
          <caption className="mb-1 text-left text-sm font-medium text-slate-700">Pipeline stages</caption>
          <thead className="text-left text-xs text-slate-500">
            <tr><th className="py-1">Stage</th><th className="text-right">Count</th><th className="text-right">p50</th><th className="text-right">p95</th></tr>
          </thead>
          <tbody>
            {s.stages.map((stage) => (
              <tr key={stage.stage} className="border-t border-slate-100">
                <td className="py-1">{stage.stage.replace("_", " ")}</td>
                <td className="text-right">{count(stage.count)}</td>
                <td className="text-right">{ms(stage.p50_ms)}</td>
                <td className="text-right">{ms(stage.p95_ms)}</td>
              </tr>
            ))}
            {s.stages.length === 0 && <tr><td colSpan={4} className="py-1 text-slate-500">No answers yet.</td></tr>}
          </tbody>
        </table>
      </div>
    </Section>
  );
}

function CostSection({ loaded }: { loaded: Loaded<Summary> }) {
  const models = loaded.data?.models ?? [];
  return (
    <Section title="LLM usage" note="Tokens, not dollars: prices change and differ by contract.">
      {loaded.error && !loaded.data && <Failed error={loaded.error} />}
      {models.length === 0 ? (
        <p className="text-sm text-slate-500">No LLM calls since start-up (offline extractive answers cost nothing).</p>
      ) : (
        <table className="w-full text-sm">
          <thead className="text-left text-xs text-slate-500">
            <tr><th className="py-1">Model</th><th className="text-right">Calls</th><th className="text-right">Input tokens</th><th className="text-right">Output tokens</th><th className="text-right">Per call</th></tr>
          </thead>
          <tbody>
            {models.map((m) => (
              <tr key={m.model} className="border-t border-slate-100">
                <td className="py-1 font-mono text-xs">{m.model}</td>
                <td className="text-right">{count(m.calls)}</td>
                <td className="text-right">{count(m.input_tokens)}</td>
                <td className="text-right">{count(m.output_tokens)}</td>
                <td className="text-right">{m.calls ? count(Math.round((m.input_tokens + m.output_tokens) / m.calls)) : "–"}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </Section>
  );
}

function FeedbackSection({ loaded }: { loaded: Loaded<FeedbackView> }) {
  const f = loaded.data;
  if (!f) return <Section title="Feedback">{loaded.error ? <Failed error={loaded.error} /> : <p className="text-sm text-slate-500">Loading…</p>}</Section>;
  const rated = f.counts.up + f.counts.down;
  return (
    <Section title="Feedback" note="From the feedback file: survives restarts.">
      <div className="mb-3 grid grid-cols-3 gap-3">
        <Tile label="Helpful" value={pct(f.counts.up, rated)} sub={`${count(f.counts.up)} 👍 · ${count(f.counts.down)} 👎`} />
        <Tile label="Ratings" value={count(rated)} sub={f.malformed ? `${f.malformed} damaged lines skipped` : undefined} />
        <Tile label="With comment" value={count(f.recent.filter((r) => r.comment).length)} sub={`in the last ${f.recent.length}`} />
      </div>
      <ul className="max-h-64 space-y-2 overflow-y-auto text-sm" aria-label="Recent feedback">
        {f.recent.map((r) => (
          <li key={`${r.ts}-${r.answer_request_id}`} className="rounded border border-slate-100 p-2">
            <div className="flex justify-between gap-2 text-xs text-slate-500">
              <span>
                <span aria-label={r.rating === "up" ? "helpful" : "not helpful"}>{r.rating === "up" ? "👍" : "👎"}</span> {dateTime(r.ts)}
              </span>
              <span className="font-mono" title="Find this answer in the API log">{r.answer_request_id}</span>
            </div>
            {r.comment && <p className="mt-1 text-slate-800">{r.comment}</p>}
          </li>
        ))}
        {f.recent.length === 0 && <li className="text-slate-500">No feedback yet.</li>}
      </ul>
    </Section>
  );
}

const isRate = (metric: string) => !/gates_passed|llm_calls|tokens/.test(metric);

function EvaluationSection({ loaded }: { loaded: Loaded<EvaluationsView> }) {
  const runs = loaded.data?.runs ?? [];
  const kinds = [...new Set(runs.map((r) => r.kind))]; // newest run's kind first: the default
  const [kindChoice, setKind] = useState<string>();
  const kind = kindChoice && kinds.includes(kindChoice) ? kindChoice : kinds[0];
  const names = kind ? metricNames(runs, kind) : [];
  const [metricChoice, setMetric] = useState<string>();
  const metric = metricChoice && names.includes(metricChoice) ? metricChoice : names[0];
  const { points, dataVersion, excluded } = kind && metric ? trend(runs, kind, metric) : { points: [], dataVersion: undefined, excluded: 0 };
  const format = (v: number) => (metric && isRate(metric) ? v.toFixed(3) : String(Math.round(v * 10) / 10));

  return (
    <Section title="Evaluation runs" note="Tracked with --track jsonl. Only runs on the same data version are joined in the trend.">
      {loaded.error && <Failed error={loaded.error} />}
      {runs.length === 0 && !loaded.error && (
        <p className="text-sm text-slate-500">
          No tracked runs yet. Run <code className="rounded bg-slate-100 px-1">python scripts/evaluate.py --track jsonl</code>.
        </p>
      )}
      {runs.length > 0 && (
        <>
          <div className="mb-3 flex flex-wrap gap-3 text-sm">
            <label className="flex items-center gap-2">
              Kind
              <select value={kind} onChange={(e) => { setKind(e.target.value); setMetric(undefined); }} className="rounded border border-slate-300 px-2 py-1">
                {kinds.map((k) => <option key={k}>{k}</option>)}
              </select>
            </label>
            <label className="flex items-center gap-2">
              Metric
              <select value={metric} onChange={(e) => setMetric(e.target.value)} className="rounded border border-slate-300 px-2 py-1">
                {names.map((n) => <option key={n}>{n}</option>)}
              </select>
            </label>
            {excluded > 0 && <span className="self-center text-xs text-amber-700">{excluded} run(s) on other data not joined</span>}
          </div>
          {points.length > 0 && metric && (
            <LineChart
              title={`${metric} (data ${dataVersion})`}
              points={points.map(({ run, value }) => ({ key: run.run_id, label: dateTime(run.time), value, detail: `${run.name} · ${run.git_commit}${run.git_dirty ? " (dirty)" : ""}` }))}
              format={format}
              yMax={isRate(metric) ? 1 : undefined}
            />
          )}
          <RunsTable runs={runs.filter((r) => r.kind === kind)} metric={metric} format={format} latestData={dataVersion} />
        </>
      )}
    </Section>
  );
}

function RunsTable({ runs, metric, format, latestData }: { runs: Run[]; metric?: string; format: (v: number) => string; latestData?: string }) {
  return (
    <div className="mt-3 max-h-72 overflow-auto">
      <table className="w-full text-sm">
        <thead className="sticky top-0 bg-white text-left text-xs text-slate-500">
          <tr><th className="py-1">Time</th><th>Run</th><th>Commit</th><th>Data</th><th className="text-right">{metric ?? "Metric"}</th><th className="text-right">Gates</th></tr>
        </thead>
        <tbody>
          {runs.map((r) => (
            <tr key={r.run_id} className="border-t border-slate-100">
              <td className="py-1 whitespace-nowrap">{dateTime(r.time)}</td>
              <td>{r.name}</td>
              <td className="font-mono text-xs">
                {r.git_commit}
                {r.git_dirty && <span className="ml-1 rounded bg-amber-100 px-1 text-amber-800" title="Measured with uncommitted changes">dirty</span>}
              </td>
              <td className={`font-mono text-xs ${r.data_version === latestData ? "" : "text-amber-700"}`}>{r.data_version}</td>
              <td className="text-right">{metric && typeof r.metrics[metric] === "number" ? format(r.metrics[metric]) : "–"}</td>
              <td className="text-right">{r.gates ?? "–"}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function IngestionSection({ loaded }: { loaded: Loaded<IngestionView> }) {
  const report = loaded.data?.report;
  return (
    <Section title="Ingestion" note="The last scripts/index.py run.">
      {loaded.error && <Failed error={loaded.error} />}
      {!report && loaded.data && <p className="text-sm text-slate-500">{loaded.data.reason}</p>}
      {report && (
        <>
          <div className="mb-3 grid grid-cols-2 gap-3 sm:grid-cols-4">
            <Tile label="Last run" value={dateTime(report.time)} sub={report.seconds !== null ? `took ${report.seconds} s` : undefined} />
            <Tile label="Loaded" value={count(report.loaded)} sub={`${report.sync.added ?? 0} added · ${report.sync.updated ?? 0} updated · ${report.sync.removed ?? 0} removed`} />
            <Tile label="Failed" value={count(report.failed.length)} sub="keep their previous version" tone={report.failed.length ? "bad" : "good"} />
            <Tile label="Duplicates" value={count(report.duplicates.length)} sub={`${report.skipped.length} skipped`} />
          </div>
          {report.failed.length > 0 && (
            <ul className="space-y-1 text-sm" aria-label="Failed files">
              {report.failed.map((f) => (
                <li key={f.source} className="rounded border border-red-100 bg-red-50 p-2">
                  <span className="font-mono text-xs">{f.source}</span>: <span className="text-red-800">{f.reason}</span>
                </li>
              ))}
            </ul>
          )}
          {report.duplicates.length > 0 && (
            <ul className="mt-2 space-y-1 text-sm text-slate-600" aria-label="Duplicate files">
              {report.duplicates.map((d) => (
                <li key={d.source}><span className="font-mono text-xs">{d.source}</span> has the same content as <span className="font-mono text-xs">{d.same_as}</span></li>
              ))}
            </ul>
          )}
        </>
      )}
    </Section>
  );
}
