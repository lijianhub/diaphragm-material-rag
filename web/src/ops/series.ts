/**
 * Turning cumulative counters into something to plot.
 *
 * The API's counters only go up, from process start. Traffic per minute is the difference
 * between two snapshots divided by the time between them. A restart resets the counters, so
 * a drop in uptime or in a counter starts a new series instead of producing a negative rate.
 */
import type { Run, Summary } from "./api";

export interface Sample {
  at: number; // ms since epoch, when the snapshot was taken
  summary: Summary;
}

export interface RatePoint {
  at: number;
  perMinute: number;
  serverErrors: number;
}

export function trafficRates(samples: Sample[]): RatePoint[] {
  const points: RatePoint[] = [];
  for (let i = 1; i < samples.length; i++) {
    const before = samples[i - 1]!;
    const after = samples[i]!;
    const restarted =
      after.summary.started_at !== before.summary.started_at || after.summary.traffic.requests < before.summary.traffic.requests;
    if (restarted) {
      points.length = 0; // earlier points belong to the previous process
      continue;
    }
    const minutes = (after.at - before.at) / 60_000;
    if (minutes <= 0) continue;
    points.push({
      at: after.at,
      perMinute: (after.summary.traffic.requests - before.summary.traffic.requests) / minutes,
      serverErrors: after.summary.traffic.server_errors - before.summary.traffic.server_errors,
    });
  }
  return points;
}

/** Metric names present in runs of one kind, most common first. */
export function metricNames(runs: Run[], kind: string): string[] {
  const seen = new Map<string, number>();
  for (const run of runs) {
    if (run.kind !== kind) continue;
    for (const name of Object.keys(run.metrics)) seen.set(name, (seen.get(name) ?? 0) + 1);
  }
  return [...seen.entries()].sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0])).map(([name]) => name);
}

export interface TrendPoint {
  run: Run;
  value: number;
}

/**
 * The runs to plot for one metric: same kind, have the metric, and measured on the same
 * data as the newest such run. Runs on other data are not comparable (a score change could
 * come from the data, not the system), so they are listed but never joined into the line.
 */
export function trend(runs: Run[], kind: string, metric: string): { points: TrendPoint[]; dataVersion?: string; excluded: number } {
  const candidates = runs.filter((r) => r.kind === kind && typeof r.metrics[metric] === "number");
  if (candidates.length === 0) return { points: [], excluded: 0 };
  const newest = candidates.reduce((a, b) => (a.time >= b.time ? a : b));
  const comparable = candidates.filter((r) => r.data_version === newest.data_version);
  const points = comparable.sort((a, b) => a.time.localeCompare(b.time)).map((run) => ({ run, value: run.metrics[metric]! }));
  return { points, dataVersion: newest.data_version, excluded: candidates.length - comparable.length };
}
