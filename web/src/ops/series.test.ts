import { describe, expect, it } from "vitest";
import { evaluations, summary } from "./fixtures";
import { metricNames, trafficRates, trend } from "./series";

const at = (seconds: number, requests: number, startedAt = 1_000, serverErrors = 0) => ({
  at: seconds * 1000,
  summary: summary({ started_at: startedAt, traffic: { requests, client_errors: 0, server_errors: serverErrors } }),
});

describe("trafficRates", () => {
  it("turns cumulative counters into requests per minute between snapshots", () => {
    const points = trafficRates([at(0, 100), at(30, 110, 1_000, 1), at(60, 140, 1_000, 3)]);

    expect(points.map((p) => p.perMinute)).toEqual([20, 60]);
    expect(points.map((p) => p.serverErrors)).toEqual([1, 2]);
  });

  it("starts a new series after a restart instead of producing a negative rate", () => {
    const points = trafficRates([at(0, 100), at(30, 130), at(60, 5, 2_000), at(90, 15, 2_000)]);

    expect(points).toEqual([{ at: 90_000, perMinute: 20, serverErrors: 0 }]);
  });

  it("needs two snapshots", () => {
    expect(trafficRates([at(0, 100)])).toEqual([]);
  });
});

describe("evaluation trend", () => {
  it("lists metric names for one kind", () => {
    expect(metricNames(evaluations.runs, "retrieval")).toEqual(["hit@1", "mrr@1"]);
    expect(metricNames(evaluations.runs, "generation")).toEqual(["answerable.correct", "gates_passed"]);
  });

  it("joins only runs on the newest run's data version, oldest first", () => {
    const { points, dataVersion, excluded } = trend(evaluations.runs, "retrieval", "hit@1");

    expect(points.map((p) => p.run.run_id)).toEqual(["r3", "r4"]);
    expect(dataVersion).toBe("d1");
    expect(excluded).toBe(1); // r2 was measured on other data
  });

  it("is empty for a metric no run has", () => {
    expect(trend(evaluations.runs, "retrieval", "nope")).toEqual({ points: [], excluded: 0 });
  });
});
