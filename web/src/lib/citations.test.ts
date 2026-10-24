import { describe, expect, it } from "vitest";
import { splitCitations } from "./citations";

describe("splitCitations", () => {
  it("turns valid markers into citations", () => {
    expect(splitCitations("Overbend the die [1]. Then anneal [2].", new Set([1, 2]))).toEqual([
      { kind: "text", text: "Overbend the die " },
      { kind: "citation", number: 1 },
      { kind: "text", text: ". Then anneal " },
      { kind: "citation", number: 2 },
      { kind: "text", text: "." },
    ]);
  });

  it("splits grouped markers", () => {
    expect(splitCitations("Yes [1, 3]", new Set([1, 3]))).toEqual([
      { kind: "text", text: "Yes " },
      { kind: "citation", number: 1 },
      { kind: "citation", number: 3 },
    ]);
  });

  it("leaves markers with unknown source numbers as text", () => {
    expect(splitCitations("Claim [7] and [1, 9].", new Set([1]))).toEqual([{ kind: "text", text: "Claim [7] and [1, 9]." }]);
  });

  it("returns no segments for an empty answer", () => {
    expect(splitCitations("", new Set())).toEqual([]);
  });
});
