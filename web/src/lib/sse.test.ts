import { describe, expect, it } from "vitest";
import { SSEParser } from "./sse";

describe("SSEParser", () => {
  it("parses one complete event", () => {
    expect(new SSEParser().push('event: delta\ndata: {"text":"hi"}\n\n')).toEqual([{ event: "delta", data: '{"text":"hi"}' }]);
  });

  it("keeps an event split across chunks until it is complete", () => {
    const parser = new SSEParser();
    expect(parser.push("event: del")).toEqual([]);
    expect(parser.push('ta\ndata: {"te')).toEqual([]);
    expect(parser.push('xt":"a"}\n')).toEqual([]);
    expect(parser.push("\n")).toEqual([{ event: "delta", data: '{"text":"a"}' }]);
  });

  it("returns several events from one chunk and keeps the remainder", () => {
    const parser = new SSEParser();
    const events = parser.push("event: a\ndata: 1\n\nevent: b\ndata: 2\n\nevent: c\ndata: 3");
    expect(events.map((e) => e.event)).toEqual(["a", "b"]);
    expect(parser.push("\n\n")).toEqual([{ event: "c", data: "3" }]);
  });

  it("accepts CRLF line endings, even split between chunks", () => {
    const parser = new SSEParser();
    expect(parser.push("event: x\r\ndata: 1\r")).toEqual([]);
    expect(parser.push("\n\r\n")).toEqual([{ event: "x", data: "1" }]);
  });

  it("joins multi-line data, ignores comments and defaults the event name", () => {
    expect(new SSEParser().push(": keep-alive\n\ndata: line 1\ndata: line 2\n\n")).toEqual([{ event: "message", data: "line 1\nline 2" }]);
  });
});
