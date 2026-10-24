/**
 * Incremental parser for Server-Sent Events.
 *
 * `EventSource` only supports GET, and asking a question is a POST, so the stream is read
 * with `fetch` and parsed here. Network chunks do not respect event boundaries: one chunk
 * can hold half an event or several events, so unfinished text is kept for the next chunk.
 */
export interface SSEEvent {
  event: string;
  data: string;
}

export class SSEParser {
  private buffer = "";

  /** Add decoded text; return every event completed by it. */
  push(text: string): SSEEvent[] {
    this.buffer += text.replace(/\r\n?/g, "\n");
    const events: SSEEvent[] = [];
    let boundary = this.buffer.indexOf("\n\n");
    while (boundary !== -1) {
      const block = this.buffer.slice(0, boundary);
      this.buffer = this.buffer.slice(boundary + 2);
      const parsed = parseBlock(block);
      if (parsed) events.push(parsed);
      boundary = this.buffer.indexOf("\n\n");
    }
    return events;
  }
}

function parseBlock(block: string): SSEEvent | null {
  let event = "message";
  const data: string[] = [];
  for (const line of block.split("\n")) {
    if (line === "" || line.startsWith(":")) continue; // comment or keep-alive
    const colon = line.indexOf(":");
    const field = colon === -1 ? line : line.slice(0, colon);
    let value = colon === -1 ? "" : line.slice(colon + 1);
    if (value.startsWith(" ")) value = value.slice(1);
    if (field === "event") event = value;
    else if (field === "data") data.push(value);
  }
  return data.length ? { event, data: data.join("\n") } : null;
}
