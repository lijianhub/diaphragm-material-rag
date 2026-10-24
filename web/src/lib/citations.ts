/** Split an answer into text and citation markers, so `[2]` can become a link to source 2. */
export type Segment = { kind: "text"; text: string } | { kind: "citation"; number: number };

const MARKER = /\[(\d+(?:\s*,\s*\d+)*)\]/g;

/**
 * Markers whose numbers are not valid sources stay plain text: the API reports them as
 * `invalid_citations`, and a link to a source that does not exist would mislead.
 */
export function splitCitations(text: string, validNumbers: ReadonlySet<number>): Segment[] {
  const segments: Segment[] = [];
  let last = 0;
  for (const match of text.matchAll(MARKER)) {
    const numbers = match[1]!.split(",").map((n) => Number(n.trim()));
    if (!numbers.every((n) => validNumbers.has(n))) continue;
    if (match.index > last) segments.push({ kind: "text", text: text.slice(last, match.index) });
    for (const number of numbers) segments.push({ kind: "citation", number });
    last = match.index + match[0].length;
  }
  if (last < text.length) segments.push({ kind: "text", text: text.slice(last) });
  return segments;
}
