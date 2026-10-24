import type { Answer } from "../lib/api";
import { ms } from "../lib/format";

/** Provenance of the answer: which model and prompt wrote it, what it cost, how long it took. */
export function RunDetails({ answer }: { answer: Answer }) {
  const t = answer.timings_ms;
  return (
    <details className="text-xs text-slate-500">
      <summary className="cursor-pointer select-none">Details</summary>
      <dl className="mt-2 grid grid-cols-[auto_1fr] gap-x-4 gap-y-1">
        <dt>Model</dt>
        <dd>{answer.model}</dd>
        <dt>Prompt</dt>
        <dd>{answer.prompt_version}</dd>
        <dt>Tokens</dt>
        <dd>{answer.usage ? `${answer.usage.input_tokens} in / ${answer.usage.output_tokens} out` : "–"}</dd>
        <dt>Time</dt>
        <dd>
          retrieval {ms(t.retrieval_ms)} · first text {ms(t.first_token_ms)} · generation {ms(t.generation_ms)}
        </dd>
        <dt>Request ID</dt>
        <dd className="font-mono">{answer.request_id}</dd>
      </dl>
    </details>
  );
}
