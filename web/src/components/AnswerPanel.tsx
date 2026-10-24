import type { Answer } from "../lib/api";
import { splitCitations } from "../lib/citations";

interface Props {
  streamedText: string;
  final?: Answer;
  streaming: boolean;
  stopped: boolean;
  onCite: (number: number) => void;
}

export function AnswerPanel({ streamedText, final, streaming, stopped, onCite }: Props) {
  if (!final && !streamedText && !streaming) return null;

  const valid = new Set(final?.citations.map((c) => c.number) ?? []);
  const tone = final?.refused ? "border-amber-300 bg-amber-50" : "border-slate-200 bg-white";

  return (
    <section aria-labelledby="answer-heading" className={`rounded-lg border p-4 shadow-sm ${tone}`}>
      <h2 id="answer-heading" className="mb-2 text-sm font-semibold text-slate-600">
        {final?.refused ? "Not found in the documents" : "Answer"}
      </h2>
      {/* aria-live: screen readers announce the text as it streams in */}
      <p aria-live="polite" aria-busy={streaming} className="leading-relaxed whitespace-pre-wrap text-slate-900" data-testid="answer-text">
        {final
          ? splitCitations(final.answer, valid).map((segment, i) =>
              segment.kind === "text" ? (
                <span key={i}>{segment.text}</span>
              ) : (
                <button
                  key={i}
                  type="button"
                  onClick={() => onCite(segment.number)}
                  aria-label={`Show source ${segment.number}`}
                  className="mx-0.5 rounded bg-sky-100 px-1.5 align-baseline text-xs font-semibold text-sky-800 hover:bg-sky-200"
                >
                  {segment.number}
                </button>
              ),
            )
          : streamedText}
        {streaming && <span className="ml-0.5 inline-block h-4 w-2 animate-pulse bg-slate-400 align-middle" aria-hidden />}
      </p>
      {stopped && !final && <p className="mt-2 text-sm text-slate-500">Stopped. The answer above is incomplete.</p>}
      {final?.fallback_reason && (
        <p className="mt-2 text-sm text-amber-700">
          The language model was unavailable, so these are quoted sentences from the sources.
        </p>
      )}
      {final && final.invalid_citations.length > 0 && (
        <p className="mt-2 text-sm text-red-700">The answer referred to sources that do not exist: {final.invalid_citations.join(", ")}.</p>
      )}
    </section>
  );
}
