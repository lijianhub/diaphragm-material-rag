import { useEffect, useRef, useState } from "react";
import type { Source } from "../lib/api";

interface Props {
  sources: Source[];
  cited: ReadonlySet<number>;
  active?: number;
}

export function SourceList({ sources, cited, active }: Props) {
  if (sources.length === 0) return null;
  return (
    <section aria-labelledby="sources-heading" className="space-y-2">
      <h2 id="sources-heading" className="text-sm font-semibold text-slate-600">
        Sources ({sources.length})
      </h2>
      <ol className="space-y-2">
        {sources.map((source) => (
          <SourceCard key={source.number} source={source} cited={cited.has(source.number)} active={active === source.number} />
        ))}
      </ol>
    </section>
  );
}

function SourceCard({ source, cited, active }: { source: Source; cited: boolean; active: boolean }) {
  const [open, setOpen] = useState(false);
  const [wasActive, setWasActive] = useState(active);
  const ref = useRef<HTMLLIElement>(null);

  // Expand when the card becomes active (adjusted during render, not in an effect).
  if (active !== wasActive) {
    setWasActive(active);
    if (active) setOpen(true);
  }

  useEffect(() => {
    if (active) ref.current?.scrollIntoView?.({ behavior: "smooth", block: "nearest" });
  }, [active]);

  return (
    <li
      ref={ref}
      id={`source-${source.number}`}
      data-active={active || undefined}
      className={`rounded-lg border bg-white p-3 text-sm shadow-sm transition ${active ? "border-sky-500 ring-2 ring-sky-200" : "border-slate-200"}`}
    >
      <div className="flex items-start justify-between gap-2">
        <div>
          <span className="mr-2 inline-block rounded bg-slate-100 px-1.5 text-xs font-semibold text-slate-700">{source.number}</span>
          <span className="font-medium text-slate-800">{source.label}</span>
          {cited && <span className="ml-2 rounded bg-sky-50 px-1.5 text-xs text-sky-700">cited</span>}
        </div>
        <button type="button" onClick={() => setOpen((o) => !o)} aria-expanded={open} className="shrink-0 text-xs text-sky-700 hover:underline">
          {open ? "Hide text" : "Show text"}
        </button>
      </div>
      <p className={`mt-2 whitespace-pre-wrap text-slate-600 ${open ? "" : "line-clamp-2"}`}>{source.text}</p>
    </li>
  );
}
