import { type FormEvent, useState } from "react";

const MAX_CHARS = 2000;

interface Props {
  busy: boolean;
  onAsk: (question: string) => void;
  onStop: () => void;
}

export function AskForm({ busy, onAsk, onStop }: Props) {
  const [question, setQuestion] = useState("");
  const trimmed = question.trim();

  const submit = (event: FormEvent) => {
    event.preventDefault();
    if (trimmed && !busy) onAsk(trimmed);
  };

  return (
    <form onSubmit={submit} className="space-y-2">
      <label htmlFor="question" className="block text-sm font-medium text-slate-700">
        Ask about the diaphragm and alloy documents
      </label>
      <textarea
        id="question"
        value={question}
        onChange={(e) => setQuestion(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === "Enter" && !e.shiftKey) submit(e);
        }}
        maxLength={MAX_CHARS}
        rows={3}
        placeholder="e.g. How do toolmakers compensate for springback in cobalt alloys?"
        className="w-full rounded-lg border border-slate-300 bg-white p-3 text-slate-900 shadow-sm focus:border-sky-500 focus:outline-none focus:ring-2 focus:ring-sky-200"
      />
      <div className="flex items-center justify-between">
        <span className="text-xs text-slate-500">
          {question.length}/{MAX_CHARS} · Enter to ask, Shift+Enter for a new line
        </span>
        {busy ? (
          <button type="button" onClick={onStop} className="rounded-lg bg-slate-200 px-4 py-2 text-sm font-medium text-slate-800 hover:bg-slate-300">
            Stop
          </button>
        ) : (
          <button
            type="submit"
            disabled={!trimmed}
            className="rounded-lg bg-sky-600 px-4 py-2 text-sm font-medium text-white hover:bg-sky-700 disabled:cursor-not-allowed disabled:bg-slate-300"
          >
            Ask
          </button>
        )}
      </div>
    </form>
  );
}
