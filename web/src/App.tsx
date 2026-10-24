import { useRef, useState } from "react";
import { AnswerPanel } from "./components/AnswerPanel";
import { AskForm } from "./components/AskForm";
import { FeedbackBar } from "./components/FeedbackBar";
import { RunDetails } from "./components/RunDetails";
import { SourceList } from "./components/SourceList";
import { type Answer, ApiError, type Source, streamAnswer } from "./lib/api";

type Status = "idle" | "streaming" | "done" | "stopped" | "error";

export default function App() {
  const [status, setStatus] = useState<Status>("idle");
  const [question, setQuestion] = useState("");
  const [sources, setSources] = useState<Source[]>([]);
  const [streamedText, setStreamedText] = useState("");
  const [final, setFinal] = useState<Answer>();
  const [error, setError] = useState<{ message: string; requestId?: string }>();
  const [activeSource, setActiveSource] = useState<number>();
  const controller = useRef<AbortController | null>(null);

  const ask = async (text: string) => {
    controller.current?.abort();
    const current = new AbortController();
    controller.current = current;
    setQuestion(text);
    setStatus("streaming");
    setSources([]);
    setStreamedText("");
    setFinal(undefined);
    setError(undefined);
    setActiveSource(undefined);
    try {
      await streamAnswer(
        text,
        {
          onSources: (s) => setSources(s),
          onDelta: (piece) => setStreamedText((previous) => previous + piece),
          // The final answer is checked (citations validated, refusals cleaned): it replaces the streamed text.
          onFinal: (answer) => {
            setFinal(answer);
            setSources(answer.sources);
          },
        },
        current.signal,
      );
      setStatus("done");
    } catch (e) {
      if (current.signal.aborted) {
        setStatus("stopped");
        return;
      }
      setError(e instanceof ApiError ? { message: e.message, requestId: e.requestId } : { message: "Could not reach the service." });
      setStatus("error");
    }
  };

  const stop = () => controller.current?.abort(); // the BFF and API stop the model too

  const cited = new Set(final?.citations.map((c) => c.number) ?? []);

  return (
    <div className="min-h-screen bg-slate-50">
      <header className="border-b border-slate-200 bg-white">
        <div className="mx-auto max-w-3xl px-4 py-4">
          <h1 className="text-lg font-semibold text-slate-900">Diaphragm Material Assistant</h1>
          <p className="text-sm text-slate-500">Answers from the engineering documents, with the sources they come from.</p>
        </div>
      </header>
      <main className="mx-auto max-w-3xl space-y-6 px-4 py-6">
        <AskForm busy={status === "streaming"} onAsk={ask} onStop={stop} />
        {question && status !== "idle" && <p className="text-sm text-slate-500">Question: {question}</p>}
        {error && (
          <div role="alert" className="rounded-lg border border-red-200 bg-red-50 p-3 text-sm text-red-800">
            {error.message}
            {error.requestId && <span className="mt-1 block font-mono text-xs text-red-600">Request ID: {error.requestId}</span>}
          </div>
        )}
        {status === "streaming" && sources.length === 0 && <p className="text-sm text-slate-500">Searching the documents…</p>}
        <AnswerPanel streamedText={streamedText} final={final} streaming={status === "streaming"} stopped={status === "stopped"} onCite={setActiveSource} />
        {final && (
          <div className="space-y-3">
            <FeedbackBar key={final.request_id} requestId={final.request_id} />
            <RunDetails answer={final} />
          </div>
        )}
        <SourceList sources={sources} cited={cited} active={activeSource} />
      </main>
    </div>
  );
}
