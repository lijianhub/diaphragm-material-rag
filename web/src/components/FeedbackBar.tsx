import { useState } from "react";
import { ApiError, sendFeedback } from "../lib/api";

type State = "idle" | "comment" | "sending" | "sent" | "failed";

export function FeedbackBar({ requestId }: { requestId: string }) {
  const [state, setState] = useState<State>("idle");
  const [rating, setRating] = useState<"up" | "down">();
  const [comment, setComment] = useState("");
  const [error, setError] = useState("");

  const send = async (value: "up" | "down", text?: string) => {
    setRating(value);
    setState("sending");
    try {
      await sendFeedback(requestId, value, text?.trim() || undefined);
      setState("sent");
    } catch (e) {
      setError(e instanceof ApiError ? e.message : "Could not send feedback.");
      setState("failed");
    }
  };

  if (state === "sent") {
    return <p className="text-sm text-slate-600">Thanks for the feedback{rating === "down" ? "; it helps improve the answers" : ""}.</p>;
  }

  return (
    <div className="space-y-2 text-sm">
      <div className="flex items-center gap-2">
        <span className="text-slate-600">Was this answer helpful?</span>
        <button
          type="button"
          aria-label="Helpful"
          disabled={state === "sending"}
          onClick={() => send("up")}
          className="rounded-lg border border-slate-200 bg-white px-2 py-1 hover:bg-slate-50 disabled:opacity-50"
        >
          👍
        </button>
        <button
          type="button"
          aria-label="Not helpful"
          disabled={state === "sending"}
          onClick={() => setState("comment")}
          className="rounded-lg border border-slate-200 bg-white px-2 py-1 hover:bg-slate-50 disabled:opacity-50"
        >
          👎
        </button>
      </div>
      {(state === "comment" || (state === "sending" && rating === "down")) && (
        <div className="flex gap-2">
          <input
            aria-label="What was wrong? (optional)"
            placeholder="What was wrong? (optional)"
            maxLength={500}
            value={comment}
            onChange={(e) => setComment(e.target.value)}
            className="flex-1 rounded-lg border border-slate-300 px-2 py-1"
          />
          <button type="button" onClick={() => send("down", comment)} disabled={state === "sending"} className="rounded-lg bg-slate-700 px-3 py-1 text-white hover:bg-slate-800 disabled:opacity-50">
            Send
          </button>
        </div>
      )}
      {state === "failed" && <p className="text-red-700">{error}</p>}
    </div>
  );
}
