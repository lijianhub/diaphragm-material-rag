"""The agent loop: a plain-Python state machine around a planner and a set of tools.

    start(question) ──► running ──► completed           (the planner answered)
                          │  ▲  └──► stopped             (a step, tool-call, token or time limit)
                          │  │  └──► failed              (the planner raised)
           side effect ──►│  │
                          ▼  │ decide(approve | reject)
                    awaiting_approval

Each turn the planner (an LLM with native tool calling, or the offline rule planner)
returns text and/or tool calls. Every call is checked before it runs: the tool must
exist, the arguments must match its schema, and the same call must not repeat. Read-only
tools run with a timeout; a tool with a side effect is **not run**: the run is stored and
pauses until a person approves or rejects that exact call. All of this is enforced here,
in code, whatever the model was told in its prompt.
"""
from __future__ import annotations

import json
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Optional, Sequence

from generation.llm import ToolCall, ToolTurn
from generation.prompt import check_citations, strip_refusal_citations
from generation.rag_chain import ExtractiveAnswerer

from .prompts import AGENT_SYSTEM_PROMPT, agent_prompt_version, user_message
from .store import RunStore
from .tools import MAX_RESULT_CHARS, Tool, ToolContext, ToolError, validate

FINAL_STATUSES = ("completed", "stopped", "failed")


class RunNotFound(KeyError):
    pass


class NotAwaitingApproval(RuntimeError):
    """The run is not waiting for a decision (already decided, finished, or unknown state)."""


@dataclass
class Limits:
    max_steps: int = 6  # planner turns
    max_tool_calls: int = 12
    max_tokens: int = 40_000  # input + output, over the whole run
    max_active_s: float = 90.0  # time spent working; waiting for approval does not count


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ---- planners --------------------------------------------------------------------------


class LLMPlanner:
    """A model with native tool calling (``tool_turn``), e.g. ``OpenAILLM`` or ``AnthropicLLM``."""

    def __init__(self, llm: Any):
        self.llm = llm
        self.name = getattr(llm, "name", "llm")
        self.provider = self.name.split(":", 1)[0]

    def next_turn(self, system: str, transcript: List[dict], tools: List[Tool], run: dict) -> ToolTurn:
        return self.llm.tool_turn(system, transcript, [t.spec() for t in tools])


class RulePlanner:
    """Offline planner for tests and demos: search for the question (and look up any alloy
    it names), then answer with quoted sentences. It never calls a side-effecting tool."""

    name = "rule-based"
    provider = "rule-based"

    def next_turn(self, system: str, transcript: List[dict], tools: List[Tool], run: dict) -> ToolTurn:
        by_name = {t.name: t for t in tools}
        if not any(entry["role"] == "tool" for entry in transcript):
            calls = []
            if "search_documents" in by_name:
                calls.append(ToolCall("c1", "search_documents", {"query": run["question"][:300]}))
            if "lookup_alloy" in by_name:
                for alloy in by_name["lookup_alloy"].parameters["properties"]["alloy"]["enum"]:
                    if alloy.lower() in run["question"].lower():
                        calls.append(ToolCall(f"c{len(calls) + 1}", "lookup_alloy", {"alloy": alloy}))
            if calls:
                return ToolTurn("", calls, [], self.name, "tool_use")
        results = [SimpleNamespace(text=s["text"]) for s in run["sources"]]
        return ToolTurn(ExtractiveAnswerer().answer(run["question"], results), [], [], self.name, "end_turn")


# ---- the agent -------------------------------------------------------------------------


class Agent:
    def __init__(
        self,
        planner: Any,
        tools: Sequence[Tool],
        store: RunStore,
        limits: Optional[Limits] = None,
        system: str = AGENT_SYSTEM_PROMPT,
        on_event: Optional[Callable[[str, Dict[str, Any]], None]] = None,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.planner = planner
        self.tools = {t.name: t for t in tools}
        self.store = store
        self.limits = limits or Limits()
        self.system = system
        self.on_event = on_event or (lambda kind, data: None)
        self.clock = clock
        # Tools run in worker threads so a slow one can be abandoned at its timeout.
        # Python cannot kill a thread: a stuck tool keeps its worker until it returns.
        self._pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="agent-tool")

    # -- public ------------------------------------------------------------------------

    def start(self, question: str, run_id: Optional[str] = None) -> dict:
        """``run_id`` is for reproducible benchmarks (a cached rerun needs the same IDs);
        normally it is a random UUID."""
        run = {
            "id": run_id or uuid.uuid4().hex,
            "question": question,
            "status": "running",
            "stop_reason": None,
            "error_type": None,
            "created": _now(),
            "updated": _now(),
            "planner": self.planner.name,
            "prompt_version": agent_prompt_version(self.system),
            "limits": asdict(self.limits),
            "steps": 0,
            "tool_calls": 0,
            "usage": {"input_tokens": 0, "output_tokens": 0},
            "active_ms": 0.0,
            "trace": [],
            "transcript": [{"role": "user", "content": user_message(question)}],
            "sources": [],
            "executed": [],
            "pending": None,
            "approvals": [],
            "answer": None,
            "citations": [],
            "invalid_citations": [],
            "refused": False,
        }
        self.store.save(run)
        return self._drive(run)

    def get(self, run_id: str) -> dict:
        run = self.store.get(run_id)
        if run is None:
            raise RunNotFound(run_id)
        return run

    def decide(self, run_id: str, decision: str, reviewer: str, reason: str = "") -> dict:
        """Approve or reject the pending call, then continue the run."""
        if decision not in ("approve", "reject"):
            raise ValueError("decision must be 'approve' or 'reject'")
        self.get(run_id)
        # Compare-and-set: of two concurrent decisions, exactly one proceeds.
        if not self.store.claim(run_id, "awaiting_approval", "deciding"):
            raise NotAwaitingApproval(run_id)
        run = self.get(run_id)
        pending = run["pending"]
        run["pending"] = None
        run["approvals"].append({"call_id": pending["call_id"], "tool": pending["tool"], "arguments": pending["arguments"],
                                 "decision": decision, "reviewer": reviewer, "reason": reason, "ts": _now()})
        self.on_event("approval", {"decision": decision, "tool": pending["tool"]})
        call = ToolCall(pending["call_id"], pending["tool"], pending["arguments"])
        if decision == "approve":
            content, status, ms = self._execute(self.tools[pending["tool"]], call, run)
            is_error = status != "ok"
            trace_status = "approved" if status == "ok" else status
        else:
            note = f" Reason: {reason}" if reason else ""
            content, is_error, ms, trace_status = f"Rejected by the reviewer ({reviewer}). The action was not performed.{note} Do not retry it.", True, 0.0, "rejected"
        run["transcript"].append({"role": "tool", "call_id": call.id, "name": call.name, "content": content, "is_error": is_error})
        run["trace"].append({"step": run["steps"], "type": "approval", "tool": call.name, "arguments": call.arguments,
                             "status": trace_status, "summary": content[:300], "ms": ms, "reviewer": reviewer})
        self.on_event("tool", {"tool": call.name, "outcome": trace_status})
        run["status"] = "running"
        return self._drive(run)

    # -- the loop ----------------------------------------------------------------------

    def _drive(self, run: dict) -> dict:
        started = self.clock()
        try:
            while True:
                stop = self._limit_reached(run, started)
                if stop:
                    return self._finish(run, "stopped", stop_reason=stop)
                try:
                    turn = self.planner.next_turn(self.system, run["transcript"], list(self.tools.values()), run)
                except Exception as exc:  # the model or its API failed: stop cleanly, keep the trace
                    return self._finish(run, "failed", stop_reason="planner_error", error_type=type(exc).__name__)
                run["steps"] += 1
                run["usage"]["input_tokens"] += turn.input_tokens
                run["usage"]["output_tokens"] += turn.output_tokens
                run["transcript"].append({
                    "role": "assistant", "provider": getattr(self.planner, "provider", "unknown"), "items": turn.items,
                    "text": turn.text, "calls": [{"id": c.id, "name": c.name, "arguments": c.arguments} for c in turn.calls],
                })
                if not turn.calls:
                    return self._answer(run, turn.text)
                if self._handle_calls(run, turn.calls):
                    return self._pause(run)
        finally:
            run["active_ms"] = round(run["active_ms"] + (self.clock() - started) * 1000, 1)
            run["updated"] = _now()
            self.store.save(run)

    def _limit_reached(self, run: dict, started: float) -> Optional[str]:
        limits = self.limits
        if run["steps"] >= limits.max_steps:
            return "step_limit"
        if run["usage"]["input_tokens"] + run["usage"]["output_tokens"] >= limits.max_tokens:
            return "token_budget"
        if run["active_ms"] / 1000 + (self.clock() - started) >= limits.max_active_s:
            return "time_limit"
        return None

    def _handle_calls(self, run: dict, calls: List[ToolCall]) -> bool:
        """Run or refuse each call; True if the run must pause for an approval.

        Every call gets a result in the transcript (the APIs require one per call), so
        calls after a paused one are answered with "not run"."""
        paused = False
        for call in calls:
            def reply(content: str, status: str, ms: float = 0.0, is_error: bool = True) -> None:
                run["transcript"].append({"role": "tool", "call_id": call.id, "name": call.name, "content": content[:MAX_RESULT_CHARS], "is_error": is_error})
                run["trace"].append({"step": run["steps"], "type": "tool", "tool": call.name, "arguments": call.arguments,
                                     "status": status, "summary": content[:300], "ms": ms})
                self.on_event("tool", {"tool": call.name if call.name in self.tools else "unknown", "outcome": status})

            tool = self.tools.get(call.name)
            if paused:
                reply("Not run: an earlier action in this turn is waiting for a person's approval. Call it again later if it is still needed.", "not_run")
                continue
            if tool is None:
                reply(f"Unknown tool {call.name!r}. Available: {', '.join(self.tools)}.", "unknown_tool")
                continue
            if run["tool_calls"] >= self.limits.max_tool_calls:
                reply("Not run: the tool-call limit for this run is reached. Answer with what you have.", "limit")
                continue
            run["tool_calls"] += 1
            if call.arguments is None:
                reply(f"Invalid arguments: not a JSON object ({(call.raw_arguments or '')[:100]!r}).", "invalid_arguments")
                continue
            errors = validate(tool.parameters, call.arguments)
            if errors:
                reply("Invalid arguments: " + "; ".join(errors[:5]), "invalid_arguments")
                continue
            signature = call.name + json.dumps(call.arguments, sort_keys=True)
            if signature in run["executed"]:
                reply("This exact call was already made; use its earlier result.", "duplicate")
                continue
            run["executed"].append(signature)
            if tool.side_effect:
                run["pending"] = {"call_id": call.id, "tool": call.name, "arguments": call.arguments, "requested": _now()}
                run["trace"].append({"step": run["steps"], "type": "tool", "tool": call.name, "arguments": call.arguments,
                                     "status": "pending_approval", "summary": "waiting for a person's approval", "ms": 0.0})
                self.on_event("tool", {"tool": call.name, "outcome": "pending_approval"})
                paused = True
                continue
            content, status, ms = self._execute(tool, call, run)
            reply(content, status, ms, is_error=status != "ok")
        return paused

    def _execute(self, tool: Tool, call: ToolCall, run: dict):
        def add_sources(sources: List[dict]) -> List[int]:
            numbers = []
            for source in sources:
                match = next((s["number"] for s in run["sources"] if s["source"] == source["source"] and s["text"] == source["text"]), None)
                if match is None:
                    match = len(run["sources"]) + 1
                    run["sources"].append({"number": match, **source})
                numbers.append(match)
            return numbers

        ctx = ToolContext(run_id=run["id"], call_id=call.id, add_sources=add_sources)
        started = time.perf_counter()
        future = self._pool.submit(tool.func, call.arguments, ctx)
        try:
            content, status = future.result(timeout=tool.timeout_s), "ok"
        except FutureTimeout:
            content, status = f"The tool timed out after {tool.timeout_s:g} s.", "timeout"
        except ToolError as exc:
            content, status = f"Tool error: {exc}", "error"
        except Exception as exc:  # no internals to the model or the user
            content, status = f"The tool failed ({type(exc).__name__}).", "error"
        return str(content)[:MAX_RESULT_CHARS], status, round((time.perf_counter() - started) * 1000, 1)

    def _answer(self, run: dict, text: str) -> dict:
        # The same guards as the RAG pipeline: no citations on "not in the documents" sentences,
        # and citation numbers checked against the passages this run actually retrieved.
        text, refused = strip_refusal_citations(text)
        run["refused"] = refused
        valid, invalid = check_citations(text, len(run["sources"]))
        by_number = {s["number"]: s for s in run["sources"]}
        run["answer"] = text
        run["citations"] = [{"number": n, "label": by_number[n]["label"], "source": by_number[n]["source"]} for n in valid]
        run["invalid_citations"] = invalid
        run["trace"].append({"step": run["steps"], "type": "answer", "tool": None, "arguments": None, "status": "ok", "summary": text[:300], "ms": 0.0})
        return self._finish(run, "completed")

    def _pause(self, run: dict) -> dict:
        run["status"] = "awaiting_approval"
        self.on_event("run", {"status": "awaiting_approval"})
        return run

    def _finish(self, run: dict, status: str, stop_reason: Optional[str] = None, error_type: Optional[str] = None) -> dict:
        run["status"], run["stop_reason"], run["error_type"] = status, stop_reason, error_type
        if status == "stopped":
            run["trace"].append({"step": run["steps"], "type": "limit", "tool": None, "arguments": None, "status": stop_reason, "summary": f"stopped: {stop_reason}", "ms": 0.0})
        self.on_event("run", {"status": status})
        return run
