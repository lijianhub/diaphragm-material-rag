"""Language-model backends for generation and query rewriting.

An LLM here is any object that can be called with a prompt string and returns
text. Backends may also offer ``chat(system, user)``, which the RAG pipeline
prefers so the grounding rules travel as a real system prompt, and
``tool_turn(system, transcript, tools)`` for the agent: one model turn with native
tool calling over a provider-neutral transcript (see ``ToolTurn``).
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Union

DEFAULT_MODEL = "claude-opus-5-5"
FALLBACK_BETA = "server-side-fallback-2026-07-01"


class LLMRefusal(Exception):
    """The model declined the request (``stop_reason == "refusal"``)."""


@dataclass
class LLMCall:
    model: str  # the model that actually served the request (may differ after a fallback)
    stop_reason: str
    input_tokens: int
    output_tokens: int


@dataclass
class ToolCall:
    """A tool call requested by the model. ``arguments`` is None when the model sent
    arguments that are not a JSON object; ``raw_arguments`` keeps what it sent."""

    id: str
    name: str
    arguments: Optional[Dict[str, Any]]
    raw_arguments: Optional[str] = None


@dataclass
class ToolTurn:
    """One model turn: text and/or tool calls, plus ``items``, the provider's own output
    as plain JSON, which must be sent back unchanged on the next turn (reasoning and
    thinking blocks included)."""

    text: str
    calls: List[ToolCall]
    items: List[Any]
    model: str
    stop_reason: str
    input_tokens: int = 0
    output_tokens: int = 0


def plain(value: Any) -> Any:
    """SDK objects (pydantic models, SimpleNamespace in tests) to JSON-able data."""
    if hasattr(value, "model_dump"):
        return value.model_dump(exclude_none=True)
    if isinstance(value, dict):
        return {k: plain(v) for k, v in value.items() if v is not None}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    if hasattr(value, "__dict__"):
        return {k: plain(v) for k, v in vars(value).items() if v is not None and not k.startswith("_")}
    return value


# Transcript entries, shared by both providers (JSON-able, so a paused run can be stored):
#   {"role": "user", "content": str}
#   {"role": "assistant", "provider": "openai" | "anthropic" | other, "items": [...], "text": str, "calls": [...]}
#   {"role": "tool", "call_id": str, "name": str, "content": str, "is_error": bool}


class AnthropicLLM:
    """Claude through the Anthropic API.

    - Opus 5.5 always thinks; depth and cost are set with ``effort`` (its default is
      ``medium``, set explicitly here). Sampling parameters such as temperature are
      not accepted by this model and are never sent.
    - ``fallbacks=True`` enables the server-side refusal fallback, which re-runs a
      declined request on a suitable model inside the same call. A request that is
      still declined raises ``LLMRefusal``, so callers can degrade gracefully.
    - Only text blocks are returned; thinking blocks are ignored.
    - Retries for 408/409/429/5xx and connection errors come from the SDK
      (``max_retries``); ``timeout`` is in seconds.
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        effort: str = "medium",
        max_tokens: int = 16000,
        timeout: float = 120.0,
        max_retries: int = 2,
        fallbacks: bool = True,
        client: Any = None,
    ):
        if client is None:
            try:
                import anthropic
            except ImportError as exc:  # pragma: no cover - needs the optional extra
                raise ImportError("AnthropicLLM needs: pip install -e .[llm]") from exc
            # Credentials come from the environment (ANTHROPIC_API_KEY or a CLI profile); never hard-code a key.
            client = anthropic.Anthropic(timeout=timeout, max_retries=max_retries)
        self.client = client
        self.model = model
        self.effort = effort
        self.max_tokens = max_tokens
        self.fallbacks = fallbacks
        self.last_call: Optional[LLMCall] = None

    @property
    def name(self) -> str:
        return f"anthropic:{self.model}"

    def chat(self, system: Optional[str], user: str) -> str:
        request = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": [{"role": "user", "content": user}],
            "output_config": {"effort": self.effort},
        }
        if system:
            request["system"] = system
        if self.fallbacks:
            response = self.client.beta.messages.create(**request, betas=[FALLBACK_BETA], fallbacks="default")
        else:
            response = self.client.messages.create(**request)

        usage = getattr(response, "usage", None)
        self.last_call = LLMCall(
            model=getattr(response, "model", self.model),
            stop_reason=response.stop_reason,
            input_tokens=getattr(usage, "input_tokens", 0),
            output_tokens=getattr(usage, "output_tokens", 0),
        )
        if response.stop_reason == "refusal":
            details = getattr(response, "stop_details", None)
            raise LLMRefusal(f"model declined (category: {getattr(details, 'category', None)})")
        return "".join(block.text for block in response.content if block.type == "text").strip()

    def stream_chat(self, system: Optional[str], user: str) -> Iterator[str]:
        """Yield the answer's text as it is generated; ``last_call`` is set at the end.

        Closing the generator early (the user went away) leaves the ``with`` block, which
        closes the HTTP stream, so the model stops generating and billing stops too.
        """
        request = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": [{"role": "user", "content": user}],
            "output_config": {"effort": self.effort},
        }
        if system:
            request["system"] = system
        if self.fallbacks:
            manager = self.client.beta.messages.stream(**request, betas=[FALLBACK_BETA], fallbacks="default")
        else:
            manager = self.client.messages.stream(**request)
        with manager as stream:
            for text in stream.text_stream:
                yield text
            response = stream.get_final_message()
        usage = getattr(response, "usage", None)
        self.last_call = LLMCall(
            model=getattr(response, "model", self.model),
            stop_reason=response.stop_reason,
            input_tokens=getattr(usage, "input_tokens", 0),
            output_tokens=getattr(usage, "output_tokens", 0),
        )
        if response.stop_reason == "refusal":
            details = getattr(response, "stop_details", None)
            raise LLMRefusal(f"model declined (category: {getattr(details, 'category', None)})")

    def tool_turn(self, system: Optional[str], transcript: List[dict], tools: List[dict]) -> ToolTurn:
        """One turn with tool use. Assistant content (thinking blocks included) is sent
        back as received; consecutive tool results become one user message of
        ``tool_result`` blocks, as the Messages API requires."""
        messages: List[dict] = []
        for entry in transcript:
            role = entry["role"]
            if role == "user":
                messages.append({"role": "user", "content": entry["content"]})
            elif role == "assistant":
                content = entry["items"] if entry.get("provider") == "anthropic" else _anthropic_blocks(entry)
                messages.append({"role": "assistant", "content": content})
            else:
                block = {"type": "tool_result", "tool_use_id": entry["call_id"], "content": entry["content"], "is_error": entry["is_error"]}
                if messages and messages[-1]["role"] == "user" and isinstance(messages[-1]["content"], list):
                    messages[-1]["content"].append(block)
                else:
                    messages.append({"role": "user", "content": [block]})
        request = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": messages,
            "tools": [{"name": t["name"], "description": t["description"], "input_schema": t["parameters"]} for t in tools],
            "output_config": {"effort": self.effort},
        }
        if system:
            request["system"] = system
        if self.fallbacks:
            response = self.client.beta.messages.create(**request, betas=[FALLBACK_BETA], fallbacks="default")
        else:
            response = self.client.messages.create(**request)
        usage = getattr(response, "usage", None)
        self.last_call = LLMCall(
            model=getattr(response, "model", self.model),
            stop_reason=response.stop_reason,
            input_tokens=getattr(usage, "input_tokens", 0),
            output_tokens=getattr(usage, "output_tokens", 0),
        )
        if response.stop_reason == "refusal":
            details = getattr(response, "stop_details", None)
            raise LLMRefusal(f"model declined (category: {getattr(details, 'category', None)})")
        calls = []
        for block in response.content:
            if block.type == "tool_use":
                arguments = block.input if isinstance(block.input, dict) else None
                calls.append(ToolCall(block.id, block.name, arguments, None if arguments is not None else str(block.input)))
        text = "".join(block.text for block in response.content if block.type == "text").strip()
        return ToolTurn(text, calls, plain(list(response.content)), self.last_call.model, response.stop_reason,
                        self.last_call.input_tokens, self.last_call.output_tokens)

    def __call__(self, prompt: str) -> str:
        return self.chat(None, prompt)


def _anthropic_blocks(entry: dict) -> List[dict]:
    """An assistant turn from another planner, as Messages API content blocks."""
    blocks: List[dict] = [{"type": "text", "text": entry["text"]}] if entry.get("text") else []
    blocks += [{"type": "tool_use", "id": c["id"], "name": c["name"], "input": c.get("arguments") or {}} for c in entry.get("calls", [])]
    return blocks


def _openai_items(entry: dict) -> List[dict]:
    """An assistant turn from another planner, as Responses API input items."""
    items: List[dict] = []
    if entry.get("text"):
        items.append({"role": "assistant", "content": entry["text"]})
    items += [{"type": "function_call", "call_id": c["id"], "name": c["name"], "arguments": json.dumps(c.get("arguments") or {})}
              for c in entry.get("calls", [])]
    return items


class OpenAILLM:
    """OpenAI models through the official SDK's Responses API.

    The grounding rules go in ``instructions`` and the sources and question in
    ``input``. No sampling parameters are sent, because reasoning models reject
    them; ``reasoning_effort`` is sent only when set, for models that support it.
    A refusal or a content-filter stop raises ``LLMRefusal``; an answer cut off by
    the output limit is returned and marked in ``last_call.stop_reason``.
    """

    def __init__(
        self,
        model: str,
        max_output_tokens: int = 16000,
        timeout: float = 120.0,
        max_retries: int = 2,
        reasoning_effort: Optional[str] = None,
        client: Any = None,
    ):
        if not model:
            raise ValueError("OpenAILLM needs a model ID, e.g. openai:<model-id>")
        if client is None:
            try:
                import openai
            except ImportError as exc:  # pragma: no cover - needs the optional extra
                raise ImportError("OpenAILLM needs: pip install -e .[llm]") from exc
            # Credentials come from the environment (OPENAI_API_KEY); never hard-code a key.
            client = openai.OpenAI(timeout=timeout, max_retries=max_retries)
        self.client = client
        self.model = model
        self.max_output_tokens = max_output_tokens
        self.reasoning_effort = reasoning_effort
        self.last_call: Optional[LLMCall] = None

    @property
    def name(self) -> str:
        return f"openai:{self.model}" + (f"@{self.reasoning_effort}" if self.reasoning_effort else "")

    def chat(self, system: Optional[str], user: str) -> str:
        request = {"model": self.model, "input": user, "max_output_tokens": self.max_output_tokens}
        if system:
            request["instructions"] = system
        if self.reasoning_effort:
            request["reasoning"] = {"effort": self.reasoning_effort}
        response = self.client.responses.create(**request)

        texts, refusals = [], []
        for item in response.output or []:
            if getattr(item, "type", None) != "message":
                continue  # reasoning items and tool calls are not part of the answer
            for part in item.content:
                if part.type == "output_text":
                    texts.append(part.text)
                elif part.type == "refusal":
                    refusals.append(part.refusal)

        details = getattr(response, "incomplete_details", None)
        reason = getattr(details, "reason", None)
        usage = getattr(response, "usage", None)
        self.last_call = LLMCall(
            model=getattr(response, "model", self.model),
            stop_reason=reason or response.status or "completed",
            input_tokens=getattr(usage, "input_tokens", 0),
            output_tokens=getattr(usage, "output_tokens", 0),
        )
        if refusals and not texts:
            raise LLMRefusal(f"model declined: {refusals[0][:200]}")
        if reason == "content_filter":
            raise LLMRefusal("model output stopped by content_filter")
        return "".join(texts).strip()

    def stream_chat(self, system: Optional[str], user: str) -> Iterator[str]:
        """Yield the answer's text deltas as they arrive (Responses API events);
        ``last_call`` is set from the final ``response.completed`` event.

        The stream is closed in ``finally``, also when the consumer stops early, so an
        abandoned request stops generating instead of running (and billing) to the end.
        """
        request = {"model": self.model, "input": user, "max_output_tokens": self.max_output_tokens, "stream": True}
        if system:
            request["instructions"] = system
        if self.reasoning_effort:
            request["reasoning"] = {"effort": self.reasoning_effort}
        stream = self.client.responses.create(**request)
        final, refusal, produced = None, [], False
        try:
            for event in stream:
                kind = getattr(event, "type", "")
                if kind == "response.output_text.delta":
                    produced = True
                    yield event.delta
                elif kind == "response.refusal.delta":
                    refusal.append(event.delta)
                elif kind in ("response.completed", "response.incomplete"):
                    final = event.response
                elif kind in ("response.failed", "error"):
                    raise RuntimeError(f"OpenAI stream failed: {getattr(event, 'message', kind)}")
        finally:
            close = getattr(stream, "close", None)
            if close:
                close()
        details = getattr(final, "incomplete_details", None) if final else None
        reason = getattr(details, "reason", None)
        usage = getattr(final, "usage", None)
        self.last_call = LLMCall(
            model=getattr(final, "model", self.model),
            stop_reason=reason or getattr(final, "status", None) or "completed",
            input_tokens=getattr(usage, "input_tokens", 0),
            output_tokens=getattr(usage, "output_tokens", 0),
        )
        if refusal and not produced:
            raise LLMRefusal(f"model declined: {''.join(refusal)[:200]}")
        if reason == "content_filter":
            raise LLMRefusal("model output stopped by content_filter")

    def tool_turn(self, system: Optional[str], transcript: List[dict], tools: List[dict]) -> ToolTurn:
        """One turn with function calling. The previous output items (reasoning items
        included) are sent back as received, followed by ``function_call_output`` items.
        Arguments are validated by the caller, so ``strict`` mode is not used."""
        items: List[Any] = []
        for entry in transcript:
            role = entry["role"]
            if role == "user":
                items.append({"role": "user", "content": entry["content"]})
            elif role == "assistant":
                items += entry["items"] if entry.get("provider") == "openai" else _openai_items(entry)
            else:
                items.append({"type": "function_call_output", "call_id": entry["call_id"], "output": entry["content"]})
        request = {
            "model": self.model,
            "input": items,
            "tools": [{"type": "function", "name": t["name"], "description": t["description"], "parameters": t["parameters"], "strict": False}
                      for t in tools],
            "max_output_tokens": self.max_output_tokens,
        }
        if system:
            request["instructions"] = system
        if self.reasoning_effort:
            request["reasoning"] = {"effort": self.reasoning_effort}
        response = self.client.responses.create(**request)

        texts, refusals, calls = [], [], []
        for item in response.output or []:
            kind = getattr(item, "type", None)
            if kind == "function_call":
                try:
                    arguments = json.loads(item.arguments or "{}")
                except json.JSONDecodeError:
                    arguments = None
                if isinstance(arguments, dict):
                    calls.append(ToolCall(item.call_id, item.name, arguments))
                else:
                    calls.append(ToolCall(item.call_id, item.name, None, item.arguments))
            elif kind == "message":
                for part in item.content:
                    if part.type == "output_text":
                        texts.append(part.text)
                    elif part.type == "refusal":
                        refusals.append(part.refusal)
        details = getattr(response, "incomplete_details", None)
        reason = getattr(details, "reason", None)
        usage = getattr(response, "usage", None)
        self.last_call = LLMCall(
            model=getattr(response, "model", self.model),
            stop_reason=reason or response.status or "completed",
            input_tokens=getattr(usage, "input_tokens", 0),
            output_tokens=getattr(usage, "output_tokens", 0),
        )
        if refusals and not texts and not calls:
            raise LLMRefusal(f"model declined: {refusals[0][:200]}")
        if reason == "content_filter":
            raise LLMRefusal("model output stopped by content_filter")
        return ToolTurn("".join(texts).strip(), calls, plain(list(response.output or [])), self.last_call.model,
                        self.last_call.stop_reason, self.last_call.input_tokens, self.last_call.output_tokens)

    def __call__(self, prompt: str) -> str:
        return self.chat(None, prompt)


def create_llm(spec: str, effort: Optional[str] = None, **kwargs: Any):
    """Build an LLM from a spec:

        anthropic:<model-id>[@effort]   (or a bare claude-* model ID)  → AnthropicLLM
        openai:<model-id>[@effort]                                      → OpenAILLM

    The ``@effort`` suffix (e.g. ``openai:<model>@low``) sets Anthropic's ``effort``
    or OpenAI's ``reasoning_effort``; lower effort spends fewer reasoning tokens.
    The ``effort`` argument is a default for Anthropic when the spec has none.
    """
    body, _, suffix = spec.partition("@")
    if body.startswith("anthropic:") or body.startswith("claude-"):
        model = body[len("anthropic:") :] if body.startswith("anthropic:") else body
        chosen = suffix or effort
        return AnthropicLLM(model=model, **({"effort": chosen} if chosen else {}), **kwargs)
    if body.startswith("openai:"):
        if suffix:
            kwargs.setdefault("reasoning_effort", suffix)
        return OpenAILLM(model=body[len("openai:") :], **kwargs)
    raise ValueError(f"unknown LLM {spec!r}; expected anthropic:<model-id> or openai:<model-id>")


class CachedLLM:
    """On-disk cache in front of any LLM, keyed by (model name, system, user).

    Identical requests (re-running an evaluation, repeated questions) are served
    from a JSONL file and cost nothing. Errors and refusals are never cached. A
    cached reply sets ``last_call`` with zero tokens, so cost accounting stays honest.
    The cache file may contain document text; keep it out of version control.
    """

    def __init__(self, inner: Any, path: Union[str, Path]):
        self.inner = inner
        self.path = Path(path)
        self.hits = 0
        self.misses = 0
        self.last_call: Optional[LLMCall] = None
        self._entries: Dict[str, dict] = {}
        if self.path.is_file():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    entry = json.loads(line)
                    self._entries[entry["key"]] = entry

    @property
    def name(self) -> str:
        return getattr(self.inner, "name", type(self.inner).__name__)

    def _key(self, system: Optional[str], user: str) -> str:
        payload = json.dumps([self.name, system or "", user], ensure_ascii=False)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def chat(self, system: Optional[str], user: str) -> str:
        key = self._key(system, user)
        entry = self._entries.get(key)
        if entry is not None:
            self.hits += 1
            self.last_call = LLMCall(model=entry["model"], stop_reason="cached", input_tokens=0, output_tokens=0)
            return entry["text"]

        self.misses += 1
        chat = getattr(self.inner, "chat", None)
        text = chat(system, user) if chat else self.inner(user if system is None else f"{system}\n\n{user}")
        call = getattr(self.inner, "last_call", None)
        self.last_call = call
        entry = {
            "key": key,
            "model": call.model if call else self.name,
            "text": text,
            "input_tokens": call.input_tokens if call else 0,
            "output_tokens": call.output_tokens if call else 0,
        }
        self._entries[key] = entry
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
        return text

    def tool_turn(self, system: Optional[str], transcript: List[dict], tools: List[dict]) -> ToolTurn:
        """Cached tool-calling turn, keyed by model, system, the whole transcript and the
        tool specs: any change in the conversation or the tools is a miss. A cached turn
        reports zero tokens."""
        payload = json.dumps([self.name, "tool_turn", system or "", transcript, tools], ensure_ascii=False, sort_keys=True)
        key = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        entry = self._entries.get(key)
        if entry is not None:
            self.hits += 1
            turn = entry["turn"]
            self.last_call = LLMCall(model=turn["model"], stop_reason="cached", input_tokens=0, output_tokens=0)
            return ToolTurn(turn["text"], [ToolCall(**c) for c in turn["calls"]], turn["items"], turn["model"], turn["stop_reason"], 0, 0)
        self.misses += 1
        turn = self.inner.tool_turn(system, transcript, tools)
        self.last_call = getattr(self.inner, "last_call", None)
        entry = {"key": key, "model": turn.model, "turn": asdict(turn)}
        self._entries[key] = entry
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
        return turn

    def __call__(self, prompt: str) -> str:
        return self.chat(None, prompt)
