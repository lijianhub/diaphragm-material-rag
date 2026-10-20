"""Language-model backends for generation and query rewriting.

An LLM here is any object that can be called with a prompt string and returns
text. Backends may also offer ``chat(system, user)``, which the RAG pipeline
prefers so the grounding rules travel as a real system prompt.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

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

    def __call__(self, prompt: str) -> str:
        return self.chat(None, prompt)


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
        return f"openai:{self.model}"

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

    def __call__(self, prompt: str) -> str:
        return self.chat(None, prompt)


def create_llm(spec: str, effort: Optional[str] = None, **kwargs: Any):
    """Build an LLM from a spec:

        anthropic:<model-id>   (or a bare claude-* model ID)  → AnthropicLLM
        openai:<model-id>                                       → OpenAILLM

    ``effort`` is the Anthropic effort level; it is not forwarded to OpenAI models,
    where the matching setting (``reasoning_effort``) only exists on some models.
    """
    if spec.startswith("anthropic:") or spec.startswith("claude-"):
        model = spec[len("anthropic:") :] if spec.startswith("anthropic:") else spec
        return AnthropicLLM(model=model, **({"effort": effort} if effort else {}), **kwargs)
    if spec.startswith("openai:"):
        return OpenAILLM(model=spec[len("openai:") :], **kwargs)
    raise ValueError(f"unknown LLM {spec!r}; expected anthropic:<model-id> or openai:<model-id>")
