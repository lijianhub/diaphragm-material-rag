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


def create_llm(spec: str, **kwargs: Any) -> AnthropicLLM:
    """``anthropic:<model-id>`` (or a bare ``claude-*`` model ID) → ``AnthropicLLM``."""
    if spec.startswith("anthropic:"):
        return AnthropicLLM(model=spec[len("anthropic:") :], **kwargs)
    if spec.startswith("claude-"):
        return AnthropicLLM(model=spec, **kwargs)
    raise ValueError(f"unknown LLM {spec!r}; expected anthropic:<model-id>")
