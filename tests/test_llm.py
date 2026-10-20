import os

import pytest

from generation import RAGPipeline, build_prompt
from generation.llm import DEFAULT_MODEL, AnthropicLLM, CachedLLM, LLMRefusal, OpenAILLM, create_llm
from ingestion.loader import Document
from retrieval.index import SearchIndex
from tests.fakes import FakeAnthropicClient, FakeOpenAIClient

DOCS = [
    Document(source="forming.txt", content="Springback is compensated by over-forming the profile by a small angle."),
    Document(source="lots.txt", content="Lot L-302 was rejected for tearing at the rim."),
]


def _index():
    index = SearchIndex()
    index.sync(DOCS)
    return index


def test_request_uses_system_prompt_effort_and_refusal_fallback_without_sampling_params():
    client = FakeAnthropicClient()
    llm = AnthropicLLM(client=client, effort="medium")

    llm.chat("You are grounded.", "Sources: ... Question: q")

    call = client.calls[0]
    assert call["path"] == "beta"
    assert call["model"] == DEFAULT_MODEL == "claude-opus-5-5"
    assert call["system"] == "You are grounded."
    assert call["messages"] == [{"role": "user", "content": "Sources: ... Question: q"}]
    assert call["output_config"] == {"effort": "medium"}
    assert call["fallbacks"] == "default" and call["betas"] == ["server-side-fallback-2026-07-01"]
    assert call["max_tokens"] >= 4096
    assert "temperature" not in call and "thinking" not in call  # rejected on this model


def test_only_text_blocks_are_returned_and_usage_is_recorded():
    llm = AnthropicLLM(client=FakeAnthropicClient(text="Over-forming [1]."))

    text = llm.chat("s", "u")

    assert text == "Over-forming [1]."  # the thinking block is ignored
    assert (llm.last_call.input_tokens, llm.last_call.output_tokens) == (1200, 45)
    assert llm.last_call.model == "claude-opus-5-5" and llm.last_call.stop_reason == "end_turn"


def test_refusal_raises_with_its_category():
    llm = AnthropicLLM(client=FakeAnthropicClient(text="", stop_reason="refusal", refusal_category="cyber"))

    with pytest.raises(LLMRefusal, match="cyber"):
        llm.chat("s", "u")


def test_fallbacks_can_be_disabled_for_platforms_without_them():
    client = FakeAnthropicClient()

    AnthropicLLM(client=client, fallbacks=False).chat("s", "u")

    assert client.calls[0]["path"] == "messages"
    assert "fallbacks" not in client.calls[0] and "betas" not in client.calls[0]


def test_single_string_calls_work_for_query_rewriting():
    client = FakeAnthropicClient(text="rephrased question")

    assert AnthropicLLM(client=client)("Rewrite: q") == "rephrased question"
    assert client.calls[0]["messages"] == [{"role": "user", "content": "Rewrite: q"}]
    assert "system" not in client.calls[0]


def test_create_llm_from_spec():
    llm = create_llm("anthropic:claude-sonnet-5-5", client=FakeAnthropicClient())

    assert llm.model == "claude-sonnet-5-5" and llm.name == "anthropic:claude-sonnet-5-5"
    with pytest.raises(ValueError):
        create_llm("mistral:large")


# ---- versioning and traceability in the pipeline ---------------------------


def test_pipeline_records_served_model_prompt_version_and_usage():
    client = FakeAnthropicClient(text="Dies are over-formed [1].", model="claude-opus-5-5")

    answer = RAGPipeline(_index(), complete=AnthropicLLM(client=client), top_k=2).answer("How is springback compensated?")

    assert answer.used_llm and answer.model == "claude-opus-5-5"
    assert answer.prompt_version == build_prompt("q", []).version
    assert answer.usage == {"input_tokens": 1200, "output_tokens": 45}
    sent = client.calls[0]
    assert sent["system"] == answer.prompt.system  # system and sources sent separately
    assert sent["messages"][0]["content"].startswith("Sources:")


def test_extractive_answers_are_versioned_too():
    answer = RAGPipeline(_index(), top_k=2).answer("Which lot was rejected for tearing?")

    assert answer.model == "extractive" and answer.prompt_version


def test_prompt_version_changes_when_the_system_prompt_text_changes(monkeypatch):
    import generation.prompt as prompt_module

    before = build_prompt("q", []).version
    monkeypatch.setattr(prompt_module, "SYSTEM_PROMPT", prompt_module.SYSTEM_PROMPT + "\n- Be brief.")
    after = build_prompt("q", []).version

    assert before != after
    assert before.split("+")[0] == after.split("+")[0]  # same label, different fingerprint


def test_refusal_falls_back_to_an_extractive_answer_with_the_reason():
    client = FakeAnthropicClient(text="", stop_reason="refusal", refusal_category="bio")

    answer = RAGPipeline(_index(), complete=AnthropicLLM(client=client), top_k=2).answer("Which lot was rejected for tearing?")

    assert not answer.used_llm and "LLMRefusal" in answer.fallback_reason
    assert "L-302" in answer.text


@pytest.mark.skipif(not os.environ.get("RUN_LLM_TESTS"), reason="calls the Anthropic API; set RUN_LLM_TESTS=1 and ANTHROPIC_API_KEY")
def test_real_model_answers_with_a_valid_citation():
    answer = RAGPipeline(_index(), complete=AnthropicLLM(), top_k=2).answer("How is springback compensated?")

    assert answer.used_llm and answer.citations and not answer.invalid_citations


# ---- OpenAI backend ----------------------------------------------------------



def test_openai_request_uses_instructions_input_and_output_limit():
    client = FakeOpenAIClient()

    OpenAILLM("model-under-test", client=client).chat("You are grounded.", "Sources: ... Question: q")

    call = client.calls[0]
    assert call["model"] == "model-under-test"
    assert call["instructions"] == "You are grounded."
    assert call["input"] == "Sources: ... Question: q"
    assert call["max_output_tokens"] >= 4096
    assert "temperature" not in call and "reasoning" not in call  # not every model accepts them


def test_openai_returns_message_text_and_records_usage():
    llm = OpenAILLM("model-under-test", client=FakeOpenAIClient(text="Over-forming [1]."))

    assert llm.chat("s", "u") == "Over-forming [1]."  # the reasoning item is ignored
    assert (llm.last_call.input_tokens, llm.last_call.output_tokens, llm.last_call.model) == (900, 60, "model-under-test")


def test_openai_refusal_and_content_filter_raise_llm_refusal():
    with pytest.raises(LLMRefusal):
        OpenAILLM("m", client=FakeOpenAIClient(refusal="I can't help with that.")).chat("s", "u")
    with pytest.raises(LLMRefusal, match="content_filter"):
        OpenAILLM("m", client=FakeOpenAIClient(text="", status="incomplete", incomplete_reason="content_filter")).chat("s", "u")


def test_openai_truncated_answer_is_kept_but_marked():
    llm = OpenAILLM("m", client=FakeOpenAIClient(text="Partial answer [1]", status="incomplete", incomplete_reason="max_output_tokens"))

    assert llm.chat("s", "u") == "Partial answer [1]"
    assert llm.last_call.stop_reason == "max_output_tokens"


def test_openai_reasoning_effort_is_sent_only_when_asked():
    client = FakeOpenAIClient()

    OpenAILLM("m", reasoning_effort="low", client=client).chat("s", "u")

    assert client.calls[0]["reasoning"] == {"effort": "low"}


def test_create_llm_builds_openai_and_requires_a_model_id():
    llm = create_llm("openai:model-under-test", client=FakeOpenAIClient())

    assert isinstance(llm, OpenAILLM) and llm.name == "openai:model-under-test"
    with pytest.raises(ValueError, match="model"):
        create_llm("openai:", client=FakeOpenAIClient())


def test_pipeline_works_unchanged_with_the_openai_backend():
    client = FakeOpenAIClient(text="Dies are over-formed [1].")

    answer = RAGPipeline(_index(), complete=OpenAILLM("model-under-test", client=client), top_k=2).answer("How is springback compensated?")

    assert answer.used_llm and answer.model == "model-under-test" and answer.citations[0].source == "forming.txt"
    assert client.calls[0]["instructions"] == answer.prompt.system


# ---- cost control: effort in the spec, on-disk cache ------------------------



def test_spec_suffix_sets_reasoning_effort_for_openai_and_effort_for_anthropic():
    openai_llm = create_llm("openai:model-under-test@low", client=FakeOpenAIClient())
    anthropic_llm = create_llm("anthropic:claude-opus-5-5@low", client=FakeAnthropicClient())

    assert openai_llm.model == "model-under-test" and openai_llm.reasoning_effort == "low"
    assert anthropic_llm.model == "claude-opus-5-5" and anthropic_llm.effort == "low"
    assert openai_llm.name == "openai:model-under-test@low"  # the effort is part of the cache key and the record


class CountingLLM:
    name = "counting"

    def __init__(self, error=None):
        self.calls = 0
        self.error = error
        self.last_call = None

    def chat(self, system, user):
        from generation.llm import LLMCall

        self.calls += 1
        if self.error:
            raise self.error
        self.last_call = LLMCall(model="m", stop_reason="end", input_tokens=100, output_tokens=10)
        return f"reply to {user}"


def test_cache_serves_repeated_requests_without_calling_the_model(tmp_path):
    inner = CountingLLM()
    llm = CachedLLM(inner, tmp_path / "llm.jsonl")

    first = llm.chat("s", "q1")
    second = llm.chat("s", "q1")

    assert first == second == "reply to q1"
    assert inner.calls == 1
    assert llm.last_call.input_tokens == 0 and llm.hits == 1 and llm.misses == 1  # a cached reply costs nothing


def test_cache_key_covers_model_system_and_user(tmp_path):
    inner = CountingLLM()
    llm = CachedLLM(inner, tmp_path / "llm.jsonl")

    llm.chat("s", "q1")
    llm.chat("s", "q2")
    llm.chat("other system", "q1")

    assert inner.calls == 3


def test_cache_persists_across_runs_and_never_stores_errors(tmp_path):
    path = tmp_path / "llm.jsonl"
    CachedLLM(CountingLLM(), path).chat("s", "q1")
    failing = CountingLLM(error=TimeoutError("slow"))

    again = CachedLLM(CountingLLM(), path)
    assert again.chat("s", "q1") == "reply to q1" and again.inner.calls == 0

    broken = CachedLLM(failing, path)
    with pytest.raises(TimeoutError):
        broken.chat("s", "new question")
    assert "new question" not in path.read_text(encoding="utf-8")


def test_cached_llm_works_as_a_plain_function_for_rewriting(tmp_path):
    llm = CachedLLM(CountingLLM(), tmp_path / "llm.jsonl")

    assert llm("rewrite this") == "reply to rewrite this"
    assert llm.name == "counting"
