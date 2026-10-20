import os

import pytest

from generation import RAGPipeline, build_prompt
from generation.llm import DEFAULT_MODEL, AnthropicLLM, LLMRefusal, create_llm
from ingestion.loader import Document
from retrieval.index import SearchIndex
from tests.fakes import FakeAnthropicClient

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
        create_llm("openai:gpt")


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
