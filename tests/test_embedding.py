import json
import math
import os

import pytest

from embedding import BedrockEmbedder, SentenceTransformerEmbedder, create_embedder
from embedding.embedder import SimpleEmbedder
from tests.fakes import FakeBedrockClient, FakeSentenceModel


def test_simple_embedder_builds_deterministic_vector():
    embedder = SimpleEmbedder()
    texts = ["alpha beta", "beta gamma"]

    vectors = embedder.encode_many(texts)

    assert len(vectors) == 2
    assert len(vectors[0]) > 0
    assert vectors[0] == embedder.encode("alpha beta")
    assert vectors[0] != vectors[1]



def test_sentence_transformer_embedder_batches_and_normalizes():
    model = FakeSentenceModel()
    embedder = SentenceTransformerEmbedder("sentence-transformers/all-MiniLM-L6-v2", model=model, batch_size=8)

    vectors = embedder.encode_many(["ab", "abcd"])

    assert embedder.dim == 3
    assert embedder.name == "st:sentence-transformers/all-MiniLM-L6-v2"
    assert model.calls == [{"texts": ["ab", "abcd"], "batch_size": 8, "normalize": True}]
    assert len(vectors) == 2 and len(vectors[0]) == 3


def test_query_and_passage_prefixes_are_applied_for_asymmetric_models():
    model = FakeSentenceModel()
    embedder = SentenceTransformerEmbedder("intfloat/e5-small-v2", model=model, query_prefix="query: ", passage_prefix="passage: ")

    embedder.encode_many(["oxide film"])
    embedder.encode_query("why no rust")

    assert model.calls[0]["texts"] == ["passage: oxide film"]
    assert model.calls[1]["texts"] == ["query: why no rust"]


def test_bedrock_embedder_sends_titan_v2_request():
    client = FakeBedrockClient(dim=256)
    embedder = BedrockEmbedder("amazon.titan-embed-text-v2:0", dimensions=256, client=client)

    vectors = embedder.encode_many(["first chunk", "second chunk"])

    assert vectors == [[0.5] * 256, [0.5] * 256]
    assert embedder.name == "bedrock:amazon.titan-embed-text-v2:0@256"
    assert [r["body"] for r in client.requests] == [
        {"inputText": "first chunk", "dimensions": 256, "normalize": True},
        {"inputText": "second chunk", "dimensions": 256, "normalize": True},
    ]
    assert {r["modelId"] for r in client.requests} == {"amazon.titan-embed-text-v2:0"}


def test_bedrock_embedder_retries_throttling_then_succeeds():
    client = FakeBedrockClient(throttle_times=2)
    embedder = BedrockEmbedder("amazon.titan-embed-text-v2:0", dimensions=256, client=client, backoff_seconds=0)

    assert embedder.encode("text") == [0.5] * 256
    assert len(client.requests) == 3


def test_bedrock_embedder_gives_up_after_max_retries():
    client = FakeBedrockClient(throttle_times=10)
    embedder = BedrockEmbedder("amazon.titan-embed-text-v2:0", dimensions=256, client=client, max_retries=2, backoff_seconds=0)

    with pytest.raises(Exception, match="Rate exceeded"):
        embedder.encode("text")
    assert len(client.requests) == 3


def test_bedrock_embedder_rejects_dimensions_titan_v2_does_not_support():
    with pytest.raises(ValueError):
        BedrockEmbedder("amazon.titan-embed-text-v2:0", dimensions=300, client=FakeBedrockClient())


@pytest.mark.parametrize(
    "name, expected_type",
    [
        ("simple-hash-256", "SimpleEmbedder"),
        ("simple-hash", "SimpleEmbedder"),
    ],
)
def test_create_embedder_rebuilds_offline_embedders_from_their_name(name, expected_type):
    embedder = create_embedder(name)

    assert type(embedder).__name__ == expected_type
    assert create_embedder(embedder.name).name == embedder.name


def test_create_embedder_passes_injected_backends_through():
    st = create_embedder("st:sentence-transformers/all-MiniLM-L6-v2", model=FakeSentenceModel())
    bedrock = create_embedder("bedrock:amazon.titan-embed-text-v2:0@512", client=FakeBedrockClient(dim=512))

    assert st.name == "st:sentence-transformers/all-MiniLM-L6-v2"
    assert bedrock.name == "bedrock:amazon.titan-embed-text-v2:0@512" and bedrock.dim == 512


def test_create_embedder_rejects_unknown_names():
    with pytest.raises(ValueError):
        create_embedder("word2vec:google-news")


@pytest.mark.skipif(not os.environ.get("RUN_MODEL_TESTS"), reason="downloads a real model; set RUN_MODEL_TESTS=1")
def test_real_model_puts_paraphrases_closer_than_unrelated_text():
    embedder = create_embedder("st:sentence-transformers/all-MiniLM-L6-v2")
    rust, oxide, lot = embedder.encode_many(
        ["Why don't these alloys rust in the sea?", "A chromium-rich oxide film protects them in seawater.", "A lot number links parts to the coil."]
    )

    def cos(a, b):
        return sum(x * y for x, y in zip(a, b)) / math.sqrt(sum(x * x for x in a) * sum(y * y for y in b))

    assert cos(rust, oxide) > cos(rust, lot)
