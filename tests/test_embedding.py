from embedding.embedder import SimpleEmbedder


def test_simple_embedder_builds_deterministic_vector():
    embedder = SimpleEmbedder()
    texts = ["alpha beta", "beta gamma"]

    vectors = embedder.encode_many(texts)

    assert len(vectors) == 2
    assert len(vectors[0]) > 0
    assert vectors[0] == embedder.encode("alpha beta")
    assert vectors[0] != vectors[1]
