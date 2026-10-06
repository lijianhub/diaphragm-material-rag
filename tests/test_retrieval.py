from embedding.embedder import SimpleEmbedder
from retrieval.hybrid import hybrid_search


def test_hybrid_search_returns_relevant_chunk():
    texts = [
        "The policy covers employee vacation days.",
        "The engineering manual explains diaphragm forming.",
        "The project uses Python and RAG evaluation.",
    ]
    embedder = SimpleEmbedder()
    vectors = embedder.encode_many(texts)

    results = hybrid_search("diaphragm forming", texts, vectors, top_k=2)

    assert results
    assert results[0][0] == "The engineering manual explains diaphragm forming."
