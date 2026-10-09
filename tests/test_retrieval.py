from embedding.embedder import SimpleEmbedder
from ingestion.loader import Document
from retrieval.hybrid import hybrid_search
from retrieval.index import SearchIndex


def test_hybrid_search_returns_relevant_chunk():
    texts = [
        "The policy covers employee vacation days.",
        "The engineering manual explains diaphragm forming.",
        "The project uses Python and RAG evaluation.",
    ]
    embedder = SimpleEmbedder()
    vectors = embedder.encode_many(texts)

    results = hybrid_search("diaphragm forming", embedder.encode("diaphragm forming"), texts, vectors, top_k=2)

    assert results
    assert results[0][0] == "The engineering manual explains diaphragm forming."


def test_hybrid_scores_depend_on_the_query():
    texts = ["Springback is elastic recovery.", "The certificate lists hardness."]
    embedder = SimpleEmbedder()
    vectors = embedder.encode_many(texts)

    springback = dict(hybrid_search("springback", embedder.encode("springback"), texts, vectors, top_k=2))
    certificate = dict(hybrid_search("certificate", embedder.encode("certificate"), texts, vectors, top_k=2))

    assert springback[texts[0]] > springback[texts[1]]
    assert certificate[texts[1]] > certificate[texts[0]]


def test_lexical_match_uses_whole_words_not_substrings():
    texts = ["Forming is done in a press.", "Information about lubrication."]
    embedder = SimpleEmbedder()
    vectors = embedder.encode_many(texts)

    results = hybrid_search("form", embedder.encode("form"), texts, vectors, top_k=2, alpha=0.0)

    assert all(score == 0.0 for _, score in results)


def test_search_index_returns_chunks_with_their_source():
    index = SearchIndex()
    index.add_documents(
        [
            Document(source="defects.txt", content="Springback is the elastic recovery of the part after forming."),
            Document(source="qa.txt", content="The material certificate lists hardness and tensile strength."),
        ]
    )

    results = index.search("What does the certificate list?", top_k=1)

    assert len(index) == 2
    assert results[0].source == "qa.txt"
    assert results[0].score > 0


def test_search_index_on_empty_index_returns_nothing():
    assert SearchIndex().search("anything") == []
