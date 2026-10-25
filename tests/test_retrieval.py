import pytest

from embedding.embedder import SimpleEmbedder
from ingestion.loader import Document
from ingestion.parser import Section
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


def test_search_index_bm25_with_stemming_matches_inflected_query():
    documents = [
        Document(source="testing.txt", content="Pressure cycling tests verify fatigue performance of each lot."),
        Document(source="alloy.txt", content="Fatigue performance depends on the alloy and its surface condition."),
    ]
    stemmed = SearchIndex(lexical="bm25", stem_tokens=True, alpha=0.0)
    stemmed.add_documents(documents)

    assert stemmed.search("How is fatigue performance verified?", top_k=1)[0].source == "testing.txt"


def test_search_index_rejects_unknown_lexical_scorer():
    with pytest.raises(ValueError):
        SearchIndex(lexical="tfidf")


def test_search_index_rrf_prefers_chunks_both_retrievers_rank_well():
    documents = [
        Document(source="springs.txt", content="Shot peening introduces compressive residual stress in springs."),
        Document(source="forming.txt", content="Residual stress after forming causes springback in the part."),
        Document(source="other.txt", content="Material certificates list hardness and tensile strength."),
    ]
    index = SearchIndex(fusion="rrf")
    index.add_documents(documents)

    results = index.search("How does shot peening change residual stress in springs?", top_k=3)

    assert results[0].source == "springs.txt"
    assert [r.score for r in results] == sorted((r.score for r in results), reverse=True)


def test_search_index_rejects_unknown_fusion():
    with pytest.raises(ValueError):
        SearchIndex(fusion="max")


# ---- metadata filter -------------------------------------------------------------------

FILTER_DOCS = [
    Document(source="notes.txt", content="Springback is compensated by over-forming the die.", metadata={"doc_type": "txt"}),
    Document(source="faq.md", content="Springback compensation: over-form the profile slightly.", metadata={"doc_type": "md"}),
    Document(source="guide.md", content="Springback grows with strength; dies are over-formed.", metadata={"doc_type": "md"}),
    # A page number comes from a section (as for a PDF page), not from document metadata.
    Document(source="sheet.pdf", content="HAVAR holds pressure in pressure windows.", metadata={"doc_type": "pdf"},
             sections=[Section(text="HAVAR holds pressure in pressure windows.", metadata={"page": 2})]),
]


def _filter_index(**kwargs):
    index = SearchIndex(**kwargs)
    index.sync(FILTER_DOCS)
    return index


@pytest.mark.parametrize("fusion", ["weighted", "rrf"])
def test_filter_acts_inside_retrieval_so_top_k_is_filled_with_matching_chunks(fusion):
    index = _filter_index(fusion=fusion)

    unfiltered = index.search("springback over-forming", top_k=2)
    filtered = index.search("springback over-forming", top_k=2, filter={"doc_type": "md"})

    assert unfiltered[0].source == "notes.txt"  # the best match overall is excluded below
    assert [r.source for r in filtered] == ["faq.md", "guide.md"] or [r.source for r in filtered] == ["guide.md", "faq.md"]
    assert all(r.metadata["doc_type"] == "md" for r in filtered)


def test_filter_values_lists_record_fields_and_strings():
    index = _filter_index()

    assert {r.source for r in index.search("springback", top_k=5, filter={"source": ["notes.txt", "sheet.pdf"]})} <= {"notes.txt", "sheet.pdf"}
    assert [r.source for r in index.search("HAVAR pressure", top_k=5, filter={"page": "2"})] == ["sheet.pdf"]
    assert [r.source for r in index.search("HAVAR pressure", top_k=5, filter={"page": 2})] == ["sheet.pdf"]  # compared as strings
    assert index.search("springback", top_k=5, filter={"doc_type": "docx"}) == []
    assert index.search("springback", top_k=5, filter={"owner": "quality"}) == []  # unknown field: excluded, not ignored
    assert index.search("springback", top_k=5, filter={}) == index.search("springback", top_k=5)  # empty filter: everything


def test_filter_applies_before_reranking():
    from tests.fakes import FakeCrossEncoder
    from reranking import CrossEncoderReranker

    # The fake reranker loves the excluded document; it must never even see it.
    model = FakeCrossEncoder(preferred="over-forming the die")
    index = _filter_index(reranker=CrossEncoderReranker(model=model))

    results = index.search("springback", top_k=2, filter={"doc_type": "md"})

    assert {r.source for r in results} <= {"faq.md", "guide.md"}
    assert all("over-forming the die" not in passage for batch in model.calls for _, passage in batch)
