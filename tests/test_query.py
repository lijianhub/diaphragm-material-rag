import pytest

from ingestion.loader import Document
from query import LLMQueryRewriter, decompose, expansion_terms, interleave, retrieve
from reranking import CrossEncoderReranker
from retrieval.bm25 import BM25
from retrieval.index import SearchIndex
from tests.fakes import FakeCrossEncoder, FakeLLM


# ---- decomposition ---------------------------------------------------------


@pytest.mark.parametrize(
    "question, parts",
    [
        (
            "What does the supplier send with each coil, and what do inspectors do with it?",
            ["What does the supplier send with each coil?", "what do inspectors do with it?"],
        ),
        ("Why is excessive thinning a problem and how is it checked?", ["Why is excessive thinning a problem?", "how is it checked?"]),
    ],
)
def test_decompose_splits_two_questions_joined_by_and(question, parts):
    assert decompose(question) == [question] + parts


@pytest.mark.parametrize(
    "question",
    [
        "Which material properties do strength and ductility affect?",  # "and" joins nouns, not questions
        "How can wrinkling in the flange be prevented?",
        "What do certificates list and inspectors compare?",  # no question word after "and"
    ],
)
def test_decompose_leaves_single_questions_alone(question):
    assert decompose(question) == [question]


# ---- pseudo-relevance feedback ----------------------------------------------


def test_expansion_terms_prefer_words_common_in_feedback_but_rare_in_corpus():
    corpus = [
        "Cool the parts in the furnace atmosphere before exposing them to air.",
        "Cooling in the furnace atmosphere prevents oxidation before air exposure.",
        "The parts are formed in a press.",
        "The parts are inspected for flatness.",
    ]
    bm25 = BM25(corpus, stem_tokens=True)

    terms = expansion_terms("Can parts go into the air after aging?", corpus[:2], bm25.idf, n_terms=3)

    assert "furnace" in terms and "atmosphere" in terms
    assert "parts" not in terms and "air" not in terms  # already in the question
    assert all(" " not in t and t.isalpha() for t in terms)  # surface words, not stems


def test_expansion_terms_are_empty_without_feedback():
    assert expansion_terms("anything", [], lambda t: 1.0, n_terms=5) == []


# ---- LLM rewriting ----------------------------------------------------------


def test_llm_rewriter_parses_numbered_variants_and_keeps_the_original_first():
    llm = FakeLLM("1. How should formed parts be cooled after aging?\n2) Cooling procedure after heat treatment\n- Exposure to air after aging\n")

    queries = LLMQueryRewriter(llm, n_variants=2)("Can parts go into the air after aging?")

    assert queries == [
        "Can parts go into the air after aging?",
        "How should formed parts be cooled after aging?",
        "Cooling procedure after heat treatment",
    ]
    assert "Can parts go into the air after aging?" in llm.prompts[0]


def test_llm_rewriter_hyde_mode_searches_with_a_hypothetical_answer():
    llm = FakeLLM("Parts are cooled in the furnace atmosphere before they are exposed to air.")

    queries = LLMQueryRewriter(llm, mode="hyde")("Can parts go into the air after aging?")

    assert queries == ["Can parts go into the air after aging?", "Parts are cooled in the furnace atmosphere before they are exposed to air."]
    assert "passage" in llm.prompts[0].lower()


def test_llm_rewriter_falls_back_to_the_original_question_when_the_llm_fails():
    queries = LLMQueryRewriter(FakeLLM(error=TimeoutError("slow")))("What is galling?")

    assert queries == ["What is galling?"]


def test_llm_rewriter_rejects_unknown_mode():
    with pytest.raises(ValueError):
        LLMQueryRewriter(FakeLLM(), mode="summarize")


# ---- retrieval with transformed queries -----------------------------------


DOCS = [
    Document(source="supplier.txt", content="Each coil ships with a material certificate from the supplier."),
    Document(source="inspection.txt", content="Inspectors compare certificate values with the purchase specification."),
    Document(source="forming.txt", content="Springback is compensated by over-forming the die profile."),
    Document(source="lots.txt", content="Every production lot is assigned a lot number."),
]


def test_retrieve_without_transforms_matches_plain_search():
    index = SearchIndex()
    index.sync(DOCS)

    question = "What does the supplier send with each coil?"
    assert [r.source for r in retrieve(index, question, top_k=2)] == [r.source for r in index.search(question, top_k=2)]


def test_decomposed_question_retrieves_evidence_for_both_parts():
    # Two extra documents match the first half of the question strongly, so plain search
    # fills its top 2 with supplier material and never reaches the inspection document.
    docs = DOCS + [
        Document(source="supplier-faq.txt", content="What does the supplier send? The supplier sends each coil with packing notes."),
        Document(source="supplier-terms.txt", content="Each coil the supplier sends is labelled; the supplier sends coils weekly."),
    ]
    index = SearchIndex()
    index.sync(docs)
    question = "What does the supplier send with each coil, and what do inspectors compare it with?"

    plain = {r.source for r in index.search(question, top_k=2)}
    assert "inspection.txt" not in plain  # premise: one search misses the second part

    decomposed = {r.source for r in retrieve(index, question, top_k=2, transforms=[decompose], fusion="interleave")}
    assert "inspection.txt" in decomposed


def test_reranker_judges_against_the_original_question_not_the_variants():
    model = FakeCrossEncoder()
    index = SearchIndex(reranker=CrossEncoderReranker("fake", model=model))
    index.sync(DOCS)
    question = "What does the supplier send with each coil, and what do inspectors compare it with?"

    retrieve(index, question, top_k=2, transforms=[decompose], fusion="interleave")

    assert len(model.calls) == 1
    assert {query for query, _ in model.calls[0]} == {question}


def test_prf_expansion_runs_a_second_search_with_added_terms(monkeypatch):
    index = SearchIndex()
    index.sync(DOCS)
    searched = []
    original_search = index.search

    def recording_search(query, top_k=3, alpha=None, rerank=True, filter=None):
        searched.append((query, top_k, rerank))
        return original_search(query, top_k=top_k, alpha=alpha, rerank=rerank, filter=filter)

    monkeypatch.setattr(index, "search", recording_search)
    results = retrieve(index, "How is springback handled?", top_k=1, prf_terms=3, prf_docs=2)

    feedback, final = searched
    assert feedback == ("How is springback handled?", 2, False)  # first pass: feedback documents
    assert final[0].startswith("How is springback handled? ") and len(final[0].split()) > 4  # expanded query
    assert results[0].source == "forming.txt"


def test_interleave_gives_each_sub_query_its_top_result_where_rrf_rewards_consensus():
    from retrieval.fusion import reciprocal_rank_fusion

    # "x" is mediocre everywhere but appears in all three rankings; a, b, c each top one ranking.
    rankings = [["a", "p", "q", "x"], ["b", "r", "s", "x"], ["c", "t", "u", "x"]]

    fused = reciprocal_rank_fusion(rankings)
    assert max(fused, key=fused.get) == "x"  # consensus wins under RRF

    assert interleave(rankings)[:3] == ["a", "b", "c"]  # coverage: every sub-query's best first
    assert interleave(rankings) == ["a", "b", "c", "p", "r", "t", "q", "s", "u", "x"]  # no duplicates


def test_retrieve_rejects_unknown_fusion():
    index = SearchIndex()
    index.sync(DOCS)

    with pytest.raises(ValueError):
        retrieve(index, "a, and what b?", transforms=[decompose], fusion="max")


def test_retrieve_passes_the_filter_to_every_search(monkeypatch):
    index = SearchIndex()
    index.sync([Document(source="a.md", content="Springback is compensated by over-forming.", metadata={"doc_type": "md"}),
                Document(source="b.txt", content="Springback compensation uses over-formed dies.", metadata={"doc_type": "txt"})])
    filters = []
    original_search = index.search

    def recording_search(query, top_k=3, alpha=None, rerank=True, filter=None):
        filters.append(filter)
        return original_search(query, top_k=top_k, alpha=alpha, rerank=rerank, filter=filter)

    monkeypatch.setattr(index, "search", recording_search)

    results = retrieve(index, "How is springback compensated, and what dies are used?", transforms=[decompose], prf_terms=2,
                       filter={"doc_type": "md"})

    assert len(filters) > 2 and all(f == {"doc_type": "md"} for f in filters)  # feedback, sub-queries, every one
    assert {r.source for r in results} == {"a.md"}
