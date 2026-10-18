import pytest

from ingestion.loader import Document
from reranking import BedrockReranker, CrossEncoderReranker, create_reranker
from retrieval.index import SearchIndex
from tests.fakes import FakeBedrockAgentClient, FakeCrossEncoder


def test_cross_encoder_scores_each_query_passage_pair():
    model = FakeCrossEncoder()
    reranker = CrossEncoderReranker("cross-encoder/ms-marco-MiniLM-L-6-v2", model=model)

    scores = reranker.score("oxide film", ["an oxide film forms", "lot numbers", "film"])

    assert scores == [2.0, 0.0, 1.0]
    assert model.calls == [[("oxide film", "an oxide film forms"), ("oxide film", "lot numbers"), ("oxide film", "film")]]
    assert reranker.name == "ce:cross-encoder/ms-marco-MiniLM-L-6-v2"


def test_bedrock_reranker_sends_inline_documents_and_restores_input_order():
    client = FakeBedrockAgentClient(scores=[0.1, 0.9, 0.5])
    reranker = BedrockReranker("amazon.rerank-v1:0", region="us-west-2", client=client)

    scores = reranker.score("why no rust", ["a", "b", "c"])

    assert scores == [0.1, 0.9, 0.5]  # aligned with the input, although the API returned b, c, a
    request = client.requests[0]
    assert request["queries"] == [{"type": "TEXT", "textQuery": {"text": "why no rust"}}]
    assert request["sources"][1] == {"type": "INLINE", "inlineDocumentSource": {"type": "TEXT", "textDocument": {"text": "b"}}}
    config = request["config"]["bedrockRerankingConfiguration"]
    assert config["numberOfResults"] == 3
    assert config["modelConfiguration"]["modelArn"] == "arn:aws:bedrock:us-west-2::foundation-model/amazon.rerank-v1:0"
    assert reranker.name == "bedrock-rerank:amazon.rerank-v1:0"


def test_create_reranker_builds_from_name_and_rejects_unknown():
    assert create_reranker("ce:cross-encoder/ms-marco-MiniLM-L-6-v2", model=FakeCrossEncoder()).name == "ce:cross-encoder/ms-marco-MiniLM-L-6-v2"
    assert create_reranker("bedrock-rerank:amazon.rerank-v1:0", region="us-west-2", client=FakeBedrockAgentClient([])).name == "bedrock-rerank:amazon.rerank-v1:0"
    with pytest.raises(ValueError):
        create_reranker("colbert:v2")


DOCS = [
    # The distractor shares most of the question's words; the answer shares only "aging".
    Document(source="heat.txt", content="Parts must be aged after forming, and aging parts before forming is avoided."),
    Document(source="clean.txt", content="An alkaline wash removes lubricant, which would otherwise leave carbon deposits during aging."),
    Document(source="lots.txt", content="Every lot has a lot number."),
]


def test_reranker_reorders_first_stage_candidates():
    index = SearchIndex(reranker=CrossEncoderReranker("fake", model=FakeCrossEncoder(preferred="carbon deposits")))
    index.sync(DOCS)
    first_stage = SearchIndex()
    first_stage.sync(DOCS)

    question = "Why must parts be cleaned before aging?"
    assert first_stage.search(question, top_k=1)[0].source == "heat.txt"  # premise: first stage gets it wrong

    top = index.search(question, top_k=1)[0]
    assert top.source == "clean.txt"
    assert "retrieval_score" in top.metadata  # the first-stage score is kept for debugging


def test_reranker_only_scores_rerank_depth_candidates():
    model = FakeCrossEncoder()
    index = SearchIndex(reranker=CrossEncoderReranker("fake", model=model), rerank_depth=2)
    index.sync(DOCS)

    results = index.search("lot number", top_k=3)

    assert len(model.calls) == 1 and len(model.calls[0]) == 2
    assert len(results) == 2  # never more than were reranked


def _overlaps(a, b):
    return a.source == b.source and a.metadata.get("section") == b.metadata.get("section") and a.start < b.end and b.start < a.end


def test_dedupe_replaces_overlap_fragments_with_the_full_chunk():
    from chunking import RecursiveChunker

    text = " ".join(f"Sentence {i} explains springback compensation by over-forming the die profile." for i in range(6))
    plain = SearchIndex(chunker=RecursiveChunker(chunk_size=120, overlap=60))
    deduped = SearchIndex(chunker=RecursiveChunker(chunk_size=120, overlap=60), dedupe_overlaps=True)
    for index in (plain, deduped):
        index.sync([Document(source="s.txt", content=text)])

    before = plain.search("springback over-forming", top_k=4)
    after = deduped.search("springback over-forming", top_k=4)

    # Premise: without dedupe the top hits are overlap fragments, i.e. strict sub-spans of other chunks.
    spans = [(r.start, r.end) for r in plain.store.records()]
    assert any(s < before[0].start and before[0].end <= e for s, e in spans)
    # With dedupe each slot holds the longer chunk covering the fragment, and no two results overlap.
    assert after[0].start <= before[0].start and after[0].end >= before[0].end
    assert after[0].end - after[0].start > before[0].end - before[0].start
    assert not any(_overlaps(a, b) for i, a in enumerate(after) for b in after[i + 1 :])


def test_rerank_settings_survive_save_and_load(tmp_path):
    index = SearchIndex(dedupe_overlaps=True, rerank_depth=7)
    index.sync(DOCS)
    index.save(tmp_path / "index")

    loaded = SearchIndex.load(tmp_path / "index", reranker=CrossEncoderReranker("fake", model=FakeCrossEncoder()))

    assert loaded.dedupe_overlaps is True and loaded.rerank_depth == 7
    assert loaded.reranker is not None


@pytest.mark.skipif(not __import__("os").environ.get("RUN_MODEL_TESTS"), reason="downloads a real model; set RUN_MODEL_TESTS=1")
def test_real_cross_encoder_prefers_the_answering_passage():
    reranker = create_reranker("ce:cross-encoder/ms-marco-MiniLM-L-6-v2")

    answer, distractor = reranker.score(
        "Why don't these alloys rust in the sea?",
        ["A thin chromium-rich oxide film protects them in seawater.", "Oxidation during aging discolours the surface."],
    )

    assert answer > distractor


def test_dedupe_collapses_fragments_even_when_one_chunk_contains_several():
    from retrieval.index import SearchIndex as Index
    from vectorstore import ChunkRecord

    def rec(i, start, end):
        return ChunkRecord(id=str(i), doc_id="d", text="", source="d", start=start, end=end, metadata={"section": 0})

    # Three fragments ranked first, then one long chunk (#3) that overlaps all of them.
    records = [rec(0, 0, 40), rec(1, 50, 90), rec(2, 30, 60), rec(3, 0, 95)]

    kept = Index._dedupe([0, 1, 2, 3], records, limit=4)

    assert kept == [3]


def test_dedupe_keeps_neighbouring_chunks_that_only_share_the_overlap_border():
    from retrieval.index import SearchIndex as Index
    from vectorstore import ChunkRecord

    # Normal consecutive chunks: 300 characters each, sharing a 50-character border.
    a = ChunkRecord(id="a", doc_id="d", text="", source="d", start=0, end=300, metadata={"section": 0})
    b = ChunkRecord(id="b", doc_id="d", text="", source="d", start=250, end=560, metadata={"section": 0})

    assert Index._dedupe([0, 1], [a, b], limit=2) == [0, 1]
