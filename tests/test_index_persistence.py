import importlib.util

import pytest

from embedding.embedder import SimpleEmbedder
from ingestion.loader import Document
from retrieval.index import SearchIndex

HAS_FAISS = importlib.util.find_spec("faiss") is not None
STORE_KINDS = ["memory", pytest.param("faiss", marks=pytest.mark.skipif(not HAS_FAISS, reason="faiss-cpu not installed"))]

DOCS = [
    Document(source="forming.txt", content="Springback is the elastic recovery of the part after forming."),
    Document(source="quality.txt", content="The material certificate lists hardness and tensile strength."),
    Document(source="springs.txt", content="Shot peening introduces compressive residual stress in springs."),
]


def _ranked(index, query):
    return [(r.source, r.start, round(r.score, 6)) for r in index.search(query, top_k=3)]


@pytest.mark.parametrize("kind", STORE_KINDS)
def test_saved_index_answers_like_the_original(kind, tmp_path):
    index = SearchIndex(store=kind)
    index.sync(DOCS)
    index.save(tmp_path / "index")

    loaded = SearchIndex.load(tmp_path / "index")

    assert len(loaded) == len(index)
    for query in ["What does the certificate list?", "shot peening springs", "springback"]:
        assert _ranked(loaded, query) == _ranked(index, query)


@pytest.mark.parametrize("kind", STORE_KINDS)
def test_faiss_and_memory_stores_rank_identically(kind):
    reference = SearchIndex(store="memory")
    candidate = SearchIndex(store=kind)
    reference.sync(DOCS)
    candidate.sync(DOCS)

    assert _ranked(candidate, "residual stress after forming") == _ranked(reference, "residual stress after forming")


def test_sync_only_reindexes_what_changed():
    index = SearchIndex()
    first = index.sync(DOCS)
    assert (first.added, first.updated, first.removed, first.unchanged) == (3, 0, 0, 0)

    edited = Document(source="quality.txt", content="Certificates also list the heat number of the strip.")
    second = index.sync([DOCS[0], edited])

    assert (second.added, second.updated, second.removed, second.unchanged) == (0, 1, 1, 1)
    assert {r.source for r in index.search("heat number certificate", top_k=5)} <= {"forming.txt", "quality.txt"}
    assert index.search("heat number certificate", top_k=1)[0].source == "quality.txt"
    assert "springs.txt" not in index.document_ids()


def test_resyncing_identical_documents_is_a_no_op():
    index = SearchIndex()
    index.sync(DOCS)
    chunks_before = len(index)

    stats = index.sync(DOCS)

    assert (stats.added, stats.updated, stats.removed, stats.unchanged) == (0, 0, 0, 3)
    assert len(index) == chunks_before


def test_load_rejects_an_embedder_that_does_not_match_the_index(tmp_path):
    index = SearchIndex(embedder=SimpleEmbedder(dim=512))
    index.sync(DOCS)
    index.save(tmp_path / "index")

    with pytest.raises(ValueError, match="embedder"):
        SearchIndex.load(tmp_path / "index", embedder=SimpleEmbedder(dim=256))


def test_load_missing_index_raises_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        SearchIndex.load(tmp_path / "nothing-here")
