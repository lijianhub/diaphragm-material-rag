import importlib.util

import pytest

from vectorstore import ChunkRecord, create_store, load_store

HAS_FAISS = importlib.util.find_spec("faiss") is not None
STORE_KINDS = ["memory", pytest.param("faiss", marks=pytest.mark.skipif(not HAS_FAISS, reason="faiss-cpu not installed"))]


def _record(chunk_id, doc_id="doc.txt", text="text"):
    return ChunkRecord(id=chunk_id, doc_id=doc_id, text=text, source=doc_id, start=0, end=len(text))


@pytest.fixture(params=STORE_KINDS)
def store(request):
    return create_store(request.param)


def test_search_returns_nearest_vectors_by_cosine(store):
    store.add([_record("a"), _record("b"), _record("c")], [[1, 0, 0], [0, 1, 0], [0.8, 0.6, 0]])

    hits = store.search([1, 0, 0], top_k=2)

    assert [hit.id for hit in hits] == ["a", "c"]
    assert hits[0].score == pytest.approx(1.0)
    assert hits[1].score == pytest.approx(0.8)


def test_search_uses_cosine_not_raw_dot_product(store):
    # By dot product the long vector [5, 0] would win; by angle [1, 1] is closer to the query.
    store.add([_record("long"), _record("close")], [[5, 0], [1, 1]])

    hits = store.search([1, 0.9], top_k=2)

    assert hits[0].id == "close"


def test_top_k_larger_than_store_returns_everything(store):
    store.add([_record("a"), _record("b")], [[1, 0], [0, 1]])

    assert len(store.search([1, 0], top_k=10)) == 2


def test_empty_store_returns_no_hits(store):
    assert store.search([1, 0], top_k=3) == []
    assert len(store) == 0


def test_delete_document_removes_all_its_chunks(store):
    store.add(
        [_record("a#0", "a.txt"), _record("a#1", "a.txt"), _record("b#0", "b.txt")],
        [[1, 0], [0.9, 0.1], [0, 1]],
    )

    removed = store.delete_document("a.txt")

    assert removed == 2
    assert len(store) == 1
    assert [hit.id for hit in store.search([1, 0], top_k=5)] == ["b#0"]
    assert [record.id for record in store.records()] == ["b#0"]


def test_records_keep_insertion_order_and_metadata(store):
    record = ChunkRecord(id="x#0", doc_id="x.pdf", text="hello", source="x.pdf", start=3, end=8, metadata={"page": 2})
    store.add([record, _record("y#0", "y.txt")], [[1, 0], [0, 1]])

    assert [r.id for r in store.records()] == ["x#0", "y#0"]
    assert store.get("x#0") == record


def test_save_and_load_round_trip(store, tmp_path):
    store.add([_record("a"), _record("b", text="other")], [[1, 0, 0], [0, 1, 0]])

    store.save(tmp_path / "store")
    loaded = load_store(tmp_path / "store")

    assert type(loaded) is type(store)
    assert loaded.records() == store.records()
    assert [(h.id, round(h.score, 6)) for h in loaded.search([0, 1, 0], 2)] == [
        (h.id, round(h.score, 6)) for h in store.search([0, 1, 0], 2)
    ]


def test_rejects_dimension_mismatch(store):
    store.add([_record("a")], [[1, 0, 0]])

    with pytest.raises(ValueError):
        store.add([_record("b")], [[1, 0]])
    with pytest.raises(ValueError):
        store.search([1, 0], top_k=1)


def test_rejects_duplicate_ids_and_misaligned_input(store):
    store.add([_record("a")], [[1, 0]])

    with pytest.raises(ValueError):
        store.add([_record("a")], [[0, 1]])
    with pytest.raises(ValueError):
        store.add([_record("b"), _record("c")], [[0, 1]])


def test_unknown_store_kind_is_rejected():
    with pytest.raises(ValueError):
        create_store("pinecone")
