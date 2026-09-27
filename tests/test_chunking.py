from chunking.fixed_size import FixedSizeChunker
from chunking.recursive import RecursiveChunker


def test_fixed_size_chunker_splits_long_text():
    chunker = FixedSizeChunker(chunk_size=10, overlap=2)
    text = "alpha beta gamma delta epsilon zeta eta"

    chunks = chunker.chunk(text, source="sample.txt")

    assert len(chunks) > 1
    assert all(chunk.text for chunk in chunks)
    assert chunks[0].source == "sample.txt"
    assert chunks[0].start == 0


def test_recursive_chunker_creates_sentence_aware_chunks():
    chunker = RecursiveChunker(chunk_size=80, overlap=20)
    text = (
        "Diaphragm materials are used in precision forming operations. "
        "The forming process depends on material strength and fatigue resistance. "
        "Co-Cr-Ni-Mo alloys are common in diaphragm applications."
    )

    chunks = chunker.chunk(text, source="diaphragm.txt")

    assert len(chunks) >= 2
    assert all(chunk.text for chunk in chunks)
    assert chunks[0].source == "diaphragm.txt"
