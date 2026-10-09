import pytest

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


def test_recursive_chunker_leaves_no_gaps_between_chunks():
    chunker = RecursiveChunker(chunk_size=60, overlap=10)
    text = " ".join(f"Sentence number {i} talks about forming pressure." for i in range(12))

    chunks = chunker.chunk(text, source="gaps.txt")

    for previous, current in zip(chunks, chunks[1:]):
        assert current.start <= previous.end
    assert chunks[0].start == 0
    assert chunks[-1].end == len(text)


def test_recursive_chunker_keeps_sentence_terminator():
    chunker = RecursiveChunker(chunk_size=40, overlap=0)

    chunks = chunker.chunk("First short sentence here. Second sentence follows it.")

    assert chunks[0].text == "First short sentence here."


def test_fixed_size_chunker_rejects_overlap_not_smaller_than_chunk_size():
    with pytest.raises(ValueError):
        FixedSizeChunker(chunk_size=10, overlap=10).chunk("x" * 50)


def test_recursive_chunker_splits_on_paragraph_breaks():
    chunker = RecursiveChunker(chunk_size=80, overlap=0)
    # No full stop before the break, so only the paragraph separator can explain the split.
    text = "Heading about forming\n\nSecond paragraph. It covers inspection and testing of finished diaphragms."

    chunks = chunker.chunk(text)

    assert chunks[0].text == "Heading about forming"


def test_recursive_chunker_overlap_starts_on_a_word_boundary():
    chunker = RecursiveChunker(chunk_size=60, overlap=15)
    text = " ".join(f"Sentence number {i} talks about forming pressure." for i in range(12))

    chunks = chunker.chunk(text)

    words = set(text.replace(".", " ").split())
    for chunk in chunks:
        assert chunk.text.split()[0].rstrip(".") in words
