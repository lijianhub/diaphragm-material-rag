import pytest

from retrieval.bm25 import BM25


def test_bm25_scores_zero_without_term_overlap():
    bm25 = BM25(["springback after forming", "material certificate lists hardness"])

    assert bm25.scores("lubrication") == [0.0, 0.0]


def test_bm25_ranks_rare_terms_above_common_terms():
    corpus = [
        "fatigue performance of the alloy",
        "fatigue performance under pressure cycling",
        "fatigue performance and corrosion",
    ]
    bm25 = BM25(corpus)

    scores = bm25.scores("pressure cycling fatigue")

    assert scores.index(max(scores)) == 1
    assert bm25.idf("pressure") > bm25.idf("fatigue")


def test_bm25_term_frequency_saturates():
    bm25 = BM25(["wrinkling", "wrinkling wrinkling wrinkling wrinkling wrinkling wrinkling", "unrelated text"], b=0.0)

    once, many, _ = bm25.scores("wrinkling")

    assert many > once
    assert many < 6 * once
    assert many < once * (bm25.k1 + 1)


def test_bm25_length_normalization_prefers_shorter_documents():
    short = "springback compensation"
    long = "springback compensation " + " ".join(f"filler{i}" for i in range(30))
    bm25 = BM25([short, long, "unrelated"])

    short_score, long_score, _ = bm25.scores("springback")

    assert short_score > long_score


def test_bm25_matches_inflected_forms_when_stemming():
    corpus = ["Pressure cycling tests verify fatigue performance.", "Inspectors compare values."]

    assert BM25(corpus).scores("verified")[0] == 0.0
    assert BM25(corpus, stem_tokens=True).scores("verified")[0] > 0.0


def test_bm25_on_empty_corpus_returns_no_scores():
    assert BM25([]).scores("anything") == []


def test_bm25_rejects_invalid_parameters():
    with pytest.raises(ValueError):
        BM25(["text"], k1=-1)
    with pytest.raises(ValueError):
        BM25(["text"], b=1.5)
