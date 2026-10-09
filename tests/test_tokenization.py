import pytest

from tokenization import stem, tokenize


@pytest.mark.parametrize(
    "variants",
    [
        ["verify", "verified", "verifies"],
        ["form", "forms", "formed", "forming"],
        ["age", "aged", "aging"],
        ["stop", "stopped", "stopping"],
        ["alloy", "alloys"],
        ["diaphragm", "diaphragms"],
        ["cycle", "cycled", "cycling"],
    ],
)
def test_stem_conflates_inflected_forms(variants):
    assert len({stem(word) for word in variants}) == 1, [stem(word) for word in variants]


@pytest.mark.parametrize("word", ["process", "corpus", "analysis", "bed", "red", "sing", "bring"])
def test_stem_leaves_words_that_only_look_inflected(word):
    assert stem(word) == word


def test_stem_keeps_short_words_intact():
    assert stem("bed") == "bed"
    assert stem("is") == "is"


def test_tokenize_with_stemming_is_opt_in():
    assert tokenize("Forming parts") == ["forming", "parts"]
    assert tokenize("Forming parts", stem_tokens=True) == ["form", "part"]


def test_tokenize_keeps_hyphenated_alloy_names_unstemmed():
    assert tokenize("Co-Cr-Ni-Mo alloys", stem_tokens=True) == ["co-cr-ni-mo", "alloy"]
