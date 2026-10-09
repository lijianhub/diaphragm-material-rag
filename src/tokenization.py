"""Shared tokenizer so embedding, lexical scoring and BM25 agree on what a term is."""
from __future__ import annotations

import re
from typing import List

# Keeps hyphenated alloy names such as "co-cr-ni-mo" as one token.
_TOKEN_RE = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")

STOPWORDS = frozenset(
    """
    a an and are as at be been by can do does for from has have how in is it its
    of on or should that the their then there these this to was were what when
    where which who why will with
    """.split()
)

_VOWELS = frozenset("aeiou")


def _has_vowel(text: str) -> bool:
    # As in Porter, "y" after a consonant acts as a vowel ("cycl" in "cycling").
    return any(
        ch in _VOWELS or (ch == "y" and i > 0 and text[i - 1] not in _VOWELS)
        for i, ch in enumerate(text)
    )


def stem(word: str) -> str:
    """Light suffix-stripping stemmer, a small subset of Porter's step 1.

    It conflates the inflections that matter for retrieval ("verify",
    "verified", "verifies" -> "verifi"). Stems are index keys, not words, so
    they need only be consistent, not readable. Hyphenated terms and numbers
    are left alone so alloy names stay exact.
    """
    if len(word) <= 2 or "-" in word or not word.isalpha():
        return word

    # Plurals: "processes" -> "processe" (the trailing-e rule finishes it), "ies" -> "i".
    if word.endswith("sses"):
        word = word[:-2]
    elif word.endswith("ies"):
        word = word[:-3] + "i"
    elif word.endswith("s") and not word.endswith(("ss", "us", "is")):
        word = word[:-1]

    # Past tense and progressive, only if a real stem (two letters with a vowel) remains.
    for suffix in ("ing", "ed"):
        if word.endswith(suffix) and not word.endswith("eed"):
            base = word[: -len(suffix)]
            if len(base) >= 2 and _has_vowel(base):
                word = base
                # "stopped" -> "stopp" -> "stop", but keep "ll", "ss", "zz".
                if len(word) >= 3 and word[-1] == word[-2] and word[-1] not in _VOWELS | set("lsz"):
                    word = word[:-1]
            break

    # Consonant + y -> i, so "verify" meets "verified"; "alloy" keeps its y.
    if word.endswith("y") and len(word) > 2 and word[-2] not in _VOWELS:
        word = word[:-1] + "i"

    # Drop a final e so "age", "aged" and "aging" all become "ag".
    if word.endswith("e") and len(word) >= 3:
        word = word[:-1]

    return word


def tokenize(text: str, remove_stopwords: bool = True, stem_tokens: bool = False) -> List[str]:
    tokens = _TOKEN_RE.findall(text.lower())
    if remove_stopwords:
        tokens = [token for token in tokens if token not in STOPWORDS]
    if stem_tokens:
        tokens = [stem(token) for token in tokens]
    return tokens
