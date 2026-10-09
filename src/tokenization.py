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


def tokenize(text: str, remove_stopwords: bool = True) -> List[str]:
    tokens = _TOKEN_RE.findall(text.lower())
    if remove_stopwords:
        tokens = [token for token in tokens if token not in STOPWORDS]
    return tokens
