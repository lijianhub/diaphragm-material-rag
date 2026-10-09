from __future__ import annotations

import math
from collections import Counter
from typing import List, Sequence

from tokenization import tokenize


class BM25:
    """Okapi BM25 over an in-memory corpus.

    score(q, d) = sum over query terms t of
        idf(t) * tf(t, d) * (k1 + 1) / (tf(t, d) + k1 * (1 - b + b * |d| / avgdl))

    ``k1`` controls how quickly repeated terms stop adding score (term-frequency
    saturation), and ``b`` controls how strongly long documents are penalised.
    IDF uses the Lucene form ln(1 + (N - df + 0.5) / (df + 0.5)), which is never
    negative, so a term that appears in most documents still counts a little
    instead of lowering the score.
    """

    def __init__(self, texts: Sequence[str], k1: float = 1.5, b: float = 0.75, stem_tokens: bool = False):
        if k1 < 0:
            raise ValueError("k1 must be non-negative")
        if not 0.0 <= b <= 1.0:
            raise ValueError("b must be between 0 and 1")
        self.k1 = k1
        self.b = b
        self.stem_tokens = stem_tokens

        self._term_counts = [Counter(tokenize(text, stem_tokens=stem_tokens)) for text in texts]
        self._lengths = [sum(counts.values()) for counts in self._term_counts]
        self._avg_length = sum(self._lengths) / len(self._lengths) if self._lengths else 0.0
        self._doc_freq: Counter = Counter()
        for counts in self._term_counts:
            self._doc_freq.update(counts.keys())

    def _idf(self, term: str) -> float:
        n_docs = len(self._term_counts)
        df = self._doc_freq.get(term, 0)
        return math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))

    def idf(self, term: str) -> float:
        """IDF of ``term`` after the same normalisation as the corpus."""
        tokens = tokenize(term, stem_tokens=self.stem_tokens)
        return self._idf(tokens[0]) if tokens else 0.0

    def scores(self, query: str) -> List[float]:
        # Each distinct query term counts once; repeating a word in the query does not boost it.
        terms = [t for t in dict.fromkeys(tokenize(query, stem_tokens=self.stem_tokens)) if t in self._doc_freq]
        idfs = {term: self._idf(term) for term in terms}

        results: List[float] = []
        for counts, length in zip(self._term_counts, self._lengths):
            norm = self.k1 * (1.0 - self.b + self.b * length / self._avg_length) if self._avg_length else self.k1
            score = 0.0
            for term in terms:
                tf = counts.get(term, 0)
                if tf:
                    score += idfs[term] * tf * (self.k1 + 1.0) / (tf + norm)
            results.append(score)
        return results
