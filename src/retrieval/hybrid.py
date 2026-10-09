from __future__ import annotations

from typing import List, Sequence, Tuple

from tokenization import tokenize

from .dense import cosine_similarity


def lexical_coverage(query_terms: Sequence[str], text: str) -> float:
    """Fraction of distinct query terms that appear as whole words in ``text``."""
    terms = set(query_terms)
    if not terms:
        return 0.0
    return len(terms & set(tokenize(text))) / len(terms)


def hybrid_scores(
    query: str,
    query_vector: Sequence[float],
    texts: Sequence[str],
    vectors: Sequence[Sequence[float]],
    alpha: float = 0.5,
) -> List[float]:
    """Blend dense cosine similarity (weight ``alpha``) with lexical coverage.

    Both components lie in [0, 1] for non-negative embeddings, so ``alpha``
    is a true mixing weight.
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be between 0 and 1")
    query_terms = tokenize(query)
    return [
        alpha * cosine_similarity(query_vector, vector) + (1.0 - alpha) * lexical_coverage(query_terms, text)
        for text, vector in zip(texts, vectors)
    ]


def hybrid_search(
    query: str,
    query_vector: Sequence[float],
    texts: Sequence[str],
    vectors: Sequence[Sequence[float]],
    top_k: int = 3,
    alpha: float = 0.5,
) -> List[Tuple[str, float]]:
    scored = list(zip(texts, hybrid_scores(query, query_vector, texts, vectors, alpha)))
    scored.sort(key=lambda item: item[1], reverse=True)
    return scored[:top_k]
