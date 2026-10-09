from __future__ import annotations

from typing import List, Sequence, Tuple

from tokenization import tokenize

from .dense import cosine_similarity


def lexical_coverage(query_terms: Sequence[str], text: str, stem_tokens: bool = False) -> float:
    """Fraction of distinct query terms that appear as whole words in ``text``.

    ``query_terms`` must be tokenized with the same ``stem_tokens`` setting.
    """
    terms = set(query_terms)
    if not terms:
        return 0.0
    return len(terms & set(tokenize(text, stem_tokens=stem_tokens))) / len(terms)


def max_normalize(scores: Sequence[float]) -> List[float]:
    """Scale scores into [0, 1] by the best score, e.g. to blend unbounded BM25 with cosine."""
    top = max(scores, default=0.0)
    return [score / top for score in scores] if top > 0 else [0.0 for _ in scores]


def combine_scores(dense: Sequence[float], lexical: Sequence[float], alpha: float = 0.5) -> List[float]:
    """Weighted sum ``alpha * dense + (1 - alpha) * lexical``; both inputs should lie in [0, 1]."""
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be between 0 and 1")
    if len(dense) != len(lexical):
        raise ValueError("dense and lexical score lists differ in length")
    return [alpha * d + (1.0 - alpha) * s for d, s in zip(dense, lexical)]


def hybrid_scores(
    query: str,
    query_vector: Sequence[float],
    texts: Sequence[str],
    vectors: Sequence[Sequence[float]],
    alpha: float = 0.5,
    stem_tokens: bool = False,
) -> List[float]:
    """Blend dense cosine similarity (weight ``alpha``) with lexical coverage."""
    query_terms = tokenize(query, stem_tokens=stem_tokens)
    dense = [cosine_similarity(query_vector, vector) for vector in vectors]
    lexical = [lexical_coverage(query_terms, text, stem_tokens) for text in texts]
    return combine_scores(dense, lexical, alpha)


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
