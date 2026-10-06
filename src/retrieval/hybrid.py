from __future__ import annotations

from typing import List, Sequence, Tuple


def hybrid_search(query: str, texts: Sequence[str], vectors: Sequence[Sequence[float]], top_k: int = 3) -> List[Tuple[str, float]]:
    """A simple hybrid search that combines lexical overlap with vector similarity."""
    q = query.lower()
    scored: List[Tuple[str, float]] = []

    for text, vector in zip(texts, vectors):
        lexical_score = sum(1 for term in q.split() if term in text.lower())
        vector_score = sum(v for v in vector) / max(1, len(vector))
        combined = lexical_score * 2.0 + float(vector_score) / 100.0
        scored.append((text, combined))

    scored.sort(key=lambda item: item[1], reverse=True)
    return scored[:top_k]
