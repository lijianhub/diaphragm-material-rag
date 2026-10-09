from __future__ import annotations

import math
from typing import List, Sequence, Tuple


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) != len(b):
        raise ValueError(f"vector dimensions differ: {len(a)} != {len(b)}")
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    if norm == 0:
        return 0.0
    return sum(x * y for x, y in zip(a, b)) / norm


def dense_search(
    query_vector: Sequence[float],
    texts: Sequence[str],
    vectors: Sequence[Sequence[float]],
    top_k: int = 3,
) -> List[Tuple[str, float]]:
    scored = [(text, cosine_similarity(query_vector, vector)) for text, vector in zip(texts, vectors)]
    scored.sort(key=lambda item: item[1], reverse=True)
    return scored[:top_k]
