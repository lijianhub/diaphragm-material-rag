"""Rank fusion for combining retrievers whose scores live on different scales."""
from __future__ import annotations

from typing import Dict, Hashable, List, Optional, Sequence


def rank_by_score(scores: Sequence[float]) -> List[int]:
    """Indices ordered by descending score, dropping items with no signal (score <= 0).

    A zero BM25 or cosine score means "not retrieved"; giving such items a rank
    would let RRF reward arbitrary tie order.
    """
    candidates = [i for i, score in enumerate(scores) if score > 0]
    return sorted(candidates, key=lambda i: scores[i], reverse=True)


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[Hashable]],
    k: int = 60,
    weights: Optional[Sequence[float]] = None,
) -> Dict[Hashable, float]:
    """Reciprocal Rank Fusion (Cormack et al., 2009).

    score(d) = sum over rankings r of  weight_r / (k + rank_r(d)),  with rank starting at 1.

    Only ranks are used, so retrievers with incompatible score scales (unbounded
    BM25, cosine in [-1, 1]) can be fused without normalization. ``k`` damps the
    advantage of the very top ranks; 60 is the value from the original paper.
    """
    if k < 0:
        raise ValueError("k must be non-negative")
    if weights is None:
        weights = [1.0] * len(rankings)
    if len(weights) != len(rankings):
        raise ValueError("weights must have one entry per ranking")

    fused: Dict[Hashable, float] = {}
    for ranking, weight in zip(rankings, weights):
        for rank, item in enumerate(ranking, start=1):
            fused[item] = fused.get(item, 0.0) + weight / (k + rank)
    return fused
