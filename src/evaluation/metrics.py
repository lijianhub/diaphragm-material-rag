"""Retrieval metrics over ranked results.

Each metric takes ``matches``: one entry per retrieved item, in rank order,
holding the indices of the relevant items (evidence phrases) that the
retrieved item contains. An empty set means the item is not relevant.
"""
from __future__ import annotations

from typing import AbstractSet, Sequence

Matches = Sequence[AbstractSet[int]]


def hit_at_k(matches: Matches, k: int) -> float:
    """1.0 if any of the top-k items is relevant, else 0.0."""
    return 1.0 if any(matches[:k]) else 0.0


def recall_at_k(matches: Matches, num_relevant: int, k: int) -> float:
    """Fraction of the relevant items covered by the top-k results."""
    if num_relevant <= 0:
        raise ValueError("num_relevant must be positive")
    found = set().union(*matches[:k]) if matches[:k] else set()
    return len(found) / num_relevant


def reciprocal_rank(matches: Matches) -> float:
    """1 / rank of the first relevant item, or 0.0 if none is relevant."""
    for rank, match in enumerate(matches, start=1):
        if match:
            return 1.0 / rank
    return 0.0
