from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Sequence, Set

from .dataset import EvalExample
from .metrics import hit_at_k, recall_at_k, reciprocal_rank

# Takes (question, top_k) and returns retrieved passages in rank order.
SearchFn = Callable[[str, int], Sequence[str]]


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def match_evidence(passages: Sequence[str], evidence: Sequence[str]) -> List[Set[int]]:
    """For each passage, the indices of evidence phrases it contains verbatim.

    Matching on phrases instead of chunk ids keeps the ground truth valid when
    the chunking strategy changes.
    """
    phrases = [_normalize(phrase) for phrase in evidence]
    matches: List[Set[int]] = []
    for passage in passages:
        normalized = _normalize(passage)
        matches.append({i for i, phrase in enumerate(phrases) if phrase in normalized})
    return matches


@dataclass
class QuestionResult:
    id: str
    question: str
    category: str
    hit: float
    recall: float
    reciprocal_rank: float
    retrieved: List[str]


@dataclass
class CategoryScore:
    count: int
    hit_rate: float
    recall: float
    mrr: float


def _aggregate(results: Sequence[QuestionResult]) -> CategoryScore:
    count = len(results)
    return CategoryScore(
        count=count,
        hit_rate=sum(r.hit for r in results) / count,
        recall=sum(r.recall for r in results) / count,
        mrr=sum(r.reciprocal_rank for r in results) / count,
    )


@dataclass
class EvalReport:
    k: int
    hit_rate: float
    recall: float
    mrr: float
    results: List[QuestionResult] = field(default_factory=list)

    @property
    def misses(self) -> List[QuestionResult]:
        return [result for result in self.results if not result.hit]

    def by_category(self) -> Dict[str, CategoryScore]:
        """Metrics per question category, so a method's weak spots are not averaged away."""
        groups: Dict[str, List[QuestionResult]] = {}
        for result in self.results:
            groups.setdefault(result.category, []).append(result)
        return {category: _aggregate(group) for category, group in sorted(groups.items())}


def evaluate_retrieval(examples: Sequence[EvalExample], search: SearchFn, k: int = 3) -> EvalReport:
    if not examples:
        raise ValueError("examples must not be empty")

    results: List[QuestionResult] = []
    for example in examples:
        retrieved = list(search(example.question, k))[:k]
        matches = match_evidence(retrieved, example.evidence)
        results.append(
            QuestionResult(
                id=example.id,
                question=example.question,
                category=example.category,
                hit=hit_at_k(matches, k),
                recall=recall_at_k(matches, len(example.evidence), k),
                reciprocal_rank=reciprocal_rank(matches),
                retrieved=retrieved,
            )
        )

    overall = _aggregate(results)
    return EvalReport(k=k, hit_rate=overall.hit_rate, recall=overall.recall, mrr=overall.mrr, results=results)
