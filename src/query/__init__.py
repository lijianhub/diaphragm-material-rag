"""Query transformation: turn one question into better or more search queries.

    decompose               rule-based split of compound questions (offline)
    expansion_terms         pseudo-relevance feedback terms (offline)
    LLMQueryRewriter        multi-query paraphrases or a HyDE passage from a language model
    retrieve                run the transformed queries against a SearchIndex and fuse them
"""
from __future__ import annotations

import dataclasses
from typing import Callable, Dict, Hashable, List, Sequence

from retrieval.fusion import reciprocal_rank_fusion
from retrieval.index import SearchIndex, SearchResult

from .decomposer import decompose
from .rewriter import LLMQueryRewriter, expansion_terms

Transform = Callable[[str], List[str]]
FUSIONS = ("rrf", "interleave")


def interleave(rankings: Sequence[Sequence[Hashable]]) -> List[Hashable]:
    """Round-robin merge: every ranking's #1, then every #2, and so on, skipping repeats.

    Use it when the queries ask for *different* things (decomposed sub-questions), so
    each part keeps its best evidence. RRF instead rewards items that appear in many
    rankings, which suits paraphrases of one question but lets a mediocre, generic
    chunk that every sub-query retrieves crowd out each part's best match.
    """
    merged: List[Hashable] = []
    seen = set()
    for position in range(max((len(r) for r in rankings), default=0)):
        for ranking in rankings:
            if position < len(ranking) and ranking[position] not in seen:
                seen.add(ranking[position])
                merged.append(ranking[position])
    return merged


def _key(result: SearchResult) -> Hashable:
    return (result.source, result.metadata.get("section"), result.start, result.end)


def retrieve(
    index: SearchIndex,
    question: str,
    top_k: int = 3,
    transforms: Sequence[Transform] = (),
    prf_terms: int = 0,
    prf_docs: int = 3,
    fusion: str = "rrf",
    fusion_k: int = 60,
) -> List[SearchResult]:
    """Search with transformed queries.

    1. Each transform maps every current query to one or more queries (originals kept).
    2. With ``prf_terms``, each query gets the top expansion terms from its own
       first-pass results (pseudo-relevance feedback).
    3. One query: a normal search. Several: each is searched (first stage only) and the
       rankings are fused: ``fusion="rrf"`` (consensus, for paraphrases of one question)
       or ``"interleave"`` (coverage, for decomposed sub-questions).
    4. If the index has a reranker, the fused candidates are reranked against the
       **original** question, never against a variant.
    """
    if fusion not in FUSIONS:
        raise ValueError(f"fusion must be one of {FUSIONS}")
    queries = [question]
    for transform in transforms:
        expanded: List[str] = []
        for query in queries:
            expanded.extend(transform(query))
        queries = list(dict.fromkeys(expanded))

    if prf_terms:
        with_feedback = []
        for query in queries:
            feedback = index.search(query, top_k=prf_docs, rerank=False)
            terms = expansion_terms(query, [r.text for r in feedback], index.term_idf, prf_terms)
            with_feedback.append(f"{query} {' '.join(terms)}" if terms else query)
        queries = list(dict.fromkeys(with_feedback))

    rerank = index.reranker is not None
    if len(queries) == 1:
        if not rerank:
            return index.search(queries[0], top_k=top_k)
        candidates = index.search(queries[0], top_k=index.rerank_depth, rerank=False)
    else:
        pool = index.rerank_depth if rerank else max(top_k, 10)
        rankings = [index.search(query, top_k=pool, rerank=False) for query in queries]
        by_key: Dict[Hashable, SearchResult] = {}
        for ranking in rankings:
            for result in ranking:
                by_key.setdefault(_key(result), result)
        keyed = [[_key(r) for r in ranking] for ranking in rankings]
        if fusion == "rrf":
            fused = reciprocal_rank_fusion(keyed, k=fusion_k)
            order = sorted(fused, key=lambda key: fused[key], reverse=True)[:pool]
            candidates = [dataclasses.replace(by_key[key], score=fused[key]) for key in order]
        else:
            order = interleave(keyed)[:pool]
            # Position-based score, so later code can still sort and report a number.
            candidates = [dataclasses.replace(by_key[key], score=1.0 / (n + 1)) for n, key in enumerate(order)]

    if not rerank:
        return candidates[:top_k]
    scores = index.reranker.score(question, [c.text for c in candidates])
    reranked = sorted(zip(candidates, scores), key=lambda pair: pair[1], reverse=True)[:top_k]
    return [
        dataclasses.replace(c, score=s, metadata={**c.metadata, "retrieval_score": c.score}) for c, s in reranked
    ]


__all__ = ["FUSIONS", "LLMQueryRewriter", "decompose", "expansion_terms", "interleave", "retrieve"]
