"""Evaluate retrieval quality on the bundled question set.

Usage:
    python scripts/evaluate.py
    python scripts/evaluate.py --k 1 --alpha 0.3 --chunk-size 200
    python scripts/evaluate.py --lexical coverage --no-stem   # step-3 configuration
"""
from __future__ import annotations

import argparse
import os
import json
import sys
from dataclasses import asdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config import load_env_file

load_env_file()  # local .env (git-ignored); real environment variables win

from chunking import RecursiveChunker
from embedding import create_embedder
from evaluation import benchmark_corpus, evaluate_retrieval, load_dataset
from ingestion.loader import load_documents
from generation import CachedLLM, create_llm
from query import LLMQueryRewriter, decompose, retrieve
from reranking import create_reranker
from retrieval.index import FUSION_METHODS, LEXICAL_SCORERS, SearchIndex
from vectorstore import STORE_KINDS


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--k", type=int, default=3, help="number of retrieved chunks to score")
    parser.add_argument("--alpha", type=float, default=0.5, help="dense weight in hybrid scoring (0 = lexical only)")
    parser.add_argument("--lexical", choices=LEXICAL_SCORERS, default="bm25", help="lexical scorer blended with dense")
    parser.add_argument("--stem", action=argparse.BooleanOptionalAction, default=True, help="stem terms for lexical scoring")
    parser.add_argument("--fusion", choices=FUSION_METHODS, default="weighted", help="how dense and lexical rankings are combined")
    parser.add_argument("--rrf-k", type=int, default=60, help="RRF rank constant")
    parser.add_argument(
        "--embedder",
        default="simple-hash",
        help="simple-hash | st:<hugging-face model> | bedrock:<model>[@dims], e.g. st:sentence-transformers/all-MiniLM-L6-v2",
    )
    parser.add_argument("--reranker", default=None, help="second-stage reranker, e.g. ce:cross-encoder/ms-marco-MiniLM-L-6-v2")
    parser.add_argument("--rerank-depth", type=int, default=20, help="first-stage candidates passed to the reranker")
    parser.add_argument("--dedupe", action=argparse.BooleanOptionalAction, default=False, help="collapse overlapping chunks of the same section")
    parser.add_argument("--decompose", action="store_true", help="split compound questions and fuse the sub-query rankings")
    parser.add_argument("--query-fusion", choices=("rrf", "interleave"), default="interleave", help="how sub-query rankings are merged")
    parser.add_argument("--rewrite", choices=("multi-query", "hyde"), default=None, help="LLM query rewriting (needs --llm)")
    parser.add_argument("--llm", default=os.environ.get("RAG_LLM") or "anthropic:claude-opus-5-5", help="model used by --rewrite (default: RAG_LLM)")
    parser.add_argument("--prf", type=int, default=0, help="pseudo-relevance feedback: number of expansion terms (0 = off)")
    parser.add_argument("--prf-docs", type=int, default=3, help="first-pass passages used for feedback")
    parser.add_argument("--store", choices=STORE_KINDS, default="memory", help="vector store backend")
    parser.add_argument("--chunk-size", type=int, default=300)
    parser.add_argument("--overlap", type=int, default=50)
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "evaluation" / "results.jsonl")
    args = parser.parse_args()

    # Only the tracked sample files (txt, md, html, csv), so scores are reproducible on every clone.
    corpus = benchmark_corpus(REPO_ROOT / "data" / "raw")
    index = SearchIndex(
        embedder=create_embedder(args.embedder),
        chunker=RecursiveChunker(chunk_size=args.chunk_size, overlap=args.overlap),
        alpha=args.alpha,
        lexical=args.lexical,
        stem_tokens=args.stem,
        fusion=args.fusion,
        rrf_k=args.rrf_k,
        store=args.store,
        reranker=create_reranker(args.reranker) if args.reranker else None,
        rerank_depth=args.rerank_depth,
        dedupe_overlaps=args.dedupe,
    )
    index.add_documents(load_documents(corpus))

    examples = load_dataset(REPO_ROOT / "evaluation" / "questions.jsonl", REPO_ROOT / "evaluation" / "ground_truth.jsonl")
    transforms = [decompose] if args.decompose else []
    if args.rewrite:
        # Cached on disk: re-running the evaluation does not pay for the same rewrites again.
        rewriter_llm = CachedLLM(create_llm(args.llm, effort="low"), REPO_ROOT / ".cache" / "llm.jsonl")
        transforms.append(LLMQueryRewriter(rewriter_llm, mode=args.rewrite))

    def search(question, k):
        results = retrieve(
            index, question, top_k=k, transforms=transforms, prf_terms=args.prf, prf_docs=args.prf_docs, fusion=args.query_fusion
        )
        return [r.text for r in results]

    report = evaluate_retrieval(examples, search, k=args.k)

    print(f"documents={len(corpus)} chunks={len(index)} questions={len(examples)} k={report.k} alpha={args.alpha} lexical={args.lexical} stem={args.stem} fusion={args.fusion} store={args.store} embedder={index.embedder.name} reranker={args.reranker} dedupe={args.dedupe} decompose={args.decompose} prf={args.prf} rewrite={args.rewrite}")
    print(f"hit@{report.k}={report.hit_rate:.3f}  recall@{report.k}={report.recall:.3f}  mrr={report.mrr:.3f}")
    if args.rewrite:
        print(f"LLM calls: {rewriter_llm.misses} new, {rewriter_llm.hits} from cache")
    for category, score in report.by_category().items():
        print(f"  {category:<11} n={score.count:<3} hit={score.hit_rate:.3f}  recall={score.recall:.3f}  mrr={score.mrr:.3f}")
    for miss in report.misses:
        print(f"  miss {miss.id}: {miss.question}")

    with args.output.open("w", encoding="utf-8") as handle:
        for result in report.results:
            handle.write(json.dumps(asdict(result)) + "\n")
    shown = args.output.resolve()
    if shown.is_relative_to(REPO_ROOT):
        shown = shown.relative_to(REPO_ROOT)
    print(f"per-question results written to {shown}")


if __name__ == "__main__":
    main()
