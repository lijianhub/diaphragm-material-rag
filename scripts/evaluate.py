"""Evaluate retrieval quality on the bundled question set.

Usage:
    python scripts/evaluate.py
    python scripts/evaluate.py --k 1 --alpha 0.3 --chunk-size 200
    python scripts/evaluate.py --lexical coverage --no-stem   # step-3 configuration
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from chunking import RecursiveChunker
from evaluation import evaluate_retrieval, load_dataset
from ingestion.loader import load_documents
from retrieval.index import FUSION_METHODS, LEXICAL_SCORERS, SearchIndex


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--k", type=int, default=3, help="number of retrieved chunks to score")
    parser.add_argument("--alpha", type=float, default=0.5, help="dense weight in hybrid scoring (0 = lexical only)")
    parser.add_argument("--lexical", choices=LEXICAL_SCORERS, default="bm25", help="lexical scorer blended with dense")
    parser.add_argument("--stem", action=argparse.BooleanOptionalAction, default=True, help="stem terms for lexical scoring")
    parser.add_argument("--fusion", choices=FUSION_METHODS, default="weighted", help="how dense and lexical rankings are combined")
    parser.add_argument("--rrf-k", type=int, default=60, help="RRF rank constant")
    parser.add_argument("--chunk-size", type=int, default=300)
    parser.add_argument("--overlap", type=int, default=50)
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "evaluation" / "results.jsonl")
    args = parser.parse_args()

    # Only the tracked *.txt samples, so scores are reproducible on every clone.
    corpus = sorted((REPO_ROOT / "data" / "raw").glob("*.txt"))
    index = SearchIndex(
        chunker=RecursiveChunker(chunk_size=args.chunk_size, overlap=args.overlap),
        alpha=args.alpha,
        lexical=args.lexical,
        stem_tokens=args.stem,
        fusion=args.fusion,
        rrf_k=args.rrf_k,
    )
    index.add_documents(load_documents(corpus))

    examples = load_dataset(REPO_ROOT / "evaluation" / "questions.jsonl", REPO_ROOT / "evaluation" / "ground_truth.jsonl")
    report = evaluate_retrieval(examples, lambda question, k: [r.text for r in index.search(question, top_k=k)], k=args.k)

    print(f"documents={len(corpus)} chunks={len(index)} questions={len(examples)} k={report.k} alpha={args.alpha} lexical={args.lexical} stem={args.stem} fusion={args.fusion}")
    print(f"hit@{report.k}={report.hit_rate:.3f}  recall@{report.k}={report.recall:.3f}  mrr={report.mrr:.3f}")
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
