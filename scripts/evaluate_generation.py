"""Evaluate generated answers: correctness, citations, refusals, ambiguity and injection.

Usage:
    python scripts/evaluate_generation.py                                  # offline extractive answers
    python scripts/evaluate_generation.py --llm openai:<model>@low         # LLM answers (cached in .cache/)
    python scripts/evaluate_generation.py --llm openai:<model>@low --gate  # exit 1 if a gate fails

Answerable cases reuse the retrieval benchmark questions; see src/evaluation/generation.py
for the case types, the scoring rules and the gate thresholds.
"""
from __future__ import annotations

import argparse
import json
import math
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
from evaluation import CASE_TYPES, benchmark_corpus, evaluate_generation, load_generation_cases
from generation import CachedLLM, RAGPipeline, create_llm
from ingestion.loader import load_documents
from retrieval.index import SearchIndex


def _fmt(value: float) -> str:
    return "n/a" if isinstance(value, float) and math.isnan(value) else f"{value:.3f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--llm", default=None, help="LLM spec, e.g. openai:<model>@low; omit for offline extractive answers")
    parser.add_argument("--embedder", default="simple-hash", help="simple-hash | st:<model> | bedrock:<model>")
    parser.add_argument("--top-k", type=int, default=3, help="sources offered to the answerer")
    parser.add_argument("--types", nargs="*", choices=CASE_TYPES, default=list(CASE_TYPES), help="case types to run")
    parser.add_argument("--gate", action="store_true", help="exit with status 1 if any gate fails")
    parser.add_argument("--show", choices=("failures", "all", "none"), default="failures", help="which answers to print")
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "evaluation" / "generation_results.jsonl")
    args = parser.parse_args()

    corpus = benchmark_corpus(REPO_ROOT / "data" / "raw")
    index = SearchIndex(embedder=create_embedder(args.embedder), chunker=RecursiveChunker(chunk_size=300, overlap=50))
    index.add_documents(load_documents(corpus))

    llm = None
    if args.llm:
        # Cached on disk keyed by model, system prompt and user prompt: reruns are free,
        # and a changed prompt is a cache miss, so a new prompt version is really measured.
        llm = CachedLLM(create_llm(args.llm), REPO_ROOT / ".cache" / "llm.jsonl")
    pipeline = RAGPipeline(index, complete=llm, top_k=args.top_k)

    cases = load_generation_cases(
        REPO_ROOT / "evaluation" / "generation_cases.jsonl",
        REPO_ROOT / "evaluation" / "questions.jsonl",
        REPO_ROOT / "evaluation" / "ground_truth.jsonl",
    )
    cases = [c for c in cases if c.type in args.types]
    report = evaluate_generation(cases, pipeline.answer)

    print(f"answerer={llm.name if llm else 'extractive'} embedder={index.embedder.name} top_k={args.top_k} cases={len(cases)}")
    for kind in args.types:
        group = report.of_type(kind)
        if group:
            print(f"  {kind:<12} n={len(group):<3} correct={sum(r.correct for r in group)}/{len(group)}")
    print("metrics:")
    for name, value in report.metrics().items():
        print(f"  {name:<32} {_fmt(value)}")
    print("gates:")
    for name, (value, threshold, passed) in report.gates().items():
        print(f"  {'PASS' if passed else 'FAIL'}  {name:<32} {_fmt(value)} (threshold {threshold})")
    if llm:
        tokens_in = sum((r.usage or {}).get("input_tokens", 0) for r in report.results)
        tokens_out = sum((r.usage or {}).get("output_tokens", 0) for r in report.results)
        print(f"LLM calls: {llm.misses} new, {llm.hits} from cache; new tokens: {tokens_in} in, {tokens_out} out")

    if args.show != "none":
        for r in report.results:
            if args.show == "all" or not r.correct:
                failed = [k for k, ok in r.checks.items() if not ok]
                mark = "ok  " if r.correct else "FAIL"
                print(f"\n{mark} {r.id} [{r.type}] {r.question}\n     -> {r.answer}\n     failed checks: {failed or '-'}")

    with args.output.open("w", encoding="utf-8") as handle:
        for r in report.results:
            handle.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")
    shown = args.output.resolve()
    print(f"\nper-case results written to {shown.relative_to(REPO_ROOT) if shown.is_relative_to(REPO_ROOT) else shown}")
    if args.gate and not report.passed:
        sys.exit(1)


if __name__ == "__main__":
    main()
