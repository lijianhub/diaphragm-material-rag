"""Evaluate the agent: tools used, actions proposed (or not), behaviour after rejection,
injection resistance, answers and cost, on the frozen cases in evaluation/agent_cases.jsonl.

Usage:
    python scripts/evaluate_agent.py                                   # offline rule planner (baseline)
    python scripts/evaluate_agent.py --llm openai:<model>@low          # LLM planner (turns cached in .cache/)
    python scripts/evaluate_agent.py --llm openai:<model>@low --gate   # exit 1 if a gate fails
    python scripts/evaluate_agent.py --llm ... --track jsonl           # record the run (commit first)

Every case runs on a fresh agent with its own run store and report file in a temporary
folder; approval requests are answered by a simulated reviewer (the case's decision for the
requested lot, otherwise a rejection). See src/evaluation/agent.py for the scoring and gates.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config import load_env_file

load_env_file()  # local .env (git-ignored); real environment variables win

from agent import Agent, LLMPlanner, NcrStore, RulePlanner, RunStore, agent_prompt_version, default_tools
from chunking import RecursiveChunker
from embedding import create_embedder
from evaluation import benchmark_corpus
from evaluation.agent import CATEGORIES, evaluate_agent, load_agent_cases, result_rows
from generation import CachedLLM, create_llm
from ingestion.loader import Document, load_documents
from retrieval.index import SearchIndex
from tracking import TRACKER_KINDS, RunRecord, create_tracker, log_safely, version_tags

CASES = REPO_ROOT / "evaluation" / "agent_cases.jsonl"
ALLOYS = REPO_ROOT / "data" / "raw" / "alloy_overview.csv"


def _fmt(value: float) -> str:
    return "n/a" if isinstance(value, float) and math.isnan(value) else (f"{value:.3f}" if isinstance(value, float) and not value.is_integer() else f"{value:g}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--llm", default=None, help="planner model, e.g. openai:<model>@low; omit for the offline rule planner")
    parser.add_argument("--embedder", default="simple-hash")
    parser.add_argument("--categories", nargs="*", choices=CATEGORIES, default=list(CATEGORIES))
    parser.add_argument("--gate", action="store_true", help="exit with status 1 if any gate fails")
    parser.add_argument("--show", choices=("failures", "all", "none"), default="failures")
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "evaluation" / "agent_results.jsonl")
    parser.add_argument("--track", choices=TRACKER_KINDS, default="none")
    parser.add_argument("--run-name", default=None)
    args = parser.parse_args()

    corpus = benchmark_corpus(REPO_ROOT / "data" / "raw")
    documents = load_documents(corpus)
    embedder = create_embedder(args.embedder)

    def build_index(extra):
        index = SearchIndex(embedder=embedder, chunker=RecursiveChunker(chunk_size=300, overlap=50))
        index.add_documents(documents + [Document(source=d["source"], content=d["content"]) for d in extra])
        return index

    base_index = build_index([])
    llm = CachedLLM(create_llm(args.llm, timeout=30, max_retries=1), REPO_ROOT / ".cache" / "llm.jsonl") if args.llm else None
    planner = LLMPlanner(llm) if llm else RulePlanner()

    cases = [c for c in load_agent_cases(CASES) if c.category in args.categories]
    with tempfile.TemporaryDirectory(prefix="agent-eval-") as work:
        def make_agent(case):
            folder = Path(work) / case.id
            index = build_index(case.extra_documents) if case.extra_documents else base_index
            return Agent(planner, default_tools(index, ALLOYS, NcrStore(folder / "ncr.jsonl")), RunStore(folder / "runs.db"))

        report = evaluate_agent(cases, make_agent)

    metrics = report.metrics()
    print(f"planner={planner.name} prompt={agent_prompt_version()} embedder={embedder.name} cases={len(cases)}")
    for category in args.categories:
        group = report.of(category)
        if group:
            print(f"  {category:<13} n={len(group):<3} correct={sum(r.correct for r in group)}/{len(group)}")
    print("metrics:")
    for name, value in metrics.items():
        print(f"  {name:<26} {_fmt(value)}")
    print("gates:")
    for name, (value, threshold, passed) in report.gates().items():
        print(f"  {'PASS' if passed else 'FAIL'}  {name:<26} {_fmt(value)} (threshold {threshold})")
    if llm:
        print(f"LLM turns: {llm.misses} new, {llm.hits} from cache; new tokens: {metrics['tokens_in']:g} in, {metrics['tokens_out']:g} out")

    if args.show != "none":
        for r in report.results:
            if args.show == "all" or not r.correct:
                failed = [k for k, ok in r.checks.items() if not ok]
                tools = ", ".join(f"{t['tool'] or t['type']}:{t['status']}" for t in r.trace)
                print(f"\n{'ok  ' if r.correct else 'FAIL'} {r.id} [{r.category}] {r.question}\n     tools: {tools}")
                if r.proposals:
                    print(f"     proposals: {json.dumps(r.proposals, ensure_ascii=False)[:400]}")
                print(f"     -> {(r.answer or '')[:400]}\n     failed checks: {failed or '-'}")

    rows = result_rows(report)
    for row in rows:
        row.pop("trace")  # the trace has document text; the scored fields are enough to review
    with args.output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    shown = args.output.resolve()
    print(f"\nper-case results written to {shown.relative_to(REPO_ROOT) if shown.is_relative_to(REPO_ROOT) else shown}")

    if args.track != "none":
        params = {"kind": "agent", "planner": planner.name, "prompt_version": agent_prompt_version(), "embedder": embedder.name,
                  "categories": ",".join(args.categories), "cases": len(cases)}
        values = {k: v for k, v in metrics.items() if v == v}
        for category in args.categories:
            group = report.of(category)
            if group:
                values[f"{category}.correct"] = sum(r.correct for r in group) / len(group)
        gates = report.gates()
        values.update({f"gate.{name}": float(passed) for name, (_, _, passed) in gates.items()})
        values["gates_passed"] = float(sum(passed for _, _, passed in gates.values()))
        if llm:
            values.update({"llm_calls_new": llm.misses, "llm_calls_cached": llm.hits})
        tags = version_tags(REPO_ROOT, {"agent_cases": CASES}, corpus)
        tags["gates"] = "pass" if report.passed else "fail"
        name = args.run_name or f"agent {planner.name} {agent_prompt_version()}"
        run_id = log_safely(create_tracker(args.track, REPO_ROOT), RunRecord(name, {k: str(v) for k, v in params.items()}, values, tags, [str(args.output)]))
        if run_id:
            print(f"tracked run {run_id} with {args.track}" + (" (working tree has uncommitted changes)" if tags["git_dirty"] == "true" else ""))
    if args.gate and not report.passed:
        sys.exit(1)


if __name__ == "__main__":
    main()
