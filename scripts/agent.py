"""Run the agent from the command line, and decide on actions waiting for approval.

The agent searches the documents, looks up alloys and can propose a nonconformance report
(NCR). Proposing is all it can do: the report is filed only after a person approves it,
here or on the ops dashboard (both use the same run store).

Usage:
    python scripts/agent.py "What causes edge burrs?"                       # offline rule planner
    python scripts/agent.py "Lot L-302 has edge burrs: raise an NCR." --llm openai:<model>@low
    python scripts/agent.py --pending                                       # runs waiting for a decision
    python scripts/agent.py --approve <run-id>
    python scripts/agent.py --reject <run-id> --reason "wrong lot"

Files (git-ignored): logs/agent.db (runs), logs/ncr.jsonl (filed reports). Needs data/index.
"""
from __future__ import annotations

import argparse
import getpass
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
# Always first, also after `pip install -e .` (which appends src/ at the end): otherwise this
# script's own folder wins and `import query` / `import agent` find the scripts of that name.
sys.path.insert(0, str(SRC_DIR))

from config import load_env_file

load_env_file()  # local .env (git-ignored); real environment variables win

from agent import Agent, Limits, LLMPlanner, NcrStore, RulePlanner, RunStore, default_tools
from retrieval.index import SearchIndex


def show(run: dict) -> None:
    for t in run["trace"]:
        what = t["tool"] or t["type"]
        args = f" {json.dumps(t['arguments'], ensure_ascii=False)}" if t.get("arguments") else ""
        print(f"  step {t['step']}: {what}{args} -> {t['status']} ({t['ms']:.0f} ms)")
    print(f"status: {run['status']}" + (f" ({run['stop_reason']})" if run["stop_reason"] else ""))
    if run["pending"]:
        p = run["pending"]
        print(f"waiting for approval: {p['tool']} {json.dumps(p['arguments'], ensure_ascii=False)}")
        print(f"  decide with: python scripts/agent.py --approve {run['id']}   (or --reject {run['id']} --reason ...)")
    if run["answer"]:
        print(f"\n{run['answer']}")
        for c in run["citations"]:
            print(f"  [{c['number']}] {c['label']}")
        if run["invalid_citations"]:
            print(f"  invalid citations: {run['invalid_citations']}")
    u = run["usage"]
    print(f"\nplanner={run['planner']} steps={run['steps']} tool_calls={run['tool_calls']} tokens in/out={u['input_tokens']}/{u['output_tokens']} active={run['active_ms']:.0f} ms run={run['id']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("question", nargs="?")
    parser.add_argument("--llm", default=os.environ.get("RAG_AGENT_LLM", ""), help="planner model, e.g. openai:<model>@low (default: offline rules)")
    parser.add_argument("--index", type=Path, default=REPO_ROOT / "data" / "index")
    parser.add_argument("--max-steps", type=int, default=Limits.max_steps)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--pending", action="store_true", help="list runs waiting for a decision")
    group.add_argument("--approve", metavar="RUN_ID")
    group.add_argument("--reject", metavar="RUN_ID")
    parser.add_argument("--reason", default="", help="why (required for --reject)")
    args = parser.parse_args()

    store = RunStore(REPO_ROOT / "logs" / "agent.db")
    if args.pending:
        for run in store.list(status="awaiting_approval"):
            print(f"{run['id']}  {run['pending']['tool']}  {json.dumps(run['pending']['arguments'], ensure_ascii=False)}")
        return
    if args.reject and not args.reason.strip():
        parser.error("--reject needs --reason")
    if not (args.question or args.approve or args.reject):
        parser.error("give a question, --pending, --approve or --reject")

    planner = RulePlanner()
    if args.llm:
        from generation import create_llm

        planner = LLMPlanner(create_llm(args.llm, timeout=30, max_retries=1))
    index = SearchIndex.load(args.index)
    agent = Agent(planner, default_tools(index, REPO_ROOT / "data" / "raw" / "alloy_overview.csv", NcrStore(REPO_ROOT / "logs" / "ncr.jsonl")),
                  store, limits=Limits(max_steps=args.max_steps))
    if args.question:
        show(agent.start(args.question))
    else:
        decision = "approve" if args.approve else "reject"
        show(agent.decide(args.approve or args.reject, decision, reviewer=f"cli:{getpass.getuser()}", reason=args.reason))


if __name__ == "__main__":
    main()
