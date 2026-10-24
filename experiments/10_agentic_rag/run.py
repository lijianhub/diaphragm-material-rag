"""Offline demo of the agent's approval flow: no API key, no files outside a temp folder.

A scripted planner stands in for the model: it searches, proposes a nonconformance
report, and answers once the decision is known. The point is the control flow:
the report is not filed until a person approves, and a second approval is refused.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from agent import Agent, NcrStore, NotAwaitingApproval, RunStore, default_tools
from generation.llm import ToolCall, ToolTurn
from ingestion.loader import load_documents
from retrieval.index import SearchIndex


class ScriptedPlanner:
    """Plays the model's part: search, propose a report, then answer."""

    name = provider = "scripted"

    def next_turn(self, system, transcript, tools, run):
        tool_results = [e for e in transcript if e["role"] == "tool"]
        if not tool_results:
            return ToolTurn("", [ToolCall("c1", "search_documents", {"query": "tearing near a tight radius"})], [], self.name, "tool_use")
        if len(tool_results) == 1:
            report = {"lot": "L-302", "defect": "tearing near a tight radius",
                      "root_cause": "sheet stretched beyond its forming limit at the radius [1]",
                      "corrective_action": "larger radius, better lubrication, lower blank holder force"}
            return ToolTurn("", [ToolCall("c2", "raise_nonconformance_report", report)], [], self.name, "tool_use")
        return ToolTurn(f"Likely cause: the sheet was stretched beyond its forming limit [1]. {tool_results[-1]['content']}", [], [], self.name, "end_turn")


def main() -> None:
    work = Path(tempfile.mkdtemp(prefix="agent-demo-"))
    index = SearchIndex()
    index.sync(load_documents(REPO_ROOT / "data" / "raw" / "forming_defects_notes.txt"))
    ncr = NcrStore(work / "ncr.jsonl")
    agent = Agent(ScriptedPlanner(), default_tools(index, REPO_ROOT / "data" / "raw" / "alloy_overview.csv", ncr), RunStore(work / "agent.db"))

    run = agent.start("Lot L-302 tore near a tight radius. Find the cause and raise an NCR.")
    print(f"1. status: {run['status']}; proposed: {run['pending']['tool']} for lot {run['pending']['arguments']['lot']}")
    print(f"   reports filed so far: {len(ncr.list())}")

    run = agent.decide(run["id"], "approve", reviewer="demo-reviewer")
    print(f"2. approved -> status: {run['status']}; reports filed: {len(ncr.list())}")
    print(f"   answer: {run['answer']}")

    try:
        agent.decide(run["id"], "approve", reviewer="someone-else")
    except NotAwaitingApproval:
        print("3. a second approval is refused; still", len(ncr.list()), "report")


if __name__ == "__main__":
    main()
