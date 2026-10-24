"""Run the HTTP API.

Usage:
    python scripts/index.py                     # build data/index first
    python scripts/serve.py                     # http://127.0.0.1:8000, docs at /docs
    python scripts/serve.py --port 8080 --host 0.0.0.0

Configuration comes from the environment (or the git-ignored .env):
    RAG_INDEX_DIR        index directory (default: data/index)
    RAG_LLM              e.g. openai:<model>@low; unset → offline extractive answers
    RAG_LLM_TIMEOUT      seconds per LLM call (default 30)
    RAG_LLM_MAX_RETRIES  SDK retries per LLM call (default 1)
    RAG_FEEDBACK_PATH    feedback file (default: logs/feedback.jsonl)
    RAG_RUNS_PATH        tracked evaluation runs for the ops dashboard (default: runs/runs.jsonl)

Logs are one JSON object per line on stdout, ready for a log collector.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config import load_env_file

load_env_file()  # local .env (git-ignored); real environment variables win


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    os.environ.setdefault("RAG_INDEX_DIR", str(REPO_ROOT / "data" / "index"))
    os.environ.setdefault("RAG_FEEDBACK_PATH", str(REPO_ROOT / "logs" / "feedback.jsonl"))
    os.environ.setdefault("RAG_RUNS_PATH", str(REPO_ROOT / "runs" / "runs.jsonl"))
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)

    import uvicorn

    # The app writes its own JSON request log, so uvicorn's access log is switched off.
    uvicorn.run("api.main:create_app", factory=True, host=args.host, port=args.port, access_log=False, log_level="warning")


if __name__ == "__main__":
    main()
