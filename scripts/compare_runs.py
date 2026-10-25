"""Compare tracked evaluation runs side by side.

Usage:
    python scripts/compare_runs.py                       # MLflow runs (mlflow.db at the repo root)
    python scripts/compare_runs.py --source jsonl        # runs/runs.jsonl
    python scripts/compare_runs.py --kind generation     # only generation runs
    python scripts/compare_runs.py --metrics hit_at_1 mrr_at_1

Each row shows the run name, the metrics asked for, and the versions it was measured on
(git commit, uncommitted changes, dataset hash), so two numbers are only compared when
they were measured on the same data.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
# Always first, also after `pip install -e .` (which appends src/ at the end): otherwise this
# script's own folder wins and `import query` / `import agent` find the scripts of that name.
sys.path.insert(0, str(SRC_DIR))

from tracking import DEFAULT_EXPERIMENT, JsonlTracker, metric_name

DEFAULT_METRICS = {
    "retrieval": ["hit_at_1", "hit_at_3", "recall_at_3", "mrr_at_3", "paraphrase.hit_at_3"],
    "generation": ["answer_accuracy", "refusal_accuracy", "injection_resistance", "ambiguity_handling", "gates_passed"],
}


def _mlflow_rows(experiment: str):
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    import mlflow  # optional extra

    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI") or f"sqlite:///{(REPO_ROOT / 'mlflow.db').as_posix()}")
    for run in mlflow.search_runs(experiment_names=[experiment], output_format="list"):
        yield run.info.run_name, run.data.params, run.data.metrics, run.data.tags, run.info.start_time


def _jsonl_rows():
    for run in JsonlTracker(REPO_ROOT / "runs" / "runs.jsonl").runs():
        metrics = {metric_name(k): v for k, v in run["metrics"].items()}
        yield run["name"], run["params"], metrics, run["tags"], run["time"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", choices=("mlflow", "jsonl"), default="mlflow")
    parser.add_argument("--experiment", default=DEFAULT_EXPERIMENT)
    parser.add_argument("--kind", choices=("retrieval", "generation"), default=None, help="filter by run kind")
    parser.add_argument("--metrics", nargs="*", default=None, help="metric names (MLflow form, e.g. hit_at_3)")
    args = parser.parse_args()

    rows = list(_mlflow_rows(args.experiment) if args.source == "mlflow" else _jsonl_rows())
    kinds = [args.kind] if args.kind else ["retrieval", "generation"]
    for kind in kinds:
        selected = sorted((r for r in rows if r[1].get("kind") == kind), key=lambda r: (str(r[0]), str(r[4])))
        if not selected:
            continue
        names = args.metrics or DEFAULT_METRICS[kind]
        print(f"\n{kind} runs ({len(selected)})")
        print(f"  {'run':<44}" + "".join(f"{m:>22}" for m in names) + "   commit   dirty  questions")
        for name, params, metrics, tags, _ in selected:
            values = "".join(f"{metrics[m]:>22.3f}" if m in metrics else f"{'-':>22}" for m in names)
            data = tags.get("data.questions.sha256", "")[:8]
            print(f"  {str(name)[:44]:<44}{values}   {tags.get('git_commit', '')[:7]}  {tags.get('git_dirty', '?'):<5}  {data}")


if __name__ == "__main__":
    main()
