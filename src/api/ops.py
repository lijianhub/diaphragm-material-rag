"""Readers for the ops dashboard: recent feedback and tracked evaluation runs.

Both read append-only JSONL files written elsewhere (``POST /feedback`` and the evaluation
scripts with ``--track jsonl``). A damaged line is counted and skipped rather than failing
the whole page: a dashboard that breaks on one bad line is useless exactly when something
has gone wrong.
"""
from __future__ import annotations

import hashlib
import json
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union


def _read_jsonl(path: Union[str, Path], keep_last: int = 0) -> Tuple[List[dict], int, int]:
    """(records, total lines, malformed lines); with ``keep_last``, only the last n records."""
    path = Path(path)
    if not path.exists():
        return [], 0, 0
    records: Any = deque(maxlen=keep_last) if keep_last else []
    total = malformed = 0
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            total += 1
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                malformed += 1
                continue
            if isinstance(record, dict):
                records.append(record)
            else:
                malformed += 1
    return list(records), total, malformed


def read_feedback(path: Union[str, Path], limit: int = 50) -> Dict[str, Any]:
    """Totals over the whole file (they survive restarts, unlike the metrics counters) and
    the ``limit`` most recent ratings, newest first."""
    records, total, malformed = _read_jsonl(path)
    counts = {"up": 0, "down": 0}
    for record in records:
        if record.get("rating") in counts:
            counts[record["rating"]] += 1
    fields = ("ts", "rating", "comment", "answer_request_id")
    recent = [{k: r.get(k) for k in fields} for r in reversed(records[-limit:])]
    return {"total": total, "malformed": malformed, "counts": counts, "recent": recent}


def data_version(tags: Dict[str, str]) -> str:
    """A short ID of the evaluation data a run was measured on. Runs with different data
    versions are not comparable: a score change may come from the data, not the system."""
    parts = sorted((k, v) for k, v in tags.items() if k.startswith("data."))
    if not parts:
        return "unknown"
    return hashlib.sha256(json.dumps(parts).encode("utf-8")).hexdigest()[:8]


def read_runs(path: Union[str, Path], limit: int = 100) -> Dict[str, Any]:
    """Tracked runs, newest first, with their parameters, metrics and version tags."""
    records, total, malformed = _read_jsonl(path, keep_last=limit)
    runs = []
    for r in reversed(records):
        tags = r.get("tags") or {}
        params = r.get("params") or {}
        runs.append({
            "run_id": r.get("run_id"),
            "time": r.get("time"),
            "name": r.get("name"),
            "kind": params.get("kind", "unknown"),
            "params": params,
            "metrics": r.get("metrics") or {},
            "git_commit": (tags.get("git_commit") or "unknown")[:7],
            "git_dirty": tags.get("git_dirty") == "true",
            "data_version": data_version(tags),
            "gates": tags.get("gates"),
        })
    return {"total": total, "malformed": malformed, "runs": runs}
