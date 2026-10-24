"""The ingestion report: what the last indexing run loaded, skipped and failed.

``scripts/index.py`` prints this, but console output is gone once the run ends. Writing it
next to the index lets the ops dashboard show failed files (with the reason) and duplicates
until the next run, which is when someone can fix them.
"""
from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Union

from .loader import IngestResult

REPORT_FILE = "ingest_report.json"


def build_report(
    result: IngestResult,
    sync_stats: Any = None,
    seconds: Optional[float] = None,
    portable: Callable[[str], str] = str,
) -> Dict[str, Any]:
    """``portable`` turns absolute paths into repository-relative ones, so the report does
    not reveal the machine's directory layout and stays the same on every clone."""
    stats = asdict(sync_stats) if is_dataclass(sync_stats) else dict(sync_stats or {})
    return {
        "time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "loaded": len(result.documents),
        "failed": [{"source": portable(f.source), "reason": f.reason} for f in result.failures],
        "skipped": [portable(s) for s in result.skipped],
        "duplicates": [{"source": portable(d), "same_as": portable(o)} for d, o in result.duplicates],
        "sync": stats,
        "seconds": None if seconds is None else round(seconds, 2),
    }


def write_report(report: Dict[str, Any], index_dir: Union[str, Path]) -> Path:
    path = Path(index_dir) / REPORT_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def read_report(index_dir: Union[str, Path]) -> Optional[Dict[str, Any]]:
    path = Path(index_dir) / REPORT_FILE
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))
