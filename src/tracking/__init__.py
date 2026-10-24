"""Experiment tracking: record what was run (configuration and versions) and what it scored.

Every evaluation run becomes a ``RunRecord``: params (the configuration), metrics (the
scores), tags (versions of code, data and prompt) and artifacts (per-question results).
Trackers write it somewhere:

- ``NullTracker``: records nothing (the default, so scripts behave as before);
- ``JsonlTracker``: appends one JSON line per run to a local file, no dependencies;
- ``MlflowTracker``: logs to MLflow (optional ``[tracking]`` extra, imported lazily).

``create_tracker("none" | "jsonl" | "mlflow")`` builds one. Tracking must never break an
evaluation, so scripts call ``log_safely`` and get a warning instead of an exception.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Union

TRACKER_KINDS = ("none", "jsonl", "mlflow")
DEFAULT_EXPERIMENT = "diaphragm-material-rag"


@dataclass
class RunRecord:
    name: str
    params: Dict[str, str] = field(default_factory=dict)
    metrics: Dict[str, float] = field(default_factory=dict)
    tags: Dict[str, str] = field(default_factory=dict)
    artifacts: List[str] = field(default_factory=list)


_METRIC_NAME = re.compile(r"[^A-Za-z0-9_\-. /]")


def metric_name(name: str) -> str:
    """MLflow-safe metric name: 'hit@3' -> 'hit_at_3'; other symbols become '_'."""
    return _METRIC_NAME.sub("_", name.replace("@", "_at_"))


def flatten(prefix: str, values: Dict[str, Any]) -> Dict[str, Any]:
    """{'a': 1, 'b': {'c': 2}} -> {'prefix.a': 1, 'prefix.b.c': 2}."""
    out: Dict[str, Any] = {}
    for key, value in values.items():
        name = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            out.update(flatten(name, value))
        else:
            out[name] = value
    return out


# ---- versions -----------------------------------------------------------------


def file_sha256(path: Union[str, Path]) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fingerprint(paths: Iterable[Union[str, Path]], root: Optional[Union[str, Path]] = None) -> str:
    """One hash over several files: their relative names and contents, in sorted order.
    Adding, removing, renaming or editing any file changes it."""
    digest = hashlib.sha256()
    base = Path(root) if root else None
    for path in sorted(Path(p) for p in paths):
        name = path.relative_to(base).as_posix() if base else path.name
        digest.update(name.encode("utf-8") + b"\0" + file_sha256(path).encode("ascii") + b"\n")
    return digest.hexdigest()


def git_state(repo: Union[str, Path]) -> Dict[str, str]:
    """{'git_commit': <sha or 'unknown'>, 'git_dirty': 'true'|'false'|'unknown'}.

    A dirty tree means the scores may come from uncommitted code or data, so the run is
    flagged rather than refused."""
    def git(*args: str) -> Optional[str]:
        try:
            done = subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True, timeout=10)
        except (OSError, subprocess.SubprocessError):
            return None
        return done.stdout.strip() if done.returncode == 0 else None

    commit = git("rev-parse", "HEAD")
    status = git("status", "--porcelain")
    return {
        "git_commit": commit or "unknown",
        "git_dirty": "unknown" if status is None else ("true" if status else "false"),
    }


def version_tags(repo: Union[str, Path], datasets: Dict[str, Union[str, Path]], corpus: Iterable[Union[str, Path]]) -> Dict[str, str]:
    """Tags that identify exactly what a score was measured on."""
    corpus = list(corpus)
    tags = dict(git_state(repo))
    for name, path in datasets.items():
        tags[f"data.{name}.sha256"] = file_sha256(path)[:16]
    tags["data.corpus.fingerprint"] = fingerprint(corpus)[:16]
    tags["data.corpus.files"] = str(len(corpus))
    return tags


# ---- trackers -----------------------------------------------------------------


class NullTracker:
    name = "none"

    def log_run(self, record: RunRecord) -> Optional[str]:
        return None


class JsonlTracker:
    """Appends one JSON object per run: zero dependencies, readable, diff-able."""

    name = "jsonl"

    def __init__(self, path: Union[str, Path]):
        self.path = Path(path)

    def log_run(self, record: RunRecord) -> str:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        run_id = hashlib.sha256(json.dumps(asdict(record), sort_keys=True).encode("utf-8")).hexdigest()[:12]
        entry = {"run_id": run_id, "time": datetime.now(timezone.utc).isoformat(timespec="seconds"), **asdict(record)}
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
        return run_id

    def runs(self) -> List[dict]:
        if not self.path.exists():
            return []
        with self.path.open(encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]


class MlflowTracker:
    """Logs a run to MLflow: params, metrics, tags and artifact files.

    ``client`` is the ``mlflow`` module (injected in tests). The tracking URI comes from
    the argument, else MLflow's own default or ``MLFLOW_TRACKING_URI``.
    """

    name = "mlflow"

    def __init__(
        self,
        experiment: str = DEFAULT_EXPERIMENT,
        tracking_uri: Optional[str] = None,
        artifact_location: Optional[str] = None,
        client: Any = None,
    ):
        if client is None:
            try:
                import mlflow as client  # optional extra
            except ImportError as exc:  # pragma: no cover - needs the optional extra
                raise ImportError("MlflowTracker needs: pip install -e .[tracking]") from exc
        self.mlflow = client
        if tracking_uri:
            self.mlflow.set_tracking_uri(tracking_uri)
        # A new experiment gets an explicit artifact folder; otherwise MLflow picks one
        # relative to the current directory, and artifacts scatter across launch folders.
        if artifact_location and self.mlflow.get_experiment_by_name(experiment) is None:
            self.mlflow.create_experiment(experiment, artifact_location=artifact_location)
        self.mlflow.set_experiment(experiment)

    def log_run(self, record: RunRecord) -> str:
        with self.mlflow.start_run(run_name=record.name) as run:
            # MLflow limits param values; long values are cut rather than failing the run.
            self.mlflow.log_params({k: str(v)[:500] for k, v in record.params.items()})
            self.mlflow.log_metrics({metric_name(k): float(v) for k, v in record.metrics.items()})
            self.mlflow.set_tags(record.tags)
            for path in record.artifacts:
                if Path(path).exists():
                    self.mlflow.log_artifact(str(path))
            return run.info.run_id


def create_tracker(kind: str, repo_root: Union[str, Path] = ".", **kwargs: Any):
    """Build a tracker whose files live under ``repo_root`` (not the current directory):
    ``runs/runs.jsonl`` for jsonl; ``mlflow.db`` and ``mlartifacts/`` for MLflow, unless
    ``MLFLOW_TRACKING_URI`` points elsewhere (e.g. a shared tracking server)."""
    root = Path(repo_root).resolve()
    if kind == "none":
        return NullTracker()
    if kind == "jsonl":
        return JsonlTracker(kwargs.get("jsonl_path", root / "runs" / "runs.jsonl"))
    if kind == "mlflow":
        uri = kwargs.pop("tracking_uri", None) or os.environ.get("MLFLOW_TRACKING_URI") or f"sqlite:///{(root / 'mlflow.db').as_posix()}"
        kwargs.setdefault("artifact_location", (root / "mlartifacts").as_uri())
        return MlflowTracker(tracking_uri=uri, **kwargs)
    raise ValueError(f"unknown tracker {kind!r}; expected one of {TRACKER_KINDS}")


def log_safely(tracker, record: RunRecord) -> Optional[str]:
    """Log a run, but never let tracking fail an evaluation: print a warning instead."""
    try:
        return tracker.log_run(record)
    except Exception as exc:  # tracking is observability, not part of the result
        print(f"warning: tracking failed ({type(exc).__name__}: {exc}); the evaluation results above are unaffected")
        return None


__all__ = [
    "DEFAULT_EXPERIMENT",
    "JsonlTracker",
    "MlflowTracker",
    "NullTracker",
    "RunRecord",
    "TRACKER_KINDS",
    "create_tracker",
    "file_sha256",
    "fingerprint",
    "flatten",
    "git_state",
    "log_safely",
    "metric_name",
    "version_tags",
]
