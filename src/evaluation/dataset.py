from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Union


@dataclass
class EvalExample:
    id: str
    question: str
    answer: str
    evidence: List[str]


def _read_jsonl(path: Path) -> List[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_dataset(questions_path: Union[str, Path], ground_truth_path: Union[str, Path]) -> List[EvalExample]:
    """Join questions and ground truth on ``id``, keeping question order."""
    truths: Dict[str, dict] = {row["id"]: row for row in _read_jsonl(Path(ground_truth_path))}

    examples: List[EvalExample] = []
    for row in _read_jsonl(Path(questions_path)):
        truth = truths.get(row["id"])
        if truth is None:
            raise ValueError(f"No ground truth for question {row['id']!r}")
        if not truth.get("evidence"):
            raise ValueError(f"Ground truth for {row['id']!r} has no evidence phrases")
        examples.append(
            EvalExample(
                id=row["id"],
                question=row["question"],
                answer=truth.get("answer", ""),
                evidence=list(truth["evidence"]),
            )
        )
    return examples
