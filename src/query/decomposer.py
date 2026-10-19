from __future__ import annotations

import re
from typing import List

# "..., and what ...", "... and how ...": a conjunction followed by a word that starts a
# new question. "strength and ductility" does not match, because a noun follows "and".
_QUESTION_START = r"(?:what|how|why|which|where|when|who|whom|whose|is|are|was|were|does|do|did|can|could|should|will|would)\b"
_SPLIT = re.compile(rf",?\s+(?:and|but)\s+(?={_QUESTION_START})", re.IGNORECASE)


def decompose(question: str) -> List[str]:
    """Split a compound question into its parts, keeping the original first.

    "What does the supplier send, and what do inspectors do with it?" becomes
    [original, "What does the supplier send?", "what do inspectors do with it?"].
    Each part retrieves its own evidence; the original keeps the full context.
    A question with nothing to split is returned unchanged as a one-item list.
    """
    parts = [part.strip().rstrip(",") for part in _SPLIT.split(question.strip())]
    parts = [part for part in parts if part]
    if len(parts) < 2:
        return [question]
    return [question] + [part if part.endswith("?") else part + "?" for part in parts]
