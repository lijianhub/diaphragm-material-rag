from __future__ import annotations

import logging
import re
from collections import Counter
from typing import Callable, Dict, List, Sequence

from tokenization import stem, tokenize

logger = logging.getLogger(__name__)


def expansion_terms(question: str, passages: Sequence[str], idf: Callable[[str], float], n_terms: int = 5) -> List[str]:
    """Pseudo-relevance feedback (a light RM3): pick terms to add to the question.

    Assume the top first-pass passages are relevant. A term scores high when it is
    frequent in those passages (term frequency normalised by passage length) and
    rare in the corpus (IDF). Terms already in the question are skipped, and terms
    are compared by stem but returned as their most common surface form, so the
    expanded query still reads as words for the embedding model.
    """
    if not passages or n_terms <= 0:
        return []
    question_stems = set(tokenize(question, stem_tokens=True))
    weights: Counter = Counter()
    surfaces: Dict[str, Counter] = {}
    for passage in passages:
        tokens = [t for t in tokenize(passage) if t.isalpha() and len(t) >= 3]
        if not tokens:
            continue
        for token, count in Counter(tokens).items():
            key = stem(token)
            if key in question_stems:
                continue
            weights[key] += count / len(tokens) * idf(token)
            surfaces.setdefault(key, Counter())[token] += count
    return [surfaces[key].most_common(1)[0][0] for key, _ in weights.most_common(n_terms)]


MULTI_QUERY_PROMPT = """You rewrite questions for a search engine over technical engineering documents.
Write {n} alternative phrasings of the question below. Use the different words a specification,
procedure or datasheet might use for the same thing. Return one phrasing per line, with no
numbering and no commentary.

Question: {question}"""

HYDE_PROMPT = """Write a short passage (two or three sentences) from a technical engineering document
that would answer the question below. Write it as the document would, without restating the question.

Question: {question}"""

_LIST_MARKER = re.compile(r"^\s*(?:\d+[.)]|[-*•])\s*")


class LLMQueryRewriter:
    """Use a language model to produce extra search queries.

    ``mode="multi-query"``: ``n_variants`` paraphrases; each is searched and the
    rankings are fused. ``mode="hyde"`` (Hypothetical Document Embeddings): the model
    writes a plausible answer passage, which is searched for alongside the question.
    Answers in document language often match better than questions do.

    ``complete`` is any text-in, text-out function. If it fails, the original
    question is used alone, so retrieval degrades gracefully instead of failing.
    """

    MODES = ("multi-query", "hyde")

    def __init__(self, complete: Callable[[str], str], n_variants: int = 3, mode: str = "multi-query"):
        if mode not in self.MODES:
            raise ValueError(f"mode must be one of {self.MODES}")
        self.complete = complete
        self.n_variants = n_variants
        self.mode = mode

    def __call__(self, question: str) -> List[str]:
        template = HYDE_PROMPT if self.mode == "hyde" else MULTI_QUERY_PROMPT
        try:
            reply = self.complete(template.format(question=question, n=self.n_variants))
        except Exception as exc:  # any LLM failure: fall back to the plain question
            logger.warning("query rewriting failed, using the original question: %s", exc)
            return [question]

        if self.mode == "hyde":
            passage = reply.strip()
            return [question, passage] if passage else [question]

        variants: List[str] = []
        seen = {question.strip().lower()}
        for line in reply.splitlines():
            text = _LIST_MARKER.sub("", line).strip()
            if text and text.lower() not in seen:
                seen.add(text.lower())
                variants.append(text)
            if len(variants) >= self.n_variants:
                break
        return [question] + variants
